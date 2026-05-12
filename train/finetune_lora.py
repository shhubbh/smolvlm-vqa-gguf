"""Custom CPU LoRA fine-tune loop for SmolVLM-500M-Instruct.

Resumable via `--resume auto`. Checkpoints every CHECKPOINT_EVERY_STEPS optimizer steps or
CHECKPOINT_EVERY_SECONDS wall-clock, whichever first.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.bf16 import choose_compute_dtype
from common.config import (
    CHECKPOINT_EVERY_SECONDS,
    CHECKPOINT_EVERY_STEPS,
    GRAD_ACCUM_STEPS,
    LR,
    MAX_OPTIMIZER_STEPS,
    MEMORY_GUARD_GB,
    NUM_THREADS,
    PATHS,
    SEED,
)
from common.guards import MemoryGuardError, check_memory
from common.hashing import hash_dict
from common.logging_setup import setup_logging
from common.seed import seed_everything
from common.status import StatusWriter
from train.checkpoint import (
    TrainingState,
    discover_latest,
    load_checkpoint,
    save_checkpoint,
)
from train.collate import build_training_inputs, to_model_kwargs
from train.dataset import VQAManifestDataset
from train.model_loader import attach_lora, load_base_model_and_processor

log = logging.getLogger("train.finetune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=PATHS["train_manifest"])
    p.add_argument("--val", default=PATHS["val_manifest"])
    p.add_argument("--output", default=PATHS["adapter"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-steps", type=int, default=MAX_OPTIMIZER_STEPS)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", default="auto", choices=("auto", "no"))
    return p.parse_args()


def config_signature(args: argparse.Namespace) -> dict:
    return {
        "train": str(args.train),
        "val": str(args.val),
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
    }


def make_run_dir(run_id: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_id or f"train-{timestamp}"
    path = Path(PATHS["runs"]) / name
    path.mkdir(parents=True, exist_ok=True)
    latest = Path(PATHS["runs"]) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        pass
    return path


def evaluate(peft_model, processor, val_dataset, dtype: torch.dtype) -> float:
    peft_model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for sample in val_dataset:
            batch = build_training_inputs(
                processor, sample["image"], sample["question"], sample["answer"]
            )
            with torch.autocast(device_type="cpu", dtype=dtype, enabled=dtype != torch.float32):
                outputs = peft_model(**to_model_kwargs(batch))
            total += float(outputs.loss.item())
            count += 1
    peft_model.train()
    return total / max(count, 1)


def main() -> int:
    setup_logging()
    args = parse_args()

    if not Path(args.train).exists():
        log.error("train manifest %s missing; run data/prepare_dataset.py first", args.train)
        return 2
    if not Path(args.val).exists():
        log.error("val manifest %s missing; run data/prepare_dataset.py first", args.val)
        return 2

    seed_everything(args.seed)
    torch.set_num_threads(NUM_THREADS)

    run_dir = make_run_dir(args.run_id)
    setup_logging(str(run_dir / "train.log"))
    status = StatusWriter(run_dir)
    status.update(stage="load_model")

    dtype = choose_compute_dtype()
    log.info("loading base model %s with dtype=%s", "SmolVLM-500M-Instruct", dtype)
    base_model, processor = load_base_model_and_processor(dtype=dtype)
    peft_model = attach_lora(base_model)
    peft_model.train()

    train_data = VQAManifestDataset(args.train)
    val_data = VQAManifestDataset(args.val)
    log.info("train=%d val=%d", len(train_data), len(val_data))

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    state = TrainingState(config_hash=hash_dict(config_signature(args)))

    if args.resume == "auto":
        latest = discover_latest(Path(PATHS["runs"]) / "checkpoints")
        if latest is not None:
            try:
                resumed = load_checkpoint(latest, peft_model=peft_model, optimizer=optimizer)
                if resumed.config_hash != state.config_hash:
                    log.warning(
                        "config hash drift (%s -> %s); ignoring checkpoint",
                        resumed.config_hash,
                        state.config_hash,
                    )
                else:
                    state = resumed
                    log.info("resumed from step %d, sample_cursor %d", state.step, state.sample_cursor)
            except Exception as exc:
                log.warning("checkpoint load failed (%s); starting fresh", exc)

    checkpoint_root = Path(PATHS["runs"]) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    last_checkpoint_time = time.monotonic()
    micro_step_idx = state.sample_cursor
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_count = 0

    status.update(stage="train", step=state.step)
    try:
        while state.step < args.max_steps:
            if micro_step_idx >= len(train_data):
                micro_step_idx = 0
            sample = train_data[micro_step_idx]
            micro_step_idx += 1

            batch = build_training_inputs(
                processor, sample["image"], sample["question"], sample["answer"]
            )
            with torch.autocast(device_type="cpu", dtype=dtype, enabled=dtype != torch.float32):
                outputs = peft_model(**to_model_kwargs(batch))
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accumulated_loss += float(loss.item()) * args.grad_accum
            accumulated_count += 1

            if accumulated_count >= args.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                state.sample_cursor = micro_step_idx
                mean_loss = accumulated_loss / accumulated_count
                accumulated_loss = 0.0
                accumulated_count = 0

                try:
                    check_memory(MEMORY_GUARD_GB)
                except MemoryGuardError as exc:
                    status.fail(
                        str(exc),
                        resume_command="python train/finetune_lora.py --resume auto",
                    )
                    log.error("memory guard tripped: %s", exc)
                    return 5

                status.update(stage="train", step=state.step, loss=round(mean_loss, 4))
                log.info("step=%d loss=%.4f", state.step, mean_loss)

                now = time.monotonic()
                if (
                    state.step % CHECKPOINT_EVERY_STEPS == 0
                    or now - last_checkpoint_time >= CHECKPOINT_EVERY_SECONDS
                ):
                    checkpoint_dir = checkpoint_root / f"step-{state.step:06d}"
                    save_checkpoint(
                        peft_model=peft_model,
                        optimizer=optimizer,
                        state=state,
                        output_dir=checkpoint_dir,
                    )
                    status.update(last_checkpoint=str(checkpoint_dir))
                    last_checkpoint_time = now

        val_loss = evaluate(peft_model, processor, val_data, dtype)
        state.best_val = (
            val_loss if state.best_val is None else min(state.best_val, val_loss)
        )
        status.update(stage="validate", val_metric=round(val_loss, 4))
        log.info("validation loss = %.4f", val_loss)

        final_dir = Path(args.output)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(final_dir))
        log.info("saved final adapter to %s", final_dir)

        final_checkpoint = checkpoint_root / f"step-{state.step:06d}-final"
        save_checkpoint(
            peft_model=peft_model,
            optimizer=optimizer,
            state=state,
            output_dir=final_checkpoint,
        )
        status.update(stage="complete", last_checkpoint=str(final_checkpoint))
        return 0
    except Exception as exc:
        status.fail(repr(exc), resume_command="python train/finetune_lora.py --resume auto")
        log.exception("training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
