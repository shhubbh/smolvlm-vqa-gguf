"""End-to-end resumable runner. Each stage writes a `.done` marker on success."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from common.config import (
    DEFAULT_TEST_SAMPLES,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
    LLAMA_CPP_TAG,
    NUM_THREADS,
    PATHS,
    SEED,
)
from common.logging_setup import setup_logging
from common.status import StatusWriter
from pipeline.state import (
    compute_signature,
    invalidate,
    is_done,
    load_recorded,
    mark_done,
    stages_to_invalidate,
    write_recorded,
)

log = logging.getLogger("pipeline.run_all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    p.add_argument("--skip-build-llama", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    log.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd)


def stage(runs_dir: Path, name: str, fn) -> None:
    if is_done(runs_dir, name):
        log.info("[%s] already done; skipping", name)
        return
    log.info("[%s] running", name)
    rc = fn()
    if rc != 0:
        raise RuntimeError(f"stage {name} failed with rc={rc}")
    mark_done(runs_dir, name)
    log.info("[%s] done", name)


def main() -> int:
    setup_logging("runs/latest.log")
    args = parse_args()
    runs_dir = Path(PATHS["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(runs_dir / "run_all")

    config = {
        "seed": args.seed,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
    }
    new_signature = compute_signature(config=config, llama_cpp_tag=LLAMA_CPP_TAG, runs_dir=runs_dir)
    old_signature = load_recorded(runs_dir)
    if old_signature is not None and args.resume:
        invalidated = stages_to_invalidate(old_signature, new_signature)
        if invalidated:
            log.warning("invalidating %d stage marker(s) due to signature drift: %s",
                        len(invalidated), invalidated)
            invalidate(runs_dir, invalidated)
    write_recorded(runs_dir, new_signature)

    try:
        stage(runs_dir, "data_prepare", lambda: run([
            sys.executable, "data/prepare_dataset.py",
            "--train-samples", str(args.train_samples),
            "--val-samples", str(args.val_samples),
            "--test-samples", str(args.test_samples),
            "--seed", str(args.seed),
        ]))
        status.update(stage="preflight")
        stage(runs_dir, "preflight", lambda: run([
            sys.executable, "pipeline/preflight.py",
        ]))
        status.update(stage="train")
        stage(runs_dir, "train", lambda: run([
            sys.executable, "train/finetune_lora.py", "--resume", "auto",
        ]))
        status.update(stage="merge")
        stage(runs_dir, "merge", lambda: run([
            sys.executable, "train/merge_lora.py",
        ]))
        if not args.skip_build_llama:
            status.update(stage="build_llama_cpp")
            stage(runs_dir, "build_llama_cpp", lambda: run([
                sys.executable, "convert/build_llama_cpp.py", "--jobs", str(NUM_THREADS),
            ]))
        status.update(stage="convert_base")
        stage(runs_dir, "convert_base", lambda: run([
            sys.executable, "convert/convert_to_gguf.py",
            "--model-dir", _resolve_base_hf_dir(),
            "--variant", "base",
        ]))
        status.update(stage="convert_finetuned")
        stage(runs_dir, "convert_finetuned", lambda: run([
            sys.executable, "convert/convert_to_gguf.py",
            "--model-dir", PATHS["merged_hf"],
            "--variant", "finetuned",
        ]))
        finetuned_f16 = f"{PATHS['gguf_finetuned']}/text-f16.gguf"
        mmproj_ft = f"{PATHS['gguf_finetuned']}/mmproj-f16.gguf"
        smoke_image = _first_train_image()
        status.update(stage="quantize_q8")
        stage(runs_dir, "quantize_q8", lambda: run([
            sys.executable, "convert/quantize_gguf.py",
            "--input", finetuned_f16, "--quant", "Q8_0",
            *(("--mmproj", mmproj_ft, "--smoke-image", smoke_image) if smoke_image else ()),
        ]))
        status.update(stage="quantize_q4")
        stage(runs_dir, "quantize_q4", lambda: run([
            sys.executable, "convert/quantize_gguf.py",
            "--input", finetuned_f16, "--quant", "Q4_K_M",
            *(("--mmproj", mmproj_ft, "--smoke-image", smoke_image) if smoke_image else ()),
        ]))
        status.update(stage="comparison")
        stage(runs_dir, "comparison", lambda: run([
            sys.executable, "pipeline/run_comparison.py",
            "--threads", str(NUM_THREADS),
        ]))
        status.update(stage="complete")
        log.info("pipeline complete")
        return 0
    except Exception as exc:
        status.fail(repr(exc), resume_command="python pipeline/run_all.py --resume")
        log.exception("pipeline failed: %s", exc)
        return 1


def _resolve_base_hf_dir() -> str:
    """Use HuggingFace cache snapshot of the base model.

    convert_hf_to_gguf.py accepts a Hub repo ID directly in recent llama.cpp; for older
    tags we may need to snapshot_download first. Try both.
    """
    from common.config import MODEL_ID
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL_ID)
        return path
    except Exception as exc:
        log.warning("snapshot_download failed (%s); falling back to repo ID", exc)
        return MODEL_ID


def _first_train_image() -> str | None:
    import json
    manifest = Path(PATHS["train_manifest"])
    if not manifest.exists():
        return None
    with open(manifest) as fh:
        line = fh.readline().strip()
    if not line:
        return None
    row = json.loads(line)
    return row.get("image_path")


if __name__ == "__main__":
    sys.exit(main())
