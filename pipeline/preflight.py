"""Preflight checks. Refuses to start the pipeline if any check fails."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.atomic import atomic_write_json
from common.bf16 import cpu_supports_bf16
from common.config import (
    DATASET_ID,
    PATHS,
    PREFLIGHT_FREE_DISK_GB,
    PREFLIGHT_FREE_RAM_GB,
)
from common.guards import snapshot
from common.logging_setup import setup_logging

log = logging.getLogger("pipeline.preflight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run a tiny model-forward check")
    p.add_argument("--llama-cpp-dir", default=PATHS["llama_cpp_dir"])
    return p.parse_args()


def check_cpu_only() -> tuple[bool, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        return False, f"CUDA_VISIBLE_DEVICES={visible!r}; expected ''"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return False, f"torch sees {torch.cuda.device_count()} CUDA devices"
    except Exception:
        pass
    return True, "ok"


def check_resources() -> tuple[bool, str]:
    snap = snapshot()
    if snap.available_gb < PREFLIGHT_FREE_RAM_GB:
        return False, f"available RAM {snap.available_gb:.1f}GB < {PREFLIGHT_FREE_RAM_GB}GB"
    if snap.free_disk_gb < PREFLIGHT_FREE_DISK_GB:
        return False, f"free disk {snap.free_disk_gb:.1f}GB < {PREFLIGHT_FREE_DISK_GB}GB"
    return True, f"ram_avail={snap.available_gb:.1f}GB disk_free={snap.free_disk_gb:.1f}GB"


def check_hf_reachable() -> tuple[bool, str]:
    try:
        socket.gethostbyname("huggingface.co")
    except OSError as exc:
        return False, f"DNS lookup failed: {exc}"
    return True, "huggingface.co resolves"


def check_imports() -> tuple[bool, str]:
    required = ["torch", "transformers", "peft", "datasets", "PIL", "psutil", "httpx"]
    missing = []
    for name in required:
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name} ({exc})")
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "all required imports succeed"


def check_dataset_decode() -> tuple[bool, str]:
    train_manifest = Path(PATHS["train_manifest"])
    if not train_manifest.exists():
        return True, "no train manifest yet; will be created by data prep"
    import json

    try:
        from PIL import Image
    except ImportError as exc:
        return False, f"PIL import failed: {exc}"
    with open(train_manifest) as fh:
        first = fh.readline().strip()
    if not first:
        return False, "train manifest is empty"
    row = json.loads(first)
    image_path = row.get("image_path")
    if not image_path or not Path(image_path).exists():
        return False, f"image referenced by first manifest row not found: {image_path}"
    Image.open(image_path).convert("RGB")
    return True, f"decoded {image_path}"


def check_bf16_status() -> tuple[bool, str]:
    if cpu_supports_bf16():
        return True, "AVX512_BF16 available"
    return True, "AVX512_BF16 NOT available; FP32 fallback will be used"


def check_llama_cpp(llama_cpp_dir: Path) -> tuple[bool, str]:
    if not llama_cpp_dir.exists():
        return True, "llama.cpp not cloned yet; will be built in pipeline"
    binaries = ["llama-server", "llama-quantize", "llama-mtmd-cli"]
    missing = []
    for name in binaries:
        path = llama_cpp_dir / "build" / "bin" / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            subprocess.run([str(path), "--version"], check=False, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError):
            missing.append(f"{name} (cannot exec)")
    if missing:
        return True, "llama.cpp partially built; will rebuild: " + ", ".join(missing)
    return True, "llama.cpp binaries OK"


def check_atomic_writes() -> tuple[bool, str]:
    target = Path("runs") / "preflight_atomic.json"
    payload = {"hello": "world"}
    try:
        atomic_write_json(target, payload)
    except Exception as exc:
        return False, f"atomic write failed: {exc}"
    finally:
        if target.exists():
            target.unlink()
    return True, "atomic write round-trip ok"


def check_required_disk_for_pipeline() -> tuple[bool, str]:
    snap = snapshot()
    return True, f"disk free {snap.free_disk_gb:.1f}GB (informational)"


def run_smoke_forward() -> tuple[bool, str]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from common.config import MODEL_ID
    except Exception as exc:
        return False, f"smoke imports failed: {exc}"

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to("cpu")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        return False, f"failed to load model: {exc}"

    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, config)
    peft_model.train()

    from PIL import Image
    image = Image.new("RGB", (128, 128), color=(127, 127, 127))
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Gray square."}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    max_side = getattr(processor.image_processor, "max_image_size", None)
    if isinstance(max_side, dict):
        max_side = max_side.get("longest_edge")
    extra = {"size": {"longest_edge": max_side}} if isinstance(max_side, int) else {}
    inputs = processor(text=text, images=[image], return_tensors="pt", **extra)
    labels = inputs.input_ids.clone()
    pad = processor.tokenizer.pad_token_id
    if pad is not None:
        labels[labels == pad] = -100
    out = peft_model(**inputs, labels=labels)
    out.loss.backward()
    return True, f"smoke forward ok (loss={out.loss.item():.4f})"


def main() -> int:
    setup_logging()
    args = parse_args()
    Path("runs").mkdir(exist_ok=True)

    checks = [
        ("cpu_only", check_cpu_only),
        ("imports", check_imports),
        ("resources", check_resources),
        ("hf_reachable", check_hf_reachable),
        ("bf16_status", check_bf16_status),
        ("dataset_decode", check_dataset_decode),
        ("llama_cpp", lambda: check_llama_cpp(Path(args.llama_cpp_dir))),
        ("atomic_writes", check_atomic_writes),
        ("disk_summary", check_required_disk_for_pipeline),
    ]
    if args.smoke:
        checks.append(("smoke_forward", run_smoke_forward))

    failures: list[str] = []
    summary: dict[str, dict] = {}
    for name, fn in checks:
        try:
            ok, message = fn()
        except Exception as exc:
            ok, message = False, f"check raised: {exc}"
        log.info("%s %s: %s", "OK " if ok else "FAIL", name, message)
        summary[name] = {"ok": ok, "message": message}
        if not ok:
            failures.append(name)

    atomic_write_json("runs/preflight.json", summary)

    if failures:
        log.error("preflight FAILED: %s", failures)
        return 1
    log.info("preflight PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
