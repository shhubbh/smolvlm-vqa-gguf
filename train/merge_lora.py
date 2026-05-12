"""Merge a saved LoRA adapter into the base SmolVLM model and write a standalone HF checkpoint."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from peft import PeftModel

from common.config import MODEL_ID, MODEL_REVISION, PATHS
from common.logging_setup import setup_logging

log = logging.getLogger("train.merge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=PATHS["adapter"])
    p.add_argument("--output", default=PATHS["merged_hf"])
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        log.error("adapter dir %s missing", adapter_path)
        return 2

    from transformers import AutoModelForVision2Seq, AutoProcessor

    log.info("loading base model %s", MODEL_ID)
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    log.info("loading adapter from %s", adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    log.info("merging LoRA weights into base")
    merged = peft_model.merge_and_unload()

    output_dir = Path(args.output)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info("saving merged HF checkpoint to %s", output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    processor.save_pretrained(str(output_dir))
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
