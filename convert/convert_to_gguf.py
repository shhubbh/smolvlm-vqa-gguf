"""Convert an HF SmolVLM checkpoint to F16 GGUF (text) and F16 mmproj using llama.cpp."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from common.config import PATHS
from common.logging_setup import setup_logging

log = logging.getLogger("convert.gguf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="HF model directory")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--llama-cpp-dir", default=PATHS["llama_cpp_dir"])
    p.add_argument("--variant", choices=("base", "finetuned"), required=True)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    setup_logging()
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        log.error("model dir %s missing", model_dir)
        return 2

    llama_dir = Path(args.llama_cpp_dir)
    converter = llama_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        log.error("converter script not found at %s", converter)
        return 2

    output_dir = Path(args.output_dir or (PATHS["gguf_base"] if args.variant == "base" else PATHS["gguf_finetuned"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    text_out = output_dir / "text-f16.gguf"
    mmproj_out = output_dir / "mmproj-f16.gguf"

    run([
        sys.executable,
        str(converter),
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(text_out),
    ])

    run([
        sys.executable,
        str(converter),
        str(model_dir),
        "--mmproj",
        "--outtype", "f16",
        "--outfile", str(mmproj_out),
    ])

    for path in (text_out, mmproj_out):
        if not path.is_file() or path.stat().st_size == 0:
            log.error("expected GGUF artifact missing or empty: %s", path)
            return 3

    log.info("wrote %s (%.1f MB) and %s (%.1f MB)",
             text_out, text_out.stat().st_size / (1024**2),
             mmproj_out, mmproj_out.stat().st_size / (1024**2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
