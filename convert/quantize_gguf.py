"""Quantize a text GGUF to a target format with llama-quantize.

After quantization, run a one-shot decode through llama-mtmd-cli as a sanity check.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import PATHS
from common.logging_setup import setup_logging

log = logging.getLogger("convert.quantize")

SUPPORTED_QUANT = ("Q8_0", "Q4_K_M")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to F16 text GGUF")
    p.add_argument("--quant", choices=SUPPORTED_QUANT, required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--mmproj", default=None, help="optional mmproj GGUF for the smoke decode")
    p.add_argument("--smoke-image", default=None, help="optional image path for smoke decode")
    p.add_argument("--llama-cpp-dir", default=PATHS["llama_cpp_dir"])
    return p.parse_args()


def llama_bin(llama_cpp_dir: Path, name: str) -> Path:
    return llama_cpp_dir / "build" / "bin" / name


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, timeout=timeout)


def main() -> int:
    setup_logging()
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        log.error("input GGUF missing: %s", input_path)
        return 2

    llama_dir = Path(args.llama_cpp_dir)
    quantize_exe = llama_bin(llama_dir, "llama-quantize")
    if not quantize_exe.is_file():
        log.error("llama-quantize not built at %s", quantize_exe)
        return 2

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem.replace("text-f16", f"text-{args.quant.lower()}") + ".gguf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run([str(quantize_exe), str(input_path), str(output_path), args.quant])

    if not output_path.is_file() or output_path.stat().st_size == 0:
        log.error("quantized GGUF missing or empty: %s", output_path)
        return 3
    in_size = input_path.stat().st_size
    out_size = output_path.stat().st_size
    log.info("quantized %s -> %s (%.1f MB -> %.1f MB, %.0f%%)",
             input_path.name, output_path.name,
             in_size / (1024**2), out_size / (1024**2),
             100 * out_size / max(in_size, 1))
    if out_size >= in_size:
        log.warning("quantized file is not smaller than F16 input; check params")

    if args.mmproj and args.smoke_image:
        mtmd_exe = llama_bin(llama_dir, "llama-mtmd-cli")
        if mtmd_exe.is_file():
            log.info("running smoke decode with %s", mtmd_exe.name)
            try:
                run([
                    str(mtmd_exe),
                    "-m", str(output_path),
                    "--mmproj", str(args.mmproj),
                    "--image", str(args.smoke_image),
                    "-p", "What is shown in the image?",
                    "--n-predict", "16",
                    "--temp", "0",
                    "--seed", "42",
                ], timeout=120)
            except subprocess.CalledProcessError as exc:
                log.error("smoke decode failed: %s", exc)
                return 4
        else:
            log.warning("llama-mtmd-cli not built; skipping smoke decode")
    else:
        log.info("no mmproj/smoke-image provided; skipping smoke decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
