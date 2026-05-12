"""Run all 3 evaluation variants sequentially and write comparison CSV/MD."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import NUM_THREADS, PATHS, VARIANTS
from common.logging_setup import setup_logging
from eval.metrics import write_comparison_csv, write_comparison_md
from eval.run_variant import evaluate_variant

log = logging.getLogger("pipeline.compare")


VARIANT_SPECS = {
    "base-f16": {
        "quantization": "F16",
        "text_gguf": f"{PATHS['gguf_base']}/text-f16.gguf",
        "mmproj_gguf": f"{PATHS['gguf_base']}/mmproj-f16.gguf",
    },
    "finetuned-q8_0": {
        "quantization": "Q8_0",
        "text_gguf": f"{PATHS['gguf_finetuned']}/text-q8_0.gguf",
        "mmproj_gguf": f"{PATHS['gguf_finetuned']}/mmproj-f16.gguf",
    },
    "finetuned-q4_k_m": {
        "quantization": "Q4_K_M",
        "text_gguf": f"{PATHS['gguf_finetuned']}/text-q4_k_m.gguf",
        "mmproj_gguf": f"{PATHS['gguf_finetuned']}/mmproj-f16.gguf",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default=PATHS["test_manifest"])
    p.add_argument("--threads", type=int, default=NUM_THREADS)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--csv", default=PATHS["results_csv"])
    p.add_argument("--md", default=PATHS["results_md"])
    return p.parse_args()


def print_terminal_table(rows: list[dict]) -> None:
    if not rows:
        print("(no results)")
        return
    columns = ["variant", "samples", "exact_match", "bertscore_f1", "latency_ms_mean", "peak_ram_mb"]
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(_fmt(row.get(c, "")).ljust(widths[c]) for c in columns))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    setup_logging()
    args = parse_args()

    if not Path(args.test).exists():
        log.error("test manifest %s missing", args.test)
        return 2

    results = []
    for variant in args.variants:
        if variant not in VARIANT_SPECS:
            log.error("unknown variant %s; valid: %s", variant, list(VARIANT_SPECS))
            return 3
        spec = VARIANT_SPECS[variant]
        text_gguf = Path(spec["text_gguf"])
        mmproj_gguf = Path(spec["mmproj_gguf"])
        for path in (text_gguf, mmproj_gguf):
            if not path.is_file():
                log.error("variant %s missing artifact: %s", variant, path)
                return 4
        log.info("evaluating %s", variant)
        metrics = evaluate_variant(
            variant=variant,
            quantization=spec["quantization"],
            text_gguf=text_gguf,
            mmproj_gguf=mmproj_gguf,
            test_manifest=args.test,
            threads=args.threads,
        )
        results.append(metrics)
        per_variant_path = Path("results") / f"{variant}.json"
        per_variant_path.parent.mkdir(parents=True, exist_ok=True)
        per_variant_path.write_text(json.dumps(metrics, indent=2, default=str))

    write_comparison_csv(results, args.csv)
    write_comparison_md(results, args.md)
    print_terminal_table(results)
    log.info("wrote %s and %s", args.csv, args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
