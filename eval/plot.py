"""Generate a single-figure summary chart from results/comparison.csv.

Three panels: BERTScore F1 (quality), peak RAM (memory), mean latency (speed).
Saves to results/comparison.png. No external deps beyond matplotlib (already
pulled in by bert-score).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import PATHS
from common.logging_setup import setup_logging

log = logging.getLogger("eval.plot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=PATHS["results_csv"])
    p.add_argument("--out", default="results/comparison.png")
    p.add_argument("--title", default="SmolVLM-500M VQA: 3-Variant Comparison (n=10)")
    return p.parse_args()


def load_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


COLORS = {
    "base-f16": "#1f77b4",
    "finetuned-q8_0": "#ff7f0e",
    "finetuned-q4_k_m": "#2ca02c",
}


def bar_panel(ax, variants, values, fmt, ylabel, title):
    colors = [COLORS.get(v, "#888888") for v in variants]
    bars = ax.bar(variants, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=15)
    ymax = max(values) * 1.15 if values else 1.0
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> int:
    setup_logging()
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        log.error("results CSV not found: %s", csv_path)
        return 2

    rows = load_rows(csv_path)
    if not rows:
        log.error("results CSV has no rows")
        return 3

    variants = [r["variant"] for r in rows]
    f1 = [float(r["bertscore_f1"]) for r in rows]
    ram_mb = [float(r["peak_ram_mb"]) for r in rows]
    latency_s = [float(r["latency_ms_mean"]) / 1000.0 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    bar_panel(
        axes[0],
        variants,
        f1,
        lambda v: f"{v:.3f}",
        "BERTScore F1",
        "Quality (higher = better)",
    )
    bar_panel(
        axes[1],
        variants,
        ram_mb,
        lambda v: f"{v:.0f} MB",
        "Peak RAM (MB)",
        "Memory (lower = better)",
    )
    bar_panel(
        axes[2],
        variants,
        latency_s,
        lambda v: f"{v:.1f}s",
        "Mean latency / query (s)",
        "Speed (lower = better)",
    )

    base_ram = ram_mb[0] if ram_mb else 1.0
    if len(ram_mb) >= 3 and base_ram > 0:
        savings = 100.0 * (1.0 - ram_mb[-1] / base_ram)
        axes[1].annotate(
            f"-{savings:.0f}% vs F16",
            xy=(len(ram_mb) - 1, ram_mb[-1]),
            xytext=(len(ram_mb) - 1, ram_mb[-1] - base_ram * 0.18),
            ha="center",
            fontsize=9,
            color="#2ca02c",
            fontweight="bold",
        )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
