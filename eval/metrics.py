"""Metrics: exact match, BERTScore F1, latency percentiles, CSV/MD writers."""

from __future__ import annotations

import csv
import logging
import statistics
from pathlib import Path
from typing import Iterable

from common.config import BERTSCORE_MODEL, BERTSCORE_NUM_LAYERS
from data.normalize import exact_match

log = logging.getLogger(__name__)


CSV_HEADER = (
    "variant",
    "quantization",
    "samples",
    "exact_match",
    "bertscore_f1",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "peak_ram_mb",
    "failures",
    "text_gguf",
    "mmproj_gguf",
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_exact_match(predictions: Iterable[str], references: Iterable[str]) -> float:
    preds = list(predictions)
    refs = list(references)
    if not preds:
        return 0.0
    hits = sum(1 for p, r in zip(preds, refs) if exact_match(p, r))
    return hits / len(preds)


def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = BERTSCORE_MODEL,
    num_layers: int = BERTSCORE_NUM_LAYERS,
) -> float:
    if not predictions:
        return 0.0
    try:
        from bert_score import score
    except ImportError:
        log.warning("bert-score not installed; returning 0.0")
        return 0.0
    _p, _r, f1 = score(
        predictions,
        references,
        model_type=model_type,
        num_layers=num_layers,
        lang="en",
        rescale_with_baseline=False,
        device="cpu",
        verbose=False,
    )
    return float(f1.mean().item())


def write_comparison_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def write_comparison_md(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(CSV_HEADER)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def summarize_latency(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    return (
        statistics.fmean(latencies),
        percentile(latencies, 50.0),
        percentile(latencies, 95.0),
    )
