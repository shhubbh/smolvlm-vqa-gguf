"""Evaluate one variant against a held-out test manifest, return metrics dict."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from eval.client import query_chat_completion
from eval.metrics import (
    compute_bertscore_f1,
    compute_exact_match,
    summarize_latency,
)
from eval.server import llama_server

log = logging.getLogger(__name__)


def evaluate_variant(
    *,
    variant: str,
    quantization: str,
    text_gguf: str | Path,
    mmproj_gguf: str | Path,
    test_manifest: str | Path,
    threads: int,
) -> dict[str, Any]:
    rows: list[dict] = []
    with open(test_manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    predictions: list[str] = []
    references: list[str] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    per_type_hits: dict[str, list[bool]] = defaultdict(list)

    with llama_server(
        text_gguf=text_gguf,
        mmproj_gguf=mmproj_gguf,
        threads=threads,
    ) as server:
        base_url = server["base_url"]
        rss_holder = server["rss"]
        for row in rows:
            try:
                pred, latency_ms = query_chat_completion(
                    base_url, row["image_path"], row["question"]
                )
            except Exception as exc:
                failures.append({"id": row["id"], "error": repr(exc)})
                continue
            predictions.append(pred.strip())
            references.append(row["answer"])
            latencies.append(latency_ms)
            from data.normalize import exact_match as em
            per_type_hits[row.get("qa_type", "unknown")].append(em(pred, row["answer"]))
        peak_rss_mb = rss_holder.get("peak_rss_mb", 0.0)

    em_score = compute_exact_match(predictions, references)
    f1_score = compute_bertscore_f1(predictions, references) if predictions else 0.0
    mean_ms, p50_ms, p95_ms = summarize_latency(latencies)

    per_type = {
        t: sum(hits) / len(hits) if hits else 0.0 for t, hits in per_type_hits.items()
    }

    return {
        "variant": variant,
        "quantization": quantization,
        "samples": len(predictions),
        "exact_match": em_score,
        "bertscore_f1": f1_score,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50_ms,
        "latency_ms_p95": p95_ms,
        "peak_ram_mb": peak_rss_mb,
        "failures": len(failures),
        "text_gguf": str(text_gguf),
        "mmproj_gguf": str(mmproj_gguf),
        "per_qa_type": per_type,
        "failure_samples": failures[:10],
    }
