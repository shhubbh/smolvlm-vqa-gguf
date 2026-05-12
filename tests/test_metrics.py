from __future__ import annotations

import csv
import json

from eval.metrics import (
    CSV_HEADER,
    compute_exact_match,
    percentile,
    summarize_latency,
    write_comparison_csv,
    write_comparison_md,
)
from eval.prompts import EVAL_USER_TEMPLATE, render_user


def test_exact_match_score():
    em = compute_exact_match(["yes.", "No"], ["YES", "no"])
    assert em == 1.0
    em = compute_exact_match(["forceps", "yes"], ["scalpel", "no"])
    assert em == 0.0


def test_percentile_known_values():
    assert percentile([10, 20, 30, 40, 50], 50.0) == 30
    assert percentile([], 50.0) == 0.0
    assert percentile([7], 95.0) == 7


def test_summarize_latency_shapes():
    mean, p50, p95 = summarize_latency([100.0, 200.0, 300.0])
    assert 100.0 <= mean <= 300.0
    assert 100.0 <= p50 <= 300.0
    assert 100.0 <= p95 <= 300.0


def test_csv_schema(tmp_path):
    rows = [
        {
            "variant": "base-f16",
            "quantization": "F16",
            "samples": 2,
            "exact_match": 0.5,
            "bertscore_f1": 0.7,
            "latency_ms_mean": 120.0,
            "latency_ms_p50": 110.0,
            "latency_ms_p95": 200.0,
            "peak_ram_mb": 1024.0,
            "failures": 0,
            "text_gguf": "a.gguf",
            "mmproj_gguf": "b.gguf",
        }
    ]
    csv_path = tmp_path / "c.csv"
    write_comparison_csv(rows, csv_path)
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        body = list(reader)
    assert tuple(header) == CSV_HEADER
    assert body[0]["variant"] == "base-f16"


def test_markdown_writes_table(tmp_path):
    rows = [
        {
            "variant": "x",
            "quantization": "Q4_K_M",
            "samples": 1,
            "exact_match": 0.0,
            "bertscore_f1": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "peak_ram_mb": 0.0,
            "failures": 0,
            "text_gguf": "",
            "mmproj_gguf": "",
        }
    ]
    md_path = tmp_path / "m.md"
    write_comparison_md(rows, md_path)
    body = md_path.read_text()
    assert "| variant |" in body
    assert "x" in body


def test_prompt_template_is_pinned_and_used_via_helper():
    assert "{question}" in EVAL_USER_TEMPLATE
    assert "What?" in render_user("What?")
