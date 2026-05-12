"""Slow smoke test: runs a 1-step training pass on a tiny synthetic manifest.

Skipped by default. Run with `RUN_SLOW=1 pytest tests/test_training_smoke.py`.
Downloads the SmolVLM-500M model on first run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

if os.environ.get("RUN_SLOW") != "1":
    pytest.skip("set RUN_SLOW=1 to run the smoke test", allow_module_level=True)


@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory):
    from PIL import Image

    root = tmp_path_factory.mktemp("synthetic")
    images = root / "images"
    images.mkdir()
    rows = []
    for i in range(4):
        img_path = images / f"s{i}.jpg"
        Image.new("RGB", (64, 64), (i * 30, 100, 200)).save(img_path, "JPEG")
        rows.append({
            "id": f"smoke-{i:02d}",
            "image_path": str(img_path),
            "question": "What color is this image?",
            "answer": "Blue.",
            "qa_type": "color",
            "split": "train",
            "video_id": None,
            "frame_id": None,
        })
    train = root / "train.jsonl"
    val = root / "val.jsonl"
    with open(train, "w") as fh:
        for r in rows[:2]:
            fh.write(json.dumps(r) + "\n")
    with open(val, "w") as fh:
        for r in rows[2:]:
            fh.write(json.dumps(r) + "\n")
    return train, val, root


def test_one_optimizer_step(synthetic_manifest, tmp_path):
    train, val, _ = synthetic_manifest
    cmd = [
        sys.executable,
        "train/finetune_lora.py",
        "--train", str(train),
        "--val", str(val),
        "--output", str(tmp_path / "adapter"),
        "--run-id", "smoke-test",
        "--max-steps", "1",
        "--grad-accum", "2",
        "--resume", "no",
    ]
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert (tmp_path / "adapter").exists()
