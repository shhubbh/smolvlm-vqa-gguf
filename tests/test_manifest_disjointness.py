"""Synthetic manifest fixtures: verify split disjointness check logic in isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def write_manifest(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i in ids:
            fh.write(json.dumps({"id": i, "image_path": str(path.with_name(f"{i}.jpg")), "question": "q", "answer": "a", "qa_type": "x", "split": "train", "video_id": None, "frame_id": None}) + "\n")


def check_disjoint(*manifests: Path) -> tuple[bool, set[str]]:
    seen: set[str] = set()
    dup: set[str] = set()
    for path in manifests:
        with open(path) as fh:
            for line in fh:
                obj = json.loads(line)
                if obj["id"] in seen:
                    dup.add(obj["id"])
                seen.add(obj["id"])
    return not dup, dup


def test_disjoint_when_unique(tmp_path):
    a = tmp_path / "train.jsonl"
    b = tmp_path / "val.jsonl"
    write_manifest(a, ["s1", "s2"])
    write_manifest(b, ["s3", "s4"])
    ok, dups = check_disjoint(a, b)
    assert ok
    assert dups == set()


def test_overlap_detected(tmp_path):
    a = tmp_path / "train.jsonl"
    b = tmp_path / "val.jsonl"
    write_manifest(a, ["s1", "s2"])
    write_manifest(b, ["s2", "s3"])
    ok, dups = check_disjoint(a, b)
    assert not ok
    assert dups == {"s2"}
