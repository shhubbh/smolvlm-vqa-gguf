"""Test checkpoint discovery prefers the latest valid checkpoint and ignores .tmp."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from train.checkpoint import discover_latest


def make_ckpt(parent: Path, name: str, valid: bool) -> Path:
    p = parent / name
    p.mkdir(parents=True)
    (p / "adapter").mkdir()
    if valid:
        (p / "VALID").touch()
    return p


def test_discover_latest_picks_newest_valid(tmp_path):
    older = make_ckpt(tmp_path, "step-000001", valid=True)
    time.sleep(0.01)
    newer = make_ckpt(tmp_path, "step-000002", valid=True)
    chosen = discover_latest(tmp_path)
    assert chosen == newer


def test_discover_latest_skips_invalid(tmp_path):
    valid_ckpt = make_ckpt(tmp_path, "step-001", valid=True)
    time.sleep(0.01)
    make_ckpt(tmp_path, "step-002", valid=False)  # no VALID sentinel
    chosen = discover_latest(tmp_path)
    assert chosen == valid_ckpt


def test_discover_latest_none_when_empty(tmp_path):
    assert discover_latest(tmp_path) is None


def test_discover_latest_ignores_tmp_suffix(tmp_path):
    valid_ckpt = make_ckpt(tmp_path, "step-001", valid=True)
    tmp_dir = tmp_path / "step-002.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "adapter").mkdir()
    chosen = discover_latest(tmp_path)
    assert chosen == valid_ckpt
