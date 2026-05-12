from __future__ import annotations

import pytest

from common.guards import (
    DiskGuardError,
    MemoryGuardError,
    check_disk,
    check_memory,
    snapshot,
)


def test_snapshot_returns_positive_numbers(tmp_path):
    snap = snapshot(str(tmp_path))
    assert snap.rss_gb > 0
    assert snap.available_gb > 0
    assert snap.free_disk_gb >= 0


def test_check_memory_under_threshold_passes():
    snap = check_memory(threshold_gb=10_000.0)
    assert snap.rss_gb < 10_000.0


def test_check_memory_trips_when_below_current_rss():
    with pytest.raises(MemoryGuardError):
        check_memory(threshold_gb=0.0001)


def test_check_disk_under_threshold_trips(tmp_path):
    with pytest.raises(DiskGuardError):
        check_disk(str(tmp_path), min_free_gb=10_000.0)


def test_check_disk_passes_with_normal_threshold(tmp_path):
    snap = check_disk(str(tmp_path), min_free_gb=0.0)
    assert snap.free_disk_gb >= 0
