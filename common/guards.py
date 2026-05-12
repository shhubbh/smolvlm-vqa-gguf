"""Belt-and-suspenders memory and disk guards. Primary OOM defense is cgroups v2."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import psutil

from common.config import DISK_GUARD_FREE_GB, MEMORY_GUARD_GB


class MemoryGuardError(RuntimeError):
    pass


class DiskGuardError(RuntimeError):
    pass


@dataclass
class ResourceSnapshot:
    rss_gb: float
    available_gb: float
    free_disk_gb: float


def snapshot(path: str = ".") -> ResourceSnapshot:
    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    free_disk_gb = shutil.disk_usage(path).free / (1024**3)
    return ResourceSnapshot(rss_gb=rss_gb, available_gb=available_gb, free_disk_gb=free_disk_gb)


def check_memory(threshold_gb: float = MEMORY_GUARD_GB) -> ResourceSnapshot:
    snap = snapshot()
    if snap.rss_gb > threshold_gb:
        raise MemoryGuardError(
            f"RSS {snap.rss_gb:.2f}GB exceeded guard {threshold_gb:.2f}GB"
        )
    return snap


def check_disk(path: str = ".", min_free_gb: float = DISK_GUARD_FREE_GB) -> ResourceSnapshot:
    snap = snapshot(path)
    if snap.free_disk_gb < min_free_gb:
        raise DiskGuardError(
            f"Free disk {snap.free_disk_gb:.2f}GB below minimum {min_free_gb:.2f}GB at {path}"
        )
    return snap
