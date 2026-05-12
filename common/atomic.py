"""Atomic file and directory write helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_json(path: str | os.PathLike, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_replace_dir(src_dir: str | os.PathLike, dest_dir: str | os.PathLike) -> None:
    src = Path(src_dir)
    dest = Path(dest_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        backup = dest.with_name(dest.name + ".prev")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(dest, backup)
        try:
            os.rename(src, dest)
        except Exception:
            os.rename(backup, dest)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(src, dest)


def write_done_marker(path: str | os.PathLike, payload: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload or {"status": "done"})


def is_done(path: str | os.PathLike) -> bool:
    return Path(path).is_file()
