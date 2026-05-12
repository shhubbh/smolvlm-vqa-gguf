"""Atomic checkpoint save/restore for LoRA training resume."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TrainingState:
    step: int = 0
    sample_cursor: int = 0
    best_val: float | None = None
    config_hash: str | None = None


def _rng_state() -> dict:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().tolist(),
    }


def _load_rng(payload: dict) -> None:
    import numpy as np
    import torch

    try:
        random.setstate(tuple(payload["python"]))
    except Exception:
        log.warning("could not restore python RNG state")
    try:
        np.random.set_state(payload["numpy"])
    except Exception:
        log.warning("could not restore numpy RNG state")
    try:
        torch.random.set_rng_state(torch.tensor(payload["torch"], dtype=torch.uint8))
    except Exception:
        log.warning("could not restore torch RNG state")


def save_checkpoint(
    *,
    peft_model,
    optimizer,
    state: TrainingState,
    output_dir: str | Path,
) -> Path:
    import torch
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_dir.with_suffix(output_dir.suffix + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    peft_model.save_pretrained(str(tmp / "adapter"))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")

    payload = {
        "step": state.step,
        "sample_cursor": state.sample_cursor,
        "best_val": state.best_val,
        "config_hash": state.config_hash,
        "rng": _rng_state(),
    }
    with open(tmp / "state.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    (tmp / "VALID").touch()

    if output_dir.exists():
        backup = output_dir.with_suffix(output_dir.suffix + ".prev")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(output_dir, backup)
        try:
            os.rename(tmp, output_dir)
        except Exception:
            os.rename(backup, output_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(tmp, output_dir)

    log.info("saved checkpoint to %s (step=%d)", output_dir, state.step)
    return output_dir


def discover_latest(runs_dir: str | Path) -> Path | None:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    candidates = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "VALID").exists() and (child / "adapter").is_dir():
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    peft_model,
    optimizer,
) -> TrainingState:
    import torch
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "VALID").exists():
        raise RuntimeError(f"checkpoint at {checkpoint_dir} is missing VALID marker")

    if isinstance(peft_model, PeftModel):
        peft_model.load_adapter(str(checkpoint_dir / "adapter"), adapter_name="default")
    else:
        raise RuntimeError("peft_model must be a PeftModel for checkpoint load")

    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu"))

    with open(checkpoint_dir / "state.json") as fh:
        payload = json.load(fh)
    _load_rng(payload.get("rng", {}))
    state = TrainingState(
        step=int(payload.get("step", 0)),
        sample_cursor=int(payload.get("sample_cursor", 0)),
        best_val=payload.get("best_val"),
        config_hash=payload.get("config_hash"),
    )
    log.info("loaded checkpoint %s (step=%d)", checkpoint_dir, state.step)
    return state
