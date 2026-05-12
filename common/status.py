"""Per-run status.json writer used by all pipeline stages."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.atomic import atomic_write_json
from common.guards import snapshot


@dataclass
class StatusFields:
    stage: str = ""
    step: int = 0
    loss: float | None = None
    val_metric: float | None = None
    rss_gb: float = 0.0
    available_gb: float = 0.0
    free_disk_gb: float = 0.0
    elapsed_seconds: float = 0.0
    last_checkpoint: str | None = None
    last_error: str | None = None
    resume_command: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class StatusWriter:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "status.json"
        self._start = time.monotonic()
        self.state = StatusFields()

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self.state, key):
                self.state.extra[key] = value
            else:
                setattr(self.state, key, value)
        snap = snapshot(str(self.run_dir))
        self.state.rss_gb = round(snap.rss_gb, 3)
        self.state.available_gb = round(snap.available_gb, 3)
        self.state.free_disk_gb = round(snap.free_disk_gb, 3)
        self.state.elapsed_seconds = round(time.monotonic() - self._start, 2)
        atomic_write_json(self.path, asdict(self.state))

    def fail(self, message: str, resume_command: str | None = None) -> None:
        self.update(last_error=message, resume_command=resume_command)
