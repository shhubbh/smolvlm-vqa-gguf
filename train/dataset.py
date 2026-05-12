"""JSONL-backed VQA dataset that yields processed model inputs for SmolVLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class VQAManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.rows: list[dict[str, Any]] = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "id": row["id"],
            "image": image,
            "question": row["question"],
            "answer": row["answer"],
        }
