"""Download PitVQA SAGE, normalize a CPU-scale subset, and write JSONL manifests.

Resumable: each image is written with a .complete sentinel; re-runs only fetch missing ones.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from common.config import (
    DATASET_ID,
    DEFAULT_TEST_SAMPLES,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
    PATHS,
    SEED,
)
from common.logging_setup import setup_logging
from data.normalize import MalformedRow, NormalizedRow, extract_qa_from_messages

log = logging.getLogger("data.prepare")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--data-dir", default=PATHS["data_processed"])
    p.add_argument(
        "--max-rows-scan",
        type=int,
        default=None,
        help="Cap how many rows to scan per split before picking the subset; default no cap.",
    )
    return p.parse_args()


def image_path_for(images_dir: Path, sample_id: str) -> Path:
    return images_dir / f"{sample_id}.jpg"


def sentinel_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".complete")


def save_image_atomic(image, target: Path) -> None:
    sentinel = sentinel_path_for(target)
    if sentinel.exists() and target.exists():
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    image.convert("RGB").save(tmp, format="JPEG", quality=95)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    sentinel.touch()


def normalize_row(row: dict, idx: int, split: str) -> NormalizedRow | None:
    try:
        question, answer = extract_qa_from_messages(row.get("messages") or [])
    except MalformedRow as exc:
        log.warning("skipping row %d (%s): %s", idx, split, exc)
        return None
    video_id = str(row.get("video_id")) if row.get("video_id") is not None else None
    frame_id = str(row.get("frame_id")) if row.get("frame_id") is not None else None
    qa_type = str(row.get("qa_type") or "unknown")
    sample_id = f"{split}-{idx:06d}"
    return NormalizedRow(
        id=sample_id,
        image_path="",
        question=question,
        answer=answer,
        qa_type=qa_type,
        split=split,
        video_id=video_id,
        frame_id=frame_id,
    )


def process_split(
    ds_split, split: str, sample_count: int, images_dir: Path, manifest_path: Path, seed: int
) -> int:
    from datasets import Dataset

    if not isinstance(ds_split, Dataset):
        raise RuntimeError("expected datasets.Dataset, got something else")

    if sample_count > len(ds_split):
        log.warning(
            "requested %d %s samples but split only has %d; truncating",
            sample_count,
            split,
            len(ds_split),
        )
        sample_count = len(ds_split)

    shuffled = ds_split.shuffle(seed=seed)
    selected = shuffled.select(range(sample_count))

    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(manifest_path, "w") as fh:
        for idx, row in enumerate(selected):
            normalized = normalize_row(row, idx, split)
            if normalized is None:
                continue
            image = row.get("image")
            if image is None:
                log.warning("row %d has no image; skipping", idx)
                continue
            target_image = image_path_for(images_dir, normalized.id)
            save_image_atomic(image, target_image)
            normalized.image_path = str(target_image)
            fh.write(json.dumps(normalized.to_jsonl()) + "\n")
            written += 1
    log.info("wrote %d rows to %s", written, manifest_path)
    return written


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed; pip install -r requirements.txt")
        return 2

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"

    log.info("loading %s", args.dataset)
    ds = load_dataset(args.dataset, download_mode="reuse_dataset_if_exists")

    splits = {
        "train": args.train_samples,
        "validation": args.val_samples,
        "test": args.test_samples,
    }
    output_names = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}

    total = 0
    written_per_split: dict[str, int] = {}
    for split_name, count in splits.items():
        if split_name not in ds:
            log.error("dataset missing split %s; available: %s", split_name, list(ds.keys()))
            return 3
        manifest_path = data_dir / output_names[split_name]
        n = process_split(
            ds[split_name],
            split_name if split_name != "validation" else "val",
            count,
            images_dir,
            manifest_path,
            args.seed,
        )
        written_per_split[split_name] = n
        total += n

    log.info("done. total rows written: %d (%s)", total, written_per_split)

    seen_ids: set[str] = set()
    for split_name in splits:
        manifest = data_dir / output_names[split_name]
        with open(manifest) as fh:
            for line in fh:
                obj = json.loads(line)
                if obj["id"] in seen_ids:
                    log.error("duplicate id across splits: %s", obj["id"])
                    return 4
                seen_ids.add(obj["id"])
    log.info("split disjointness verified across %d rows", len(seen_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
