"""pipeline_state.json — guards stage `.done` markers against config/script/dataset drift."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.atomic import atomic_write_json
from common.hashing import hash_dict, hash_text_file

log = logging.getLogger(__name__)


@dataclass
class PipelineSignature:
    config_hash: str
    train_script_hash: str
    eval_script_hash: str
    dataset_manifest_hash: str
    llama_cpp_tag: str


def file_or_empty_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    return hash_text_file(p)


def compute_signature(*, config: dict[str, Any], llama_cpp_tag: str, runs_dir: str | Path) -> PipelineSignature:
    train_hash = file_or_empty_hash("train/finetune_lora.py")
    eval_hash = file_or_empty_hash("eval/run_variant.py")
    manifest_paths = [
        "data/processed/train.jsonl",
        "data/processed/val.jsonl",
        "data/processed/test.jsonl",
    ]
    manifest_hash = hash_dict({p: file_or_empty_hash(p) for p in manifest_paths})
    return PipelineSignature(
        config_hash=hash_dict(config),
        train_script_hash=train_hash,
        eval_script_hash=eval_hash,
        dataset_manifest_hash=manifest_hash,
        llama_cpp_tag=llama_cpp_tag,
    )


STAGE_DEPS = {
    "data_prepare": (),
    "preflight": ("data_prepare",),
    "train": ("data_prepare", "preflight"),
    "merge": ("train",),
    "build_llama_cpp": ("preflight",),
    "convert_base": ("build_llama_cpp",),
    "convert_finetuned": ("merge", "build_llama_cpp"),
    "quantize_q8": ("convert_finetuned",),
    "quantize_q4": ("convert_finetuned",),
    "eval_base": ("convert_base",),
    "eval_q8": ("quantize_q8",),
    "eval_q4": ("quantize_q4",),
    "comparison": ("eval_base", "eval_q8", "eval_q4"),
}

STAGE_INVALIDATION = {
    "config_hash": ("train", "merge", "convert_base", "convert_finetuned", "quantize_q8", "quantize_q4", "eval_base", "eval_q8", "eval_q4", "comparison"),
    "train_script_hash": ("train", "merge", "convert_finetuned", "quantize_q8", "quantize_q4", "eval_q8", "eval_q4", "comparison"),
    "eval_script_hash": ("eval_base", "eval_q8", "eval_q4", "comparison"),
    "dataset_manifest_hash": ("train", "merge", "eval_base", "eval_q8", "eval_q4", "comparison"),
    "llama_cpp_tag": ("build_llama_cpp", "convert_base", "convert_finetuned", "quantize_q8", "quantize_q4", "eval_base", "eval_q8", "eval_q4", "comparison"),
}


def load_recorded(runs_dir: str | Path) -> PipelineSignature | None:
    path = Path(runs_dir) / "pipeline_state.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    try:
        return PipelineSignature(**payload)
    except TypeError:
        log.warning("pipeline_state.json schema mismatch; treating as missing")
        return None


def write_recorded(runs_dir: str | Path, signature: PipelineSignature) -> None:
    atomic_write_json(Path(runs_dir) / "pipeline_state.json", asdict(signature))


def stages_to_invalidate(old: PipelineSignature, new: PipelineSignature) -> set[str]:
    invalidated: set[str] = set()
    for field, stages in STAGE_INVALIDATION.items():
        if getattr(old, field) != getattr(new, field):
            invalidated.update(stages)
    return invalidated


def done_path(runs_dir: str | Path, stage: str) -> Path:
    return Path(runs_dir) / "done" / f"{stage}.json"


def mark_done(runs_dir: str | Path, stage: str, payload: dict | None = None) -> None:
    target = done_path(runs_dir, stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload or {"stage": stage, "status": "done"})


def is_done(runs_dir: str | Path, stage: str) -> bool:
    return done_path(runs_dir, stage).is_file()


def invalidate(runs_dir: str | Path, stages: set[str]) -> None:
    for stage in stages:
        path = done_path(runs_dir, stage)
        if path.exists():
            log.warning("invalidating .done marker for stage %s", stage)
            path.unlink()
