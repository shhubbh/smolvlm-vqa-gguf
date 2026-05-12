from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.state import (
    PipelineSignature,
    STAGE_INVALIDATION,
    invalidate,
    is_done,
    mark_done,
    stages_to_invalidate,
)


def make_sig(**overrides) -> PipelineSignature:
    base = dict(
        config_hash="cfg",
        train_script_hash="train",
        eval_script_hash="eval",
        dataset_manifest_hash="ds",
        llama_cpp_tag="b9000",
    )
    base.update(overrides)
    return PipelineSignature(**base)


def test_no_drift_no_invalidation():
    sig = make_sig()
    assert stages_to_invalidate(sig, sig) == set()


def test_config_drift_invalidates_train_onward():
    old = make_sig()
    new = make_sig(config_hash="cfg2")
    invalidated = stages_to_invalidate(old, new)
    assert "train" in invalidated
    assert "data_prepare" not in invalidated


def test_eval_script_drift_only_invalidates_eval_stages():
    old = make_sig()
    new = make_sig(eval_script_hash="eval2")
    invalidated = stages_to_invalidate(old, new)
    assert invalidated == set(STAGE_INVALIDATION["eval_script_hash"])
    assert "train" not in invalidated


def test_llama_tag_drift_invalidates_build_and_downstream():
    old = make_sig()
    new = make_sig(llama_cpp_tag="b9001")
    invalidated = stages_to_invalidate(old, new)
    assert "build_llama_cpp" in invalidated
    assert "quantize_q4" in invalidated


def test_mark_and_invalidate_done(tmp_path):
    mark_done(tmp_path, "train")
    assert is_done(tmp_path, "train")
    invalidate(tmp_path, {"train"})
    assert not is_done(tmp_path, "train")
