"""Loader for the base SmolVLM model + processor, plus LoRA injection with scope assertions."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

from common.config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_ID,
    MODEL_REVISION,
)

log = logging.getLogger(__name__)

MIN_TRAINABLE_PARAMS = 1_000_000
MAX_TRAINABLE_PARAMS = 10_000_000


def load_base_model_and_processor(dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to("cpu")
    model.eval()
    return model, processor


def _freeze_vision_and_connector(model: torch.nn.Module) -> None:
    frozen_count = 0
    for name, param in model.named_parameters():
        if (
            ".vision_model." in name
            or name.startswith("vision_model.")
            or ".connector." in name
            or name.startswith("connector.")
        ):
            param.requires_grad = False
            frozen_count += 1
    log.info("explicitly froze %d vision/connector params", frozen_count)


def attach_lora(
    model: torch.nn.Module, target_modules: Sequence[str] | None = None
) -> torch.nn.Module:
    target_modules = list(target_modules or LORA_TARGET_MODULES)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    _freeze_vision_and_connector(peft_model)
    _assert_lora_scope(peft_model)
    return peft_model


def _assert_lora_scope(peft_model: torch.nn.Module) -> None:
    trainable: list[tuple[str, int]] = []
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel()))
    total_trainable = sum(n for _, n in trainable)
    log.info(
        "LoRA trainable params: %d across %d tensors", total_trainable, len(trainable)
    )
    if not (MIN_TRAINABLE_PARAMS <= total_trainable <= MAX_TRAINABLE_PARAMS):
        raise RuntimeError(
            f"trainable param count {total_trainable} outside expected range "
            f"[{MIN_TRAINABLE_PARAMS}, {MAX_TRAINABLE_PARAMS}]"
        )
    bad = [name for name, _ in trainable if not _is_in_text_decoder(name)]
    if bad:
        sample = ", ".join(bad[:5])
        raise RuntimeError(
            f"{len(bad)} trainable params are outside the text decoder; first few: {sample}"
        )


_TEXT_DECODER_PATTERNS = (
    re.compile(r"\.text_model\."),
    re.compile(r"^text_model\."),
    re.compile(r"\.language_model\."),
    re.compile(r"^language_model\."),
)


def _is_in_text_decoder(name: str) -> bool:
    return any(p.search(name) for p in _TEXT_DECODER_PATTERNS)
