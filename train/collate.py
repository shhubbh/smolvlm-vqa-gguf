"""Batch construction for SmolVLM teacher-forced answer-token loss.

Builds two passes through the processor: one for the prompt prefix (user turn + assistant
generation marker) and one for the full sequence (prompt + answer). Tokens belonging to
the prompt prefix are masked out of the labels (-100), so loss only flows through the
assistant answer tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

LABEL_IGNORE = -100


@dataclass
class BatchTensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    pixel_attention_mask: torch.Tensor | None
    labels: torch.Tensor
    extras: dict[str, Any]


def _build_messages(question: str, answer: str | None) -> list[dict[str, Any]]:
    user_turn = {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }
    if answer is None:
        return [user_turn]
    assistant_turn = {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }
    return [user_turn, assistant_turn]


def _images_kwargs(processor) -> dict:
    """Cap longest_edge to processor.image_processor.max_image_size.

    SmolVLM/Idefics3 default `size["longest_edge"]` is larger than `max_image_size`
    (designed for tiling). transformers >=4.46 guards against this in
    `get_resize_output_image_size`, raising ValueError. We disable tiling by passing
    `size["longest_edge"] = max_image_size["longest_edge"]`.
    """
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is None:
        return {}
    max_side = getattr(img_proc, "max_image_size", None)
    if isinstance(max_side, dict):
        max_side = max_side.get("longest_edge")
    if not isinstance(max_side, int):
        return {}
    return {"size": {"longest_edge": max_side}}


def build_training_inputs(processor, image, question: str, answer: str) -> BatchTensors:
    prompt_text = processor.apply_chat_template(
        _build_messages(question, None),
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        _build_messages(question, answer),
        add_generation_prompt=False,
    )

    img_kwargs = _images_kwargs(processor)
    prompt_inputs = processor(text=prompt_text, images=[image], return_tensors="pt", **img_kwargs)
    full_inputs = processor(text=full_text, images=[image], return_tensors="pt", **img_kwargs)

    prompt_len = int(prompt_inputs.input_ids.shape[1])
    full_ids = full_inputs.input_ids
    if full_ids.shape[1] < prompt_len:
        raise RuntimeError(
            f"full sequence ({full_ids.shape[1]}) shorter than prompt ({prompt_len}); "
            "chat template mismatch"
        )

    labels = full_ids.clone()
    labels[:, :prompt_len] = LABEL_IGNORE
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = LABEL_IGNORE

    extras = {
        "prompt_len": prompt_len,
        "full_len": int(full_ids.shape[1]),
    }
    return BatchTensors(
        input_ids=full_inputs.input_ids,
        attention_mask=full_inputs.attention_mask,
        pixel_values=getattr(full_inputs, "pixel_values", None),
        pixel_attention_mask=getattr(full_inputs, "pixel_attention_mask", None),
        labels=labels,
        extras=extras,
    )


def to_model_kwargs(batch: BatchTensors) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
    }
    if batch.pixel_values is not None:
        kwargs["pixel_values"] = batch.pixel_values
    if batch.pixel_attention_mask is not None:
        kwargs["pixel_attention_mask"] = batch.pixel_attention_mask
    return kwargs
