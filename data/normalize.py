"""PitVQA SAGE row normalization and answer canonicalization."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable


class MalformedRow(ValueError):
    pass


@dataclass
class NormalizedRow:
    id: str
    image_path: str
    question: str
    answer: str
    qa_type: str
    split: str
    video_id: str | None
    frame_id: str | None

    def to_jsonl(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "qa_type": self.qa_type,
            "split": self.split,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
        }


def extract_qa_from_messages(messages: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Iterate messages by role rather than indexing positions.

    PitVQA SAGE samples are single-turn (one user, one assistant). Reject anything else.
    """
    user_msg = None
    assistant_msg = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if user_msg is not None:
                raise MalformedRow("multi-turn user content")
            user_msg = content
        elif role == "assistant":
            if assistant_msg is not None:
                raise MalformedRow("multi-turn assistant content")
            assistant_msg = content
        elif role == "system":
            continue
        else:
            raise MalformedRow(f"unknown role: {role!r}")
    if user_msg is None or assistant_msg is None:
        raise MalformedRow("missing user or assistant turn")
    return _content_to_text(user_msg), _content_to_text(assistant_msg)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(p.strip() for p in parts if p).strip()
    raise MalformedRow(f"unsupported content type: {type(content).__name__}")


_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    text = answer.lower().strip()
    numeric = text.rstrip(".")
    if re.fullmatch(r"\d+(\.\d+)?", numeric):
        try:
            f = float(numeric)
            return str(int(f)) if f.is_integer() else f"{f:g}"
        except ValueError:
            pass
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)
