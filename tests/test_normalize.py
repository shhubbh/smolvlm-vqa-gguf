from __future__ import annotations

import pytest

from data.normalize import (
    MalformedRow,
    exact_match,
    extract_qa_from_messages,
    normalize_answer,
)


def test_extract_qa_basic():
    messages = [
        {"role": "user", "content": "What is shown?"},
        {"role": "assistant", "content": "A scalpel."},
    ]
    q, a = extract_qa_from_messages(messages)
    assert q == "What is shown?"
    assert a == "A scalpel."


def test_extract_qa_reverse_order_is_role_based():
    messages = [
        {"role": "assistant", "content": "Answer first."},
        {"role": "user", "content": "Question second."},
    ]
    q, a = extract_qa_from_messages(messages)
    assert q == "Question second."
    assert a == "Answer first."


def test_extract_qa_skips_system():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    q, a = extract_qa_from_messages(messages)
    assert q == "Q"
    assert a == "A"


def test_extract_qa_rejects_multi_turn():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    with pytest.raises(MalformedRow):
        extract_qa_from_messages(messages)


def test_extract_qa_handles_list_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    q, a = extract_qa_from_messages(messages)
    assert q == "Hello world"
    assert a == "Hi"


def test_normalize_answer():
    assert normalize_answer(" Hello, World! ") == "hello world"
    assert normalize_answer("42.0") == "42"
    assert normalize_answer("YES.") == "yes"
    assert normalize_answer(None) == ""


def test_exact_match():
    assert exact_match("Yes.", "yes") is True
    assert exact_match("forceps", "scalpel") is False
    assert exact_match("3.0", "3") is True
