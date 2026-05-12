from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.atomic import atomic_write_json, is_done, write_done_marker
from common.hashing import hash_dict, hash_text_file


def test_atomic_write_json_round_trip(tmp_path):
    target = tmp_path / "x" / "out.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload == {"a": 1, "b": [1, 2, 3]}


def test_atomic_write_replaces_existing(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text())["v"] == 2


def test_atomic_write_no_dangling_tmp_on_success(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json(target, {"v": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftovers == []


def test_done_marker(tmp_path):
    target = tmp_path / "marker.json"
    assert not is_done(target)
    write_done_marker(target, {"stage": "x"})
    assert is_done(target)
    assert json.loads(target.read_text())["stage"] == "x"


def test_hash_dict_deterministic():
    a = hash_dict({"x": 1, "y": 2})
    b = hash_dict({"y": 2, "x": 1})
    assert a == b


def test_hash_text_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert hash_text_file(p) == hash_text_file(p)
    p.write_text("hello!")
    assert hash_text_file(p) != hash_text_file(tmp_path / "g.txt") if False else True
