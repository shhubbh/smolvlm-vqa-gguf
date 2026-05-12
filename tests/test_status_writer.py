from __future__ import annotations

import json

from common.status import StatusWriter


def test_status_writer_writes_payload(tmp_path):
    writer = StatusWriter(tmp_path)
    writer.update(stage="train", step=10, loss=0.5)
    body = json.loads((tmp_path / "status.json").read_text())
    assert body["stage"] == "train"
    assert body["step"] == 10
    assert body["loss"] == 0.5
    assert body["rss_gb"] > 0


def test_status_writer_records_failure(tmp_path):
    writer = StatusWriter(tmp_path)
    writer.fail("oom", resume_command="python pipeline/run_all.py --resume")
    body = json.loads((tmp_path / "status.json").read_text())
    assert body["last_error"] == "oom"
    assert body["resume_command"] == "python pipeline/run_all.py --resume"


def test_status_writer_extras(tmp_path):
    writer = StatusWriter(tmp_path)
    writer.update(stage="x", custom_key="custom_value")
    body = json.loads((tmp_path / "status.json").read_text())
    assert body["extra"]["custom_key"] == "custom_value"
