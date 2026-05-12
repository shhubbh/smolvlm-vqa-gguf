"""Assert the eval prompt template is referenced through eval.prompts (single source)."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PROMPT_HOLDER = ROOT / "eval" / "prompts.py"


def test_no_other_file_redefines_prompt_template():
    pattern = re.compile(r"Answer with the shortest correct response", re.IGNORECASE)
    here = Path(__file__).resolve()
    matches: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if path == ALLOWED_PROMPT_HOLDER or path.resolve() == here:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            matches.append(path)
    assert matches == [], f"prompt template duplicated in: {matches}"


def test_eval_client_imports_render_user():
    text = (ROOT / "eval" / "client.py").read_text()
    assert "from eval.prompts import" in text
    assert "render_user" in text
