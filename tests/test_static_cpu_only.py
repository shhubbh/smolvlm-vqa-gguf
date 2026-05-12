"""AST-based static scan to enforce CPU-only project policy."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXECUTABLE_DIRS = ("data", "train", "convert", "eval", "pipeline")
ENTRYPOINT_DIRS = ("data", "train", "convert", "pipeline")


def project_py_files() -> list[Path]:
    files: list[Path] = []
    for sub in EXECUTABLE_DIRS + ("common", "tests"):
        for path in (ROOT / sub).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _resolves_to_cuda(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "cuda" in node.value.lower()
    return False


class CpuOnlyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "cuda" and isinstance(node.ctx, ast.Load):
            parent_is_call_func = getattr(node, "_parent_is_call_func", False)
            if parent_is_call_func:
                self.violations.append((node.lineno, "tensor.cuda() call"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            setattr(node.func, "_parent_is_call_func", True)
            if node.func.attr == "to":
                for arg in node.args:
                    if _resolves_to_cuda(arg):
                        self.violations.append((node.lineno, ".to('cuda') call"))
                for kw in node.keywords:
                    if kw.arg == "device" and _resolves_to_cuda(kw.value):
                        self.violations.append((node.lineno, ".to(device='cuda') call"))
        for kw in node.keywords:
            if kw.arg == "device_map" and isinstance(kw.value, (ast.Dict, ast.Constant)):
                if isinstance(kw.value, ast.Constant) and _resolves_to_cuda(kw.value):
                    self.violations.append((node.lineno, "device_map='cuda'"))
                elif isinstance(kw.value, ast.Dict):
                    for v in kw.value.values:
                        if _resolves_to_cuda(v):
                            self.violations.append((node.lineno, 'device_map={"":"cuda"}'))
            if kw.arg == "n_gpu_layers" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, int) and kw.value.value > 0:
                    self.violations.append((node.lineno, f"n_gpu_layers={kw.value.value}"))
        self.generic_visit(node)


def test_no_gpu_calls_in_any_project_python():
    failures: dict[Path, list[tuple[int, str]]] = {}
    for path in project_py_files():
        if path.name == "test_static_cpu_only.py":
            continue
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        v = CpuOnlyVisitor()
        v.visit(tree)
        if v.violations:
            failures[path] = v.violations
    assert not failures, f"GPU usage detected: {failures}"


def test_main_entrypoints_set_device_cpu():
    missing: list[Path] = []
    for sub in ENTRYPOINT_DIRS:
        for path in (ROOT / sub).rglob("*.py"):
            if "__init__" in path.name or "__pycache__" in path.parts:
                continue
            src = path.read_text()
            if "if __name__ == \"__main__\"" not in src:
                continue
            if not re.search(r'^DEVICE\s*=\s*"cpu"', src, re.MULTILINE):
                missing.append(path)
    assert not missing, f"executable scripts missing DEVICE='cpu': {missing}"


def test_requirements_have_no_gpu_packages():
    text = (ROOT / "requirements.txt").read_text()
    forbidden = ["bitsandbytes", "flash-attn", "nvidia-", "auto-gptq", "llama-cpp-python"]
    for token in forbidden:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert token not in stripped, f"forbidden dep '{token}' in requirements.txt: {line}"
