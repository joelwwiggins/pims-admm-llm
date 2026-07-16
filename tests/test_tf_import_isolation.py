"""E6/E14: TensorFlow optional-dep isolation + Case-1 non-wiring contract.

These tests must pass with TensorFlow **absent**. They lock:

1. Optional-dep hygiene (``tf_available``, no hard TF import for excel path).
2. Static non-wiring: ``excel_pipeline`` does not import tensorflow / tf_linear*.
3. Package import of excel pipeline does not require tensorflow in sys.modules.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PIPELINE_SRC = (
    REPO_ROOT / "src" / "pims_admm_llm" / "models" / "excel_pipeline.py"
)
MODELS_DIR = REPO_ROOT / "src" / "pims_admm_llm" / "models"


def _import_names_from_ast(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
                names.add(node.module)
                # relative: from .tf_linear_blocks import ...
                if node.level and node.module:
                    names.add(node.module)
            for alias in node.names:
                # from . import tf_linear_blocks
                if alias.name != "*":
                    names.add(alias.name)
    return names


def test_excel_pipeline_source_has_no_tf_imports():
    """E14 static gate: excel_pipeline.py must not import TF surfaces."""
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(EXCEL_PIPELINE_SRC))
    imported = _import_names_from_ast(tree)
    offenders = sorted(
        n
        for n in imported
        if n == "tensorflow"
        or n.startswith("tensorflow.")
        or n == "tf_linear_blocks"
        or n.startswith("tf_linear")
        or n == "tf_optional"
    )
    assert offenders == [], f"excel_pipeline TF-related imports: {offenders}"


def test_excel_pipeline_source_text_no_tf_import_lines():
    """Belt-and-suspenders: no import line mentioning tensorflow / tf_linear."""
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    bad = []
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        low = stripped.lower()
        if "tensorflow" in low or "tf_linear" in low or "tf_optional" in low:
            bad.append((i, stripped))
    assert bad == [], f"forbidden import lines in excel_pipeline: {bad}"


def test_models_package_init_does_not_import_tf_linear():
    """Package models/__init__ must not eagerly pull tf_linear_blocks / tensorflow."""
    init_path = MODELS_DIR / "__init__.py"
    src = init_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(init_path))
    imported = _import_names_from_ast(tree)
    assert "tensorflow" not in imported
    assert "tf_linear_blocks" not in imported
    assert "tf_optional" not in imported


def test_import_excel_pipeline_without_tensorflow_in_sys_modules():
    """E6 always-on isolation: excel_pipeline loads without tensorflow present."""
    # Drop tensorflow if a prior test imported it (should be rare in default suite).
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "tensorflow" or k.startswith("tensorflow.")
    }
    try:
        # Force re-import of excel_pipeline after clearing TF.
        for key in list(sys.modules):
            if key == "pims_admm_llm.models.excel_pipeline":
                del sys.modules[key]
        mod = importlib.import_module("pims_admm_llm.models.excel_pipeline")
        assert hasattr(mod, "run_excel_pipeline")
        # tensorflow must not have been pulled in by that import.
        assert "tensorflow" not in sys.modules
        assert not any(k.startswith("tensorflow.") for k in sys.modules)
    finally:
        sys.modules.update(saved)


def test_tf_linear_blocks_importable_without_tensorflow():
    """Scaffold module must import even when TF is missing."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    assert callable(tlb.tf_available)
    ok = tlb.tf_available()
    assert isinstance(ok, bool)
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["tf_available"] is ok
    if not ok:
        assert meta["backend"] == "unavailable"
    else:
        assert meta["backend"] == "tensorflow"


def test_tf_available_does_not_raise_when_missing():
    from pims_admm_llm.models.tf_linear_blocks import tf_available, tf_import_error

    # Must never raise regardless of environment.
    val = tf_available()
    assert val in (True, False)
    err = tf_import_error()
    if val:
        assert err is None
    else:
        # Missing TF or broken wheel → some exception recorded after probe.
        assert err is not None


def test_coker_kernel_symbols_exist_without_tf_and_stay_offline():
    """E15: Coker factory/postprocess surface is importable offline; not wired to Case 1."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    assert callable(tlb.tf_linear_coker)
    assert callable(tlb.apply_coker_postprocess)
    assert "COKER" in tlb.UNITS and "FCC" in tlb.UNITS
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    # excel_pipeline still must not import the coker kernel module
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(EXCEL_PIPELINE_SRC))
    imported = _import_names_from_ast(tree)
    assert "tf_linear_blocks" not in imported
    assert "tf_linear_coker" not in imported
    assert "tensorflow" not in imported
    # How_to may mention the name as text (allowed); import graph must not.

def test_cdu_kernel_symbols_exist_without_tf_and_stay_offline():
    """E15: CDU factory/postprocess surface is importable offline; not wired to Case 1."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    assert callable(tlb.tf_linear_cdu)
    assert callable(tlb.apply_cdu_postprocess)
    assert "CDU" in tlb.UNITS and "FCC" in tlb.UNITS and "COKER" in tlb.UNITS
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(EXCEL_PIPELINE_SRC))
    imported = _import_names_from_ast(tree)
    assert "tf_linear_blocks" not in imported
    assert "tf_linear_cdu" not in imported
    assert "tensorflow" not in imported
    # How_to may name factories as text; import graph must not pull them.
