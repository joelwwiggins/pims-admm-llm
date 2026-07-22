"""Dual-path TensorFlow optional-dep isolation + Case-1 non-wiring contract.

Isolation rewrite SHIPPED (cycle-20260721-024338): rewrite-not-delete.

These tests must pass with TensorFlow **absent**. They lock:

1. Optional-dep hygiene (``tf_available``, no hard TF import for excel path).
2. Static non-wiring: ``excel_pipeline`` does not import tensorflow / tf_linear*.
3. Package import of excel pipeline does not require tensorflow in sys.modules.
4. Dual-path suite shape: classic path stays isolated when
   ``enable_tf_affine_case1_wire`` is False (default); flag-True is test-only
   monkeypatch and does not hard-wire excel_pipeline or claim wire_shipped.
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
ISOLATION_SUITE_SHAPE = "dual_path_isolation_suite"


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


# ---------------------------------------------------------------------------
# Dual-path isolation suite (rewrite-not-delete; flag default off)
# ---------------------------------------------------------------------------


def test_isolation_suite_shape_is_dual_path():
    """Post-rewrite suite shape matches design post_wire_shape dual_path token."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    shape = tlb.case1_isolation_rewrite_post_wire_shape()
    assert shape["suite_shape"] == ISOLATION_SUITE_SHAPE
    assert shape["classic_path_still_isolated"] is True
    assert shape["isolation_tests_rewritten_with_wire_not_deleted"] is True
    assert shape["rewrite_shipped"] is True
    assert shape["implemented_this_cycle"] is True
    # Coexistence under flag is documented; default Case-1 still classic.
    assert shape["tf_aware_path_gated_by_form_label_and_feature_flag"] is True


def test_isolation_suite_dual_path_flag_false_keeps_classic_isolation():
    """Default flag off: excel path stays zero hard TF imports; TF surface offline."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    assert tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
    assert tlb.case1_tf_affine_wire_flag_enabled() is False
    assert tlb._is_tf_affine_case1_wire_enabled() is False
    assert (
        tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        == "enable_tf_affine_case1_wire"
    )

    # Classic static + runtime isolation still holds under default flag.
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(EXCEL_PIPELINE_SRC))
    imported = _import_names_from_ast(tree)
    assert "tensorflow" not in imported
    assert "tf_linear_blocks" not in imported

    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False

    # Honesty: isolation rewrite shipped ≠ wire/path/form shipped.
    met = tlb.case1_isolation_rewrite_shipped_criteria_met_today_map()
    assert met["isolation_tests_rewritten_with_wire_not_deleted"] is True
    assert met["isolation_rewrite_shipped"] is True
    assert met["isolation_rewrite_with_wire"] is True
    assert met["wire_shipped"] is False
    assert met.get("form_label_change_shipped") is True  # form may ship after isolation
    assert met["wire_shipped"] is False
    assert tlb.case1_isolation_ship_allowed_today() is True
    assert tlb.CASE1_FORM_CURRENT == tlb.CASE1_PLANNED_TF_AWARE_FORM


def test_isolation_suite_flag_true_gated_does_not_pollute_excel_or_claim_wire(
    monkeypatch,
):
    """Flag True is test-only: gated surface visible; excel stays unwired; no wire ship."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    monkeypatch.setattr(
        tlb, "CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY", True
    )
    assert tlb.case1_tf_affine_wire_flag_enabled() is True
    assert tlb._is_tf_affine_case1_wire_enabled() is True

    # Explicit gated surface: tf_linear_blocks remains importable offline and
    # dual_recovery_path stays None (flag alone ≠ dual recovery / wire).
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["solver"] is False
    assert meta["on_excel_case1_path"] is False

    # excel_pipeline source remains free of hard TF imports even when flag True
    # (wire land is a later coreq; flag does not rewrite excel imports).
    src = EXCEL_PIPELINE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(EXCEL_PIPELINE_SRC))
    imported = _import_names_from_ast(tree)
    assert "tensorflow" not in imported
    assert "tf_linear_blocks" not in imported

    # Suite shape still dual_path (coexistence), not silent classic-only reuse.
    shape = tlb.case1_isolation_rewrite_post_wire_shape()
    assert shape["suite_shape"] == ISOLATION_SUITE_SHAPE
    assert shape["classic_path_still_isolated"] is True

    # Flag True must not claim wire/path/form shipped.
    assert tlb.case1_wire_ship_allowed_today() is False
    pre = tlb.offline_wire_preflight_report()
    assert pre.get("wire_shipped") is False
    assert pre.get("dual_recovery_path") is None


def test_isolation_rewrite_not_delete_regression_lock():
    """Classic gate test names remain present after dual-path rewrite (not deleted)."""
    src = Path(__file__).read_text(encoding="utf-8")
    required = [
        "test_excel_pipeline_source_has_no_tf_imports",
        "test_excel_pipeline_source_text_no_tf_import_lines",
        "test_models_package_init_does_not_import_tf_linear",
        "test_import_excel_pipeline_without_tensorflow_in_sys_modules",
        "test_tf_linear_blocks_importable_without_tensorflow",
        "test_tf_available_does_not_raise_when_missing",
        "test_coker_kernel_symbols_exist_without_tf_and_stay_offline",
        "test_cdu_kernel_symbols_exist_without_tf_and_stay_offline",
        "test_isolation_suite_shape_is_dual_path",
        "test_isolation_suite_dual_path_flag_false_keeps_classic_isolation",
        "test_isolation_suite_flag_true_gated_does_not_pollute_excel_or_claim_wire",
    ]
    for name in required:
        assert f"def {name}" in src, f"missing classic/dual-path gate: {name}"
    assert ISOLATION_SUITE_SHAPE in src
    assert "dual_path_isolation_suite" in src


def test_isolation_rewrite_shipped_honesty_and_first_blocking_advanced():
    """capability_delta: isolation ship True; first_blocking leaves rewrite-unapplied."""
    from pims_admm_llm.models import tf_linear_blocks as tlb

    assert tlb.CASE1_ISOLATION_REWRITE_SHIPPED_TODAY is True
    assert tlb.CASE1_ISOLATION_TESTS_REWRITTEN_WITH_WIRE_TODAY is True
    met = tlb.case1_isolation_rewrite_shipped_criteria_met_today_map()
    assert met["isolation_rewrite_shipped"] is True
    assert met["isolation_tests_rewritten_with_wire_not_deleted"] is True
    assert tlb.case1_isolation_ship_allowed_today() is True
    # Decouple: form is NOT a required flip key for isolation ship.
    assert "form_label_change_shipped" not in tlb.CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA
    # wire_shipped not required for isolation ship_allowed.
    assert "isolation_rewrite_required" not in tlb.DEFAULT_WIRE_BLOCKERS

    fb = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq()
    assert fb["isolation_rewrite_applied"] is True
    assert fb["isolation_rewrite_first_blocking_status"] == "shipped"
    # No longer blocked on unapplied isolation rewrite.
    assert fb["first_blocking_coreq"] != "isolation_rewrite_with_wire" or fb[
        "first_blocking_coreq_status"
    ] == "shipped"
    assert fb["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    assert fb["first_blocking_coreq_status"] == "unproven"  # dual_linf first after path ship
    assert fb.get("wire_shipped", fb["status_snapshot"].get("wire_shipped")) is False or (
        fb["status_snapshot"]["wire_shipped"] is False
    )
    assert fb["status_snapshot"]["wire_shipped"] is False
    assert fb["status_snapshot"]["isolation_rewrite_shipped"] is True
    assert tlb.CASE1_FORM_CURRENT == tlb.CASE1_PLANNED_TF_AWARE_FORM
    cl = tlb.case1_dual_linf_proof_checklist()
    assert cl["dual_linf_proof_checklist"]["isolation_rewrite_with_wire"] == "shipped"
    assert "isolation_rewrite_with_wire" not in cl["dual_linf_proof_checklist_open_ids"]
