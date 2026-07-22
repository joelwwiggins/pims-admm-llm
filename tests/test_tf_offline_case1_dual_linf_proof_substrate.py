"""E1/E2: offline Case-1 dual_linf proof substrate (PR-A post-#82).

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the substrate hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- substrate_present True; substrate_ok may True without proving dual_linf
- dual_linf_proof_allowed_today False; criteria_met_today False
- path_present face labeled; path_under_wire False
- first_blocking_coreq remains dual_linf_under_wire_proven
- feature flag enable_tf_affine_case1_wire remains False
- linf<=15 is diagnostic only (not proof / not ok gate)
- substrate ≠ dual L∞ under wire proof ≠ gate flip ≠ wire ≠ VERDICT
- probe/bridge/warmstart alone never prove dual_linf
- no excel_pipeline import on tf_linear_blocks substrate hot path

Charter validation companions:
  pytest tests/test_tf_offline_case1_dual_linf_proof_substrate.py \\
    tests/test_tf_offline_case1_dual_linf_under_wire_criteria_contract.py \\
    tests/test_tf_import_isolation.py -q
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_linf_proof_substrate_report()
    assert report["kind"] == tlb.CASE1_DUAL_LINF_PROOF_SUBSTRATE_KIND
    assert report["kind"] == "offline_case1_dual_linf_proof_substrate"
    assert report["annotation"] == "present"
    assert report["substrate_present"] is True
    assert report["substrate_ok"] is True
    assert report["ok"] is True
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["path_shipped"] is True
    assert report["dual_honest_tf_aware_path_present"] is True
    assert report["dual_honest_tf_aware_path_under_wire"] is False
    assert report["form_label_change_shipped"] is True
    assert report["isolation_rewrite_shipped"] is True
    assert report["form_current"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["feature_flag_enabled_today"] is False
    assert (
        report["feature_flag_name"]
        == tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        == "enable_tf_affine_case1_wire"
    )
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["gate_flip_allowed_today"] is False
    assert report["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    assert report["substrate_is_not_dual_linf_under_wire_proof"] is True
    assert report["substrate_is_not_wire"] is True
    assert report["substrate_is_not_verdict_gate"] is True
    assert report["substrate_is_not_gate_flip"] is True
    assert report["probe_bridge_are_not_proof"] is True
    assert report["linf_le_threshold_is_not_proof"] is True
    assert report["proof_composition_path_executed_today"] is False
    assert report["proof_composition_status_today"] == "not_executed"
    assert report["on_excel_case1_path"] is False
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True


def test_measurement_fields_finite_and_face_labeled():
    report = tlb.offline_case1_dual_linf_proof_substrate_report()
    assert report["measurement_ok"] is True
    assert report["bridge_ok"] is True
    assert report["probe_ok"] is True
    assert report["stream_alignment_ok"] is True
    assert report["finite_ok"] is True
    assert report["substrate_face"] == tlb.CASE1_DUAL_LINF_PROOF_SUBSTRATE_FACE
    assert report["cdu_surface"] == "offline_affine_base_delta"
    assert report["blender_surface"] == tlb.CASE1_SHAPED_BLENDER_SURFACE
    assert report["streams"] == list(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert isinstance(report["linf"], float)
    assert report["finite_ok"] is True
    # Diagnostic threshold comparison exists but never gates ok/substrate_ok alone.
    assert "linf_le_threshold_diagnostic" in report
    assert report["dual_gate_threshold_diagnostic"] == 15.0
    # Fixture path often has large L∞ — ok must not require le-threshold.
    if not report["linf_le_threshold_diagnostic"]:
        assert report["ok"] is True
        assert report["substrate_ok"] is True


def test_caller_supplied_primary_lambda_path():
    primary = {s: 0.0 for s in tlb.CASE1_SHAPED_LINKING_STREAMS}
    report = tlb.offline_case1_dual_linf_proof_substrate_report(
        case1_primary_online_lambda=primary,
        allow_fixture_fallback=False,
    )
    assert report["live_lambda_source"] in (
        tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED,
        "caller_supplied",
    )
    assert report["substrate_ok"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["wire_shipped"] is False


def test_zero_linf_still_unproven():
    """Even perfect diagnostic L∞ never flips dual_linf proven."""
    # Use same skeleton as probe would so gap can be small/zero when aligned.
    skel = tlb.extract_case1_shaped_skeleton_lambda(n_rounds=1)
    report = tlb.offline_case1_dual_linf_proof_substrate_report(
        case1_primary_online_lambda=skel,
        skeleton_lambda=skel,
        allow_fixture_fallback=False,
    )
    assert report["stream_alignment_ok"] is True
    assert report["finite_ok"] is True
    assert report["linf"] == pytest.approx(0.0, abs=1e-9)
    assert report["linf_le_threshold_diagnostic"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["substrate_is_not_dual_linf_under_wire_proof"] is True
    assert report["ok"] is True


def test_anti_criteria_and_blockers_remain():
    report = tlb.offline_case1_dual_linf_proof_substrate_report()
    anti = report["anti_criteria_today"]
    for need in (
        "this_substrate_alone",
        "probe_linf",
        "bridge_linf",
        "warmstart_linf",
        "dual_linf_criteria_alone",
        "online_linf_gate_criteria_alone",
        "path_present_alone",
    ):
        assert need in anti
    blockers = report["wire_blockers"]
    assert "dual_linf_under_wire_unproven" in blockers
    assert "wire_not_shipped" in blockers
    assert report["dual_linf_under_wire_unproven_blocker_still_true"] is True
    assert report["wire_not_shipped_blocker_still_true"] is True
    assert report["n_wire_blockers"] >= 5


def test_aliases_and_availability():
    a = tlb.case1_dual_linf_proof_substrate_report()
    b = tlb.multi_unit_case1_dual_linf_proof_substrate_report()
    c = tlb.offline_case1_dual_linf_proof_substrate_report()
    assert a["kind"] == b["kind"] == c["kind"]
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_linf_proof_substrate_available") is True
    assert "offline_case1_dual_linf_proof_substrate_report" in tlb.__all__


def test_hot_path_does_not_import_excel_or_pulp():
    fn_src = inspect.getsource(tlb.offline_case1_dual_linf_proof_substrate_report)
    for line in fn_src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            low = stripped.lower()
            assert "excel_pipeline" not in low
            assert "pulp" not in low
            assert "tensorflow" not in low
    # Runtime: substrate does not put excel/tf into sys.modules
    import sys

    before = {k for k in sys.modules if "excel_pipeline" in k or k == "tensorflow"}
    tlb.offline_case1_dual_linf_proof_substrate_report()
    after = {k for k in sys.modules if "excel_pipeline" in k or k == "tensorflow"}
    assert after == before


def test_criteria_contract_still_unproven_after_substrate():
    """Substrate does not advance dual_linf criteria or first_blocking."""
    sub = tlb.offline_case1_dual_linf_proof_substrate_report()
    assert sub["ok"] is True
    crit = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert crit["dual_linf_under_wire_status"] == "unproven"
    assert crit["dual_linf_proof_allowed_today"] is False
    assert crit["criteria_met_today"] is False
    assert crit["ok"] is True
    fb = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq()
    assert fb["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    assert tlb.CASE1_DUAL_LINF_UNDER_WIRE_STATUS == "unproven"
    assert tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False


def test_negative_ship_and_proof_flags_never_true():
    report = tlb.offline_case1_dual_linf_proof_substrate_report()
    for k in (
        "wire_shipped",
        "bundle_shipped",
        "feature_flag_enabled_today",
        "dual_linf_proof_allowed_today",
        "criteria_met_today",
        "gate_flip_allowed_today",
        "dual_honest_tf_aware_path_under_wire",
        "proof_composition_path_executed_today",
        "on_excel_case1_path",
    ):
        assert report.get(k) is False, k
    assert report["dual_linf_under_wire_status"] == "unproven"
