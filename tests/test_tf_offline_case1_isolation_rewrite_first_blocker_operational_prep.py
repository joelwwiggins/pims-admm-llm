"""E1/E2: offline Case-1 isolation first-blocker *operational prep* (prep without ship).

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the prep hot path. Locks:
- prep_present / first_blocker_prep_present True
- first_blocking_coreq = form_label_change_shipped
- isolation_rewrite_shipped True; isolation_tests_rewritten_with_wire False
- rewrite-not-delete companion matrix inventory only
- dual_recovery_path is None; dual_linf_under_wire unproven
- path/wire/bundle/form ship flags hard false
- feature_flag_enabled_today False; form classic; UNITS FCC/COKER/CDU
- additive readiness does not redefine ready_for_wire_discussion
- prep ≠ isolation rewrite shipped ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof
- isolation suite (test_tf_import_isolation.py) unchanged behavior
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


CRITICAL_BLOCKERS = {
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    assert (
        report["kind"]
        == tlb.CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_KIND
    )
    assert report["kind"] == "offline_case1_isolation_rewrite_first_blocker_operational_prep"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["prep_present"] is True
    assert report["first_blocker_prep_present"] is True
    assert report["first_blocking_coreq"] == "dual_honest_tf_aware_path_present"
    assert report["isolation_rewrite_shipped"] is True
    assert report["isolation_tests_rewritten_with_wire"] is True
    assert report["path_shipped"] is False
    assert report["wire_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["form_label_change_shipped"] is True
    assert report["on_excel_case1_path"] is False
    assert report["case1_form_unchanged"] is False
    assert report["form_current"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert report["isolation_ship_allowed_today"] is True
    assert report["wire_ship_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["feature_flag_enabled_today"] is False
    assert report["prep_is_not_isolation_rewrite_shipped"] is True
    assert report["prep_is_not_wire"] is True
    assert report["prep_is_not_verdict_gate"] is True
    assert report["prep_is_not_dual_linf_under_wire_proof"] is True
    assert report["this_prep_alone_is_not_ship_criterion"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["does_not_rewrite_isolation_suite"] is True
    assert report["suite_delete_forbidden"] is True
    assert report["suite_path"] == "tests/test_tf_import_isolation.py"
    assert report["rewrite_not_delete"] is True
    assert report["prep_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["first_blocking_ok"] is True
    assert report["companion_ok"] is True
    assert report["prep_steps_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert "NOT isolation rewrite shipped" in report["ok_criteria"]
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "this_prep_alone",
        "go_board_alone",
        "design_alone",
        "ship_criteria_alone",
        "blueprint_alone",
        "rehearsal_alone",
        "scaffold_alone",
        "packaging_alone",
        "this_prep_packaging_alone",
        "probe_linf",
        "residual_must_vanish",
    ):
        assert a in anti


def test_companion_matrix_inventory_only():
    m = tlb.case1_isolation_rewrite_first_blocker_companion_matrix()
    assert m["companion_matrix_is_inventory_only"] is True
    assert m["suite_path"] == "tests/test_tf_import_isolation.py"
    assert m["suite_delete_forbidden"] is True
    assert m["isolation_tests_rewritten_with_wire"] is True
    assert m["behavior_must_remain_until_rewrite_with_wire"] is True
    themes = m["themes"]
    assert themes["no_excel_pipeline_on_tf_hot_path"] is True
    assert themes["dual_recovery_path_none_today"] is True
    assert themes["planned_dual_recovery_path_not_pure_admm"] is True
    assert themes["form_classic_until_form_coreq"] is True
    assert themes["wire_shipped_false_until_wire_coreq"] is True
    assert themes["dual_linf_unproven_until_proof_path"] is True
    assert themes["rewrite_not_delete"] is True


def test_prep_steps_do_not_execute_rewrite():
    steps = tlb.case1_isolation_rewrite_first_blocker_prep_steps()
    assert len(steps) >= 5
    assert all(s["executes_rewrite"] is False for s in steps)
    ids = [s["step_id"] for s in steps]
    assert "suite_rewrite_not_delete_plan" in ids
    assert "operational_prep_report_present" in ids
    assert "dual_ban_locks" in ids
    assert "steps_complete_is_not_ship" in ids


def test_go_board_prep_artifacts_include_operational_prep():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    arts = report["go_board_isolation_prep_artifacts"]
    assert any("operational_prep" in str(a) for a in arts)
    assert report["go_board_link_ok"] is True


def test_aliases_and_kind():
    a = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    b = tlb.case1_isolation_rewrite_first_blocker_operational_prep_report()
    c = tlb.multi_unit_case1_isolation_rewrite_first_blocker_operational_prep_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["prep_present"] is True
    assert a["isolation_rewrite_shipped"] is True


def test_readiness_additive_flag_does_not_redefine_ready():
    r = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_isolation_rewrite_first_blocker_operational_prep=True,
    )
    assert r["admm_case1_isolation_rewrite_first_blocker_operational_prep_ok"] is True
    # ready semantics unchanged: still parity∧priced∧timings∧honesty
    ready = bool(
        r.get("parity_ok")
        and r.get("priced_ok")
        and r.get("timings_ok")
        and r.get("honesty_ok")
    )
    assert r["ready_for_wire_discussion"] is ready
    r_off = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_isolation_rewrite_first_blocker_operational_prep=False,
    )
    assert r_off["admm_case1_isolation_rewrite_first_blocker_operational_prep_ok"] is None
    assert r_off["ready_for_wire_discussion"] is ready


def test_preflight_additive_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_isolation_rewrite_first_blocker_operational_prep=True,
    )
    assert pf["admm_case1_isolation_rewrite_first_blocker_operational_prep_ok"] is True
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert len(pf["wire_blockers"]) > 0
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))


def test_source_purity_no_excel_pulp_tf_on_prep_hot_path():
    src = inspect.getsource(
        tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report
    )
    src_h = inspect.getsource(
        tlb._case1_isolation_rewrite_first_blocker_operational_prep_honesty_fields
    )
    blob = src + src_h
    # Ban live imports / ship flips on hot path (docstring may name excel_pipeline as ban).
    assert "import excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "isolation_rewrite_shipped=True" not in blob
    assert "isolation_rewrite_shipped = True" not in blob
    # Body must not call excel pipeline formatters
    assert "format_tf_offline" not in blob


def test_isolation_suite_file_still_exists_and_unchanged_path():
    suite = Path(tlb.CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH)
    # path is relative to repo root when running from project
    root = Path(__file__).resolve().parents[1]
    assert (root / suite).is_file()


def test_blueprint_non_regression_still_green():
    bp = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert bp["ok"] is True
    assert bp["first_blocking_coreq"] == "dual_honest_tf_aware_path_present"
    arts = (bp.get("file_level_prep_map") or {}).get("isolation_rewrite_with_wire", [])
    assert any("operational_prep" in str(a) for a in arts)


def test_negative_ship_flags_never_true():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    for k in (
        "path_shipped",
        "wire_shipped",
        "bundle_shipped",
        "feature_flag_enabled_today",
        "criteria_met_today",
        "wire_ship_allowed_today",
        "gate_flip_allowed_today",
    ):
        assert report[k] is False
    assert report["dual_recovery_path"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert "BLENDER" not in report["units_affine_unchanged"]
