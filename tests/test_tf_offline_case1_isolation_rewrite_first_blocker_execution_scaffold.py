"""E2/E22: offline Case-1 isolation rewrite first-blocker *execution scaffold*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the scaffold hot path. Locks:
- scaffold_present / execution_scaffold_present / isolation_rewrite_scaffold_present True
- isolation_rewrite_shipped False; isolation_ship_allowed_today False
- isolation_tests_rewritten_with_wire False; checklist open
- first_blocking_coreq = isolation_rewrite_with_wire; order_hint_index=0
- dual_recovery_path is None; dual_linf_under_wire unproven; gate open
- path/wire/bundle/form ship flags hard false; feature_flag_enabled_today False
- mutation inventory of 8 tests not applied; suite unchanged
- additive readiness does not redefine ready_for_wire_discussion
- scaffold ≠ isolation rewrite shipped ≠ design ≠ ship-met ≠ operational prep
  ≠ path scaffold ≠ wire fifth prep ≠ VERDICT
"""

from __future__ import annotations

import ast
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
    "isolation_rewrite_required",
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    assert (
        report["kind"]
        == tlb.CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_EXECUTION_SCAFFOLD_KIND
    )
    assert report["kind"] == (
        "offline_case1_isolation_rewrite_first_blocker_execution_scaffold"
    )
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["scaffold_present"] is True
    assert report["execution_scaffold_present"] is True
    assert report["isolation_rewrite_scaffold_present"] is True
    assert report["first_blocking_coreq"] == "isolation_rewrite_with_wire"
    assert report["is_first_blocking_coreq"] is True
    assert report["order_hint_index"] == 0
    assert report["isolation_rewrite_shipped"] is False
    assert report["isolation_ship_allowed_today"] is False
    assert report["isolation_tests_rewritten_with_wire"] is False
    assert report["path_shipped"] is False
    assert report["wire_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["form_label_change_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["wire_ship_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["isolation_rewrite_with_wire"] == "open"
    assert report["isolation_rewrite_still_open"] is True
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["feature_flag_enabled_today"] is False
    assert report["feature_flag_name"] == "enable_tf_affine_case1_wire"
    assert report["scaffold_is_not_isolation_rewrite_shipped"] is True
    assert report["scaffold_is_not_isolation_ship_allow"] is True
    assert report["scaffold_is_not_wire"] is True
    assert report["scaffold_is_not_verdict_gate"] is True
    assert report["scaffold_is_not_dual_linf_under_wire_proof"] is True
    assert report["this_scaffold_alone_is_not_ship_criterion"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["does_not_rewrite_isolation_suite"] is True
    assert report["suite_delete_forbidden"] is True
    assert report["suite_path"] == "tests/test_tf_import_isolation.py"
    assert report["rewrite_not_delete"] is True
    assert report["scaffold_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["first_blocking_ok"] is True
    assert report["inventory_ok"] is True
    assert report["scaffold_steps_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert "NOT isolation rewrite shipped" in report["ok_criteria"]
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    assert report["distinct_from_isolation_design"] is True
    assert report["distinct_from_isolation_ship_met"] is True
    assert report["distinct_from_isolation_operational_prep"] is True
    assert report["distinct_from_path_execution_scaffold"] is True
    assert report["distinct_from_wire_fifth_prep"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "this_scaffold_alone",
        "this_execution_scaffold_alone",
        "operational_prep_alone",
        "path_execution_scaffold_alone",
        "design_alone",
        "ship_met_criteria_alone",
        "wire_fifth_prep_alone",
        "packaging_alone",
        "probe_linf",
        "residual_must_vanish",
    ):
        assert a in anti


def test_mutation_inventory_matches_live_suite_names():
    inv = tlb.case1_isolation_rewrite_execution_scaffold_mutation_inventory()
    assert inv["suite_path"] == "tests/test_tf_import_isolation.py"
    assert inv["suite_delete_forbidden"] is True
    assert inv["rewrite_not_delete"] is True
    assert inv["isolation_tests_rewritten_with_wire"] is False
    assert inv["n_current_tests"] == 8
    assert inv["mutation_status_today"] == "not_applied"
    assert inv["feature_flag_enabled_today"] is False
    root = Path(__file__).resolve().parents[1]
    suite = root / inv["suite_path"]
    assert suite.is_file()
    tree = ast.parse(suite.read_text(encoding="utf-8"))
    live = [
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    assert live == list(inv["current_tests"])
    assert live == list(tlb.CASE1_ISOLATION_REWRITE_EXECUTION_SCAFFOLD_CURRENT_TESTS)
    for t in inv["tests"]:
        assert t["executes_rewrite"] is False
        assert t["mutation_status_today"] == "not_applied"
        assert t["must_remain_after_rewrite"] is True
        assert t["test_name"] in live


def test_scaffold_steps_do_not_execute_rewrite():
    steps = tlb.case1_isolation_rewrite_first_blocker_scaffold_steps()
    assert len(steps) >= 8
    assert all(s["executes_rewrite"] is False for s in steps)
    ids = [s["step_id"] for s in steps]
    assert "suite_path_documented" in ids
    assert "rewrite_not_delete_locked" in ids
    assert "current_tests_inventoried" in ids
    assert "planned_post_wire_assertions_named" in ids
    assert "feature_flag_reserved_named" in ids
    assert "dual_ban_locks" in ids
    assert "first_blocking_still_isolation" in ids
    assert "scaffold_complete_is_not_rewrite_shipped" in ids


def test_go_board_prep_artifacts_include_execution_scaffold():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    arts = report["go_board_isolation_prep_artifacts"]
    assert any("execution_scaffold" in str(a) for a in arts)
    bp = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert bp["ok"] is True
    assert bp["first_blocking_coreq"] == "isolation_rewrite_with_wire"
    arts2 = (bp.get("file_level_prep_map") or {}).get("isolation_rewrite_with_wire", [])
    assert any("execution_scaffold" in str(a) for a in arts2)


def test_aliases_and_kind():
    a = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    b = tlb.offline_case1_isolation_rewrite_execution_scaffold_report()
    c = tlb.case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    d = tlb.multi_unit_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    assert a["kind"] == b["kind"] == c["kind"] == d["kind"]
    assert a["scaffold_present"] is True
    assert a["isolation_rewrite_shipped"] is False


def test_distinct_from_operational_prep_kind():
    sc = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    prep = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    assert sc["kind"] != prep["kind"]
    assert sc["kind"] != tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_KIND
    assert prep["ok"] is True
    assert sc["ok"] is True


def test_readiness_additive_flag_does_not_redefine_ready():
    r = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_isolation_rewrite_first_blocker_execution_scaffold=True,
    )
    assert r["admm_case1_isolation_rewrite_first_blocker_execution_scaffold_ok"] is True
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
        include_admm_case1_isolation_rewrite_first_blocker_execution_scaffold=False,
    )
    assert r_off["admm_case1_isolation_rewrite_first_blocker_execution_scaffold_ok"] is None
    assert r_off["ready_for_wire_discussion"] is ready


def test_preflight_additive_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_isolation_rewrite_first_blocker_execution_scaffold=True,
    )
    assert pf["admm_case1_isolation_rewrite_first_blocker_execution_scaffold_ok"] is True
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert len(pf["wire_blockers"]) > 0
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))


def test_source_purity_no_excel_pulp_tf_on_scaffold_hot_path():
    src = inspect.getsource(
        tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report
    )
    src_h = inspect.getsource(
        tlb._case1_isolation_rewrite_first_blocker_execution_scaffold_honesty_fields
    )
    blob = src + src_h
    assert "import excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "isolation_rewrite_shipped=True" not in blob
    assert "isolation_rewrite_shipped = True" not in blob
    assert "format_tf_offline" not in blob


def test_isolation_suite_file_still_exists_and_unchanged_path():
    suite = Path(tlb.CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH)
    root = Path(__file__).resolve().parents[1]
    assert (root / suite).is_file()


def test_negative_ship_flags_never_true():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    for k in (
        "isolation_rewrite_shipped",
        "isolation_tests_rewritten_with_wire",
        "isolation_ship_allowed_today",
        "path_shipped",
        "wire_shipped",
        "bundle_shipped",
        "form_label_change_shipped",
        "feature_flag_enabled_today",
        "criteria_met_today",
        "wire_ship_allowed_today",
        "gate_flip_allowed_today",
        "dual_linf_proof_allowed_today",
    ):
        assert report[k] is False
    assert report["dual_recovery_path"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert "BLENDER" not in report["units_affine_unchanged"]


def test_feasibility_scaffold_present_does_not_allow_rewrite_ship():
    report = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    assert report["scaffold_present"] is True
    assert report["isolation_rewrite_shipped"] is False
    assert report["isolation_ship_allowed_today"] is False
    # AND-lock: scaffold present must not co-exist with ship true
    assert not (
        report["scaffold_present"] is True and report["isolation_rewrite_shipped"] is True
    )
    assert not (
        report["scaffold_present"] is True
        and report["isolation_ship_allowed_today"] is True
    )
