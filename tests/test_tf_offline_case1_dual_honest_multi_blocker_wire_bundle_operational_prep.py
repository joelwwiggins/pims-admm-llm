"""E2: offline Case-1 dual-honest multi-blocker wire *bundle companion operational prep*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the prep hot path. Locks:
- prep_present / bundle_prep_present / companion_bundle_prep_present True
- bundle_shipped False; bundle_ship_allowed_today False; criteria_met_today False
- bundle_land_path_executed_today False
- wire_shipped False; wire_ship_allowed_today False
- dual_linf_under_wire unproven; dual_linf_proof_allowed_today False
- gate open; gate_flip_allowed_today False
- feature_flag_enabled_today False; feature flag named
- first_blocking_coreq = form_label_change_shipped (bundle is companion)
- is_first_blocking_coreq False; companion_not_order_hint_primary True
- dual_recovery_path is None
- path/wire/bundle/isolation/form ship flags hard false
- UNITS FCC/COKER/CDU
- distinct from bundle design / ship-met criteria / wire fifth prep+scaffold
- additive readiness does not redefine ready_for_wire_discussion
- prep ≠ bundle shipped ≠ bundle allow ≠ wire shipped ≠ dual_linf proven ≠ gate flip ≠ VERDICT
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
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    assert report["kind"] == tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_OPERATIONAL_PREP_KIND
    assert report["kind"] == "offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["prep_present"] is True
    assert report["bundle_prep_present"] is True
    assert report["companion_bundle_prep_present"] is True
    assert report["operational_prep_present"] is True
    assert report["first_blocking_coreq"] == "form_label_change_shipped"
    assert report["is_first_blocking_coreq"] is False
    assert report["companion_not_order_hint_primary"] is True
    assert report["bundle_shipped"] is False
    assert report["bundle_ship_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["bundle_land_path_executed_today"] is False
    assert report["wire_shipped"] is False
    assert report["wire_ship_allowed_today"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["path_shipped"] is False
    assert report["dual_honest_tf_aware_path_present"] is False
    assert report["feature_flag_enabled_today"] is False
    assert (
        report["feature_flag_name"]
        == tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
    )
    assert report["isolation_rewrite_shipped"] is True
    assert report["form_label_change_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["prep_is_not_bundle_shipped"] is True
    assert report.get("prep_is_not_bundle_ship_allow") is True
    assert report["prep_is_not_wire_shipped"] is True
    assert report["prep_is_not_wire"] is True
    assert report["prep_is_not_verdict_gate"] is True
    assert report["prep_is_not_feature_flag_enable"] is True
    assert report["this_prep_alone_is_not_bundle_shipped"] is True
    assert report["distinct_from_bundle_design_contract"] is True
    assert report["distinct_from_bundle_ship_met_criteria_contract"] is True
    assert report["distinct_from_wire_fifth_coreq_operational_prep"] is True
    assert report["distinct_from_wire_fifth_coreq_execution_scaffold"] is True
    assert report["this_prep_formalizes_how_bundle_prep_lands_without_ship"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["does_not_rewrite_isolation_suite"] is True
    assert report["prep_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["first_blocking_ok"] is True
    assert report["companion_ok"] is True
    assert report["prep_steps_ok"] is True
    assert report["inventory_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert "NOT bundle shipped" in report["ok_criteria"] or "NOT bundle" in report["ok_criteria"]
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "this_prep_alone",
        "design_alone",
        "bundle_design_alone",
        "ship_met_criteria_alone",
        "packaging_alone",
        "preflight_alone",
        "wire_scaffold_alone",
        "wire_prep_alone",
        "dual_linf_scaffold_alone",
        "form_scaffold_alone",
        "path_scaffold_alone",
        "iso_scaffold_alone",
        "probe_linf",
        "bridge_linf",
        "warmstart_linf",
        "recovered_blender_linf",
        "residual_must_vanish",
        "blueprint_alone",
        "rehearsal_alone",
        "go_board_alone",
    ):
        assert a in anti, a


def test_how_bundle_prep_lands_inventory():
    inv = tlb.case1_bundle_companion_operational_prep_land_composition_inventory()
    assert inv["bundle_shipped"] is False
    assert inv["bundle_ship_allowed_today"] is False
    assert inv["criteria_met_today"] is False
    assert inv["bundle_land_path_executed_today"] is False
    assert inv["composition_status_today"] == "not_executed"
    assert inv["inventory_ok"] is True
    assert inv["inventory_ok_is_not_bundle_ship_allowed"] is True
    assert inv["first_blocking_coreq"] == "form_label_change_shipped"
    assert inv["is_first_blocking_coreq"] is False
    assert inv["companion_not_order_hint_primary"] is True
    assert inv["dual_recovery_path"] is None
    assert inv["dual_recovery_path_planned_when_shipped"] == (
        tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
    )
    assert inv["n_pieces"] >= 7
    assert all(p.get("executed_today") is False for p in inv["pieces"])
    assert all(p.get("ships") is False for p in inv["pieces"])
    assert all(p.get("allows_bundle") is False for p in inv["pieces"])


def test_prep_steps_do_not_execute_bundle_land():
    steps = tlb.case1_dual_honest_multi_blocker_wire_bundle_operational_prep_steps()
    assert len(steps) >= 8
    assert all(s["executes_bundle_land"] is False for s in steps)
    assert all(s["ships"] is False for s in steps)
    assert all(s["allows_bundle"] is False for s in steps)
    ids = [s["step_id"] for s in steps]
    assert "bundle_design_documented" in ids
    assert "bundle_ship_met_criteria_documented" in ids
    assert "order_hint_coreq_preps_scaffolds_inventory_only" in ids
    assert "dual_ban_locks" in ids
    assert "first_blocking_still_isolation" in ids
    assert "companion_not_order_hint_primary" in ids
    assert "prep_complete_is_not_bundle_shipped" in ids
    assert "prep_complete_is_not_wire_shipped" in ids


def test_go_board_companion_prep_artifacts_include_operational_prep():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    arts = report["go_board_bundle_prep_artifacts"]
    assert any("operational_prep" in str(a) for a in arts)
    assert any("bundle_design" in str(a) for a in arts)
    assert report["go_board_link_ok"] is True


def test_aliases_and_kind():
    a = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    b = tlb.case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    c = tlb.offline_case1_bundle_companion_operational_prep_report()
    d = tlb.case1_bundle_operational_prep_report()
    e = tlb.multi_unit_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    assert a["kind"] == b["kind"] == c["kind"] == d["kind"] == e["kind"]
    assert a["prep_present"] is True
    assert a["bundle_shipped"] is False
    assert a["bundle_ship_allowed_today"] is False
    assert a["bundle_land_path_executed_today"] is False


def test_distinct_from_design_criteria_wire_prep_scaffold():
    prep = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    design = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report()
    criteria = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    wire_prep = tlb.offline_case1_wire_fifth_coreq_operational_prep_report()
    wire_scaf = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    assert prep["kind"] != design["kind"]
    assert prep["kind"] != criteria["kind"]
    assert prep["kind"] != wire_prep["kind"]
    assert prep["kind"] != wire_scaf["kind"]
    assert prep["kind"] == "offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep"
    assert prep["distinct_from_bundle_design_contract"] is True
    assert prep["distinct_from_bundle_ship_met_criteria_contract"] is True
    assert prep["distinct_from_wire_fifth_coreq_operational_prep"] is True
    assert prep["distinct_from_wire_fifth_coreq_execution_scaffold"] is True


def test_readiness_additive_flag_does_not_redefine_ready():
    r = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep=True,
    )
    assert r["admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_ok"] is True
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
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep=False,
    )
    assert r_off["admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_ok"] is None
    assert r_off["ready_for_wire_discussion"] is ready


def test_preflight_additive_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep=True,
    )
    assert pf["admm_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_ok"] is True
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert len(pf["wire_blockers"]) > 0
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))


def test_source_purity_no_excel_pulp_tf_on_prep_hot_path():
    src = inspect.getsource(
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report
    )
    src_h = inspect.getsource(
        tlb._case1_dual_honest_multi_blocker_wire_bundle_operational_prep_honesty_fields
    )
    blob = src + src_h
    assert "import excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "bundle_shipped=True" not in blob
    assert "bundle_shipped = True" not in blob
    assert "format_tf_offline" not in blob


def test_isolation_suite_file_still_exists():
    suite = Path("tests/test_tf_import_isolation.py")
    root = Path(__file__).resolve().parents[1]
    assert (root / suite).is_file()


def test_negative_ship_and_proof_flags_never_true():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    for k in (
        "path_shipped",
        "dual_honest_tf_aware_path_present",
        "ship_met_allowed_today",
        "path_present_criteria_met_today",
        "form_label_change_shipped",
        "form_label_ship_allowed_today",
        "wire_shipped",
        "bundle_shipped",
        "bundle_ship_allowed_today",
        "bundle_land_path_executed_today",
        "feature_flag_enabled_today",
        "criteria_met_today",
        "wire_ship_allowed_today",
        "gate_flip_allowed_today",
        "dual_linf_proof_allowed_today",
        "is_first_blocking_coreq",
    ):
        assert report[k] is False, k
    assert report["dual_recovery_path"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert "BLENDER" not in report["units_affine_unchanged"]


def test_ladder_non_regression_prior_residuals():
    form = tlb.offline_case1_form_label_second_coreq_operational_prep_report()
    assert form["ok"] is True
    assert form["prep_present"] is True
    assert form["form_label_change_shipped"] is False
    iso = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    assert iso["ok"] is True
    assert iso["prep_present"] is True
    assert iso["isolation_rewrite_shipped"] is True
    dl = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert dl["ok"] is True
    assert dl["dual_linf_under_wire_status"] == "unproven"
    path = tlb.offline_case1_path_third_coreq_operational_prep_report()
    assert path["ok"] is True
    assert path["prep_present"] is True
    assert path["path_shipped"] is False
    dl4 = tlb.offline_case1_dual_linf_fourth_coreq_operational_prep_report()
    assert dl4["ok"] is True
    assert dl4["prep_present"] is True
    wire_prep = tlb.offline_case1_wire_fifth_coreq_operational_prep_report()
    assert wire_prep["ok"] is True
    assert wire_prep["wire_shipped"] is False
    wire_scaf = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    assert wire_scaf["ok"] is True
    assert wire_scaf["wire_shipped"] is False
    design = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report()
    assert design["ok"] is True
    assert design["bundle_shipped"] is False
    criteria = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    assert criteria["ok"] is True
    assert criteria["bundle_shipped"] is False
    assert criteria["bundle_ship_allowed_today"] is False


def test_feasibility_and_lock_and():
    """E4 fold: prep_present cannot coexist with ship/allow true."""
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_report()
    assert report["prep_present"] is True
    assert report["bundle_shipped"] is False
    assert report["bundle_ship_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["bundle_land_path_executed_today"] is False
    assert report["wire_shipped"] is False
    assert report["wire_ship_allowed_today"] is False
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["path_shipped"] is False
    assert report["isolation_rewrite_shipped"] is True
    assert report["form_label_change_shipped"] is False
    assert report["feature_flag_enabled_today"] is False
    assert report["dual_recovery_path"] is None
    assert report["first_blocking_coreq"] == "form_label_change_shipped"
    assert report["is_first_blocking_coreq"] is False
    assert report["companion_not_order_hint_primary"] is True
    assert report["order_hint_is_not_executor"] is True
