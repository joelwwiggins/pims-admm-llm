"""E2/E1: offline Case-1 wire fifth-coreq *execution scaffold*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the scaffold hot path. Locks:
- scaffold_present / execution_scaffold_present / wire_scaffold_present True
- wire_shipped False; wire_ship_allowed_today False; wire_land_path_executed_today False; dual_linf unproven
- criteria_met_today False; online_linf_gate open; gate_flip_allowed_today False
- wire_land_path_executed_today False
- first_blocking_coreq = isolation_rewrite_with_wire; order_hint_index=4
- is_first_blocking_coreq False
- dual_recovery_path is None; form classic
- path/wire/bundle/isolation/form ship flags hard false; feature_flag_enabled_today False
- proof-composition inventory pieces not executed
- additive readiness does not redefine ready_for_wire_discussion
- scaffold ≠ wire shipped ≠ wire allow ≠ wire fifth prep ≠ wire design ≠ preflight
  ≠ dual_linf scaffold ≠ form scaffold
  ≠ isolation scaffold ≠ path scaffold ≠ VERDICT
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
    report = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    assert (
        report["kind"]
        == tlb.CASE1_WIRE_FIFTH_COREQ_EXECUTION_SCAFFOLD_KIND
    )
    assert report["kind"] == (
        "offline_case1_wire_fifth_coreq_execution_scaffold"
    )
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["scaffold_present"] is True
    assert report["execution_scaffold_present"] is True
    assert report["wire_scaffold_present"] is True
    assert report["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    assert report["is_first_blocking_coreq"] is False
    assert report["order_hint_index"] == 4
    assert report["order_hint_coreq"] == "wire_shipped"
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["gate_flip_allowed_today"] is False
    assert report["wire_land_path_executed_today"] is False
    assert report["form_label_change_shipped"] is True
    assert report["form_label_ship_allowed_today"] is True
    assert report["isolation_rewrite_shipped"] is True
    assert report["isolation_ship_allowed_today"] is True
    assert report["isolation_tests_rewritten_with_wire"] is True
    assert report["path_shipped"] is True
    assert report["wire_shipped"] is False
    assert report["wire_ship_criteria_met_today"] is False
    assert report["wire_land_path_executed_today"] is False
    assert report["bundle_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["case1_form_unchanged"] is False
    assert report["form_current"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert report["wire_ship_allowed_today"] is False
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["feature_flag_enabled_today"] is False
    assert report["feature_flag_name"] == "enable_tf_affine_case1_wire"
    assert (
        report["wire_land_path_name"]
        == tlb.CASE1_WIRE_LAND_PATH_NAME
    )
    assert report["scaffold_is_not_wire_shipped"] is True
    assert report["scaffold_is_not_wire_allow"] is True
    assert report["scaffold_is_not_gate_flip"] is True
    assert report["scaffold_is_not_wire"] is True
    assert report["scaffold_is_not_verdict_gate"] is True
    assert report["scaffold_is_not_wire"] is True
    assert report["this_scaffold_alone_is_not_wire_ship"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["does_not_rewrite_isolation_suite"] is True
    assert report["scaffold_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["first_blocking_ok"] is True
    assert report["inventory_ok"] is True
    assert report["scaffold_steps_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert "NOT wire shipped" in report["ok_criteria"]
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    assert report["distinct_from_wire_fifth_operational_prep"] is True
    assert report["distinct_from_wire_ship_acceptance_design"] is True
    assert report["distinct_from_offline_wire_preflight"] is True
    assert report["distinct_from_dual_linf_fourth_coreq_execution_scaffold"] is True
    assert report["distinct_from_form_label_execution_scaffold"] is True
    assert report["distinct_from_isolation_execution_scaffold"] is True
    assert report["distinct_from_path_execution_scaffold"] is True
    assert report["distinct_from_wire_fifth_prep"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "this_scaffold_alone",
        "this_execution_scaffold_alone",
        "operational_prep_alone",
        "wire_fifth_prep_alone",
        "wire_ship_acceptance_design_alone",
        "preflight_alone",
        "dual_linf_scaffold_alone",
        "dual_linf_criteria_alone",
        "form_scaffold_alone",
        "isolation_scaffold_alone",
        "path_execution_scaffold_alone",
        "design_alone",
        "ship_met_criteria_alone",
        "packaging_alone",
        "probe_linf",
        "residual_must_vanish",
        "go_board_alone",
        "blueprint_alone",
        "rehearsal_alone",
    ):
        assert a in anti


def test_wire_land_composition_inventory_pieces_not_executed():
    inv = tlb.case1_wire_execution_scaffold_wire_land_composition_inventory()
    assert inv["dual_linf_under_wire_status"] == "unproven"
    assert inv["dual_linf_proof_allowed_today"] is False
    assert inv["criteria_met_today"] is False
    assert inv["gate_flip_allowed_today"] is False
    assert inv["online_linf_gate_under_tf_path"] == "open"
    assert (
        inv["wire_land_path_name"]
        == tlb.CASE1_WIRE_LAND_PATH_NAME
    )
    assert inv["wire_land_path_executed_today"] is False
    assert inv["composition_status_today"] == "not_executed"
    assert inv["dual_recovery_path"] is None
    assert inv["feature_flag_enabled_today"] is False
    assert inv["inventory_is_not_wire_shipped"] is True
    assert inv["inventory_is_not_wire_allow"] is True
    assert inv["inventory_is_not_verdict"] is True
    assert inv["n_pieces"] >= 8
    assert inv["is_first_blocking_coreq"] is False
    assert inv["order_hint_index"] == 4
    assert inv["first_blocking_coreq_unchanged"] == "dual_linf_under_wire_proven"
    for p in inv["pieces"]:
        assert p["executes_wire_land"] is False
        assert p["ships_wire"] is False
        assert p["composition_status_today"] == "not_executed"
    ids = {p["piece_id"] for p in inv["pieces"]}
    for need in (
        "wire_ship_acceptance_design",
        "wire_fifth_operational_prep",
        "offline_preflight_blockers_still_true",
        "prior_coreq_scaffolds_inventory_only",
        "feature_flag_reserved_named_false",
        "planned_dual_recovery_path_under_wire",
        "dual_linf_proof_checklist_open_ids",
        "named_wire_land_path",
    ):
        assert need in ids


def test_scaffold_steps_do_not_execute_wire_land():
    steps = tlb.case1_wire_fifth_coreq_scaffold_steps()
    assert len(steps) >= 8
    assert all(s["executes_wire_land"] is False for s in steps)
    assert all(s.get("ships_wire") is False for s in steps)
    assert all(s.get("wire_land_path_executed") is False for s in steps)
    ids = [s["step_id"] for s in steps]
    assert "wire_ship_acceptance_design_documented" in ids
    assert "wire_fifth_operational_prep_documented" in ids
    assert "preflight_blockers_documented_still_true" in ids
    assert "dual_ban_locks" in ids
    assert "dual_recovery_path_none_today_planned_under_wire_labeled" in ids
    assert "feature_flag_reserved_named" in ids
    assert "first_blocking_still_isolation" in ids
    assert "order_hint_index_4" in ids
    assert "is_first_blocking_coreq_false" in ids
    assert "scaffold_complete_is_not_wire_shipped" in ids


def test_go_board_prep_artifacts_include_execution_scaffold():
    report = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    arts = report["go_board_wire_prep_artifacts"]
    assert any("execution_scaffold" in str(a) for a in arts)
    bp = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert bp["ok"] is True
    assert bp["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    arts2 = (bp.get("file_level_prep_map") or {}).get("wire_shipped", [])
    assert any("execution_scaffold" in str(a) for a in arts2)


def test_aliases_and_kind():
    a = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    b = tlb.offline_case1_wire_execution_scaffold_report()
    c = tlb.case1_wire_fifth_coreq_execution_scaffold_report()
    d = tlb.multi_unit_case1_wire_fifth_coreq_execution_scaffold_report()
    assert a["kind"] == b["kind"] == c["kind"] == d["kind"]
    assert a["scaffold_present"] is True
    assert a["dual_linf_under_wire_status"] == "unproven"
    assert a["dual_linf_proof_allowed_today"] is False


def test_distinct_from_prep_design_preflight_dual_linf_form_iso_path_scaffold():
    sc = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    prep = tlb.offline_case1_wire_fifth_coreq_operational_prep_report()
    design = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    preflight = tlb.offline_wire_preflight_report(readiness_n_repeats=5, readiness_warmup=0)
    dual_sc = tlb.offline_case1_dual_linf_fourth_coreq_execution_scaffold_report()
    form = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    iso = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    path = tlb.offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
    assert sc["kind"] != prep["kind"]
    assert sc["kind"] != design["kind"]
    assert sc["kind"] != dual_sc["kind"]
    assert sc["kind"] != form["kind"]
    assert sc["kind"] != iso["kind"]
    assert sc["kind"] != path["kind"]
    assert prep["ok"] is True
    assert design["ok"] is True
    assert preflight.get("preflight_ok", preflight.get("ok")) is not False
    assert dual_sc["ok"] is True
    assert form["ok"] is True
    assert iso["ok"] is True
    assert path["ok"] is True
    assert sc["ok"] is True



def test_readiness_additive_flag_does_not_redefine_ready():
    r = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_wire_fifth_coreq_execution_scaffold=True,
    )
    assert r["admm_case1_wire_fifth_coreq_execution_scaffold_ok"] is True
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
        include_admm_case1_wire_fifth_coreq_execution_scaffold=False,
    )
    assert r_off["admm_case1_wire_fifth_coreq_execution_scaffold_ok"] is None
    assert r_off["ready_for_wire_discussion"] is ready


def test_preflight_additive_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_wire_fifth_coreq_execution_scaffold=True,
    )
    assert pf["admm_case1_wire_fifth_coreq_execution_scaffold_ok"] is True
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert len(pf["wire_blockers"]) > 0
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))


def test_source_purity_no_excel_pulp_tf_on_scaffold_hot_path():
    src = inspect.getsource(
        tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report
    )
    src_h = inspect.getsource(
        tlb._case1_wire_fifth_coreq_execution_scaffold_honesty_fields
    )
    blob = src + src_h
    assert "import excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "wire_shipped=True" not in blob
    assert "wire_shipped = True" not in blob
    assert "wire_ship_allowed_today = True" not in blob
    assert "wire_ship_allowed_today=True" not in blob
    assert "format_tf_offline" not in blob


def test_negative_ship_flags_never_true():
    report = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    for k in (
        "wire_shipped",
        "bundle_shipped",
        "feature_flag_enabled_today",
        "criteria_met_today",
        "wire_ship_allowed_today",
        "gate_flip_allowed_today",
        "dual_linf_proof_allowed_today",
        "wire_land_path_executed_today",
    ):
        assert report[k] is False
    assert report["dual_recovery_path"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert "BLENDER" not in report["units_affine_unchanged"]


def test_feasibility_scaffold_present_does_not_allow_wire_ship():
    report = tlb.offline_case1_wire_fifth_coreq_execution_scaffold_report()
    assert report["scaffold_present"] is True
    assert report["wire_shipped"] is False
    assert report["wire_ship_allowed_today"] is False
    assert report["wire_land_path_executed_today"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert not (
        report["scaffold_present"] is True and report["wire_shipped"] is True
    )
    assert not (
        report["scaffold_present"] is True and report["wire_ship_allowed_today"] is True
    )
    assert not (
        report["scaffold_present"] is True
        and report["wire_land_path_executed_today"] is True
    )



def test_isolation_suite_file_still_exists():
    suite = Path("tests/test_tf_import_isolation.py")
    root = Path(__file__).resolve().parents[1]
    assert (root / suite).is_file()
