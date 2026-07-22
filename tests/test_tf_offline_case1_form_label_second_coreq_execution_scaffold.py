"""E2/E1: offline Case-1 form_label second-coreq *execution scaffold*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the scaffold hot path. Locks:
- scaffold_present / execution_scaffold_present / form_label_scaffold_present True
- form_label_change_shipped False; form_label_ship_allowed_today False
- form_mutation_path_executed_today False; form classic
- first_blocking_coreq = isolation_rewrite_with_wire; order_hint_index=1
- is_first_blocking_coreq False
- dual_recovery_path is None; dual_linf_under_wire unproven; gate open
- path/wire/bundle/isolation ship flags hard false; feature_flag_enabled_today False
- mutation inventory sites not applied
- additive readiness does not redefine ready_for_wire_discussion
- scaffold ≠ form ship ≠ form criteria ≠ form operational prep ≠ isolation scaffold
  ≠ path scaffold ≠ wire fifth prep ≠ VERDICT
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
    report = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    assert (
        report["kind"]
        == tlb.CASE1_FORM_LABEL_SECOND_COREQ_EXECUTION_SCAFFOLD_KIND
    )
    assert report["kind"] == (
        "offline_case1_form_label_second_coreq_execution_scaffold"
    )
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["scaffold_present"] is True
    assert report["execution_scaffold_present"] is True
    assert report["form_label_scaffold_present"] is True
    assert report["first_blocking_coreq"] == "form_label_change_shipped"
    assert report["is_first_blocking_coreq"] is True
    assert report["order_hint_index"] == 1
    assert report["order_hint_coreq"] == "form_label_change_shipped"
    assert report["form_label_change_shipped"] is False
    assert report["form_label_ship_allowed_today"] is False
    assert report["form_mutation_path_executed_today"] is False
    assert report["isolation_rewrite_shipped"] is True
    assert report["isolation_ship_allowed_today"] is True
    assert report["isolation_tests_rewritten_with_wire"] is True
    assert report["path_shipped"] is False
    assert report["wire_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert report["wire_ship_allowed_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["feature_flag_enabled_today"] is False
    assert report["feature_flag_name"] == "enable_tf_affine_case1_wire"
    assert (
        report["form_mutation_path_name"]
        == "feature_flag_enable_tf_affine_case1_wire_then_set_model_form_to_planned"
    )
    assert report["scaffold_is_not_form_label_change_shipped"] is True
    assert report["scaffold_is_not_form_label_ship_allow"] is True
    assert report["scaffold_is_not_form_flip"] is True
    assert report["scaffold_is_not_wire"] is True
    assert report["scaffold_is_not_verdict_gate"] is True
    assert report["scaffold_is_not_dual_linf_under_wire_proof"] is True
    assert report["this_scaffold_alone_is_not_ship_criterion"] is True
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
    assert "NOT form_label_change_shipped" in report["ok_criteria"]
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    assert report["distinct_from_form_label_change_shipped_criteria_contract"] is True
    assert report["distinct_from_form_label_second_coreq_operational_prep"] is True
    assert report["distinct_from_isolation_execution_scaffold"] is True
    assert report["distinct_from_path_execution_scaffold"] is True
    assert report["distinct_from_wire_fifth_prep"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "this_scaffold_alone",
        "this_execution_scaffold_alone",
        "operational_prep_alone",
        "form_label_criteria_alone",
        "isolation_scaffold_alone",
        "path_execution_scaffold_alone",
        "design_alone",
        "ship_met_criteria_alone",
        "wire_fifth_prep_alone",
        "packaging_alone",
        "probe_linf",
        "residual_must_vanish",
    ):
        assert a in anti


def test_mutation_inventory_sites_not_applied():
    inv = tlb.case1_form_label_execution_scaffold_mutation_inventory()
    assert inv["form_current"] == "classic_2block_excel_path"
    assert inv["form_planned"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert (
        inv["form_mutation_path_name"]
        == "feature_flag_enable_tf_affine_case1_wire_then_set_model_form_to_planned"
    )
    assert inv["form_mutation_path_executed_today"] is False
    assert inv["mutation_status_today"] == "not_applied"
    assert inv["form_label_change_shipped"] is False
    assert inv["form_label_ship_allowed_today"] is False
    assert inv["feature_flag_enabled_today"] is False
    assert inv["inventory_is_not_form_ship"] is True
    assert inv["inventory_is_not_form_allow"] is True
    assert inv["inventory_is_not_form_flip"] is True
    assert inv["n_sites"] >= 5
    assert inv["is_first_blocking_coreq"] is False
    assert inv["order_hint_index"] == 1
    assert inv["first_blocking_coreq_unchanged"] == "isolation_rewrite_with_wire"
    for s in inv["sites"]:
        assert s["executes_form_flip"] is False
        assert s["mutation_status_today"] == "not_applied"
    ids = {s["site_id"] for s in inv["sites"]}
    for need in (
        "model_form_field",
        "feature_flag_enable",
        "checklist_form_label_change_shipped",
        "howto_meta_labels",
        "dual_recovery_path_tf_face",
    ):
        assert need in ids


def test_scaffold_steps_do_not_execute_form_flip():
    steps = tlb.case1_form_label_second_coreq_scaffold_steps()
    assert len(steps) >= 8
    assert all(s["executes_form_flip"] is False for s in steps)
    assert all(s.get("mutation_path_executed") is False for s in steps)
    ids = [s["step_id"] for s in steps]
    assert "form_current_documented" in ids
    assert "form_planned_documented" in ids
    assert "mutation_path_named_not_executed" in ids
    assert "feature_flag_reserved_named" in ids
    assert "dual_ban_locks" in ids
    assert "first_blocking_still_isolation" in ids
    assert "order_hint_index_1" in ids
    assert "scaffold_complete_is_not_form_label_shipped" in ids


def test_go_board_prep_artifacts_include_execution_scaffold():
    report = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    arts = report["go_board_form_label_prep_artifacts"]
    assert any("execution_scaffold" in str(a) for a in arts)
    bp = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert bp["ok"] is True
    assert bp["first_blocking_coreq"] == "form_label_change_shipped"
    arts2 = (bp.get("file_level_prep_map") or {}).get("form_label_change_shipped", [])
    assert any("execution_scaffold" in str(a) for a in arts2)


def test_aliases_and_kind():
    a = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    b = tlb.offline_case1_form_label_execution_scaffold_report()
    c = tlb.case1_form_label_second_coreq_execution_scaffold_report()
    d = tlb.multi_unit_case1_form_label_second_coreq_execution_scaffold_report()
    assert a["kind"] == b["kind"] == c["kind"] == d["kind"]
    assert a["scaffold_present"] is True
    assert a["form_label_change_shipped"] is False


def test_distinct_from_operational_prep_and_criteria_and_iso_scaffold():
    sc = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    prep = tlb.offline_case1_form_label_second_coreq_operational_prep_report()
    crit = tlb.offline_case1_form_label_change_shipped_criteria_contract_report()
    iso = tlb.offline_case1_isolation_rewrite_first_blocker_execution_scaffold_report()
    assert sc["kind"] != prep["kind"]
    assert sc["kind"] != crit["kind"]
    assert sc["kind"] != iso["kind"]
    assert sc["kind"] != tlb.CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_KIND
    assert prep["ok"] is True
    assert crit["ok"] is True
    assert iso["ok"] is True
    assert sc["ok"] is True


def test_readiness_additive_flag_does_not_redefine_ready():
    r = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_form_label_second_coreq_execution_scaffold=True,
    )
    assert r["admm_case1_form_label_second_coreq_execution_scaffold_ok"] is True
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
        include_admm_case1_form_label_second_coreq_execution_scaffold=False,
    )
    assert r_off["admm_case1_form_label_second_coreq_execution_scaffold_ok"] is None
    assert r_off["ready_for_wire_discussion"] is ready


def test_preflight_additive_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_form_label_second_coreq_execution_scaffold=True,
    )
    assert pf["admm_case1_form_label_second_coreq_execution_scaffold_ok"] is True
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert len(pf["wire_blockers"]) > 0
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))


def test_source_purity_no_excel_pulp_tf_on_scaffold_hot_path():
    src = inspect.getsource(
        tlb.offline_case1_form_label_second_coreq_execution_scaffold_report
    )
    src_h = inspect.getsource(
        tlb._case1_form_label_second_coreq_execution_scaffold_honesty_fields
    )
    blob = src + src_h
    assert "import excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "form_label_change_shipped=True" not in blob
    assert "form_label_change_shipped = True" not in blob
    assert "format_tf_offline" not in blob


def test_negative_ship_flags_never_true():
    report = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    for k in (
        "form_label_change_shipped",
        "form_label_ship_allowed_today",
        "form_mutation_path_executed_today",
                                "path_shipped",
        "wire_shipped",
        "bundle_shipped",
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


def test_feasibility_scaffold_present_does_not_allow_form_ship():
    report = tlb.offline_case1_form_label_second_coreq_execution_scaffold_report()
    assert report["scaffold_present"] is True
    assert report["form_label_change_shipped"] is False
    assert report["form_label_ship_allowed_today"] is False
    assert report["form_mutation_path_executed_today"] is False
    # AND-lock: scaffold present must not co-exist with form ship true
    assert not (
        report["scaffold_present"] is True
        and report["form_label_change_shipped"] is True
    )
    assert not (
        report["scaffold_present"] is True
        and report["form_label_ship_allowed_today"] is True
    )
    assert not (
        report["scaffold_present"] is True
        and report["form_mutation_path_executed_today"] is True
    )


def test_isolation_suite_file_still_exists():
    suite = Path("tests/test_tf_import_isolation.py")
    root = Path(__file__).resolve().parents[1]
    assert (root / suite).is_file()
