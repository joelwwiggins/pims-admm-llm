"""E1/E14: offline Case-1 dual-honest multi-blocker wire *implementation blueprint*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the blueprint hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- blueprint_present / implementation_blueprint_present / wire_go_board_present True
- first_blocking_coreq present (expected isolation_rewrite_with_wire)
- order_hint go-board / file-level prep map present
- rehearsal_present / wire_rehearsal_present True
- scaffold_present / execution_scaffold_present True (visibility link)
- path_shipped False; dual_honest_tf_aware_path_present ship-met False
- wire_shipped False; bundle_shipped False; bundle_ship_allowed_today False
- criteria_met_today False; isolation_rewrite_shipped True; form classic
- form_label_change_shipped False; online_linf_gate open; gate_flip False
- feature_flag_enabled_today False; UNITS FCC/COKER/CDU (no silent BLENDER)
- critical blockers still in DEFAULT_WIRE_BLOCKERS
- additive readiness flag does not redefine ready_for_wire_discussion
- blueprint ≠ path shipped ≠ ship-met ≠ wire ≠ bundle ≠ isolation rewrite shipped
  ≠ form ship ≠ VERDICT ≠ dual L∞ under wire proof
- order_hint is not an executor; no auto-wire
- excel_packaging_twin_deferred False on rehearsal (packaging present after #63)
- no excel_pipeline / pulp / tensorflow import on blueprint hot path
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


def test_checklist_stays_open_and_dual_linf_unproven():
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["isolation_rewrite_with_wire"] == "shipped"
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["online_linf_gate_under_tf_path"]
        == "open"
    )
    cl = tlb.case1_dual_linf_proof_checklist()
    assert "isolation_rewrite_with_wire" not in cl["dual_linf_proof_checklist_open_ids"]
    assert "online_linf_gate_under_tf_path" in cl["dual_linf_proof_checklist_open_ids"]
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert cl["dual_linf_proof_checklist_n_open"] >= 3


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert (
        report["kind"]
        == tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_KIND
    )
    assert (
        report["kind"]
        == "offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint"
    )
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["blueprint_present"] is True
    assert report["implementation_blueprint_present"] is True
    assert report["wire_go_board_present"] is True
    assert report["rehearsal_present"] is True
    assert report["wire_rehearsal_present"] is True
    assert report["scaffold_present"] is True
    assert report["execution_scaffold_present"] is True
    assert report["scaffold_compose_ok"] is True
    assert report["path_shipped"] is False
    assert report["dual_honest_tf_aware_path_present"] is False
    assert report["wire_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["bundle_ship_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["isolation_rewrite_shipped"] is True
    assert report["form_label_change_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["wire_ship_allowed_today"] is False
    assert report["ship_met_allowed_today"] is False
    assert report["isolation_ship_allowed_today"] is True
    assert report["form_label_ship_allowed_today"] is False
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["isolation_rewrite_checklist_open"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["gate_flip_allowed_today"] is False
    assert report["blueprint_is_not_path_shipped"] is True
    assert report["blueprint_is_not_path_present_for_ship"] is True
    assert report["blueprint_is_not_wire_shipped"] is True
    assert report["blueprint_is_not_wire"] is True
    assert report["blueprint_is_not_bundle_shipped"] is True
    assert report["blueprint_is_not_isolation_rewrite_shipped"] is True
    assert report["blueprint_is_not_form_label_change_shipped"] is True
    assert report["blueprint_is_not_ship_allow"] is True
    assert report["blueprint_is_not_verdict_gate"] is True
    assert report["blueprint_is_not_dual_linf_under_wire_proof"] is True
    assert report["this_blueprint_alone_is_not_ship_criterion"] is True
    assert report["this_blueprint_alone_is_not_multi_blocker_ship"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["no_blender_offline_affine_kernel_blocker_still_true"] is True
    assert report["feature_flag_enabled_today"] is False
    assert report["feature_flag_name"] == "enable_tf_affine_case1_wire"
    assert report["dual_recovery_path_today_on_tf_surface"] is None
    assert (
        report["dual_recovery_path_planned_when_shipped"]
        == "online_lambda_under_tf_aware_form_when_shipped"
    )
    assert "pure-admm" not in report["dual_recovery_path_planned_when_shipped"].lower()
    assert report["blueprint_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["go_board_ok"] is True
    assert report["first_blocking_ok"] is True
    assert report["scaffold_link_ok"] is True
    assert report["rehearsal_link_ok"] is True
    assert report["shape_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["blueprint_does_not_flip_dual_honest_tf_aware_path_present_met_today"] is True
    assert "NOT path shipped" in report["ok_criteria"]
    assert "NOT wire shipped" in report["ok_criteria"]
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["honest_pooling_path_present"] is True
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["cdu_surface"] == "offline_affine_base_delta"
    assert report["blender_surface"] == "linear_quality_pooling"
    assert report["any_ship_allowed_today"] is False
    assert report["all_ship_flags_false"] is True
    # Blueprint packaging twin present after Excel packaging ship (existence only)
    assert report["excel_packaging_twin_deferred"] is False
    assert report["excel_packaging_twin_present"] is True
    assert report["excel_rehearsal_packaging_twin_deferred"] is False
    assert report["excel_rehearsal_packaging_twin_present"] is True
    anti = set(report["anti_criteria_today"])
    for a in (
        "probe_linf",
        "bridge_linf",
        "warmstart_linf",
        "pooling_linf",
        "seed_identity_linf",
        "recovered_blender_linf",
        "residual_must_vanish",
        "packaging_alone",
        "design_contracts_alone",
        "this_scaffold_alone",
        "this_rehearsal_alone",
        "this_blueprint_alone",
        "go_board_alone",
        "first_blocking_coreq_alone",
        "prep_map_alone",
        "scaffold_plus_rehearsal_plus_blueprint_alone",
        "path_design_alone",
        "path_present_criteria_alone",
        "bundle_design_alone",
        "bundle_ship_met_criteria_alone",
        "wire_ship_acceptance_alone",
        "case1_shaped_linking_skeleton_alone",
        "diagnostic_linf_alone",
    ):
        assert a in anti


def test_first_blocking_coreq_expected_form_label():
    fb = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq()
    assert fb["first_blocking_coreq"] == "form_label_change_shipped"
    assert fb["matches_expected_today"] is True
    assert fb["order_hint_exhausted"] is False
    assert fb["order_hint_is_not_executor"] is True
    assert fb["no_auto_wire"] is True
    assert fb["does_not_set_isolation_rewrite_shipped"] is False  # isolation already shipped
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert report["first_blocking_coreq"] == "form_label_change_shipped"
    assert report["first_blocking_coreq_order_index"] == 1


def test_go_board_order_hint_coverage_and_prep_map():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    rows = report["order_hint_rows"]
    order_ids = [r["coreq_id"] for r in rows]
    assert order_ids == list(tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    assert report["order_hint"] == list(
        tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT
    )
    fb_rows = [r for r in rows if r.get("is_first_blocking")]
    assert len(fb_rows) == 1
    assert fb_rows[0]["coreq_id"] == report["first_blocking_coreq"]
    assert rows[0]["coreq_id"] == "isolation_rewrite_with_wire"
    assert rows[0]["status"] == "shipped"
    assert rows[0]["ship_flag_still_false"] is False
    assert rows[0]["is_first_blocking"] is False
    assert rows[1]["is_first_blocking"] is True
    for row in rows:
        # isolation is shipped; dual_linf uses unproven flag; remaining coreqs still false_today
        if row["coreq_id"] == "isolation_rewrite_with_wire":
            assert row["ship_flag_still_false"] is False
            assert row["status"] == "shipped"
        else:
            assert row["ship_flag_still_false"] is True or row["coreq_id"] == "dual_linf_under_wire_proven"
        assert isinstance(row["prep_artifacts"], list) and len(row["prep_artifacts"]) >= 1
    prep = report["file_level_prep_map"]
    assert "isolation_rewrite_with_wire" in prep
    assert any("test_tf_import_isolation.py" in a for a in prep["isolation_rewrite_with_wire"])
    companions = {c["coreq_id"] for c in report["companion_rows"]}
    for cid in (
        "bundle_shipped",
        "no_blender_offline_affine_kernel",
        "case1_is_cdu_blender_package_admm",
        "feature_flag_enabled_today",
        "dual_recovery_path",
    ):
        assert cid in companions
    assert report["go_board"]["does_not_rewrite_isolation_suite"] is True
    assert report["go_board"]["static_prep_only"] is True


def test_multi_blocker_coreq_visibility_without_flip():
    before_wire = tlb.case1_wire_ship_acceptance_criteria_met_today_map()
    before_path = tlb.case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    before_bundle = (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    after_wire = tlb.case1_wire_ship_acceptance_criteria_met_today_map()
    after_path = tlb.case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    after_bundle = (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    assert before_wire == after_wire
    assert before_path == after_path
    assert before_bundle == after_bundle

    coreqs = report["multi_blocker_coreqs"]
    assert coreqs["isolation_rewrite_with_wire"] == "shipped"
    assert coreqs["isolation_rewrite_shipped"] is True
    assert coreqs["form_label_change_shipped"] is False
    assert coreqs["dual_honest_tf_aware_path_present"] is False
    assert coreqs["dual_linf_under_wire"] == "unproven"
    assert coreqs["online_linf_gate_under_tf_path"] == "open"
    assert coreqs["wire_shipped"] is False
    assert coreqs["bundle_shipped"] is False
    assert coreqs["bundle_ship_allowed_today"] is False
    assert coreqs["criteria_met_today"] is False
    assert coreqs["blueprint_present"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True


def test_critical_blockers_still_present():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert "no_blender_offline_affine_kernel" in tlb.DEFAULT_WIRE_BLOCKERS
    assert "isolation_rewrite_required" not in tlb.DEFAULT_WIRE_BLOCKERS
    assert "wire_not_shipped" in tlb.DEFAULT_WIRE_BLOCKERS
    assert (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp"
        in tlb.DEFAULT_WIRE_BLOCKERS
    )
    assert report["blockers_still_documented"] is True
    assert report.get("isolation_rewrite_required_in_default_wire_blockers", False) is False
    assert report["no_blender_offline_affine_kernel_in_default_wire_blockers"] is True
    assert report["wire_not_shipped_blocker_still_true"] is True
    assert report["dual_linf_under_wire_unproven_blocker_still_true"] is True


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_KIND",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANNOTATION",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANTI_CRITERIA_TODAY",
        "case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq",
        "case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board",
        "offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
        "case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
        "multi_unit_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    b = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    c = tlb.multi_unit_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_dual_honest_multi_blocker_wire_implementation_blueprint_honesty_fields,
        tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq,
        tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board,
        tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report,
    ]
    blob = "".join(inspect.getsource(f) for f in funcs)
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "from pulp" not in blob
    assert "from tensorflow" not in blob
    assert "import excel_pipeline" not in blob
    assert "from excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "from .excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob
    assert "auto_wire_execute" not in blob
    assert "enable_tf_affine_case1_wire = True" not in blob


def test_honesty_sync_rehearsal_packaging_not_deferred():
    rehearsal = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert rehearsal["excel_packaging_twin_deferred"] is False
    assert rehearsal.get("excel_packaging_twin_present") is True
    scaffold = tlb.offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
    assert scaffold["excel_packaging_twin_deferred"] is False
    assert scaffold.get("excel_packaging_twin_present") is True
    note = rehearsal.get("note") or ""
    assert "packaging twin of rehearsal deferred" not in note.lower() or "present" in note.lower()


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint=True,
        include_admm_case1_dual_honest_multi_blocker_wire_rehearsal=True,
        include_admm_case1_dual_honest_tf_aware_path_execution_scaffold=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=True,
        include_admm_case1_wire_ship_acceptance_design_contract=True,
        include_admm_case1_isolation_rewrite_design_contract=True,
        include_admm_case1_online_linf_gate_criteria_contract=True,
        include_admm_case1_honest_blender_pooling_path=True,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=True,
        include_admm_case1_dual_space_linf_live_lambda_bridge=True,
        include_admm_case1_dual_space_linf_probe=True,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok" in rep
    assert rep["admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok"] is True
    assert rep["admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok"] is True
    assert rep["admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_blueprint_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint=False,
        include_admm_case1_dual_honest_multi_blocker_wire_rehearsal=False,
        include_admm_case1_dual_honest_tf_aware_path_execution_scaffold=False,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=False,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=False,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=False,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=False,
        include_admm_case1_wire_ship_acceptance_design_contract=False,
        include_admm_case1_isolation_rewrite_design_contract=False,
        include_admm_case1_online_linf_gate_criteria_contract=False,
        include_admm_case1_honest_blender_pooling_path=False,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=False,
        include_admm_case1_dual_space_linf_live_lambda_bridge=False,
        include_admm_case1_dual_space_linf_probe=False,
        include_admm_case1_dual_space_form_contract=False,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert rep["admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_blueprint_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint=True,
        include_admm_case1_dual_honest_multi_blocker_wire_rehearsal=True,
        include_admm_case1_dual_honest_tf_aware_path_execution_scaffold=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=True,
        include_admm_case1_wire_ship_acceptance_design_contract=True,
        include_admm_case1_isolation_rewrite_design_contract=True,
        include_admm_case1_online_linf_gate_criteria_contract=True,
        include_admm_case1_honest_blender_pooling_path=True,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=True,
        include_admm_case1_dual_space_linf_live_lambda_bridge=True,
        include_admm_case1_dual_space_linf_probe=True,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))
    assert "isolation_rewrite_required" not in pf["wire_blockers"]
    assert pf.get("admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok") is True
    assert pf.get("admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok") is True
    assert pf.get("admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_blueprint():
    meta = tlb.honesty_metadata()
    assert (
        meta.get("admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_available")
        is True
    )
    assert (
        meta.get("admm_case1_dual_honest_multi_blocker_wire_rehearsal_available")
        is True
    )
    assert meta["dual_recovery_path"] is None


def test_isolation_suite_file_not_modified_this_cycle():
    """Isolation suite behavior must stay classic — this cycle does not rewrite it."""
    path = Path(__file__).resolve().parent / "test_tf_import_isolation.py"
    assert path.is_file()


def test_rehearsal_and_scaffold_still_present_and_green():
    scaffold = tlb.offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
    assert scaffold["scaffold_present"] is True
    assert scaffold["execution_scaffold_present"] is True
    assert scaffold["compose_ok"] is True
    assert scaffold["ok"] is True
    assert scaffold["path_shipped"] is False
    assert scaffold["wire_shipped"] is False
    rehearsal = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert rehearsal["rehearsal_present"] is True
    assert rehearsal["ok"] is True
    assert rehearsal["path_shipped"] is False
    assert rehearsal["wire_shipped"] is False
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert report["scaffold_present"] is True
    assert report["scaffold_compose_ok"] is True
    assert report["scaffold_link_ok"] is True
    assert report["rehearsal_present"] is True
    assert report["rehearsal_link_ok"] is True


def test_form_contract_and_ladder_non_regression():
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
    assert contract["dual_linf_under_wire_status"] == "unproven"
    pool = tlb.offline_case1_honest_blender_pooling_path_report()
    assert pool["ok"] is True
    assert pool["dual_linf_under_wire_status"] == "unproven"
    probe = tlb.offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
    assert probe["probe_ok"] is True
    assert probe["dual_linf_under_wire_status"] == "unproven"
    bridge = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        allow_fixture_fallback=True,
        skeleton_n_rounds=1,
    )
    assert bridge["bridge_ok"] is True
    warm = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        allow_fixture_fallback=True,
        n_rounds=1,
    )
    assert warm["warmstart_ok"] is True
    assert warm["dual_linf_under_wire_status"] == "unproven"
    crit = tlb.offline_case1_online_linf_gate_criteria_contract_report()
    assert crit["online_linf_gate_under_tf_path"] == "open"
    assert crit["gate_flip_allowed_today"] is False
    design = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert design["isolation_rewrite_with_wire"] == "shipped"
    assert design["isolation_rewrite_shipped"] is True
    assert tlb.CASE1_FORM_CURRENT == "classic_2block_excel_path"
    ws = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert ws["wire_ship_allowed_today"] is False
    assert ws["wire_shipped"] is False
    path = tlb.offline_case1_dual_honest_tf_aware_path_design_contract_report()
    assert path["path_shipped"] is False
    assert path["dual_honest_tf_aware_path_present"] is False
    present = tlb.offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report()
    assert present["dual_honest_tf_aware_path_present"] is False
    assert present.get("ship_met_allowed_today", False) is False
    bundle = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert bundle["bundle_shipped"] is False
    assert bundle["bundle_ship_allowed_today"] is False
    assert bundle["criteria_met_today"] is False
    scaffold = tlb.offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
    assert scaffold["scaffold_present"] is True
    assert scaffold["path_shipped"] is False
    assert scaffold["wire_shipped"] is False
    assert scaffold["bundle_shipped"] is False
    assert scaffold["isolation_rewrite_shipped"] is True
    assert scaffold["form_label_change_shipped"] is False
    assert scaffold["dual_linf_under_wire_status"] == "unproven"
    rehearsal = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert rehearsal["rehearsal_present"] is True
    assert rehearsal["path_shipped"] is False
    assert rehearsal["wire_shipped"] is False
    assert rehearsal["bundle_shipped"] is False
    assert rehearsal["isolation_rewrite_shipped"] is True
    assert rehearsal["form_label_change_shipped"] is False
    assert rehearsal["dual_linf_under_wire_status"] == "unproven"
    blueprint = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert blueprint["blueprint_present"] is True
    assert blueprint["path_shipped"] is False
    assert blueprint["wire_shipped"] is False
    assert blueprint["bundle_shipped"] is False
    assert blueprint["isolation_rewrite_shipped"] is True
    assert blueprint["form_label_change_shipped"] is False
    assert blueprint["dual_linf_under_wire_status"] == "unproven"
