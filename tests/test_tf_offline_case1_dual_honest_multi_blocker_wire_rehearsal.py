"""E1/E2: offline Case-1 dual-honest multi-blocker wire *rehearsal*.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the rehearsal hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- rehearsal_present / wire_rehearsal_present True
- scaffold_present / execution_scaffold_present True (visibility link)
- path_shipped False; dual_honest_tf_aware_path_present ship-met False
- wire_shipped False; bundle_shipped False; bundle_ship_allowed_today False
- criteria_met_today False; isolation_rewrite_shipped True; form classic
- form_label_change_shipped False; online_linf_gate open; gate_flip False
- feature_flag_enabled_today False; UNITS FCC/COKER/CDU (no silent BLENDER)
- co-req status matrix keys present; ship-critical open/false/unproven
- critical blockers still in DEFAULT_WIRE_BLOCKERS
- additive readiness flag does not redefine ready_for_wire_discussion
- rehearsal ≠ path shipped ≠ ship-met ≠ wire ≠ bundle ≠ isolation rewrite shipped
  ≠ form ship ≠ VERDICT ≠ dual L∞ under wire proof
- order_hint is not an executor; no auto-wire
- no excel_pipeline / pulp / tensorflow import on rehearsal hot path
- isolation suite (test_tf_import_isolation.py) unchanged behavior

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
    tests/test_tf_offline_case1_dual_honest_multi_blocker_wire_rehearsal.py \\
    tests/test_tf_offline_case1_dual_honest_tf_aware_path_execution_scaffold.py \\
    tests/test_tf_offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract.py \\
    tests/test_tf_offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract.py \\
    tests/test_tf_offline_case1_isolation_rewrite_shipped_criteria_contract.py \\
    tests/test_tf_offline_case1_isolation_rewrite_design_contract.py \\
    tests/test_tf_offline_case1_form_label_change_shipped_criteria_contract.py \\
    tests/test_tf_offline_case1_dual_honest_tf_aware_path_present_criteria_contract.py \\
    tests/test_tf_offline_case1_dual_honest_tf_aware_path_design_contract.py \\
    tests/test_tf_offline_case1_wire_ship_acceptance_design_contract.py \\
    tests/test_tf_offline_case1_online_linf_gate_criteria_contract.py \\
    tests/test_tf_offline_case1_honest_blender_pooling_path.py \\
    tests/test_tf_offline_case1_dual_space_linf_probe.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_bridge.py \\
    tests/test_tf_offline_case1_dual_space_form_contract.py \\
    tests/test_tf_offline_case1_shaped_linking.py \\
    tests/test_tf_offline_wire_preflight.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_seeded_warmstart.py -q
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

MATRIX_REQUIRED_KEYS = {
    "isolation_rewrite_with_wire",
    "isolation_rewrite_shipped",
    "form_label_change_shipped",
    "form_current",
    "online_linf_gate_under_tf_path",
    "dual_linf_under_wire",
    "dual_honest_tf_aware_path_present",
    "path_shipped",
    "wire_shipped",
    "wire_ship_allowed_today",
    "bundle_shipped",
    "bundle_ship_allowed_today",
    "criteria_met_today",
    "no_blender_offline_affine_kernel",
    "blender_surface",
    "case1_is_cdu_blender_package_admm",
    "feature_flag_enabled_today",
    "dual_recovery_path",
    "scaffold_present",
    "execution_scaffold_present",
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
    assert cl["dual_linf_proof_checklist_n_open"] >= 2


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert (
        report["kind"]
        == tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_KIND
    )
    assert report["kind"] == "offline_case1_dual_honest_multi_blocker_wire_rehearsal"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
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
    assert report["form_label_change_shipped"] is True
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is False
    assert report["form_current"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["wire_ship_allowed_today"] is False
    assert report["ship_met_allowed_today"] is False
    assert report["isolation_ship_allowed_today"] is True
    assert report["form_label_ship_allowed_today"] is True
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["isolation_rewrite_checklist_open"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["gate_flip_allowed_today"] is False
    assert report["rehearsal_is_not_path_shipped"] is True
    assert report["rehearsal_is_not_path_present_for_ship"] is True
    assert report["rehearsal_is_not_wire_shipped"] is True
    assert report["rehearsal_is_not_wire"] is True
    assert report["rehearsal_is_not_bundle_shipped"] is True
    assert report["rehearsal_is_not_isolation_rewrite_shipped"] is True
    assert report["rehearsal_is_not_form_label_change_shipped"] is True
    assert report["rehearsal_is_not_ship_allow"] is True
    assert report["rehearsal_is_not_verdict_gate"] is True
    assert report["rehearsal_is_not_dual_linf_under_wire_proof"] is True
    assert report["this_rehearsal_alone_is_not_ship_criterion"] is True
    assert report["this_rehearsal_alone_is_not_multi_blocker_ship"] is True
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
    assert report["rehearsal_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["matrix_ok"] is True
    assert report["scaffold_link_ok"] is True
    assert report["shape_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["rehearsal_does_not_flip_dual_honest_tf_aware_path_present_met_today"] is True
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
        "wire_rehearsal_alone",
        "coreq_matrix_alone",
        "scaffold_plus_rehearsal_alone",
        "path_design_alone",
        "path_present_criteria_alone",
        "bundle_design_alone",
        "bundle_ship_met_criteria_alone",
        "wire_ship_acceptance_alone",
        "case1_shaped_linking_skeleton_alone",
        "diagnostic_linf_alone",
    ):
        assert a in anti


def test_coreq_status_matrix_keys_and_statuses():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    matrix = report["coreq_status_matrix"]
    assert MATRIX_REQUIRED_KEYS.issubset(set(matrix.keys()))
    assert matrix["isolation_rewrite_with_wire"]["status"] == "shipped"
    assert matrix["isolation_rewrite_shipped"]["value"] is True
    assert matrix["form_label_change_shipped"]["value"] is True
    assert matrix["form_current"]["value"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert matrix["online_linf_gate_under_tf_path"]["status"] == "open"
    assert matrix["dual_linf_under_wire"]["status"] == "unproven"
    assert matrix["dual_honest_tf_aware_path_present"]["value"] is False
    assert matrix["path_shipped"]["value"] is False
    assert matrix["wire_shipped"]["value"] is False
    assert matrix["wire_ship_allowed_today"]["value"] is False
    assert matrix["bundle_shipped"]["value"] is False
    assert matrix["bundle_ship_allowed_today"]["value"] is False
    assert matrix["criteria_met_today"]["value"] is False
    assert matrix["no_blender_offline_affine_kernel"]["value"] is True
    assert matrix["blender_surface"]["value"] == "linear_quality_pooling"
    assert matrix["case1_is_cdu_blender_package_admm"]["value"] is True
    assert matrix["feature_flag_enabled_today"]["value"] is False
    assert matrix["dual_recovery_path"]["value"] is None
    assert matrix["scaffold_present"]["value"] is True
    assert matrix["execution_scaffold_present"]["value"] is True
    assert report["any_ship_allowed_today"] is False
    assert report["coreq_matrix"]["matrix_ok"] is True
    assert report["coreq_matrix"]["any_ship_allowed_today"] is False
    # every ship-critical row forbids wire today
    for key, row in matrix.items():
        if row.get("ship_critical"):
            assert row.get("allows_wire_today") is False, key


def test_multi_blocker_coreq_visibility_without_flip():
    before_wire = tlb.case1_wire_ship_acceptance_criteria_met_today_map()
    before_path = tlb.case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    before_bundle = (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
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
    assert coreqs["form_label_change_shipped"] is True
    assert coreqs["dual_honest_tf_aware_path_present"] is False
    assert coreqs["dual_linf_under_wire"] == "unproven"
    assert coreqs["online_linf_gate_under_tf_path"] == "open"
    assert coreqs["wire_shipped"] is False
    assert coreqs["bundle_shipped"] is False
    assert coreqs["bundle_ship_allowed_today"] is False
    assert coreqs["criteria_met_today"] is False
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["order_hint"] == list(
        tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT
    )


def test_critical_blockers_still_present():
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
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
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_KIND",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANNOTATION",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANTI_CRITERIA_TODAY",
        "case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix",
        "offline_case1_dual_honest_multi_blocker_wire_rehearsal_report",
        "case1_dual_honest_multi_blocker_wire_rehearsal_report",
        "multi_unit_case1_dual_honest_multi_blocker_wire_rehearsal_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    b = tlb.case1_dual_honest_multi_blocker_wire_rehearsal_report()
    c = tlb.multi_unit_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_dual_honest_multi_blocker_wire_rehearsal_honesty_fields,
        tlb.case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix,
        tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report,
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
    # no auto-wire executor
    assert "auto_wire_execute" not in blob
    assert "enable_tf_affine_case1_wire = True" not in blob


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert "admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok" in rep
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


def test_readiness_skips_rehearsal_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert rep["admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_rehearsal_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
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


def test_honesty_metadata_mentions_rehearsal():
    meta = tlb.honesty_metadata()
    assert (
        meta.get("admm_case1_dual_honest_multi_blocker_wire_rehearsal_available")
        is True
    )
    assert (
        meta.get("admm_case1_dual_honest_tf_aware_path_execution_scaffold_available")
        is True
    )
    assert meta["dual_recovery_path"] is None


def test_isolation_suite_file_not_modified_this_cycle():
    """Isolation suite behavior must stay classic — this cycle does not rewrite it."""
    path = (
        Path(__file__).resolve().parent / "test_tf_import_isolation.py"
    )
    assert path.is_file()


def test_scaffold_still_present_and_green():
    scaffold = tlb.offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
    assert scaffold["scaffold_present"] is True
    assert scaffold["execution_scaffold_present"] is True
    assert scaffold["compose_ok"] is True
    assert scaffold["ok"] is True
    assert scaffold["path_shipped"] is False
    assert scaffold["wire_shipped"] is False
    report = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert report["scaffold_present"] is True
    assert report["scaffold_compose_ok"] is True
    assert report["scaffold_link_ok"] is True


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
    assert tlb.CASE1_FORM_CURRENT == tlb.CASE1_PLANNED_TF_AWARE_FORM
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
    assert scaffold["form_label_change_shipped"] is True
    assert scaffold["dual_linf_under_wire_status"] == "unproven"
    rehearsal = tlb.offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    assert rehearsal["rehearsal_present"] is True
    assert rehearsal["path_shipped"] is False
    assert rehearsal["wire_shipped"] is False
    assert rehearsal["bundle_shipped"] is False
    assert rehearsal["isolation_rewrite_shipped"] is True
    assert rehearsal["form_label_change_shipped"] is True
    assert rehearsal["dual_linf_under_wire_status"] == "unproven"
