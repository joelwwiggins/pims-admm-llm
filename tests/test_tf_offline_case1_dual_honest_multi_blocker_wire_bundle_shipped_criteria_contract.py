"""E1/E2: offline Case-1 multi-blocker wire bundle ship-met / flip criteria contract.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the criteria-contract hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- bundle_shipped False; bundle_ship_allowed_today False; criteria_met_today False
- criteria_present True; bundle_design_present True
- isolation_rewrite_shipped False; isolation checklist isolation_rewrite_with_wire open
- form classic_2block_excel_path; form_label_change_shipped False
- path_design_present True; path_shipped False
- dual_honest_tf_aware_path_present ship-met remains False
- ship_met_allowed_today False; wire_ship_allowed_today False
- online_linf_gate_under_tf_path remains "open"
- feature_flag_enabled_today False; order_hint is not executor
- UNITS still FCC/COKER/CDU (no silent BLENDER)
- critical blockers still in DEFAULT_WIRE_BLOCKERS
- additive readiness flag does not redefine ready_for_wire_discussion
- design *what* ≠ criteria *when* ≠ bundle shipped ≠ wire ≠ VERDICT
- anti-criteria include this_contract + this_bundle_design_alone + packaging_alone
- no excel_pipeline import on tf_linear_blocks criteria-contract hot path
- foreign isolation/form_label/path/gate/wire-ship met_today maps not flipped
- isolation suite test_tf_import_isolation.py behavior unchanged (not rewritten)

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
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
    "isolation_rewrite_required",
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}

REQUIRED_FLIP_KEYS = {
    "bundle_design_present",
    "isolation_rewrite_with_wire",
    "isolation_rewrite_shipped",
    "isolation_tests_rewritten_with_wire_not_deleted",
    "form_label_change_shipped",
    "dual_honest_tf_aware_path_present",
    "online_linf_gate_under_tf_path",
    "dual_linf_under_wire_proven",
    "wire_shipped",
    "dual_recovery_path_planned_labeled_honestly",
    "feature_flag_reserved_and_named",
    "no_silent_form_reuse",
    "rewrite_not_delete",
    "no_blender_affine_units",
    "case1_cdu_blender_package_shape_acknowledged",
}

UNDER_WIRE_ONLY = {
    "online_linf_gate_under_tf_path",
    "dual_linf_under_wire_proven",
}


def test_checklist_stays_open_and_dual_linf_unproven():
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["isolation_rewrite_with_wire"] == "open"
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["form_label_change_shipped"] == "open"
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["online_linf_gate_under_tf_path"]
        == "open"
    )
    cl = tlb.case1_dual_linf_proof_checklist()
    assert "isolation_rewrite_with_wire" in cl["dual_linf_proof_checklist_open_ids"]
    assert "form_label_change_shipped" in cl["dual_linf_proof_checklist_open_ids"]
    assert "online_linf_gate_under_tf_path" in cl["dual_linf_proof_checklist_open_ids"]
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert cl["dual_linf_proof_checklist_n_open"] >= 4


def test_flip_criteria_map_keys_and_classes():
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    flip = report["flip_criteria"]
    assert set(flip.keys()) == REQUIRED_FLIP_KEYS
    for k in REQUIRED_FLIP_KEYS:
        if k in UNDER_WIRE_ONLY:
            assert flip[k] == tlb.FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        else:
            assert flip[k] == tlb.FLIP_CRITERION_REQUIRED
    assert set(
        tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA.keys()
    ) == REQUIRED_FLIP_KEYS
    assert set(
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria().keys()
    ) == REQUIRED_FLIP_KEYS
    assert report["flip_criteria_formalized"] is True
    assert report["criteria_present"] is True
    assert report["bundle_design_present"] is True


def test_anti_criteria_include_this_contract_and_design_alone():
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
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
        "this_bundle_design_alone",
        "this_bundle_ship_criteria_contract_alone",
        "this_bundle_ship_met_criteria_alone",
        "this_contract_alone",
        "wire_ship_acceptance_design_alone",
        "isolation_design_alone",
        "isolation_ship_criteria_alone",
        "form_label_criteria_alone",
        "path_design_alone",
        "path_present_criteria_alone",
        "gate_criteria_alone",
    ):
        assert a in anti
    assert (
        set(tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY)
        == anti
    )
    # Design anti-criteria also lists criteria-alone tokens after ladder refresh.
    design_anti = set(tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ANTI_CRITERIA_TODAY)
    assert "this_bundle_ship_criteria_contract_alone" in design_anti
    assert "this_bundle_ship_met_criteria_alone" in design_anti


def test_report_always_on_honesty_locks():
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert (
        report["kind"]
        == tlb.CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_KIND
    )
    assert report["kind"] == (
        "offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract"
    )
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["bundle_shipped"] is False
    assert report["bundle_ship_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["criteria_present"] is True
    assert report["bundle_design_present"] is True
    assert report["path_design_present"] is True
    assert report["path_shipped"] is False
    assert report["dual_honest_tf_aware_path_present"] is False
    assert report["ship_met_allowed_today"] is False
    assert report["form_label_change_shipped"] is False
    assert report["form_label_ship_allowed_today"] is False
    assert report["isolation_rewrite_shipped"] is False
    assert report["isolation_ship_allowed_today"] is False
    assert report["wire_shipped"] is False
    assert report["wire_ship_allowed_today"] is False
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["isolation_rewrite_with_wire"] == "open"
    assert report["isolation_rewrite_still_open"] is True
    assert report["isolation_rewrite_checklist_open"] is True
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_is_not_bundle_shipped"] is True
    assert report["criteria_is_not_bundle_ship_allow"] is True
    assert report["criteria_is_not_wire_shipped"] is True
    assert report["criteria_is_not_wire"] is True
    assert report["criteria_is_not_isolation_rewrite_shipped"] is True
    assert report["criteria_is_not_form_flip"] is True
    assert report["criteria_is_not_gate_flip"] is True
    assert report["criteria_is_not_verdict_gate"] is True
    assert report["criteria_is_not_dual_linf_under_wire_proof"] is True
    assert report["criteria_is_not_ship_allow"] is True
    assert report["order_hint_is_not_executor"] is True
    assert report["no_auto_wire"] is True
    assert report["this_bundle_design_alone_is_not_ship_criterion"] is True
    assert report["this_bundle_ship_criteria_contract_alone_is_not_ship_criterion"] is True
    assert report["this_contract_alone_is_not_ship_criterion"] is True
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
    assert report["design_contract_ok"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["design_does_not_flip_form_label_change_shipped_met_today"] is True
    assert report["design_does_not_flip_wire_ship_met_today"] is True
    assert report["design_does_not_flip_gate_met_today"] is True
    assert report["design_does_not_flip_isolation_ship_met_today"] is True
    assert "NOT bundle shipped" in report["ok_criteria"]
    assert "NOT wire shipped" in report["ok_criteria"]
    assert "NOT isolation rewrite shipped" in report["ok_criteria"]
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["bundle_shipped_criteria_contract"] == "present"
    assert report["honest_pooling_path_present"] is True
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["cdu_surface"] == "offline_affine_base_delta"
    assert report["blender_surface"] == "linear_quality_pooling"
    assert report["form_label_change_still_open"] is True
    assert report["form_label_change_shipped_checklist"] == "open"


def test_permission_architecture_hard_false_with_structural_trues():
    assert tlb.case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today() is False
    assert (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate()
        is False
    )
    assert tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped() is False
    met = tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    # Structural labels may be True; co-reqs False; aggregate still False
    assert met["bundle_design_present"] is True
    assert met["dual_recovery_path_planned_labeled_honestly"] is True
    assert met["feature_flag_reserved_and_named"] is True
    assert met["no_silent_form_reuse"] is True
    assert met["rewrite_not_delete"] is True
    assert met["no_blender_affine_units"] is True
    assert met["case1_cdu_blender_package_shape_acknowledged"] is True
    assert met["isolation_rewrite_with_wire"] is False
    assert met["isolation_rewrite_shipped"] is False
    assert met["isolation_tests_rewritten_with_wire_not_deleted"] is False
    assert met["form_label_change_shipped"] is False
    assert met["dual_honest_tf_aware_path_present"] is False
    assert met["dual_linf_under_wire_proven"] is False
    assert met["online_linf_gate_under_tf_path"] is False
    assert met["wire_shipped"] is False
    assert met["bundle_shipped"] is False
    assert met["bundle_ship_allowed_today"] is False
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert report["bundle_ship_allowed_today"] is False
    assert report["criteria_met_today"] is False
    # Under-wire-only keys alone must not open ship_allowed
    forced = dict(met)
    forced["online_linf_gate_under_tf_path"] = True
    forced["dual_linf_under_wire_proven"] = True
    assert (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(forced)
        is False
    )
    assert (
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            forced
        )
        is False
    )
    # Foreign maps still false for isolation_rewrite_with_wire
    met_ws = tlb.case1_wire_ship_acceptance_criteria_met_today_map()
    assert met_ws["isolation_rewrite_with_wire"] is False
    assert met_ws["isolation_tests_rewritten_with_wire_not_deleted"] is False
    met_gate = tlb.case1_online_linf_gate_criteria_met_today_map()
    assert met_gate["isolation_rewrite_with_wire"] is False
    met_pp = tlb.case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    assert met_pp["isolation_rewrite_with_wire"] is False
    met_fl = tlb.case1_form_label_change_shipped_criteria_met_today_map()
    assert met_fl["isolation_rewrite_with_wire"] is False
    met_iso = tlb.case1_isolation_rewrite_shipped_criteria_met_today_map()
    assert met_iso["isolation_rewrite_shipped"] is False


def test_critical_blockers_still_present():
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert "no_blender_offline_affine_kernel" in tlb.DEFAULT_WIRE_BLOCKERS
    assert "isolation_rewrite_required" in tlb.DEFAULT_WIRE_BLOCKERS
    assert "form_label_change_required" in tlb.DEFAULT_WIRE_BLOCKERS
    assert "wire_not_shipped" in tlb.DEFAULT_WIRE_BLOCKERS
    assert (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp"
        in tlb.DEFAULT_WIRE_BLOCKERS
    )
    assert report["blockers_still_documented"] is True
    assert report["isolation_rewrite_required_in_default_wire_blockers"] is True
    assert report["form_label_change_required_in_default_wire_blockers"] is True
    assert report["no_blender_offline_affine_kernel_in_default_wire_blockers"] is True
    assert report["wire_not_shipped_blocker_still_true"] is True
    assert report["dual_linf_under_wire_unproven_blocker_still_true"] is True
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_KIND",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA",
        "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY",
        "case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria",
        "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map",
        "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate",
        "offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
        "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
        "multi_unit_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    b = tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    c = tlb.multi_unit_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_honesty_fields,
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report,
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria,
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map,
        tlb.case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate,
        tlb.case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today,
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


def test_composes_form_checklist_not_maximizer():
    src = inspect.getsource(
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report
    )
    assert "case1_form_label_contract" in src
    assert "case1_dual_linf_proof_checklist" in src
    assert "DEFAULT_WIRE_BLOCKERS" in src
    assert "case1_online_linf_gate_criteria_met_today_map" in src
    assert "case1_wire_ship_acceptance_criteria_met_today_map" in src
    assert "case1_dual_honest_tf_aware_path_present_criteria_met_today_map" in src
    assert "case1_form_label_change_shipped_criteria_met_today_map" in src
    assert "case1_isolation_rewrite_shipped_criteria_met_today_map" in src
    assert "case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria" in src
    # ok criteria must not depend on multi-round maximizer residual vanish
    assert "offline_case1_shaped_cdu_blender_linking_report" not in src
    assert "offline_case1_dual_space_linf_probe_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_bridge_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report" not in src
    # Prefer lower helpers / constants over recursive foreign reports
    assert "offline_case1_isolation_rewrite_design_contract_report" not in src
    assert "offline_case1_wire_ship_acceptance_design_contract_report" not in src
    assert "offline_case1_dual_honest_tf_aware_path_design_contract_report" not in src
    assert (
        "offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report"
        not in src
    )
    assert (
        "offline_case1_form_label_change_shipped_criteria_contract_report" not in src
    )
    assert (
        "offline_case1_isolation_rewrite_shipped_criteria_contract_report" not in src
    )
    assert (
        "offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report"
        not in src
    )


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=True,
        include_admm_case1_isolation_rewrite_shipped_criteria_contract=True,
        include_admm_case1_form_label_change_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=True,
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
    assert (
        "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"
        in rep
    )
    assert (
        rep["admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"]
        is True
    )
    assert (
        rep["admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok"]
        is True
    )
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_bundle_ship_criteria_contract_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=False,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=False,
        include_admm_case1_isolation_rewrite_shipped_criteria_contract=False,
        include_admm_case1_form_label_change_shipped_criteria_contract=False,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=False,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=False,
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
    assert (
        rep["admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"]
        is None
    )
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_bundle_ship_criteria_contract_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=True,
        include_admm_case1_isolation_rewrite_shipped_criteria_contract=True,
        include_admm_case1_form_label_change_shipped_criteria_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=True,
        include_admm_case1_dual_honest_tf_aware_path_design_contract=True,
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
    assert "form_label_change_required" in pf["wire_blockers"]
    assert "isolation_rewrite_required" in pf["wire_blockers"]
    assert (
        pf.get(
            "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"
        )
        is True
    )
    assert (
        pf.get("admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok")
        is True
    )
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_bundle_ship_criteria():
    meta = tlb.honesty_metadata()
    assert (
        meta.get(
            "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_available"
        )
        is True
    )
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "bundle" in note
    assert "ship" in note or "criteria" in note


def test_isolation_suite_still_present_and_not_rewritten():
    suite = Path(__file__).resolve().parent / "test_tf_import_isolation.py"
    assert suite.is_file()
    report = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert report["isolation_tests_rewritten_with_wire"] is False
    assert report["isolation_rewrite_shipped"] is False
    assert report["isolation_rewrite_with_wire"] == "open"


def test_form_contract_and_ladder_non_regression():
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
    assert contract["dual_linf_under_wire_status"] == "unproven"
    assert contract["dual_linf_proof_checklist_n_open"] >= 4
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
    assert crit["criteria_met_today"] is False
    design = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert design["isolation_rewrite_with_wire"] == "open"
    assert design["isolation_rewrite_shipped"] is False
    assert design["gate_flip_allowed_today"] is False
    assert design["criteria_met_today"] is False
    assert tlb.CASE1_FORM_CURRENT == "classic_2block_excel_path"
    ws = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert ws["wire_ship_allowed_today"] is False
    assert ws["wire_shipped"] is False
    assert ws["isolation_rewrite_shipped"] is False
    assert ws["dual_linf_under_wire_status"] == "unproven"
    assert ws["criteria_met_today_map"]["isolation_rewrite_with_wire"] is False
    path = tlb.offline_case1_dual_honest_tf_aware_path_design_contract_report()
    assert path["path_shipped"] is False
    assert path["dual_honest_tf_aware_path_present"] is False
    assert path["wire_ship_allowed_today"] is False
    ship_met = tlb.offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report()
    assert ship_met["dual_honest_tf_aware_path_present"] is False
    assert ship_met["ship_met_allowed_today"] is False
    assert ship_met["path_design_present"] is True
    assert ship_met["path_shipped"] is False
    assert ship_met["wire_shipped"] is False
    assert ship_met["criteria_met_today_map"]["isolation_rewrite_with_wire"] is False
    form_label = tlb.offline_case1_form_label_change_shipped_criteria_contract_report()
    assert form_label["form_label_change_shipped"] is False
    assert form_label["form_label_ship_allowed_today"] is False
    assert form_label["form_current"] == "classic_2block_excel_path"
    assert form_label["criteria_present"] is True
    isolation_ship = (
        tlb.offline_case1_isolation_rewrite_shipped_criteria_contract_report()
    )
    assert isolation_ship["isolation_rewrite_shipped"] is False
    assert isolation_ship["isolation_ship_allowed_today"] is False
    assert isolation_ship["criteria_present"] is True
    assert isolation_ship["isolation_rewrite_with_wire"] == "open"
    bundle_design = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report()
    )
    assert bundle_design["bundle_design_present"] is True
    assert bundle_design["bundle_shipped"] is False
    assert bundle_design["bundle_ship_allowed_today"] is False
    assert bundle_design["ok"] is True
    bundle_ship = (
        tlb.offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
    )
    assert bundle_ship["criteria_present"] is True
    assert bundle_ship["bundle_shipped"] is False
    assert bundle_ship["bundle_ship_allowed_today"] is False
    assert bundle_ship["criteria_met_today"] is False
    assert bundle_ship["ok"] is True
