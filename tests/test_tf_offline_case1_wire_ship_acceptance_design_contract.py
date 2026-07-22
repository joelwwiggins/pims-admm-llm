"""E1: offline Case-1 dual-honest wire-ship acceptance design/acceptance contract.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the design-contract hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- design_present True; wire_ship_allowed_today False; criteria_met_today False
- isolation_rewrite_shipped True; isolation checklist "shipped"
- online_linf_gate_under_tf_path remains "open"; gate_flip_allowed_today False
- form still classic_2block_excel_path; planned distinct
- UNITS still FCC/COKER/CDU (no silent BLENDER)
- critical blockers still in DEFAULT_WIRE_BLOCKERS (incl. isolation_rewrite,
  form_label, dual_linf unproven, no_blender, wire_not_shipped)
- additive readiness flag does not redefine ready_for_wire_discussion
- design ≠ wire shipped ≠ ship allow ≠ VERDICT ≠ dual L∞ under wire proof
- no excel_pipeline import on tf_linear_blocks design-contract hot path

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
    tests/test_tf_offline_case1_wire_ship_acceptance_design_contract.py \\
    tests/test_tf_offline_case1_isolation_rewrite_design_contract.py \\
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

REQUIRED_ACCEPTANCE_KEYS = {
    "isolation_rewrite_with_wire",
    "isolation_tests_rewritten_with_wire_not_deleted",
    "form_label_change_shipped",
    "dual_honest_tf_aware_path_present",
    "online_linf_gate_under_tf_path",
    "dual_linf_under_wire_proven",
    "wire_shipped",
    "dual_recovery_path_labeled_honestly",
    "no_silent_form_reuse",
    "no_blender_offline_affine_kernel_honesty_preserved",
    "case1_cdu_blender_package_admm_shape_acknowledged",
    "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp",
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
    report = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert report["kind"] == tlb.CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_KIND
    assert report["kind"] == "offline_case1_wire_ship_acceptance_design_contract"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is False
    assert report["form_current"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["form_label_change_required_still_true"] is False
    assert report["design_present"] is True
    assert report["wire_ship_allowed_today"] is False
    assert report["wire_ship_criteria_met_today"] is False
    assert report["isolation_rewrite_design_present"] is True
    assert report["isolation_rewrite_shipped"] is True
    assert report["isolation_tests_rewritten_with_wire"] is True
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["isolation_rewrite_checklist_open"] is False
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["design_is_not_wire_shipped"] is True
    assert report["design_is_not_wire"] is True
    assert report["design_is_not_ship_allow"] is True
    assert report["design_is_not_isolation_rewrite_shipped"] is True
    assert report["design_is_not_gate_flip"] is True
    assert report["design_is_not_verdict_gate"] is True
    assert report["design_is_not_dual_linf_under_wire_proof"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["no_blender_offline_affine_kernel_blocker_still_true"] is True
    assert report["design_contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert "NOT wire shipped" in report["ok_criteria"]
    assert "NOT ship allow" in report["ok_criteria"]
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["wire_ship_acceptance_design_contract"] == "present"
    assert report["honest_pooling_path_present"] is True
    assert report["suggested_next_wave_still_full_wire"] is True


def test_acceptance_criteria_map_keys_and_classes():
    report = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    criteria = report["criteria_map"]
    assert set(criteria.keys()) == REQUIRED_ACCEPTANCE_KEYS
    assert (
        criteria["online_linf_gate_under_tf_path"]
        == tlb.FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    )
    assert (
        criteria["dual_linf_under_wire_proven"]
        == tlb.FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    )
    for k in REQUIRED_ACCEPTANCE_KEYS:
        if k not in (
            "online_linf_gate_under_tf_path",
            "dual_linf_under_wire_proven",
        ):
            assert criteria[k] == tlb.FLIP_CRITERION_REQUIRED
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
        "this_wire_ship_acceptance_design_alone",
    ):
        assert a in anti
    assert set(tlb.CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA.keys()) == REQUIRED_ACCEPTANCE_KEYS
    assert set(tlb.case1_wire_ship_acceptance_criteria().keys()) == REQUIRED_ACCEPTANCE_KEYS
    assert report["criteria_formalized"] is True


def test_ship_permission_hard_false():
    assert tlb.case1_wire_ship_allowed_today() is False
    assert tlb.case1_wire_ship_criteria_met_today_aggregate() is False
    met = tlb.case1_wire_ship_acceptance_criteria_met_today_map()
    # Structural honesty may be True; aggregate still False
    assert met["dual_recovery_path_labeled_honestly"] is True
    assert met["no_silent_form_reuse"] is True
    assert met["isolation_rewrite_with_wire"] is True
    assert met["wire_shipped"] is False
    assert met["form_label_change_shipped"] is True
    assert met["dual_honest_tf_aware_path_present"] is True
    assert met["dual_linf_under_wire_proven"] is False
    report = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert report["wire_ship_allowed_today"] is False
    assert report["wire_ship_criteria_met_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False


def test_critical_blockers_still_present():
    report = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
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
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_KIND",
        "CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_ANNOTATION",
        "CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA",
        "CASE1_WIRE_SHIP_ANTI_CRITERIA_TODAY",
        "case1_wire_ship_acceptance_criteria",
        "case1_wire_ship_acceptance_criteria_met_today_map",
        "case1_wire_ship_allowed_today",
        "case1_wire_ship_criteria_met_today_aggregate",
        "offline_case1_wire_ship_acceptance_design_contract_report",
        "case1_wire_ship_acceptance_design_contract_report",
        "multi_unit_case1_wire_ship_acceptance_design_contract_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    b = tlb.case1_wire_ship_acceptance_design_contract_report()
    c = tlb.multi_unit_case1_wire_ship_acceptance_design_contract_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_wire_ship_acceptance_design_contract_honesty_fields,
        tlb.offline_case1_wire_ship_acceptance_design_contract_report,
        tlb.case1_wire_ship_acceptance_criteria,
        tlb.case1_wire_ship_acceptance_criteria_met_today_map,
        tlb.case1_wire_ship_allowed_today,
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


def test_composes_checklist_form_not_maximizer():
    src = inspect.getsource(
        tlb.offline_case1_wire_ship_acceptance_design_contract_report
    )
    assert "case1_dual_linf_proof_checklist" in src
    assert "case1_form_label_contract" in src
    assert "DEFAULT_WIRE_BLOCKERS" in src
    assert "case1_online_linf_gate_criteria_met_today_map" in src
    # ok criteria must not depend on multi-round maximizer residual vanish
    assert "offline_case1_shaped_cdu_blender_linking_report" not in src
    assert "offline_case1_dual_space_linf_probe_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_bridge_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report" not in src
    # Prefer lower helpers / constants over recursive isolation report
    assert "offline_case1_isolation_rewrite_design_contract_report" not in src


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert "admm_case1_wire_ship_acceptance_design_contract_ok" in rep
    assert rep["admm_case1_wire_ship_acceptance_design_contract_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_design_contract_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert rep["admm_case1_wire_ship_acceptance_design_contract_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_design_contract_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
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
    assert pf.get("admm_case1_wire_ship_acceptance_design_contract_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_wire_ship_acceptance_design():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_wire_ship_acceptance_design_contract_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "wire-ship" in note or "wire_ship" in note
    assert "acceptance" in note or "ship_allowed" in note or "may ship" in note


def test_form_contract_and_ladder_non_regression():
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
    assert contract["dual_linf_under_wire_status"] == "unproven"
    assert contract["dual_linf_proof_checklist_n_open"] >= 2
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
    assert design["isolation_rewrite_with_wire"] == "shipped"
    assert design["isolation_rewrite_shipped"] is True
    assert design["gate_flip_allowed_today"] is False
    assert design["criteria_met_today"] is False
    # form still classic
    assert tlb.CASE1_FORM_CURRENT == tlb.CASE1_PLANNED_TF_AWARE_FORM
    # wire-ship acceptance design does not allow ship or claim wire
    ws = tlb.offline_case1_wire_ship_acceptance_design_contract_report()
    assert ws["wire_ship_allowed_today"] is False
    assert ws["wire_shipped"] is False
    assert ws["isolation_rewrite_shipped"] is True
    assert ws["dual_linf_under_wire_status"] == "unproven"
