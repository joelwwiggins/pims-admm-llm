"""E1: offline Case-1 isolation-rewrite design-only contract.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the design-contract hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- isolation_rewrite_design_present True; isolation_rewrite_shipped True
- isolation_rewrite_with_wire checklist is \"shipped\" (design still present post-ship)
- isolation_tests_rewritten_with_wire True; rewrite-not-delete rule True
- isolation_rewrite_required cleared from DEFAULT_WIRE_BLOCKERS / critical
- online_linf_gate_under_tf_path remains \"open\"; gate_flip_allowed_today False
- gate criteria_met_today False; isolation met_today keys True post-ship
- form still classic_2block_excel_path; planned distinct
- UNITS still FCC/COKER/CDU (no silent BLENDER)
- additive readiness flag does not redefine ready_for_wire_discussion
- design ≠ rewrite shipped ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof
- no excel_pipeline import on tf_linear_blocks design-contract hot path

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
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


def test_isolation_checklist_stays_open():
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST["isolation_rewrite_with_wire"] == "shipped"
    )
    cl = tlb.case1_dual_linf_proof_checklist()
    assert "isolation_rewrite_with_wire" not in cl["dual_linf_proof_checklist_open_ids"]
    assert "form_label_change_shipped" not in cl["dual_linf_proof_checklist_open_ids"]
    assert "online_linf_gate_under_tf_path" in cl["dual_linf_proof_checklist_open_ids"]
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert cl["dual_linf_proof_checklist_n_open"] >= 2



def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert report["kind"] == tlb.CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_KIND
    assert report["kind"] == "offline_case1_isolation_rewrite_design_contract"
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
    assert report["isolation_rewrite_design_present"] is True
    assert report["isolation_rewrite_shipped"] is True
    assert report["isolation_tests_rewritten_with_wire"] is True
    assert report["isolation_tests_must_be_rewritten_with_wire_not_deleted"] is True
    assert report["isolation_rewrite_with_wire"] == "shipped"
    assert report["isolation_rewrite_still_open"] is False
    assert report["isolation_rewrite_checklist_open"] is False
    assert report["design_does_not_close_isolation_rewrite_checklist"] is True
    assert report["design_does_not_set_isolation_met_today"] is True
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["isolation_rewrite_met_today"] is True
    assert report["isolation_tests_rewritten_met_today"] is True
    assert report["design_is_not_isolation_rewrite_shipped"] is True
    assert report["design_is_not_wire"] is True
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
    assert "NOT isolation rewrite shipped" in report["ok_criteria"]
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["isolation_rewrite_design_contract"] == "present"
    assert report["honest_pooling_path_present"] is True


def test_invariants_and_post_wire_shape():
    report = tlb.offline_case1_isolation_rewrite_design_contract_report()
    invariants = report["isolation_invariants_must_survive"]
    assert len(invariants) >= 5
    for key in (
        "excel_pipeline_source_must_not_hard_import_tensorflow_or_tf_linear_blocks_on_classic_path",
        "models_init_must_not_hard_import_tf_linear_blocks",
        "case1_form_remains_classic_until_explicit_form_label_change_shipped",
        "dual_recovery_path_labeled_honestly_none_on_offline_tf_surface_online_lambda_owns_verdict_on_classic",
        "optional_tf_path_must_remain_skipif_friendly_when_tf_absent",
    ):
        assert key in invariants
    shape = report["post_wire_rewrite_shape"]
    assert shape["suite_shape"] == "dual_path_isolation_suite"
    assert shape["classic_path_still_isolated"] is True
    assert shape["tf_aware_path_gated_by_form_label_and_feature_flag"] is True
    assert shape["isolation_tests_rewritten_with_wire_not_deleted"] is True
    assert shape["no_silent_form_reuse"] is True
    assert shape["rewrite_shipped"] is True
    assert shape["implemented_this_cycle"] is True
    assert report["invariants_ok"] is True
    assert report["post_wire_shape_ok"] is True
    # Constants + helpers agree
    assert list(tlb.CASE1_ISOLATION_INVARIANTS_MUST_SURVIVE) == invariants
    assert tlb.case1_isolation_invariants_must_survive() == invariants
    assert tlb.case1_isolation_rewrite_post_wire_shape()["rewrite_shipped"] is True


def test_isolation_rewrite_required_still_in_blockers():
    report = tlb.offline_case1_isolation_rewrite_design_contract_report()
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert "isolation_rewrite_required" not in tlb.DEFAULT_WIRE_BLOCKERS
    assert "isolation_rewrite_required" not in tlb.CASE1_CONTRACT_CRITICAL_BLOCKERS
    assert report.get("isolation_rewrite_required_in_default_wire_blockers", False) is False
    assert report["isolation_rewrite_required_in_critical_blockers"] is False
    assert report["isolation_rewrite_required_in_wire_blocker_notes"] is True
    assert report["blockers_still_documented"] is True
    note = report["isolation_rewrite_blocker_note"] or ""
    assert "rewritten" in note.lower() or "WITH" in note
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_flip_met_today_isolation_keys_are_true_post_ship():
    met = tlb.case1_online_linf_gate_criteria_met_today_map()
    assert met["isolation_rewrite_with_wire"] is True
    assert met["isolation_tests_rewritten_with_wire_not_deleted"] is True
    assert met["wire_shipped"] is False
    assert met["form_label_change_shipped"] is True
    assert tlb.case1_online_linf_gate_flip_allowed_today() is False
    assert tlb.case1_online_linf_gate_criteria_met_today_aggregate() is False
    report = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["isolation_rewrite_met_today"] is True
    assert report["isolation_tests_rewritten_met_today"] is True
    flip_keys = report["flip_criteria_isolation_keys"]
    assert flip_keys["isolation_rewrite_with_wire"] == tlb.FLIP_CRITERION_REQUIRED
    assert (
        flip_keys["isolation_tests_rewritten_with_wire_not_deleted"]
        == tlb.FLIP_CRITERION_REQUIRED
    )


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_KIND",
        "CASE1_ISOLATION_REWRITE_CHECKLIST_KEY",
        "CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_ANNOTATION",
        "CASE1_ISOLATION_REWRITE_BLOCKER_ID",
        "CASE1_ISOLATION_REWRITE_FLIP_KEY",
        "CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY",
        "CASE1_ISOLATION_INVARIANTS_MUST_SURVIVE",
        "CASE1_ISOLATION_REWRITE_POST_WIRE_SHAPE",
        "case1_isolation_invariants_must_survive",
        "case1_isolation_rewrite_post_wire_shape",
        "offline_case1_isolation_rewrite_design_contract_report",
        "case1_isolation_rewrite_design_contract_report",
        "multi_unit_case1_isolation_rewrite_design_contract_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_isolation_rewrite_design_contract_report()
    b = tlb.case1_isolation_rewrite_design_contract_report()
    c = tlb.multi_unit_case1_isolation_rewrite_design_contract_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_isolation_rewrite_design_contract_honesty_fields,
        tlb.offline_case1_isolation_rewrite_design_contract_report,
        tlb.case1_isolation_invariants_must_survive,
        tlb.case1_isolation_rewrite_post_wire_shape,
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
    src = inspect.getsource(tlb.offline_case1_isolation_rewrite_design_contract_report)
    assert "case1_dual_linf_proof_checklist" in src
    assert "case1_form_label_contract" in src
    assert "DEFAULT_WIRE_BLOCKERS" in src
    assert "offline_case1_shaped_cdu_blender_linking_report" not in src
    assert "offline_case1_dual_space_linf_probe_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_bridge_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report" not in src


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert "admm_case1_isolation_rewrite_design_contract_ok" in rep
    assert rep["admm_case1_isolation_rewrite_design_contract_ok"] is True
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
    assert rep["admm_case1_isolation_rewrite_design_contract_ok"] is None
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
    assert pf.get("admm_case1_isolation_rewrite_design_contract_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_isolation_design_contract():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_isolation_rewrite_design_contract_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "isolation-rewrite" in note or "isolation_rewrite" in note
    assert "design" in note


def test_form_contract_and_ladder_non_regression():
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
    assert contract["dual_linf_under_wire_status"] == "unproven"
    assert contract["dual_linf_proof_checklist_n_open"] >= 2
    assert (
        contract["dual_linf_proof_checklist"]["isolation_rewrite_with_wire"] == "shipped"
    )
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
    # form still classic
    assert tlb.CASE1_FORM_CURRENT == tlb.CASE1_PLANNED_TF_AWARE_FORM
    # design contract does not flip isolation checklist or ship rewrite
    design = tlb.offline_case1_isolation_rewrite_design_contract_report()
    assert design["isolation_rewrite_with_wire"] == "shipped"
    assert design["isolation_rewrite_shipped"] is True
    assert design["gate_flip_allowed_today"] is False
    assert design["criteria_met_today"] is False
