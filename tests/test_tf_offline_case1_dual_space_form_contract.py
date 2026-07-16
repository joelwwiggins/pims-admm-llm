"""E1/E2: offline Case-1 dual-space / form-label contract (always-on numpy).

Always-on sections run without TensorFlow and without PuLP on the hot path.
Locks:
- form_current classic_2block_excel_path; planned TF-aware form distinct; form_unchanged
- streams = naphtha/distillate/gasoil/residue; stream_alignment_ok; skeleton slots match
- dual_recovery_path is None; skeleton λ ≠ Case 1 PRIMARY/SECONDARY duals
- dual_linf_under_wire status unproven; checklist open; blocker still present
- wire_shipped False; critical DEFAULT_WIRE_BLOCKERS still present
- ok True under honesty; solver False; on_excel_case1_path False
- additive readiness flag does not redefine ready_for_wire_discussion
- no residual-must-vanish; no recovered L∞ ≤15; no Case 1 form mutation
- no excel_cdu_matrix_matches_affine / excel_blender invent

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem, test_tf_offline_admm_coordination,
  test_tf_offline_admm_plant_linking, test_tf_offline_wire_preflight,
  test_tf_offline_case1_shaped_linking,
  test_excel_pipeline, test_api_excel
  EMRPS optional-only (not required for this gate).
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
    "isolation_rewrite_required",
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}


def test_form_label_registry_current_vs_planned_distinct():
    assert tlb.CASE1_FORM_CURRENT == "classic_2block_excel_path"
    assert tlb.CASE1_PLANNED_TF_AWARE_FORM == "tf_affine_cdu_blender_shaped_excel_path"
    assert tlb.CASE1_PLANNED_TF_AWARE_FORM != tlb.CASE1_FORM_CURRENT
    assert tlb.CASE1_PLANNED_TF_AWARE_FORM != "classic_2block_excel_path"
    form = tlb.case1_form_label_contract()
    assert form["form_current"] == "classic_2block_excel_path"
    assert form["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert form["form_unchanged"] is True
    assert form["planned_form_distinct"] is True
    assert form["form_label_change_required_still_true"] is True
    assert form["form_contract_ok"] is True


def test_dual_space_stream_map_alignment():
    smap = tlb.case1_dual_space_stream_map()
    assert smap["streams"] == ["naphtha", "distillate", "gasoil", "residue"]
    assert smap["linking_streams"] == list(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert smap["skeleton_lambda_slots"] == list(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert smap["stream_alignment_ok"] is True
    assert "resid" not in smap["streams"]
    assert smap["package_dual_gate"] == "online_lambda"
    assert smap["package_dual_secondary"] == "recovered_blender"
    assert smap["skeleton_lambda_is_not_case1_online_lambda"] is True
    assert smap["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
    assert smap["dual_recovery_path"] is None
    for s in smap["streams"]:
        assert s in smap["stream_dual_roles"]
        assert smap["stream_dual_roles"][s]["skeleton_lambda_slot"] == s


def test_dual_linf_proof_checklist_unproven():
    cl = tlb.case1_dual_linf_proof_checklist()
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert cl["dual_linf_under_wire"] == "unproven"
    assert cl["dual_linf_under_wire_unproven_still_true"] is True
    assert cl["dual_linf_status_unproven_ok"] is True
    assert cl["dual_linf_proof_checklist_n_open"] >= 4
    checklist = cl["dual_linf_proof_checklist"]
    for key in (
        "isolation_rewrite_with_wire",
        "form_label_change_shipped",
        "online_linf_gate_under_tf_path",
        "wire_shipped",
    ):
        assert key in checklist
    assert "dual_linf_under_wire_unproven" in tlb.DEFAULT_WIRE_BLOCKERS


def test_report_always_on_aggregate_ok_honesty_locks():
    report = tlb.offline_case1_dual_space_form_contract_report()
    assert report["kind"] == "offline_case1_dual_space_form_contract"
    assert report["kind"] == tlb.CASE1_DUAL_SPACE_FORM_CONTRACT_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is True
    assert report["form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["form_label_change_required_still_true"] is True
    assert report["form_contract_ok"] is True
    assert report["stream_alignment_ok"] is True
    assert report["wire_shipped"] is False
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_isolation_rewrite"] is True
    assert report["not_full_tf_admm_wire"] is True
    assert report["skeleton_lambda_is_not_case1_online_lambda"] is True
    assert report["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
    assert report["blender_surface"] == "linear_quality_pooling"
    assert report["blender_is_base_delta_affine_unit"] is False
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["dual_linf_status_unproven_ok"] is True
    assert report["dual_linf_proof_checklist_n_open"] > 0
    assert report["blockers_still_documented"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert set(report["streams"]) == set(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert report["package_dual_gate"] == "online_lambda"
    assert report["package_dual_secondary"] == "recovered_blender"
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS


def test_critical_blockers_still_present():
    report = tlb.offline_case1_dual_space_form_contract_report()
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_alias_and_no_tf_required():
    a = tlb.offline_case1_dual_space_form_contract_report()
    b = tlb.multi_unit_case1_dual_space_form_contract_report()
    assert a["kind"] == b["kind"]
    assert a["ok"] is True
    assert b["ok"] is True
    assert "tf_available" in a  # reported, not required


def test_source_does_not_import_pulp_or_tensorflow_on_hot_path():
    src = inspect.getsource(tlb.offline_case1_dual_space_form_contract_report)
    src2 = inspect.getsource(tlb.case1_form_label_contract)
    src3 = inspect.getsource(tlb.case1_dual_space_stream_map)
    src4 = inspect.getsource(tlb.case1_dual_linf_proof_checklist)
    blob = src + src2 + src3 + src4
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "from pulp" not in blob
    assert "from tensorflow" not in blob


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_dual_space_form_contract_ok" in rep
    assert rep["admm_case1_dual_space_form_contract_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_contract_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_form_contract=False,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert rep["admm_case1_dual_space_form_contract_ok"] is None
    # ready still structural
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_contract_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
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
    assert pf.get("admm_case1_dual_space_form_contract_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_case1_shaped_and_wire_preflight_still_green():
    c1 = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    assert c1["ok"] is True
    assert c1["case1_form_unchanged"] is True
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_shaped_linking=False,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert pf["ok"] is True
    assert pf["blockers_documented"] is True


def test_honesty_metadata_mentions_contract():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_space_form_contract_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "dual-space" in note or "form-label" in note or "dual space" in note


def test_exports_in_all():
    for name in (
        "CASE1_DUAL_SPACE_FORM_CONTRACT_KIND",
        "CASE1_FORM_CURRENT",
        "CASE1_PLANNED_TF_AWARE_FORM",
        "CASE1_DUAL_LINF_UNDER_WIRE_STATUS",
        "CASE1_DUAL_LINF_PROOF_CHECKLIST",
        "CASE1_CONTRACT_CRITICAL_BLOCKERS",
        "case1_form_label_contract",
        "case1_dual_space_stream_map",
        "case1_dual_linf_proof_checklist",
        "offline_case1_dual_space_form_contract_report",
        "multi_unit_case1_dual_space_form_contract_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_no_excel_blender_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_blender_matrix_matches_affine")
    # excel_cdu_matrix_matches_affine must not be a module-level dual-space invent
    # (excel_cdu may exist as None field only on reports)
    report = tlb.offline_case1_dual_space_form_contract_report()
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None


def test_note_language_hard_negatives():
    report = tlb.offline_case1_dual_space_form_contract_report()
    note = (report.get("note") or "").lower()
    assert "case 1" in note or "case1" in note
    assert "wire" in note
    assert "dual" in note
    assert "form" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
