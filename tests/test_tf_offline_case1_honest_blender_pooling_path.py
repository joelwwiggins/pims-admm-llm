"""E1: offline Case-1 honest blender pooling path formalization.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the pooling-path hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- blender_surface == linear_quality_pooling; blender_is_base_delta_affine_unit False
- excel_blender_matrix_matches_affine is None (not invented)
- no_blender_offline_affine_kernel still in DEFAULT_WIRE_BLOCKERS / critical
- checklist status honest_pooling_path_present (not bare open; not closed_via_affine)
- pooling path ≠ affine kernel / ≠ wire / ≠ VERDICT
- UNITS still FCC/COKER/CDU (no silent BLENDER)
- additive readiness flag does not redefine ready_for_wire_discussion
- no residual-must-vanish; no linf<=15 gate on pooling_path_ok
- no excel_pipeline import on tf_linear_blocks pooling hot path
- no form mutation

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
    tests/test_tf_offline_case1_dual_space_linf_probe.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_bridge.py \\
    tests/test_tf_offline_case1_dual_space_form_contract.py \\
    tests/test_tf_offline_case1_shaped_linking.py \\
    tests/test_tf_offline_wire_preflight.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_seeded_warmstart.py \\
    tests/test_tf_offline_case1_honest_blender_pooling_path.py -q
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


def test_checklist_status_honest_pooling_not_bare_open():
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST[
            "blender_affine_kernel_or_honest_pooling_path"
        ]
        == tlb.CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST[
            "blender_affine_kernel_or_honest_pooling_path"
        ]
        == "honest_pooling_path_present"
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST[
            "blender_affine_kernel_or_honest_pooling_path"
        ]
        != "open"
    )
    assert (
        tlb.CASE1_DUAL_LINF_PROOF_CHECKLIST[
            "blender_affine_kernel_or_honest_pooling_path"
        ]
        != "closed_via_affine_kernel"
    )
    cl = tlb.case1_dual_linf_proof_checklist()
    assert (
        "blender_affine_kernel_or_honest_pooling_path"
        not in cl["dual_linf_proof_checklist_open_ids"]
    )
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert cl["dual_linf_proof_checklist_n_open"] >= 4
    # Remaining open/false_today items still present
    for key in (
        "isolation_rewrite_with_wire",
        "form_label_change_shipped",
        "online_linf_gate_under_tf_path",
        "wire_shipped",
    ):
        assert key in cl["dual_linf_proof_checklist_open_ids"]


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_honest_blender_pooling_path_report()
    assert report["kind"] == tlb.CASE1_HONEST_BLENDER_POOLING_PATH_KIND
    assert report["kind"] == "offline_case1_honest_blender_pooling_path"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["blender_surface"] == "linear_quality_pooling"
    assert report["blender_surface"] == tlb.CASE1_SHAPED_BLENDER_SURFACE
    assert report["blender_is_base_delta_affine_unit"] is False
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None
    assert report["no_blender_offline_affine_kernel_blocker_still_true"] is True
    assert report["pooling_path_is_not_affine_kernel"] is True
    assert report["pooling_path_is_not_wire"] is True
    assert report["pooling_path_is_not_verdict_gate"] is True
    assert report["pooling_path_is_not_dual_linf_under_wire_proof"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["blender_pooling_checklist_status"] == "honest_pooling_path_present"
    assert report["pooling_path_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["recipes_source"] == "synthetic_offline_demo"
    assert report["residual_must_vanish_is_not_gate"] is True
    assert report["linf_le_15_is_not_gate"] is True
    assert "NOT linf<=15" in report["ok_criteria"] or "not" in report["ok_criteria"].lower()
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]


def test_critical_blockers_still_present_including_no_blender():
    report = tlb.offline_case1_honest_blender_pooling_path_report()
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert "no_blender_offline_affine_kernel" in tlb.DEFAULT_WIRE_BLOCKERS
    assert "no_blender_offline_affine_kernel" in tlb.CASE1_CONTRACT_CRITICAL_BLOCKERS
    assert report["no_blender_offline_affine_kernel_in_default_wire_blockers"] is True
    assert report["no_blender_offline_affine_kernel_in_critical_blockers"] is True
    assert report["blockers_still_documented"] is True
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_honest_blender_pooling_path_report()
    assert report["units_ok"] is True
    assert report["blender_is_base_delta_affine_unit"] is False


def test_aliases_and_exports():
    for name in (
        "CASE1_HONEST_BLENDER_POOLING_PATH_KIND",
        "CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS",
        "CASE1_HONEST_BLENDER_POOLING_PATH_RECIPES_SOURCE",
        "offline_case1_honest_blender_pooling_path_report",
        "case1_honest_blender_pooling_path_report",
        "multi_unit_case1_honest_blender_pooling_path_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_honest_blender_pooling_path_report()
    b = tlb.case1_honest_blender_pooling_path_report()
    c = tlb.multi_unit_case1_honest_blender_pooling_path_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_honest_blender_pooling_path_honesty_fields,
        tlb.offline_case1_honest_blender_pooling_path_report,
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


def test_composes_skeleton_honesty_not_maximizer():
    src = inspect.getsource(tlb.offline_case1_honest_blender_pooling_path_report)
    src2 = inspect.getsource(tlb._case1_honest_blender_pooling_path_honesty_fields)
    blob = src + src2
    assert "_case1_shaped_linking_honesty_fields" in blob
    assert "CASE1_SHAPED_BLENDER_SURFACE" in blob
    # ok criteria must not depend on multi-round maximizer residual vanish
    assert "residual_must_vanish" not in src or "not" in src.lower()
    assert "offline_case1_shaped_cdu_blender_linking_report" not in src


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert "admm_case1_honest_blender_pooling_path_ok" in rep
    assert rep["admm_case1_honest_blender_pooling_path_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_pooling_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
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
    assert rep["admm_case1_honest_blender_pooling_path_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_pooling_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
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
    assert pf.get("admm_case1_honest_blender_pooling_path_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_pooling_path():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_honest_blender_pooling_path_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "pooling" in note
    assert "honest" in note or "linear_quality" in note


def test_form_contract_and_ladder_non_regression():
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
    assert contract["dual_linf_under_wire_status"] == "unproven"
    assert contract["dual_linf_proof_checklist_n_open"] >= 4
    assert (
        contract["dual_linf_proof_checklist"][
            "blender_affine_kernel_or_honest_pooling_path"
        ]
        == "honest_pooling_path_present"
    )
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
    # form still classic
    assert tlb.CASE1_FORM_CURRENT == "classic_2block_excel_path"
