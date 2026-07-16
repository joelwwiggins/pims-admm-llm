"""E1: offline Case-1-shaped CDU↔Blender linking ADMM skeleton.

Always-on sections run without TensorFlow and without PuLP on the hot path.
Locks:
- dual_recovery_path is None; wire_shipped False; not Case 1 solve
- linking streams = naphtha/distillate/gasoil/residue (Case 1 intermediates)
- CDU side uses affine path / finite residuals
- blender_surface=linear_quality_pooling (not base_delta_affine_unit)
- UNITS still exactly FCC/COKER/CDU (no silent BLENDER)
- DEFAULT_WIRE_BLOCKERS still contain case1_is_cdu_blender_package_admm
  and no_blender_offline_affine_kernel (skeleton ≠ wire)
- no residual-must-vanish SLA
- no excel_cdu_matrix_matches_affine / excel_blender invent
- optional additive readiness flag does not redefine ready_for_wire_discussion
- existing plant-linking + coordination surfaces remain green when co-run

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem, test_tf_offline_admm_coordination,
  test_tf_offline_admm_plant_linking, test_tf_offline_wire_preflight,
  test_tf_linear_block, test_tf_linear_coker, test_tf_linear_cdu,
  test_excel_pipeline, test_api_excel
  EMRPS optional-only (not required for this gate).
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_case1_shaped_streams_and_cdu_map():
    streams = list(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert streams == ["naphtha", "distillate", "gasoil", "residue"]
    assert "resid" not in streams  # plant-linking spelling is separate
    cmap = tlb.case1_shaped_cdu_to_intermediate_map()
    assert cmap["cdu_naphtha_light"] == "naphtha"
    assert cmap["cdu_naphtha_heavy"] == "naphtha"
    assert cmap["cdu_distillate"] == "distillate"
    assert cmap["cdu_gasoil"] == "gasoil"
    assert cmap["cdu_resid"] == "residue"
    assert "cdu_offgas" not in cmap


def test_project_cdu_y_to_case1_intermediates_finite():
    coeffs = tlb.cached_offline_unit_coeffs("CDU")
    y0 = {p: float(coeffs.y0[i]) for i, p in enumerate(coeffs.products)}
    y_link = tlb.project_cdu_y_to_case1_intermediates(y0)
    assert set(y_link.keys()) == set(tlb.CASE1_SHAPED_LINKING_STREAMS)
    for s, v in y_link.items():
        assert math.isfinite(v)
    # light+heavy naphtha sum
    expected_n = float(y0.get("cdu_naphtha_light", 0.0)) + float(
        y0.get("cdu_naphtha_heavy", 0.0)
    )
    assert abs(y_link["naphtha"] - expected_n) <= 1e-12
    assert abs(y_link["residue"] - float(y0.get("cdu_resid", 0.0))) <= 1e-12


def test_blender_recipe_use_linear_pooling():
    y_prod = {"gasoline": 10.0, "diesel": 12.0, "fuel_oil": 8.0}
    use = tlb.blender_recipe_use_from_products(y_prod)
    assert set(use.keys()) == set(tlb.CASE1_SHAPED_LINKING_STREAMS)
    # gasoline 0.85*10 + diesel 0*n + ...
    assert abs(use["naphtha"] - 0.85 * 10.0) <= 1e-12
    assert abs(use["distillate"] - (0.15 * 10.0 + 0.70 * 12.0)) <= 1e-12
    assert abs(use["gasoil"] - (0.30 * 12.0 + 0.40 * 8.0)) <= 1e-12
    assert abs(use["residue"] - 0.60 * 8.0) <= 1e-12


def test_report_always_on_aggregate_ok_honesty_locks():
    report = tlb.offline_case1_shaped_cdu_blender_linking_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["kind"] == "offline_case1_shaped_cdu_blender_linking"
    assert report["kind"] == tlb.CASE1_SHAPED_LINKING_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_shaped_offline_only"] is True
    assert report["case1_form_unchanged"] is True
    assert report["wire_shipped"] is False
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_full_plant_blocks_feed_lp"] is True
    assert report["not_live_plant_blocks"] is True
    assert report["not_plant_linking_multi_unit_fcc_coker_cdu"] is True
    assert report["linking_lambda_is_not_case1_online_lambda"] is True
    assert report["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
    assert report["blender_surface"] == "linear_quality_pooling"
    assert report["blender_surface"] == tlb.CASE1_SHAPED_BLENDER_SURFACE
    assert report["blender_is_base_delta_affine_unit"] is False
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None
    assert report["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    assert report["no_blender_offline_affine_kernel_blocker_still_true"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["n_rounds"] == 3
    assert len(report["trajectory"]) == 3
    assert set(report["streams"]) == set(tlb.CASE1_SHAPED_LINKING_STREAMS)
    assert report["package_order"] == ["CDU", "BLENDER"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS


def test_cdu_and_blender_sides_finite_under_synthetic_lam_z_rho():
    report = tlb.offline_case1_shaped_cdu_blender_linking_report(
        n_rounds=2, rho=1.0, delta=0.5
    )
    cdu = report["packages"]["CDU"]
    blender = report["packages"]["BLENDER"]
    assert cdu["ok"] is True
    assert blender["ok"] is True
    assert cdu["final_y_link"] is not None
    assert blender["final_use"] is not None
    for s in tlb.CASE1_SHAPED_LINKING_STREAMS:
        assert math.isfinite(cdu["final_y_link"][s])
        assert math.isfinite(blender["final_use"][s])
    assert blender["blender_surface"] == "linear_quality_pooling"
    for tr in report["trajectory"]:
        assert math.isfinite(tr["r_l1_link"])
        assert math.isfinite(tr["r_linf_link"])
        assert math.isfinite(tr["sum_augmented_local"])
        assert tr["ok"] is True


def test_n_rounds_respected_no_residual_must_vanish():
    for n in (1, 2, 4):
        report = tlb.offline_case1_shaped_cdu_blender_linking_report(
            n_rounds=n, rho=1.0, delta=0.5
        )
        assert report["n_rounds"] == n
        assert len(report["trajectory"]) == n
        assert report["ok"] is True, report
        # Explicit non-gate: residual need not vanish
        for tr in report["trajectory"]:
            assert "r_l1_link" in tr
            # no assertion that residual == 0


def test_dual_ascent_uses_pre_z_linking_residual():
    streams = list(tlb.CASE1_SHAPED_LINKING_STREAMS)
    z_far = {s: 0.0 for s in streams}
    row = tlb.case1_shaped_cdu_blender_linking_round(
        z_link=z_far, rho=1.0, delta=1.0, dual_step=1.0, z_blend=1.0
    )
    assert row["ok"] is True
    # lam moves from pre residual
    moved = False
    for s in streams:
        if abs(row["lam_post"][s] - row["lam_pre"][s]) > 1e-12:
            moved = True
            expected = row["lam_pre"][s] + 1.0 * 1.0 * row["r_link_pre"][s]
            assert abs(row["lam_post"][s] - expected) <= 1e-9
    assert moved


def test_composes_cdu_subproblem_kind():
    row = tlb.case1_shaped_cdu_blender_linking_round(rho=1.0, delta=0.5)
    assert row["ok"] is True
    assert row["cdu"]["subproblem_kind"] == tlb.ADMM_SUBPROBLEM_KIND
    assert row["cdu"]["subproblem_ok"] is True
    assert row["cdu"]["not_worse_than_ref"] is True
    assert row["blender"]["blender_surface"] == "linear_quality_pooling"


def test_alias_and_no_tf_required():
    a = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    b = tlb.multi_block_case1_shaped_linking_admm_report(n_rounds=1)
    assert a["kind"] == b["kind"]
    assert a["ok"] is True
    assert b["ok"] is True
    # always-on surface reports tf_available but does not require it
    assert "tf_available" in a


def test_source_does_not_import_pulp_or_tensorflow_on_hot_path():
    src = inspect.getsource(tlb.offline_case1_shaped_cdu_blender_linking_report)
    src2 = inspect.getsource(tlb.case1_shaped_cdu_blender_linking_round)
    src3 = inspect.getsource(tlb._blender_linear_pooling_step)
    blob = src + src2 + src3
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "from pulp" not in blob
    assert "from tensorflow" not in blob


def test_units_still_fcc_coker_cdu_no_silent_blender():
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]


def test_wire_blockers_still_true_after_skeleton():
    blockers = list(tlb.DEFAULT_WIRE_BLOCKERS)
    assert "case1_is_cdu_blender_package_admm" in blockers
    assert "no_blender_offline_affine_kernel" in blockers
    assert "wire_not_shipped" in blockers
    cat = tlb.offline_wire_blocker_catalog()
    assert cat["wire_shipped"] is False
    assert "case1_is_cdu_blender_package_admm" in cat["wire_blockers"]
    assert "no_blender_offline_affine_kernel" in cat["wire_blockers"]
    # notes may mention skeleton ≠ wire but ids remain
    notes = cat["wire_blocker_notes"]
    assert "case1_is_cdu_blender_package_admm" in notes
    assert "no_blender_offline_affine_kernel" in notes


def test_existing_plant_linking_and_coordination_still_green():
    pl = tlb.multi_block_plant_linking_admm_report(n_rounds=1, mode="synthetic")
    assert pl["ok"] is True
    assert pl["kind"] == tlb.ADMM_PLANT_LINKING_KIND
    pn = tlb.multi_block_plant_linking_admm_report(n_rounds=1, mode="plant_named")
    assert pn["ok"] is True
    coord = tlb.multi_unit_admm_coordination_report(n_rounds=1)
    assert coord["not_plant_linking_coordinator"] is True
    assert coord["ok"] is True


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_shaped_linking=True,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_shaped_linking_ok" in rep
    assert rep["admm_case1_shaped_linking_ok"] is True
    # ready still structural only
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    # case1 flag is not AND-ed into ready by construction when ladder green
    assert rep["dual_recovery_path"] is None


def test_preflight_still_documents_blockers_with_case1_flag():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_shaped_linking=True,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert "case1_is_cdu_blender_package_admm" in pf["wire_blockers"]
    assert "no_blender_offline_affine_kernel" in pf["wire_blockers"]
    assert pf.get("admm_case1_shaped_linking_ok") is True
    # ready meaning unchanged: parity∧priced∧timings∧honesty
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected


def test_note_language_hard_negatives():
    report = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    note = (report.get("note") or "").lower()
    assert "case 1" in note or "case1" in note
    assert "wire" in note
    assert "dual" in note
    assert "pooling" in note or "linear" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
