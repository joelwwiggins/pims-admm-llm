"""E1/E2: offline multi-round ADMM coordination harness.

Always-on sections run without TensorFlow. Locks:
- multi_unit_admm_coordination_report lists FCC/COKER/CDU
- aggregate ok without TF
- dual_recovery_path is None; on_excel_case1_path is False; solver False
- optimand_space / z_update_space = raw_affine; synthetic λ/z/ρ
- per-unit synthetic scope (not plant linking)
- n_rounds respected; trajectory length matches; residuals finite
- n_rounds=1 reduces cleanly (one subproblem + z/λ step)
- dual ascent uses z_pre residual (not post-z zero theater)
- Coker raw honesty if exposed
- no residual-must-vanish SLA
- no excel_cdu_matrix_matches_affine invent
- no PuLP offline backend
- optional readiness flag admm_block_subproblem_ok additive only

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem,
  test_tf_linear_block, test_tf_linear_coker, test_tf_linear_cdu,
  test_excel_pipeline, test_api_excel
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_multi_unit_admm_coordination_report_always_on_aggregate_ok():
    report = tlb.multi_unit_admm_coordination_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_admm_coordination"
    assert report["kind"] == tlb.ADMM_COORDINATION_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["price_source"] == tlb.PRICE_SOURCE
    assert report["lam_source"] == tlb.PRICE_SOURCE
    assert report["z_source"] == "synthetic_offline_demo"
    assert report["rho_source"] == "synthetic_offline_demo"
    assert report["optimand_space"] == "raw_affine"
    assert report["z_update_space"] == "raw_affine"
    assert report["coordination_scope"] == tlb.ADMM_COORDINATION_SCOPE
    assert report["not_plant_linking_coordinator"] is True
    assert report["coordination_lambda_is_not_case1_online_lambda"] is True
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["n_rounds"] == 3
    assert len(report["trajectory"]) == 3
    assert report["formula"] == tlb.ADMM_COORDINATION_FORMULA

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        assert row["optimand_space"] == "raw_affine"
        assert row["z_update_space"] == "raw_affine"
        assert row["not_plant_linking_coordinator"] is True
        assert len(row["rounds"]) == 3
        for rr in row["rounds"]:
            assert math.isfinite(rr["r_l1_raw"])
            assert math.isfinite(rr["r_linf_raw"])
            assert math.isfinite(rr["augmented_local_raw"])
            assert rr["ok"] is True
            assert rr["subproblem_ok"] is True
            assert rr["not_worse_than_ref"] is True


def test_coordination_honesty_not_dual_not_case1_not_wire():
    report = tlb.multi_unit_admm_coordination_report(n_rounds=2)
    note = (report.get("note") or "").lower()
    assert "raw" in note
    assert "not" in note
    assert "case 1" in note or "classic_2block" in note
    assert "wire" in note
    assert "plant" in note or "linking" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
    assert report["lam_source"] == tlb.PRICE_SOURCE
    assert report["coordination_lambda_is_not_case1_online_lambda"] is True


def test_n_rounds_respected_and_trajectory_finite():
    for n in (1, 2, 4):
        report = tlb.multi_unit_admm_coordination_report(
            n_rounds=n, rho=1.0, delta=0.5
        )
        assert report["n_rounds"] == n
        assert len(report["trajectory"]) == n
        assert report["ok"] is True, report
        for tr in report["trajectory"]:
            assert math.isfinite(tr["sum_r_l1_raw"])
            assert math.isfinite(tr["max_r_linf_raw"])
            assert math.isfinite(tr["sum_augmented_local_raw"])
            assert tr["ok"] is True
            assert set(tr["units_ok"].keys()) == set(tlb.UNITS)


def test_n_rounds_one_reduces_cleanly():
    report = tlb.multi_unit_admm_coordination_report(n_rounds=1, delta=0.5)
    assert report["ok"] is True
    assert report["n_rounds"] == 1
    assert len(report["trajectory"]) == 1
    for unit in tlb.UNITS:
        assert len(report["units"][unit]["rounds"]) == 1
        assert report["units"][unit]["ok"] is True


def test_dual_ascent_uses_z_pre_not_post_zero_theater():
    """With z_blend=1 (full copy), post residual would be 0; trajectory uses z_pre."""
    # Seed z far from any plausible y so first residual is nonzero
    coeffs = tlb.cached_offline_unit_coeffs("FCC")
    z_far = {p: 0.0 for p in coeffs.products}
    row = tlb.admm_coordination_round_for_unit(
        "FCC", z=z_far, rho=1.0, delta=1.0, dual_step=1.0, z_blend=1.0
    )
    assert row["ok"] is True
    assert row["r_l1_raw"] > 1e-6  # pre-update residual must be nonzero
    # After full copy, z_post ≈ y_raw
    for p in coeffs.products:
        assert abs(row["z_post"][p] - row["y_raw_star"][p]) <= 1e-9
    # λ must have moved: λ_post = λ_pre + α ρ r
    moved = False
    for p in coeffs.products:
        if abs(row["lam_post"][p] - row["lam_pre"][p]) > 1e-12:
            moved = True
            expected = row["lam_pre"][p] + 1.0 * 1.0 * row["r_raw"][p]
            assert abs(row["lam_post"][p] - expected) <= 1e-9
    assert moved


def test_multi_round_lambda_and_z_update():
    report = tlb.multi_unit_admm_coordination_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["ok"] is True
    # Soft structure: residual_trend is diagnostic only
    assert report["residual_trend"] in (
        "nonincreasing",
        "nondecreasing",
        "mixed",
        "n/a",
    )
    # final_lam / final_z present and finite
    for unit in tlb.UNITS:
        row = report["units"][unit]
        assert set(row["final_lam"].keys())
        assert set(row["final_z"].keys())
        assert all(math.isfinite(v) for v in row["final_lam"].values())
        assert all(math.isfinite(v) for v in row["final_z"].values())


def test_parameter_guards():
    with pytest.raises(ValueError, match="n_rounds"):
        tlb.multi_unit_admm_coordination_report(n_rounds=0)
    with pytest.raises(ValueError, match="rho"):
        tlb.multi_unit_admm_coordination_report(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        tlb.admm_coordination_round_for_unit("FCC", rho=-1.0)
    with pytest.raises(ValueError, match="dual_step"):
        tlb.multi_unit_admm_coordination_report(dual_step=float("nan"))
    with pytest.raises(ValueError, match="z_blend"):
        tlb.multi_unit_admm_coordination_report(z_blend=1.5)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.admm_coordination_round_for_unit("BLENDER")


def test_coker_raw_honesty_preserved_when_exposed():
    row = tlb.admm_coordination_round_for_unit("COKER", rho=1.0, delta=0.5)
    assert row["ok"] is True
    assert row["optimand_space"] == "raw_affine"
    assert row["z_update_space"] == "raw_affine"
    note = (row.get("renorm_note") or "").lower()
    assert "raw" in note
    assert math.isfinite(float(row.get("raw_vs_full_r_l1_gap_star") or 0.0))


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_unit_admm_coordination_report(n_rounds=1)
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_no_residual_must_vanish_sla():
    """ok must not require residual→0 (absolute magnitude SLA forbidden)."""
    report = tlb.multi_unit_admm_coordination_report(n_rounds=2, delta=1.0)
    assert report["ok"] is True
    # Even if residuals stay nonzero, ok holds
    any_nonzero = any(tr["sum_r_l1_raw"] > 1e-12 for tr in report["trajectory"])
    # With full-seed z vs raw optimand, first residual often nonzero — soft check
    assert report["ok"] is True
    _ = any_nonzero  # structure lock only; never hard-fail on magnitude


def test_reuses_subproblem_maximizer_kind():
    row = tlb.admm_coordination_round_for_unit("FCC", delta=0.5)
    assert row["subproblem_kind"] == tlb.ADMM_SUBPROBLEM_KIND
    assert row["subproblem_ok"] is True


def test_coordination_public_exports():
    for name in (
        "ADMM_COORDINATION_KIND",
        "ADMM_COORDINATION_FORMULA",
        "ADMM_COORDINATION_SCOPE",
        "admm_coordination_round_for_unit",
        "multi_unit_admm_coordination_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_honesty_metadata_mentions_coordination():
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    assert meta.get("admm_coordination_available") is True
    note = (meta.get("note") or "").lower()
    assert "coordination" in note


def test_readiness_admm_block_subproblem_ok_additive():
    """Secondary: admm_block_subproblem_ok additive; does not redefine ready."""
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_residual=True,
        include_admm_block_subproblem=True,
        include_admm_coordination=False,
    )
    assert "admm_block_subproblem_ok" in rep
    assert rep["admm_block_subproblem_ok"] is True
    assert "admm_residual_ok" in rep
    # ready_for_wire_discussion still parity∧priced∧timings∧honesty only
    assert "ready_for_wire_discussion" in rep
    ready = bool(rep["ready_for_wire_discussion"])
    # Structural ready equals ok (same as before) — not AND subproblem
    assert rep["ok"] is ready
    note = (rep.get("note") or "").lower()
    assert "additive" in note
    assert "ready_for_wire_discussion" in note


def test_readiness_can_skip_subproblem_flag():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=3,
        warmup=0,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
        include_admm_coordination=False,
        include_admm_plant_linking=False,
    )
    assert rep["admm_block_subproblem_ok"] is None
    assert rep["admm_residual_ok"] is None
    assert rep["admm_coordination_ok"] is None
    assert rep["admm_plant_linking_ok"] is None


def test_readiness_admm_coordination_ok_additive():
    """Optional secondary: admm_coordination_ok additive; does not redefine ready."""
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_residual=True,
        include_admm_block_subproblem=True,
        include_admm_coordination=True,
    )
    assert "admm_coordination_ok" in rep
    assert rep["admm_coordination_ok"] is True
    assert "admm_block_subproblem_ok" in rep
    assert "admm_residual_ok" in rep
    # ready_for_wire_discussion still parity∧priced∧timings∧honesty only
    assert "ready_for_wire_discussion" in rep
    ready = bool(rep["ready_for_wire_discussion"])
    assert rep["ok"] is ready
    note = (rep.get("note") or "").lower()
    assert "additive" in note
    assert "coordination" in note
    assert "ready_for_wire_discussion" in note
    assert "plant" in note or "linking" in note
