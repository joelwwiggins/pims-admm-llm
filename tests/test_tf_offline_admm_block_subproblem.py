"""E1: offline multi-unit ADMM block subproblem maximizer.

Always-on sections run without TensorFlow. Locks:
- multi_unit_admm_block_subproblem_report lists FCC/COKER/CDU
- aggregate ok without TF
- dual_recovery_path is None; on_excel_case1_path is False; solver False
- optimand_space = raw_affine; synthetic λ/z/ρ (not Case 1 online λ)
- maximizer not worse than ref on raw; delta=0 ⇒ x_star≈x0
- L1 identity on raw fields; Coker raw≠full diagnostic
- no excel_cdu_matrix_matches_affine invent
- no brittle absolute residual magnitude / µs SLAs
- no PuLP offline subproblem backend

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
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


def test_multi_unit_admm_block_subproblem_report_always_on_aggregate_ok():
    report = tlb.multi_unit_admm_block_subproblem_report(rho=1.0, delta=0.5)
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_admm_block_subproblem"
    assert report["kind"] == tlb.ADMM_SUBPROBLEM_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["price_source"] == tlb.PRICE_SOURCE
    assert report["optimand_space"] == "raw_affine"
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["formula"] == tlb.ADMM_SUBPROBLEM_FORMULA_L1_RAW
    assert "lambda_dot_y_raw" in report["formula"]
    assert "y_raw" in report["formula"]
    assert report["method"] == tlb.ADMM_SUBPROBLEM_METHOD

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        assert row["optimand_space"] == "raw_affine"
        assert math.isfinite(row["augmented_local_raw"])
        assert math.isfinite(row["lambda_dot_y_raw"])
        assert math.isfinite(row["r_l1_raw"])
        assert math.isfinite(row["penalty_raw"])
        assert row["not_worse_than_ref"] is True
        assert row["augmented_local_raw"] + 1e-9 >= row["augmented_local_raw_ref"]
        # L1 identity on raw fields
        expect_aug = row["lambda_dot_y_raw"] - float(report["rho"]) * row["r_l1_raw"]
        assert abs(row["augmented_local_raw"] - expect_aug) <= 1e-9
        assert abs(row["penalty_raw"] - float(report["rho"]) * row["r_l1_raw"]) <= 1e-9
        assert set(row["y_raw_star"].keys()) == set(row["products"])
        assert len(row["x_star"]) == len(row["x0"]) == len(row["drivers"])


def test_admm_block_subproblem_honesty_not_dual_not_case1():
    report = tlb.multi_unit_admm_block_subproblem_report()
    note = (report.get("note") or "").lower()
    assert "raw" in note
    assert "not" in note and (
        "online" in note or "dual" in note or "shadow" in note or "λ" in note
    )
    assert "case 1" in note or "classic_2block" in note
    assert "wire" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
    assert report["lam_source"] == tlb.PRICE_SOURCE
    assert report["z_source"] == "synthetic_offline_demo"
    assert report["rho_source"] == "synthetic_offline_demo"
    assert "optimality" in (report.get("optimality_note") or "").lower() or (
        "coordinate" in (report.get("optimality_note") or "").lower()
    )


def test_maximizer_not_worse_than_ref():
    for unit in tlb.UNITS:
        row = tlb.admm_block_subproblem_for_unit(unit, rho=1.0, delta=1.0)
        assert row["ok"] is True
        assert row["augmented_local_raw"] + 1e-9 >= row["augmented_local_raw_ref"]
        assert row["improvement_raw"] >= -1e-9


def test_delta_zero_x_star_equals_x0():
    for unit in tlb.UNITS:
        row = tlb.admm_block_subproblem_for_unit(unit, delta=0.0)
        assert row["ok"] is True
        assert np.allclose(row["x_star"], row["x0"], atol=1e-12)
        assert abs(row["augmented_local_raw"] - row["augmented_local_raw_ref"]) <= 1e-9


def test_coker_raw_vs_full_diagnostic_honesty():
    """Coker renorm: raw optimand may differ from full diagnostic even near ref."""
    row = tlb.admm_block_subproblem_for_unit("COKER", rho=1.0, delta=0.5)
    assert row["ok"] is True
    assert row["optimand_space"] == "raw_affine"
    # Diagnostic fields present
    assert math.isfinite(row["augmented_local_full_diagnostic"])
    assert math.isfinite(row["raw_vs_full_aug_gap_star"])
    assert math.isfinite(row["raw_vs_full_r_l1_gap_star"])
    # Coker renorm typically creates raw≠full gap (may be zero only if trivial)
    note = (row.get("renorm_note") or "").lower()
    assert "raw" in note and ("full" in note or "diagnostic" in note)
    # At least renorm language or a nonzero gap somewhere under offset box
    assert "renorm" in note or row["raw_vs_full_r_l1_gap_star"] > 1e-9 or (
        row["raw_vs_full_aug_gap_star"] > 1e-9
    )


def test_rho_must_be_positive():
    with pytest.raises(ValueError, match="rho"):
        tlb.multi_unit_admm_block_subproblem_report(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        tlb.admm_block_subproblem_for_unit("FCC", rho=-1.0)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.admm_block_subproblem_for_unit("BLENDER")


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_unit_admm_block_subproblem_report()
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_admm_block_subproblem_public_exports():
    for name in (
        "ADMM_SUBPROBLEM_KIND",
        "ADMM_SUBPROBLEM_FORMULA_L1_RAW",
        "admm_block_subproblem_for_unit",
        "multi_unit_admm_block_subproblem_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_honesty_metadata_mentions_admm_block_subproblem():
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    assert meta.get("admm_block_subproblem_available") is True
    note = (meta.get("note") or "").lower()
    assert "subproblem" in note or "block subproblem" in note


def test_box_feasibility_x_star_in_box():
    delta = 0.75
    for unit in tlb.UNITS:
        row = tlb.admm_block_subproblem_for_unit(unit, delta=delta)
        x0 = np.asarray(row["x0"], dtype=float)
        xs = np.asarray(row["x_star"], dtype=float)
        assert np.all(xs >= x0 - delta - 1e-9)
        assert np.all(xs <= x0 + delta + 1e-9)


def test_structure_subproblem_not_identical_to_pure_priced_corner_when_z_offset():
    """Soft structure lock: L1 coupling can move x_star away from pure priced corner.

    Uses a strong z offset so penalty matters. Not a hard 'must always differ'
    under all ρ — only check under a clear coupling regime.
    """
    coeffs = tlb.cached_offline_unit_coeffs("FCC")
    # z far from any raw yield → penalty couples strongly
    z_far = {p: 0.0 for p in coeffs.products}
    row = tlb.admm_block_subproblem_for_unit(
        "FCC", z=z_far, rho=5.0, delta=1.0
    )
    box = tlb.local_box_direction("FCC", delta=1.0)
    x_star = np.asarray(row["x_star"], dtype=float)
    x_box = np.asarray(box["x_star"], dtype=float)
    # Soft: either differs from priced corner OR aug_raw(star) >= aug_raw(box)
    differs = float(np.max(np.abs(x_star - x_box))) > 1e-6
    aug_star = row["augmented_local_raw"]
    # evaluate box under same z/rho
    p_dict, p_vec, _ = tlb._resolve_prices("FCC", None)
    z_dict, z_vec, _ = tlb._resolve_z_vector(coeffs, z_far)
    y_box = tlb.numpy_affine_forward(coeffs, x_box, clamp_products=True)
    aug_box = float(p_vec @ y_box) - 5.0 * float(np.sum(np.abs(y_box - z_vec)))
    assert aug_star + 1e-9 >= min(row["augmented_local_raw_ref"], aug_box) - 1e-9
    # At least structure fields present for diagnostics
    assert math.isfinite(row["augmented_local_raw_priced_box"])
    assert differs or aug_star + 1e-9 >= aug_box


def test_custom_z_override_source_label():
    coeffs = tlb.cached_offline_unit_coeffs("CDU")
    z_bad = {p: 0.0 for p in coeffs.products}
    row = tlb.admm_block_subproblem_for_unit("CDU", z=z_bad, delta=0.25)
    assert row["ok"] is True
    assert row["z_source"] == "caller_override"
