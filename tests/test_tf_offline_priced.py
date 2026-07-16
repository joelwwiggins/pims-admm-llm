"""E1/E2: offline multi-unit priced residual harness + optional box direction.

Always-on sections run without TensorFlow. Locks:
- multi_unit_priced_residual_report lists FCC/COKER/CDU
- aggregate ok without TF
- dual_recovery_path is None; on_excel_case1_path is False
- prices are synthetic demo (not duals / not Case 1 shadows)
- ref residual tight (postprocess-aware)
- Coker raw priced may ≠ full evaluate even at reference
- no excel_cdu_matrix_matches_affine invent
"""

from __future__ import annotations

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


def test_default_offline_prices_cover_products_nonneg():
    for unit in tlb.UNITS:
        prices = tlb.default_offline_prices(unit)
        coeffs = tlb.offline_unit_coeffs(unit)
        assert set(prices.keys()) == set(coeffs.products)
        assert all(float(v) >= 0.0 for v in prices.values())
        vec = tlb.pack_price_vector(coeffs, prices)
        assert vec.shape == (len(coeffs.products),)
        assert float(vec.min()) >= 0.0


def test_pack_price_vector_missing_and_unknown():
    coeffs = tlb.offline_unit_coeffs("FCC")
    full = tlb.default_offline_prices("FCC")
    incomplete = {k: full[k] for k in list(full)[:2]}
    with pytest.raises(ValueError, match="Missing product prices"):
        tlb.pack_price_vector(coeffs, incomplete)
    filled = tlb.pack_price_vector(coeffs, incomplete, fill_missing=True)
    assert filled.shape[0] == len(coeffs.products)
    with pytest.raises(ValueError, match="Unknown product price"):
        tlb.pack_price_vector(coeffs, {**full, "not_a_product": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        bad = dict(full)
        bad[coeffs.products[0]] = -1.0
        tlb.pack_price_vector(coeffs, bad)


def test_unknown_unit_prices_raises():
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.default_offline_prices("BLENDER")
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.priced_residual_for_unit("REFORMER")


def test_multi_unit_priced_residual_report_always_on_aggregate_ok():
    report = tlb.multi_unit_priced_residual_report(atol=1e-9, rtol=1e-9)
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_priced_residual"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["price_source"] == tlb.PRICE_SOURCE
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        assert row["ref"]["ok"] is True, (unit, row["ref"])
        assert row["offset"]["ok"] is True, (unit, row["offset"])
        assert row["ref"]["abs_err"] <= 1e-9 + 1e-9 * abs(row["ref"]["v_eval"])


def test_priced_residual_honesty_not_dual_not_case1():
    report = tlb.multi_unit_priced_residual_report()
    note = (report.get("note") or "").lower()
    assert "not a solve" in note or "not on" in note
    assert "not admm" in note or "dual" in note
    assert "not" in note and ("shadow" in note or "λ" in note or "lambda" in note or "dual" in note)
    assert "case 1" in note or "classic_2block" in note
    # Must not claim dual recovery ownership
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
    assert "admm dual recovery" not in note or "not" in note


def test_coker_raw_vs_full_priced_gap_at_ref():
    """Coker renorm always engages → raw priced value may ≠ full evaluate at ref."""
    row = tlb.priced_residual_for_unit("COKER")
    assert row["ok"] is True  # full postprocess residual still tight
    # y0 sum ~0.92 → renorm scales liquids; with non-uniform prices raw≠full
    assert row["raw_vs_full_priced_gap"] > 1e-6, row
    assert abs(row["v_raw"] - row["v_eval"]) > 1e-6
    # But postprocess path tracks evaluate
    assert abs(row["v_aff"] - row["v_eval"]) <= 1e-9


def test_cdu_ref_full_residual_tight_often_raw_close():
    row = tlb.priced_residual_for_unit("CDU")
    assert row["ok"] is True
    assert abs(row["v_aff"] - row["v_eval"]) <= 1e-9


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_unit_priced_residual_report()
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_priced_public_exports():
    for name in (
        "default_offline_prices",
        "pack_price_vector",
        "priced_residual_for_unit",
        "multi_unit_priced_residual_report",
        "local_box_direction",
        "PRICE_SOURCE",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_local_box_direction_closed_form_and_honesty():
    box = tlb.local_box_direction("FCC", delta=1.0)
    assert box["kind"] == "offline_local_box_direction"
    assert box["solver"] is False
    assert box["dual_recovery_path"] is None
    assert box["on_excel_case1_path"] is False
    assert box["price_source"] == tlb.PRICE_SOURCE
    note = (box.get("note") or "").lower()
    assert "outside" in note and ("postprocess" in note or "linear program" in note)
    assert "not admm" in note or "not" in note
    assert "shadow" in note or "λ" in note or "lambda" in note or "dual" in note
    # Corner maximizer: raw at x_star should be ≥ raw at x0 for linear objective
    assert box["v_raw_star"] + 1e-12 >= box["v_raw_ref"]
    assert len(box["x_star"]) == len(box["x0"]) == len(box["drivers"])
    # TF arm skip when absent
    if not tlb.tf_available():
        assert box["tf"]["skipped"] is True


def test_local_box_coker_raw_vs_full_split():
    box = tlb.local_box_direction("COKER", delta=0.5)
    assert box["dual_recovery_path"] is None
    # Coker renorm typically keeps raw ≠ full
    assert box["raw_vs_full_priced_gap_star"] >= 0.0
    # Still not dual claims
    assert "ADMM λ" in box["note"] or "ADMM" in box["note"]


def test_priced_residual_for_unit_custom_prices():
    prices = tlb.default_offline_prices("FCC")
    # Uniform prices → residual still ok; raw may closer to full if renorm preserves mass
    uniform = {k: 10.0 for k in prices}
    row = tlb.priced_residual_for_unit("FCC", uniform)
    assert row["ok"] is True
    assert row["prices"] == uniform


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow optional; absent on Jetson")
def test_local_box_tf_raw_matches_numpy_when_available():
    box = tlb.local_box_direction("FCC", delta=1.0)
    assert box["tf"]["skipped"] is False
    assert box["tf"]["ok"] is True, box["tf"]
