"""E1/E2: Coker exact-linear kernel package + E7/E8 Excel honesty.

Always-on sections run without TensorFlow. TF graph tests use skipif.
Covers:
- L0 affine shape / pack_driver / manual loop
- L1: affine + postprocess_coker_yields ≡ evaluate
- L_div: raw affine ≠ evaluate (renorm always-on, including at ref)
- Excel E7/E8 MB_* match (pre-postprocess)
- honesty_metadata multi-unit dual ban
- optional TF forward ≡ numpy L0/L1
"""

from __future__ import annotations

import numpy as np
import pytest

from pims_admm_llm.models.base_delta import (
    build_coker_base_delta,
    postprocess_coker_yields,
    process_modes_coker,
)
from pims_admm_llm.models import tf_linear_blocks as tlb


# ---------------------------------------------------------------------------
# Always-on: L0 shape + Excel E7/E8 + honesty
# ---------------------------------------------------------------------------


def test_coker_affine_shape_5x6_always_on():
    """L0 foundation: COKER is 5 products × 6 drivers."""
    model = build_coker_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    assert coeffs.unit == "COKER"
    assert len(coeffs.products) == 5
    assert len(coeffs.drivers) == 6
    assert coeffs.y0.shape == (5,)
    assert coeffs.D.shape == (5, 6)
    assert coeffs.x0.shape == (6,)
    assert coeffs.y0.dtype == np.float64
    y = tlb.numpy_affine_forward(coeffs, coeffs.x0)
    np.testing.assert_allclose(y, coeffs.y0, atol=1e-15)


def test_coker_pack_driver_and_manual_loop_l0():
    """Always-on L0: pack_driver + manual BASE/DELTA loop at feed/process offset."""
    model = build_coker_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    feed = dict(model.reference_feed)
    feed["api"] = 12.0
    feed["ccr_wt"] = 10.0
    cond = {"drum_outlet_temp_f": 930.0, "recycle_ratio": 0.20}
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_vec = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    y_dict = tlb.y_raw_dict(coeffs, y_vec)

    from pims_admm_llm.models.base_delta import merge_process_conditions

    feed_m = dict(feed)
    cond_m = merge_process_conditions("COKER", cond)
    flat = dict(feed_m)
    for k, v in cond_m.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
    ref_flat = dict(model.reference_feed)
    for k, v in model.reference_conditions.items():
        if isinstance(v, (int, float)):
            ref_flat[k] = float(v)
    for p in model.products:
        base = float(model.base_yields.get(p, 0.0))
        dy = 0.0
        for drv, coef in (model.deltas.get(p) or {}).items():
            x0 = float(ref_flat.get(drv, 0.0))
            xv = float(flat.get(drv, x0))
            dy += float(coef) * (xv - x0)
        assert abs(y_dict[p] - max(0.0, base + dy)) < 1e-12


def test_excel_coker_matrix_matches_affine_e7_e8():
    """E7/E8: Submodel_Coker MB_* BASE/D_* == affine package (always-on, no TF)."""
    report = tlb.excel_coker_matrix_matches_affine(atol=1e-12)
    assert report["ok"], report.get("mismatches")
    # BASE + 6 D_* per product × 5 products
    assert report["checked"] >= 5 * 7
    assert report["n_products"] == 5
    assert report["n_drivers"] == 6
    assert report["mismatches"] == []


def test_excel_coker_report_shape_and_order():
    """E8: report shape includes checked/mismatches/n_products/n_drivers."""
    report = tlb.excel_coker_matrix_matches_affine()
    for key in ("ok", "checked", "mismatches", "n_products", "n_drivers"):
        assert key in report
    model = build_coker_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    assert list(model.products) == list(coeffs.products)
    assert list(model.drivers) == list(coeffs.drivers)


def test_coker_excel_match_scope_is_mb_rows_only():
    """Only MB_* product rows are compared — teaching/FREE rows not required."""
    from pims_admm_llm.models.excel_pipeline import base_delta_unit_submodel_tables

    tables = base_delta_unit_submodel_tables()
    matrix = tables.get("coker_pims_matrix") or []
    rows = {str(r.get("row")) for r in matrix if r.get("row")}
    assert any(r.startswith("MB_") for r in rows)
    report = tlb.excel_coker_matrix_matches_affine(atol=1e-12)
    assert report["ok"], report.get("mismatches")


def test_honesty_metadata_still_bans_duals_after_coker_helper():
    """Multi-unit surface must not invent dual recovery claims."""
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert tlb.DUAL_RECOVERY_PATH is None
    units = meta.get("units") or list(tlb.UNITS)
    assert "FCC" in units and "COKER" in units


# ---------------------------------------------------------------------------
# Always-on: postprocess extract + L1 + L_div (product-critical for Coker)
# ---------------------------------------------------------------------------


def test_postprocess_coker_yields_identity_and_evaluate_freeze():
    """Helper ≡ evaluate at ref + modes; pure copy (does not mutate input)."""
    model = build_coker_base_delta()
    y_base = {p: float(model.base_yields[p]) for p in model.products}
    y_in = dict(y_base)
    y_pp = postprocess_coker_yields(y_in)
    # pure: input not mutated
    assert y_in == y_base
    y_eval = model.evaluate(clamp_products=True)
    for p in model.products:
        assert abs(y_pp[p] - y_eval[p]) < 1e-15, (p, y_pp[p], y_eval[p])

    # golden freeze across process_modes_coker
    for m in process_modes_coker(model):
        cond = dict(m.get("conditions") or {})
        feed = dict(model.reference_feed)
        y_eval_m = model.evaluate(feed, cond, clamp_products=True)
        # re-build raw via affine then postprocess
        coeffs = tlb.affine_coeffs_from_base_delta(model)
        x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
        y_raw = tlb.y_raw_dict(
            coeffs, tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        )
        y_full = postprocess_coker_yields(y_raw)
        for p in model.products:
            assert abs(y_full[p] - y_eval_m[p]) < 1e-12, (m.get("id"), p)


def test_l1_affine_plus_postprocess_matches_evaluate_coker():
    """L1: affine + apply_coker_postprocess ≡ evaluate (ref + offset + modes)."""
    model = build_coker_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    cases = [
        (None, None),
        ({"api": 12.0, "ccr_wt": 10.0}, {"drum_outlet_temp_f": 930.0}),
        ({"api": 8.0, "sulfur_wt": 3.0}, {"recycle_ratio": 0.25, "drum_pressure_psig": 30.0}),
    ]
    for feed, cond in cases:
        feed_use = dict(
            model.reference_feed if feed is None else {**model.reference_feed, **feed}
        )
        x = tlb.pack_driver_vector(coeffs, feed=feed_use, conditions=cond)
        y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        y_full = tlb.apply_coker_postprocess(y_raw, products=coeffs.products)
        y_eval = model.evaluate(feed_use, cond, clamp_products=True)
        for p in coeffs.products:
            assert abs(y_full[p] - y_eval[p]) < 1e-9, (p, y_full[p], y_eval[p])

    # process_modes severity points
    for m in process_modes_coker(model):
        feed = dict(model.reference_feed)
        cond = dict(m.get("conditions") or {})
        x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
        y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        y_full = tlb.apply_coker_postprocess(y_raw, products=coeffs.products)
        y_eval = model.evaluate(feed, cond, clamp_products=True)
        for p in coeffs.products:
            assert abs(y_full[p] - y_eval[p]) < 1e-9, (m.get("id"), p)


def test_l_div_raw_affine_ne_evaluate_at_reference():
    """L_div honesty: renorm always engages — raw ≠ evaluate even at x0.

    Measured max_gap at ref ≈ 0.024; modes ≈ 0.015–0.038.
    """
    model = build_coker_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    # At reference
    y_raw = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, coeffs.x0, clamp_products=True)
    )
    y_eval = model.evaluate(clamp_products=True)
    max_gap = max(abs(y_raw[p] - y_eval[p]) for p in coeffs.products)
    assert max_gap > 1e-3, f"expected renorm divergence at ref, max_gap={max_gap}"

    # Feed offset + modes also diverge
    feed = dict(model.reference_feed)
    feed["api"] = 10.0
    cond = {"recycle_ratio": 0.30}
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_raw2 = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    )
    y_eval2 = model.evaluate(feed, cond, clamp_products=True)
    max_gap2 = max(abs(y_raw2[p] - y_eval2[p]) for p in coeffs.products)
    assert max_gap2 > 1e-3, f"expected renorm divergence off-ref, max_gap={max_gap2}"


def test_renorm_policy_invariants():
    """Freeze clamp bounds + liquid_vol formula + liquid sum == liquid_vol."""
    model = build_coker_base_delta()
    y_base = {p: float(model.base_yields[p]) for p in model.products}
    y_pp = postprocess_coker_yields(y_base)
    coke = y_pp["coker_coke"]
    assert 0.12 - 1e-12 <= coke <= 0.40 + 1e-12
    liquids = ["coker_dry_gas", "coker_lpg", "coker_naphtha", "coker_gasoil"]
    liquid_vol = max(0.50, min(0.80, 0.96 - coke))
    s = sum(y_pp[p] for p in liquids)
    assert abs(s - liquid_vol) < 1e-12
    # base liquids sum 0.70 → renorm to 0.74 at coke=0.22
    assert abs(sum(y_base[p] for p in liquids) - 0.70) < 1e-12
    assert abs(liquid_vol - 0.74) < 1e-12


# ---------------------------------------------------------------------------
# Optional TF path (skip if TF absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_coker_forward_matches_numpy_l0():
    block = tlb.tf_linear_coker()
    coeffs = block.coeffs
    assert coeffs.unit == "COKER"
    x = coeffs.x0 + np.array([1.0, 0.1, 0.5, 10.0, 0.05, 2.0], dtype=np.float64)
    y_np = tlb.numpy_affine_forward(coeffs, x, clamp_products=False)
    y_tf = np.asarray(block.forward(x, clamp_products=False, as_dict=False), dtype=np.float64)
    np.testing.assert_allclose(y_tf, y_np, atol=1e-10)
    meta = block.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["solver"] is False
    assert meta["on_excel_case1_path"] is False
    assert meta["unit"] == "COKER"


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_coker_l1_matches_evaluate():
    block = tlb.tf_linear_coker()
    model = build_coker_base_delta()
    feed = dict(model.reference_feed)
    feed["api"] = 11.0
    cond = {"drum_outlet_temp_f": 925.0, "recycle_ratio": 0.18}
    x = tlb.pack_driver_vector(block.coeffs, feed=feed, conditions=cond)
    y_raw = block.forward(x, clamp_products=True)
    y_full = tlb.apply_coker_postprocess(y_raw, products=block.coeffs.products)
    y_eval = model.evaluate(feed, cond, clamp_products=True)
    for p in block.coeffs.products:
        assert abs(y_full[p] - y_eval[p]) < 1e-9


@pytest.mark.skipif(tlb.tf_available(), reason="only when TF missing")
def test_tf_linear_coker_raises_when_tf_missing():
    with pytest.raises(ImportError):
        tlb.tf_linear_coker()
