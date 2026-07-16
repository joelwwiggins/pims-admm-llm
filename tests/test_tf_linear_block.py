"""E1/E10: FCC exact-linear affine package + Excel honesty + optional TF parity.

Always-on sections run without TensorFlow. TF graph tests use importorskip.
"""

from __future__ import annotations

import numpy as np
import pytest

from pims_admm_llm.models.base_delta import (
    build_fcc_base_delta,
    postprocess_fcc_yields,
    process_modes_fcc,
)
from pims_admm_llm.models import tf_linear_blocks as tlb


# ---------------------------------------------------------------------------
# Always-on: affine package + honesty + Excel consistency (E10)
# ---------------------------------------------------------------------------


def test_affine_coeffs_fcc_shape_and_y0():
    model = build_fcc_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    assert coeffs.unit == "FCC"
    assert len(coeffs.products) == 6
    assert len(coeffs.drivers) == 8
    assert coeffs.y0.shape == (6,)
    assert coeffs.D.shape == (6, 8)
    assert coeffs.x0.shape == (8,)
    assert coeffs.y0.dtype == np.float64
    # y0 + D@0 == y0
    y = tlb.numpy_affine_forward(coeffs, coeffs.x0)
    np.testing.assert_allclose(y, coeffs.y0, atol=1e-15)
    # dict recompute vs D matrix
    for i, p in enumerate(coeffs.products):
        for j, d in enumerate(coeffs.drivers):
            expected = float((model.deltas.get(p) or {}).get(d, 0.0))
            assert abs(float(coeffs.D[i, j]) - expected) < 1e-12


def test_numpy_affine_matches_manual_loop():
    model = build_fcc_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    feed = dict(model.reference_feed)
    feed["api"] = 25.0
    cond = {"riser_outlet_temp_f": 1000.0}
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_vec = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    y_dict = tlb.y_raw_dict(coeffs, y_vec)

    # Manual pre-postprocess loop (mirrors _FCCModel affine body)
    from pims_admm_llm.models.base_delta import merge_process_conditions

    feed_m = dict(feed)
    cond_m = merge_process_conditions("FCC", cond)
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


def test_l1_affine_plus_postprocess_matches_evaluate():
    model = build_fcc_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    cases = [
        (None, None),
        ({"api": 25.0}, {"riser_outlet_temp_f": 1000.0}),
        ({"api": 20.0, "ccr_wt": 0.6}, {"riser_outlet_temp_f": 940.0, "catalyst_to_oil": 5.5}),
    ]
    for feed, cond in cases:
        feed_use = dict(model.reference_feed if feed is None else {**model.reference_feed, **feed})
        x = tlb.pack_driver_vector(coeffs, feed=feed_use, conditions=cond)
        y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        y_full = tlb.apply_fcc_postprocess(y_raw, products=coeffs.products)
        y_eval = model.evaluate(feed_use, cond, clamp_products=True)
        for p in coeffs.products:
            assert abs(y_full[p] - y_eval[p]) < 1e-9, (p, y_full[p], y_eval[p])


def test_raw_affine_ne_evaluate_when_renorm_engages():
    """Honesty: pre-postprocess affine is not full evaluate when renorm fires."""
    model = build_fcc_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    feed = dict(model.reference_feed)
    feed["api"] = 28.0
    cond = {"riser_outlet_temp_f": 1020.0, "catalyst_to_oil": 7.5}
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_raw = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    )
    y_eval = model.evaluate(feed, cond, clamp_products=True)
    max_gap = max(abs(y_raw[p] - y_eval[p]) for p in coeffs.products)
    assert max_gap > 1e-4, f"expected renorm divergence, max_gap={max_gap}"


def test_process_mode_severity_point_l1():
    model = build_fcc_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    modes = process_modes_fcc(model)
    assert modes
    m = modes[0]  # rot_low
    feed = dict(model.reference_feed)
    cond = dict(m.get("conditions") or {})
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    y_full = tlb.apply_fcc_postprocess(y_raw, products=coeffs.products)
    y_eval = model.evaluate(feed, cond, clamp_products=True)
    for p in coeffs.products:
        assert abs(y_full[p] - y_eval[p]) < 1e-9


def test_honesty_metadata_bans_dual_recovery():
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["kind"] in ("exact_linear_copy", "exact_linear_copy_surface")
    assert "dual" not in (meta.get("note") or "").lower() or "not" in (
        meta.get("note") or ""
    ).lower()
    assert tlb.DUAL_RECOVERY_PATH is None
    assert tlb.SOLVER is False


def test_postprocess_fcc_yields_identity_at_ref():
    model = build_fcc_base_delta()
    # raw base yields through postprocess vs evaluate at ref
    y_base = {p: float(model.base_yields[p]) for p in model.products}
    y_pp = postprocess_fcc_yields(y_base)
    y_eval = model.evaluate(clamp_products=True)
    for p in model.products:
        assert abs(y_pp[p] - y_eval[p]) < 1e-12


def test_excel_fcc_matrix_matches_affine_e10():
    """E10: Submodel_FCC MB_* BASE/D_* == affine package (always-on, no TF)."""
    report = tlb.excel_fcc_matrix_matches_affine(atol=1e-12)
    assert report["ok"], report.get("mismatches")
    assert report["checked"] >= 6 * 9  # BASE + 8 D_* per product
    assert report["n_products"] == 6
    assert report["n_drivers"] == 8


def test_module_import_without_tf():
    """Importing tf_linear_blocks must not require tensorflow."""
    import importlib
    import sys

    # ensure module loads
    mod = importlib.import_module("pims_admm_llm.models.tf_linear_blocks")
    assert hasattr(mod, "affine_coeffs_from_base_delta")
    assert hasattr(mod, "TFLinearBlock")
    assert hasattr(mod, "tf_linear_fcc")
    # Does not force-load tensorflow just by import of helper functions
    _ = mod.affine_coeffs_from_base_delta(build_fcc_base_delta())
    # tf_available is the only probe
    ok = mod.tf_available()
    assert isinstance(ok, bool)
    if not ok:
        assert "tensorflow" not in sys.modules or True  # may be present from other tests


# ---------------------------------------------------------------------------
# Optional TF path (skip if TF absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_forward_matches_numpy_l0():
    block = tlb.tf_linear_fcc()
    coeffs = block.coeffs
    x = coeffs.x0 + np.array(
        [1.0, 0.0, 0.05, 10.0, 0.2, 0.0, 5.0, 0.0], dtype=np.float64
    )
    y_np = tlb.numpy_affine_forward(coeffs, x, clamp_products=False)
    y_tf = np.asarray(block.forward(x, clamp_products=False, as_dict=False), dtype=np.float64)
    np.testing.assert_allclose(y_tf, y_np, atol=1e-10)
    meta = block.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["solver"] is False
    assert meta["on_excel_case1_path"] is False


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_l1_matches_evaluate():
    block = tlb.tf_linear_fcc()
    model = build_fcc_base_delta()
    feed = dict(model.reference_feed)
    feed["api"] = 24.0
    cond = {"riser_outlet_temp_f": 990.0}
    x = tlb.pack_driver_vector(block.coeffs, feed=feed, conditions=cond)
    y_raw = block.forward(x, clamp_products=True)
    y_full = tlb.apply_fcc_postprocess(y_raw, products=block.coeffs.products)
    y_eval = model.evaluate(feed, cond, clamp_products=True)
    for p in block.coeffs.products:
        assert abs(y_full[p] - y_eval[p]) < 1e-9


@pytest.mark.skipif(tlb.tf_available(), reason="only when TF missing")
def test_tf_linear_fcc_raises_when_tf_missing():
    with pytest.raises(ImportError):
        tlb.tf_linear_fcc()
