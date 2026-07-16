"""E1/E2: CDU exact-linear offline kernel package.

Always-on sections run without TensorFlow. TF graph tests use skipif.
Covers:
- Nested cut_points_f.* x0 honesty (pack@ref ≡ x0; L0 affine@pack ≡ y0)
- L0 shape 6×8 float64
- L1: affine + postprocess_cdu_yields ≡ evaluate
- L_div / renorm policy (identity-at-ref likely; diverge when cuts/API move)
- Anti-claim: no excel_cdu_matrix_matches_affine; Submodel_CDU TECH+A
- honesty_metadata multi-unit dual ban (FCC+COKER+CDU)
- optional TF forward ≡ numpy L0/L1
- FCC/Coker x0 stability (no nested drivers → unchanged flatten)
"""

from __future__ import annotations

import numpy as np
import pytest

from pims_admm_llm.models.base_delta import (
    build_cdu_base_delta,
    build_coker_base_delta,
    build_fcc_base_delta,
    postprocess_cdu_yields,
    process_modes_cdu,
)
from pims_admm_llm.models import tf_linear_blocks as tlb


# ---------------------------------------------------------------------------
# Always-on: nested x0 gate + L0 shape
# ---------------------------------------------------------------------------


def test_cdu_affine_shape_6x8_and_nested_drivers():
    """L0 foundation: CDU is 6 products × 8 drivers with nested cut keys."""
    model = build_cdu_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    assert coeffs.unit == "CDU"
    assert len(coeffs.products) == 6
    assert len(coeffs.drivers) == 8
    assert coeffs.y0.shape == (6,)
    assert coeffs.D.shape == (6, 8)
    assert coeffs.x0.shape == (8,)
    assert coeffs.y0.dtype == np.float64
    nested = [d for d in coeffs.drivers if d.startswith("cut_points_f.")]
    assert nested == [
        "cut_points_f.naphtha_ep",
        "cut_points_f.distillate_ep",
        "cut_points_f.gasoil_ep",
    ]
    # Nested x0 must be real cut points, not zeros (blocking bug gate)
    by_drv = dict(zip(coeffs.drivers, coeffs.x0.tolist()))
    assert abs(by_drv["cut_points_f.naphtha_ep"] - 392.0) < 1e-9
    assert abs(by_drv["cut_points_f.distillate_ep"] - 698.0) < 1e-9
    assert abs(by_drv["cut_points_f.gasoil_ep"] - 1022.0) < 1e-9
    assert by_drv["cut_points_f.naphtha_ep"] != 0.0


def test_cdu_pack_at_ref_matches_x0_and_l0():
    """Acceptance: pack@ref ≡ x0; raw affine at pack@ref ≡ y0."""
    model = build_cdu_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    x = tlb.pack_driver_vector(
        coeffs,
        feed=model.reference_feed,
        conditions=model.reference_conditions,
    )
    np.testing.assert_allclose(x, coeffs.x0, atol=1e-12)
    y = tlb.numpy_affine_forward(coeffs, x)
    np.testing.assert_allclose(y, coeffs.y0, atol=1e-12)
    y0_only = tlb.numpy_affine_forward(coeffs, coeffs.x0)
    np.testing.assert_allclose(y0_only, coeffs.y0, atol=1e-15)


def test_cdu_manual_loop_l0_with_nested_ref_flatten():
    """Always-on L0: pack_driver + manual BASE/DELTA loop (nested-aware)."""
    model = build_cdu_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    feed = dict(model.reference_feed)
    feed["api"] = 28.0
    cond = {
        "flash_zone_temp_f": 690.0,
        "cut_points_f": {
            "naphtha_ep": 380.0,
            "distillate_ep": 670.0,
            "gasoil_ep": 1050.0,
        },
    }
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_vec = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    y_dict = tlb.y_raw_dict(coeffs, y_vec)

    from pims_admm_llm.models.base_delta import merge_process_conditions

    feed_m = dict(feed)
    cond_m = merge_process_conditions("CDU", cond)
    flat = dict(feed_m)
    for k, v in cond_m.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    flat[f"{k}.{kk}"] = float(vv)
    ref_flat = tlb._ref_flat_from_model(model)
    for p in model.products:
        base = float(model.base_yields.get(p, 0.0))
        dy = 0.0
        for drv, coef in (model.deltas.get(p) or {}).items():
            x0 = float(ref_flat.get(drv, 0.0))
            xv = float(flat.get(drv, x0))
            dy += float(coef) * (xv - x0)
        assert abs(y_dict[p] - max(0.0, base + dy)) < 1e-12


def test_fcc_coker_x0_still_flat_only_after_nested_fix():
    """FCC/Coker have no nested drivers; x0 must remain finite non-zero process refs."""
    for builder, unit in (
        (build_fcc_base_delta, "FCC"),
        (build_coker_base_delta, "COKER"),
    ):
        model = builder()
        coeffs = tlb.affine_coeffs_from_base_delta(model)
        assert coeffs.unit == unit
        assert not any("." in d for d in coeffs.drivers)
        assert np.all(np.isfinite(coeffs.x0))
        # pack@ref still ≡ x0
        x = tlb.pack_driver_vector(
            coeffs,
            feed=model.reference_feed,
            conditions=model.reference_conditions,
        )
        np.testing.assert_allclose(x, coeffs.x0, atol=1e-12)
        y = tlb.numpy_affine_forward(coeffs, coeffs.x0)
        np.testing.assert_allclose(y, coeffs.y0, atol=1e-15)


# ---------------------------------------------------------------------------
# Always-on: postprocess extract + L1 + L_div
# ---------------------------------------------------------------------------


def test_postprocess_cdu_yields_identity_and_evaluate_freeze():
    """Helper ≡ evaluate at ref + modes; pure copy (does not mutate input)."""
    model = build_cdu_base_delta()
    y_base = {p: float(model.base_yields[p]) for p in model.products}
    y_in = dict(y_base)
    y_pp = postprocess_cdu_yields(y_in, products=model.products)
    assert y_in == y_base  # pure
    y_eval = model.evaluate(clamp_products=True)
    for p in model.products:
        assert abs(y_pp[p] - y_eval[p]) < 1e-15, (p, y_pp[p], y_eval[p])

    for m in process_modes_cdu(model):
        cond = dict(m.get("conditions") or {})
        feed = dict(model.reference_feed)
        y_eval_m = model.evaluate(feed, cond, clamp_products=True)
        coeffs = tlb.affine_coeffs_from_base_delta(model)
        x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
        y_raw = tlb.y_raw_dict(
            coeffs, tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        )
        y_full = postprocess_cdu_yields(y_raw, products=model.products)
        for p in model.products:
            assert abs(y_full[p] - y_eval_m[p]) < 1e-12, (m.get("id"), p)


def test_l1_affine_plus_postprocess_matches_evaluate_cdu():
    """L1: affine + apply_cdu_postprocess ≡ evaluate (ref + offset + modes)."""
    model = build_cdu_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    cases = [
        (None, None),
        ({"api": 28.0, "sulfur_wt": 1.2}, {"flash_zone_temp_f": 690.0}),
        (
            {"api": 32.0},
            {
                "overflash_frac": 0.03,
                "cut_points_f": {
                    "naphtha_ep": 400.0,
                    "distillate_ep": 710.0,
                    "gasoil_ep": 1040.0,
                },
            },
        ),
    ]
    for feed, cond in cases:
        feed_use = dict(
            model.reference_feed if feed is None else {**model.reference_feed, **feed}
        )
        x = tlb.pack_driver_vector(coeffs, feed=feed_use, conditions=cond)
        y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        y_full = tlb.apply_cdu_postprocess(y_raw, products=coeffs.products)
        y_eval = model.evaluate(feed_use, cond, clamp_products=True)
        for p in coeffs.products:
            assert abs(y_full[p] - y_eval[p]) < 1e-9, (p, y_full[p], y_eval[p])

    for m in process_modes_cdu(model):
        feed = dict(model.reference_feed)
        cond = dict(m.get("conditions") or {})
        x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
        y_raw = tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
        y_full = tlb.apply_cdu_postprocess(y_raw, products=coeffs.products)
        y_eval = model.evaluate(feed, cond, clamp_products=True)
        for p in coeffs.products:
            assert abs(y_full[p] - y_eval[p]) < 1e-9, (m.get("id"), p)


def test_l_div_renorm_policy_cdu():
    """L_div honesty: document measured renorm behavior for CDU.

    Measured at HEAD (after nested-x0 fix):
    - Base liquids already sum to 1.0; every liquid-driver column of D sums ≈0
      across liquid products → raw affine keeps liquid sum ≈1 at typical points.
    - postprocess renorm is therefore **identity** at reference and at ordinary
      process_modes / feed offsets (unlike Coker, where renorm always engages).
    - Offgas clamp is identity when offgas stays in [0.005, 0.03].
    - L_div is still a product honesty gate: raw must equal evaluate when renorm
      is identity; if a future delta change makes liquid sum drift, renorm engages
      and L1 still holds via apply_cdu_postprocess.
    """
    model = build_cdu_base_delta()
    coeffs = tlb.affine_coeffs_from_base_delta(model)
    liquids = [p for p in coeffs.products if p != "cdu_offgas"]

    # Process / cut / API liquid columns are mass-conserving (sum≈0); feed quality
    # columns (sulfur, ccr) intentionally shift liquid mass slightly.
    for j, drv in enumerate(coeffs.drivers):
        col_sum = sum(
            float(coeffs.D[i, j])
            for i, p in enumerate(coeffs.products)
            if p != "cdu_offgas"
        )
        if drv in ("sulfur_wt", "ccr_wt"):
            # small non-zero liquid mass shift is intentional planning policy
            assert abs(col_sum) < 0.01, (drv, col_sum)
        else:
            assert abs(col_sum) < 1e-12, (drv, col_sum)

    # At reference: raw ≡ evaluate (identity renorm + offgas clamp)
    y_raw = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, coeffs.x0, clamp_products=True)
    )
    y_eval = model.evaluate(clamp_products=True)
    max_gap_ref = max(abs(y_raw[p] - y_eval[p]) for p in coeffs.products)
    assert max_gap_ref < 1e-12, f"expected identity-at-ref, max_gap={max_gap_ref}"
    assert abs(sum(y_raw[p] for p in liquids) - 1.0) < 1e-12

    # Mass-conserving off-ref (api + cuts only): raw ≡ evaluate
    feed = dict(model.reference_feed)
    feed["api"] = 35.0
    cond = {
        "flash_zone_temp_f": 710.0,
        "cut_points_f": {
            "naphtha_ep": 420.0,
            "distillate_ep": 720.0,
            "gasoil_ep": 1100.0,
        },
    }
    x = tlb.pack_driver_vector(coeffs, feed=feed, conditions=cond)
    y_raw2 = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, x, clamp_products=True)
    )
    y_eval2 = model.evaluate(feed, cond, clamp_products=True)
    max_gap2 = max(abs(y_raw2[p] - y_eval2[p]) for p in coeffs.products)
    assert max_gap2 < 1e-9, f"expected identity under mass-conserving drivers, gap={max_gap2}"

    for m in process_modes_cdu(model):
        feed_m = dict(model.reference_feed)
        cond_m = dict(m.get("conditions") or {})
        xm = tlb.pack_driver_vector(coeffs, feed=feed_m, conditions=cond_m)
        yr = tlb.y_raw_dict(
            coeffs, tlb.numpy_affine_forward(coeffs, xm, clamp_products=True)
        )
        ye = model.evaluate(feed_m, cond_m, clamp_products=True)
        gap = max(abs(yr[p] - ye[p]) for p in coeffs.products)
        assert gap < 1e-9, (m.get("id"), gap)

    # Quality offset (sulfur/ccr): liquid D sum ≠ 0 → renorm engages → raw ≠ evaluate
    feed_q = dict(model.reference_feed)
    feed_q["sulfur_wt"] = 2.5
    feed_q["ccr_wt"] = 4.0
    xq = tlb.pack_driver_vector(coeffs, feed=feed_q, conditions=None)
    y_raw_q = tlb.y_raw_dict(
        coeffs, tlb.numpy_affine_forward(coeffs, xq, clamp_products=True)
    )
    y_eval_q = model.evaluate(feed_q, None, clamp_products=True)
    raw_liq = sum(y_raw_q[p] for p in liquids)
    max_gap_q = max(abs(y_raw_q[p] - y_eval_q[p]) for p in coeffs.products)
    assert abs(raw_liq - 1.0) > 1e-6, f"expected liquid sum drift, sum={raw_liq}"
    assert max_gap_q > 1e-6, f"expected renorm L_div under quality offset, gap={max_gap_q}"
    # L1 still holds after postprocess
    y_full_q = tlb.apply_cdu_postprocess(y_raw_q, products=list(coeffs.products))
    for p in coeffs.products:
        assert abs(y_full_q[p] - y_eval_q[p]) < 1e-9

    # Synthetic L_div: force renorm by breaking liquid sum, then postprocess ≠ raw
    y_broken = dict(y_raw)
    y_broken["cdu_resid"] = float(y_broken["cdu_resid"]) + 0.05
    y_pp = postprocess_cdu_yields(y_broken, products=list(coeffs.products))
    assert abs(sum(y_pp[p] for p in liquids) - 1.0) < 1e-12
    assert max(abs(y_pp[p] - y_broken[p]) for p in liquids) > 1e-3


def test_renorm_policy_invariants_cdu():
    """Freeze offgas clamp bounds + liquid sum ≈ 1 after postprocess."""
    model = build_cdu_base_delta()
    y_base = {p: float(model.base_yields[p]) for p in model.products}
    y_pp = postprocess_cdu_yields(y_base, products=model.products)
    off = y_pp["cdu_offgas"]
    assert 0.005 - 1e-12 <= off <= 0.03 + 1e-12
    liquids = [p for p in model.products if p != "cdu_offgas"]
    s = sum(y_pp[p] for p in liquids)
    assert abs(s - 1.0) < 1e-12
    # base liquids already sum to 1
    assert abs(sum(y_base[p] for p in liquids) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Always-on: honesty / anti-fake-matrix
# ---------------------------------------------------------------------------


def test_honesty_metadata_includes_cdu_bans_duals():
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    units = meta.get("units") or list(tlb.UNITS)
    assert "FCC" in units and "COKER" in units and "CDU" in units
    note = meta.get("note") or ""
    assert "TECH+A" in note or "not a PIMS" in note or "CDU" in note


def test_no_excel_cdu_matrix_matches_affine_helper():
    """Ban inventing excel_cdu_matrix_matches_affine (Submodel_CDU is TECH+A)."""
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    assert "excel_cdu_matrix_matches_affine" not in tlb.__all__


def test_submodel_cdu_is_tech_a_not_pims_matrix():
    """Structure gate: Submodel_CDU tables are TECH+A, not fcc_pims_matrix-style MB_*."""
    from pims_admm_llm.models.excel_pipeline import base_delta_unit_submodel_tables

    tables = base_delta_unit_submodel_tables()
    # FCC/Coker have pims matrices; CDU must not claim the same shape
    assert "fcc_pims_matrix" in tables
    assert "coker_pims_matrix" in tables
    assert "cdu_pims_matrix" not in tables


# ---------------------------------------------------------------------------
# Optional TF path (skip if TF absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_cdu_forward_matches_numpy_l0():
    block = tlb.tf_linear_cdu()
    coeffs = block.coeffs
    assert coeffs.unit == "CDU"
    x = coeffs.x0 + np.array(
        [1.0, 0.1, 0.2, 5.0, 0.01, 10.0, 10.0, 10.0], dtype=np.float64
    )
    y_np = tlb.numpy_affine_forward(coeffs, x, clamp_products=False)
    y_tf = np.asarray(block.forward(x, clamp_products=False, as_dict=False), dtype=np.float64)
    np.testing.assert_allclose(y_tf, y_np, atol=1e-10)
    meta = block.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["solver"] is False
    assert meta["on_excel_case1_path"] is False
    assert meta["unit"] == "CDU"


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_cdu_l1_matches_evaluate():
    block = tlb.tf_linear_cdu()
    model = build_cdu_base_delta()
    feed = dict(model.reference_feed)
    feed["api"] = 29.0
    cond = {
        "flash_zone_temp_f": 685.0,
        "cut_points_f": {
            "naphtha_ep": 400.0,
            "distillate_ep": 700.0,
            "gasoil_ep": 1030.0,
        },
    }
    x = tlb.pack_driver_vector(block.coeffs, feed=feed, conditions=cond)
    y_raw = block.forward(x, clamp_products=True)
    y_full = tlb.apply_cdu_postprocess(y_raw, products=block.coeffs.products)
    y_eval = model.evaluate(feed, cond, clamp_products=True)
    for p in block.coeffs.products:
        assert abs(y_full[p] - y_eval[p]) < 1e-9


@pytest.mark.skipif(tlb.tf_available(), reason="only when TF missing")
def test_tf_linear_cdu_raises_when_tf_missing():
    with pytest.raises(ImportError):
        tlb.tf_linear_cdu()
