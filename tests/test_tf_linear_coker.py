"""E7/E8: Excel Submodel_Coker MB_* BASE/D_* ↔ affine package consistency.

Always-on (no TensorFlow). Scope is product mass-balance rows only, not
teaching rows (E_BASE_REF, FREE). Pattern-mirrors FCC E10 gate.
"""

from __future__ import annotations

import numpy as np

from pims_admm_llm.models.base_delta import build_coker_base_delta
from pims_admm_llm.models import tf_linear_blocks as tlb


def test_coker_affine_shape_5x6_always_on():
    """L0 foundation for E7/E8: COKER is 5 products × 6 drivers."""
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
    # Teaching / free rows may exist; excel match must still be ok
    assert any(r.startswith("MB_") for r in rows)
    report = tlb.excel_coker_matrix_matches_affine(atol=1e-12)
    assert report["ok"], report.get("mismatches")


def test_honesty_metadata_still_bans_duals_after_coker_helper():
    """Adding coker excel match must not invent dual recovery claims."""
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert tlb.DUAL_RECOVERY_PATH is None
