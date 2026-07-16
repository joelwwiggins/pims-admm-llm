"""E1/E2: offline multi-unit ADMM-style consensus residual harness.

Always-on sections run without TensorFlow. Locks:
- multi_unit_admm_residual_report lists FCC/COKER/CDU
- aggregate ok without TF
- dual_recovery_path is None; on_excel_case1_path is False; solver False
- synthetic λ/z/ρ (not Case 1 online λ / not recovered duals / not pure-ADMM dual recovery)
- residuals finite; L1 formula primary; y==z ⇒ penalty≈0
- Coker raw residual may ≠ full even when z is full-space
- no excel_cdu_matrix_matches_affine invent
- no brittle absolute residual magnitude SLAs

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_linear_block, test_tf_linear_coker,
  test_tf_linear_cdu, test_excel_pipeline, test_api_excel
"""

from __future__ import annotations

import math

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_multi_unit_admm_residual_report_always_on_aggregate_ok():
    report = tlb.multi_unit_admm_residual_report(rho=1.0, x_mode="offset")
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_admm_block_residual"
    assert report["kind"] == tlb.ADMM_RESIDUAL_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["price_source"] == tlb.PRICE_SOURCE
    assert report["not_a_solve"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["formula"] == tlb.ADMM_AUGMENTED_FORMULA_L1
    assert "lambda_dot_y" in report["formula"]
    assert "||y_full - z||_1" in report["formula"] or "l1" in report["formula"].lower()

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        assert row["not_a_solve"] is True
        assert math.isfinite(row["r_l1"])
        assert math.isfinite(row["r_l2"])
        assert math.isfinite(row["r_linf"])
        assert math.isfinite(row["augmented_local"])
        assert math.isfinite(row["lambda_dot_y"])
        assert math.isfinite(row["penalty"])
        assert set(row["consensus_residual"].keys()) == set(row["products"])
        assert row["residual_on"] == "postprocess"
        # L1 identity: augmented = lambda_dot_y - rho * r_l1
        expect_aug = row["lambda_dot_y"] - float(report["rho"]) * row["r_l1"]
        assert abs(row["augmented_local"] - expect_aug) <= 1e-9


def test_admm_residual_honesty_not_dual_not_case1():
    report = tlb.multi_unit_admm_residual_report()
    note = (report.get("note") or "").lower()
    assert "not a solve" in note or "not on" in note
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


def test_ref_mode_penalty_near_zero_when_z_is_default():
    """When x=ref and z=postprocess@ref, consensus residual ≈ 0 → penalty ≈ 0."""
    for unit in tlb.UNITS:
        row = tlb.admm_residual_for_unit(unit, rho=1.0, x_mode="ref")
        assert row["ok"] is True
        assert row["r_l1"] <= 1e-9, (unit, row["r_l1"])
        assert abs(row["penalty"]) <= 1e-9
        assert abs(row["augmented_local"] - row["lambda_dot_y"]) <= 1e-9


def test_coker_raw_vs_full_residual_gap_honesty():
    """Coker renorm: raw residual vs full-space z may differ even at reference."""
    row = tlb.admm_residual_for_unit("COKER", rho=1.0, x_mode="ref")
    assert row["ok"] is True
    # At ref with z=full, full residual ~0 but raw may not
    assert row["r_l1"] <= 1e-9
    assert row["raw_vs_full_residual_l1_gap"] > 1e-6, row
    assert "renorm" in (row.get("renorm_note") or "").lower() or row["r_raw_l1"] > 1e-6


def test_offset_mode_nontrivial_residual():
    report = tlb.multi_unit_admm_residual_report(rho=1.0, x_mode="offset")
    # At least one unit should show nonzero residual under mild offset
    any_nontrivial = any(
        report["units"][u]["r_l1"] > 1e-9 for u in report["unit_order"]
    )
    assert any_nontrivial, report


def test_box_step_mode_ok():
    report = tlb.multi_unit_admm_residual_report(rho=1.0, use_box_step=True, box_delta=0.5)
    assert report["ok"] is True
    assert report["x_mode"] == "box"
    for unit in tlb.UNITS:
        assert report["units"][unit]["ok"] is True
        assert report["units"][unit]["x_source"] == "local_box_step"


def test_rho_must_be_positive():
    with pytest.raises(ValueError, match="rho"):
        tlb.multi_unit_admm_residual_report(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        tlb.admm_residual_for_unit("FCC", rho=-1.0)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.admm_residual_for_unit("BLENDER")


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_unit_admm_residual_report()
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_admm_residual_public_exports():
    for name in (
        "ADMM_RESIDUAL_KIND",
        "ADMM_AUGMENTED_FORMULA_L1",
        "admm_residual_for_unit",
        "multi_unit_admm_residual_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_honesty_metadata_mentions_admm_residual():
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    assert meta.get("admm_residual_available") is True
    note = (meta.get("note") or "").lower()
    assert "admm" in note and "residual" in note


def test_readiness_compose_admm_residual_ok_additive():
    report = tlb.offline_block_solve_readiness_report(
        n_repeats=10, warmup=1, include_box=False, include_admm_residual=True
    )
    assert report["kind"] == "offline_block_solve_readiness"
    assert report["ready_for_wire_discussion"] is True
    assert report["admm_residual_ok"] is True
    # Additive: ready semantics still parity∧priced∧timings∧honesty
    assert report["parity_ok"] is True
    assert report["priced_ok"] is True
    assert report["timings_ok"] is True
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    note = (report.get("note") or "").lower()
    assert "admm_residual_ok" in note or "additive" in note


def test_readiness_can_skip_admm_residual_compose():
    report = tlb.offline_block_solve_readiness_report(
        n_repeats=5, warmup=0, include_box=False, include_admm_residual=False
    )
    assert report["ready_for_wire_discussion"] is True
    assert report["admm_residual_ok"] is None


def test_custom_z_override_creates_mismatch():
    """Synthetic z override away from ref yields must produce nonzero residual at ref."""
    coeffs = tlb.cached_offline_unit_coeffs("FCC")
    z_bad = {p: 0.0 for p in coeffs.products}
    row = tlb.admm_residual_for_unit("FCC", z=z_bad, x_mode="ref", rho=1.0)
    assert row["ok"] is True
    assert row["r_l1"] > 1e-6
    assert row["z_source"] == "caller_override"
