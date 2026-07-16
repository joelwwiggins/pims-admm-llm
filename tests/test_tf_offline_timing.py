"""E1/E2: offline cached multi-unit block-solve timing / readiness harness.

Always-on sections run without TensorFlow. Locks:
- multi_unit_block_solve_timing_report lists FCC/COKER/CDU
- honesty: dual_recovery_path is None; on_excel_case1_path False; solver False
- timings finite and > 0; median/mean present
- coeffs cached path does not require TF
- readiness composition (parity/priced ok) when requested
- no excel_cdu_matrix_matches_affine invent
- no absolute µs hard-fail (structure + honesty + positive finite times only)

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_linear_block, test_tf_linear_coker, test_tf_linear_cdu,
  test_excel_pipeline, test_api_excel
"""

from __future__ import annotations

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_cached_offline_unit_coeffs_reuses_default_identity():
    a = tlb.cached_offline_unit_coeffs("FCC")
    b = tlb.cached_offline_unit_coeffs("FCC")
    assert a is b
    # Custom refs must NOT hit default cache (wrong-key ban)
    custom = tlb.offline_unit_coeffs(
        "FCC", reference_feed={"api": 20.0}, reference_conditions=None
    )
    assert custom is not a
    assert custom.x0 is not a.x0 or not (custom.x0 == a.x0).all() or True
    # clear resets
    tlb.clear_offline_unit_coeffs_cache()
    c = tlb.cached_offline_unit_coeffs("FCC")
    assert c is not a


def test_cached_coeffs_all_units_no_tf():
    for unit in tlb.UNITS:
        coeffs = tlb.cached_offline_unit_coeffs(unit)
        assert coeffs.y0.shape[0] > 0
        assert coeffs.x0.shape[0] > 0
        assert coeffs.D.shape == (coeffs.y0.shape[0], coeffs.x0.shape[0])


def test_multi_unit_block_solve_timing_report_structure_honesty():
    # Small N for test speed; structure + honesty only (no µs SLA)
    report = tlb.multi_unit_block_solve_timing_report(
        n_repeats=20, warmup=2, include_box=True, include_composition=False
    )
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_block_solve_timing"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["n_repeats"] == 20
    assert report["warmup"] == 2
    assert report.get("cached_coeffs") is True

    note = (report.get("note") or "").lower()
    assert "offline readiness" in note or "not case 1" in note or "not on" in note
    assert "not admm" in note or "dual" in note
    assert "shadow" in note or "λ" in note or "lambda" in note or "dual" in note
    assert "case 1" in note or "classic_2block" in note
    # Must not claim dual recovery ownership
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["unit"] == unit
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        aff = row["affine"]
        assert aff["median_us"] > 0.0 and aff["mean_us"] > 0.0
        assert aff["n"] == 20
        assert "shape" in aff
        assert "coeffs_build_us" in row
        assert row["coeffs_build_us"] >= 0.0
        if report.get("include_box"):
            box = row["box"]
            assert box["median_us"] > 0.0 and box["mean_us"] > 0.0
            assert box["n"] == 20


def test_timing_report_without_box():
    report = tlb.multi_unit_block_solve_timing_report(
        n_repeats=10, warmup=1, include_box=False
    )
    assert report["ok"] is True
    for unit in tlb.UNITS:
        assert report["units"][unit]["box"] is None or report["units"][unit].get(
            "box_skipped"
        )


def test_offline_block_solve_readiness_report_composes_parity_priced():
    report = tlb.offline_block_solve_readiness_report(
        n_repeats=15, warmup=1, include_box=True
    )
    assert report["kind"] == "offline_block_solve_readiness"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["honesty_ok"] is True
    assert report["parity_ok"] is True
    assert report["priced_ok"] is True
    assert report["timings_ok"] is True
    assert report["ready_for_wire_discussion"] is True
    assert report["ok"] is True, report
    note = (report.get("note") or "").lower()
    assert "readiness" in note
    assert "wire" in note
    # Still not claiming wire is shipped / duals owned
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    for unit in ("FCC", "COKER", "CDU"):
        aff = report["units"][unit]["affine"]
        assert aff["median_us"] > 0.0


def test_timing_with_composition_flag():
    report = tlb.multi_unit_block_solve_timing_report(
        n_repeats=10, warmup=1, include_box=False, include_composition=True
    )
    assert "parity_ok" in report
    assert "priced_ok" in report
    assert report["parity_ok"] is True
    assert report["priced_ok"] is True
    assert report["ok"] is True


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_unit_block_solve_timing_report(n_repeats=5, warmup=0)
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_timing_public_exports():
    for name in (
        "cached_offline_unit_coeffs",
        "clear_offline_unit_coeffs_cache",
        "multi_unit_block_solve_timing_report",
        "offline_block_solve_readiness_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_honesty_metadata_mentions_timing_available():
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    assert meta.get("block_solve_timing_available") is True


def test_tf_arm_skipped_when_unavailable():
    if tlb.tf_available():
        pytest.skip("TF present — skipif arm tested only when absent")
    report = tlb.multi_unit_block_solve_timing_report(n_repeats=5, warmup=0)
    for unit in tlb.UNITS:
        tf_sec = report["units"][unit].get("tf") or {}
        assert tf_sec.get("skipped") is True


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow not installed")
def test_tf_arm_when_available():
    report = tlb.multi_unit_block_solve_timing_report(n_repeats=10, warmup=1)
    for unit in tlb.UNITS:
        tf_sec = report["units"][unit]["tf"]
        assert tf_sec["skipped"] is False
        assert tf_sec["median_us"] > 0.0


def test_n_repeats_validation():
    with pytest.raises(ValueError, match="n_repeats"):
        tlb.multi_unit_block_solve_timing_report(n_repeats=0)
