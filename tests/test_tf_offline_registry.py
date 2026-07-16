"""E1/E2: multi-unit offline registry + wiring-readiness parity harness.

Always-on sections run without TensorFlow. Optional TF multi-unit parity is
skipif. Locks:
- registry units == FCC, COKER, CDU
- offline_units_status dual-ban + Case1-off
- multi_unit_parity_report aggregate ok without TF
- CDU excel_match is None (no excel_cdu_matrix_matches_affine)
- no TF dual recovery claims
"""

from __future__ import annotations

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


def test_offline_unit_registry_exact_units():
    reg = tlb.offline_unit_registry()
    names = tuple(d.unit for d in reg)
    assert names == ("FCC", "COKER", "CDU")
    assert names == tlb.UNITS
    assert len(reg) == 3


def test_registry_descriptor_fields_and_cdu_no_excel_match():
    reg = {d.unit: d for d in tlb.offline_unit_registry()}
    assert reg["FCC"].excel_match_name == "excel_fcc_matrix_matches_affine"
    assert reg["COKER"].excel_match_name == "excel_coker_matrix_matches_affine"
    assert reg["CDU"].excel_match_name is None
    assert reg["FCC"].n_products == 6 and reg["FCC"].n_drivers == 8
    assert reg["COKER"].n_products == 5 and reg["COKER"].n_drivers == 6
    assert reg["CDU"].n_products == 6 and reg["CDU"].n_drivers == 8
    assert "tf_linear_fcc" == reg["FCC"].factory_name
    assert "apply_coker_postprocess" == reg["COKER"].postprocess_name
    assert "build_cdu_base_delta" == reg["CDU"].builder_name
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")


def test_offline_unit_coeffs_always_on_no_tf():
    for unit in tlb.UNITS:
        coeffs = tlb.offline_unit_coeffs(unit)
        assert coeffs.unit == unit
        assert coeffs.y0.shape[0] == len(coeffs.products)
        assert coeffs.D.shape == (len(coeffs.products), len(coeffs.drivers))
        assert coeffs.x0.shape[0] == len(coeffs.drivers)


def test_offline_units_status_honesty_dual_and_case1_ban():
    st = tlb.offline_units_status()
    assert st["units"] == ["FCC", "COKER", "CDU"]
    assert st["solver"] is False
    assert st["dual_recovery_path"] is None
    assert st["on_excel_case1_path"] is False
    assert isinstance(st["tf_available"], bool)
    assert st["tf_available"] == tlb.tf_available()
    assert set(st["per_unit"].keys()) == {"FCC", "COKER", "CDU"}
    assert st["per_unit"]["CDU"]["excel_match"] is False
    assert st["per_unit"]["FCC"]["excel_match"] is True
    note = st["note"].lower()
    assert "classic_2block" in note or "not on" in note
    assert "dual" in note
    assert "none" in note or "primary" in note


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.offline_unit_coeffs("BLENDER")
    with pytest.raises(ValueError, match="Unknown offline unit"):
        tlb.build_offline_unit("REFORMER")


def test_multi_unit_parity_report_always_on_aggregate_ok():
    """Wiring readiness without TF: pack@ref + affine+postprocess ≡ evaluate."""
    report = tlb.multi_unit_parity_report(atol=1e-9)
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert row["solver"] is False
        assert row["dual_recovery_path"] is None
        assert row["on_excel_case1_path"] is False
        ch = row["checks"]
        assert ch["pack_ref_eq_x0"] is True
        assert ch["affine_postprocess_eq_evaluate_ref"] is True
        assert ch["affine_postprocess_eq_evaluate_offset"] is True
        # TF section must not force failure when TF absent
        if not tlb.tf_available():
            assert row["tf"]["skipped"] is True


def test_multi_unit_parity_not_a_solve_or_dual_claim():
    report = tlb.multi_unit_parity_report()
    note = (report.get("note") or "").lower()
    assert "not a solve" in note or "not admm" in note
    assert "dual" in note
    assert report["dual_recovery_path"] is None
    # No inventing CDU Excel MB_* matcher
    assert report["units"]["CDU"]["excel_match_name"] is None


def test_honesty_metadata_still_multi_unit_dual_ban():
    meta = tlb.honesty_metadata()
    assert meta["solver"] is False
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["units"] == ["FCC", "COKER", "CDU"]


def test_registry_public_exports():
    for name in (
        "offline_unit_registry",
        "offline_unit_coeffs",
        "build_offline_unit",
        "offline_units_status",
        "multi_unit_parity_report",
        "OfflineUnitDescriptor",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


@pytest.mark.skipif(not tlb.tf_available(), reason="TensorFlow optional; absent on Jetson")
def test_build_offline_unit_and_tf_multi_unit_parity():
    for unit in tlb.UNITS:
        block = tlb.build_offline_unit(unit)
        assert block.coeffs.unit == unit
        meta = block.honesty_metadata()
        assert meta["dual_recovery_path"] is None
        assert meta["solver"] is False
    report = tlb.multi_unit_parity_report(atol=1e-9)
    assert report["ok"] is True, report
    for unit in tlb.UNITS:
        tf_sec = report["units"][unit]["tf"]
        assert tf_sec["skipped"] is False
        assert tf_sec["ok"] is True, (unit, tf_sec)
