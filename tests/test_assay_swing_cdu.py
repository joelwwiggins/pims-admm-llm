"""Assay import → heart/swing CDU: mass balance + property blend."""

from __future__ import annotations

from pathlib import Path

from pims_admm_llm.models.assay_swing import (
    build_heart_swing_library,
    cdu_yields_and_props_from_assay,
    import_crude_from_assays_package,
    import_detailed_assay_json,
    list_importable_assays,
    solve_cdu_swing_cuts,
    synthesize_from_tbp_cut_vol,
)


ROOT = Path(__file__).resolve().parents[1]
CLK = ROOT / "data" / "assays" / "cold_lake_blend_clkbl23b.json"


def test_import_cold_lake_detailed():
    assay = import_detailed_assay_json(CLK)
    assert assay.name.lower().startswith("cold")
    assert abs(assay.total_vol() - 1.0) < 1e-6
    assert len(assay.cuts) >= 10
    assert assay.whole_crude["api"] == 19.5


def test_import_by_name_cold_lake():
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    assert abs(assay.total_vol() - 1.0) < 1e-6
    hearts, swings = build_heart_swing_library(assay)
    assert len(hearts) >= 4
    assert len(swings) >= 1


def test_synthesize_from_tbp():
    crude = {
        "name": "Toy",
        "api": 30.0,
        "sulfur_wt": 1.0,
        "ccr_wt": 2.0,
        "tbp_cut_vol": {
            "naphtha_ibp_350f": 0.2,
            "distillate_350_650f": 0.25,
            "gasoil_650_1050f": 0.3,
            "resid_1050f_plus": 0.25,
        },
    }
    assay = synthesize_from_tbp_cut_vol(crude)
    assert abs(assay.total_vol() - 1.0) < 1e-6
    hearts, swings = build_heart_swing_library(assay)
    assert hearts and swings


def test_cdu_swing_mass_balance_cold_lake():
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    res = solve_cdu_swing_cuts(assay, charge_kbd=100.0, optimize=True)
    assert res.status == "Optimal"
    assert res.mass_balance["ok"] is True, res.mass_balance
    # products sum to charge * coverage
    s = sum(res.product_rates_kbd.values())
    assert abs(s - 100.0 * assay.total_vol()) < 1e-3
    # heavy crude → resid + gasoil dominate
    assert res.product_yields_vol["cdu_resid"] > 0.25
    assert res.product_yields_vol["cdu_gasoil"] > 0.20
    # resid should be sourer than naphtha
    assert res.product_properties["cdu_resid"]["sulfur_wt"] > res.product_properties["cdu_naphtha"]["sulfur_wt"]
    assert res.product_properties["cdu_resid"]["ccr_wt"] >= res.product_properties["cdu_gasoil"]["ccr_wt"]


def test_swing_allocation_respects_fixed_frac():
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    hearts, swings = build_heart_swing_library(assay)
    if not swings:
        return
    sid = swings[0].id
    res = solve_cdu_swing_cuts(
        assay,
        charge_kbd=50.0,
        optimize=False,
        swing_light_frac={sid: 1.0},
    )
    assert res.status == "Optimal"
    assert res.swing_allocations[sid]["light_frac"] > 0.99


def test_convenience_and_list():
    names = list_importable_assays()
    assert "Cold_Lake_Blend" in names or any("Cold" in n for n in names)
    d = cdu_yields_and_props_from_assay("Cold_Lake_Blend", charge_kbd=80.0)
    assert d["status"] == "Optimal"
    assert d["mass_balance"]["ok"] is True
