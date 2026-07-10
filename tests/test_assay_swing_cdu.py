"""Assay import → cut-point CDU handles: mass balance + property blend."""

from __future__ import annotations

from pathlib import Path

from pims_admm_llm.models.assay_swing import (
    allocate_cut_by_cut_points,
    build_heart_swing_library,
    cdu_cut_point_modes,
    cdu_yields_and_props_from_assay,
    import_crude_from_assays_package,
    import_detailed_assay_json,
    list_importable_assays,
    normalize_cut_points,
    solve_cdu_from_cut_points,
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


def test_cut_points_are_operational_handles():
    """Raising naphtha EP must increase naphtha yield (process handle)."""
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    light = solve_cdu_from_cut_points(
        assay,
        {"naphtha_ep_c": 220.0, "distillate_ep_c": 370.0, "gasoil_ep_c": 550.0},
        charge_kbd=100.0,
    )
    heavy = solve_cdu_from_cut_points(
        assay,
        {"naphtha_ep_c": 175.0, "distillate_ep_c": 370.0, "gasoil_ep_c": 550.0},
        charge_kbd=100.0,
    )
    assert light.status == "Optimal" and heavy.status == "Optimal"
    assert light.mass_balance["ok"] and heavy.mass_balance["ok"]
    assert light.product_yields_vol["cdu_naphtha"] > heavy.product_yields_vol["cdu_naphtha"]
    # meta exposes handles
    assert light.meta["operational_handles"] == [
        "naphtha_ep_c",
        "distillate_ep_c",
        "gasoil_ep_c",
    ]
    assert light.cut_points_c["naphtha_ep_c"] == 220.0


def test_gasoil_ep_handle_moves_resid():
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    deep = solve_cdu_from_cut_points(
        assay,
        {"naphtha_ep_c": 200.0, "distillate_ep_c": 370.0, "gasoil_ep_c": 565.0},
        charge_kbd=100.0,
    )
    shallow = solve_cdu_from_cut_points(
        assay,
        {"naphtha_ep_c": 200.0, "distillate_ep_c": 370.0, "gasoil_ep_c": 520.0},
        charge_kbd=100.0,
    )
    # Deeper gasoil cut → more gasoil, less resid
    assert deep.product_yields_vol["cdu_gasoil"] > shallow.product_yields_vol["cdu_gasoil"]
    assert deep.product_yields_vol["cdu_resid"] < shallow.product_yields_vol["cdu_resid"]


def test_cdu_swing_default_is_cut_point_mode():
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    res = solve_cdu_swing_cuts(assay, charge_kbd=100.0)  # default mode=cut_point
    assert res.status == "Optimal"
    assert res.mass_balance["ok"] is True, res.mass_balance
    assert res.mass_balance.get("driver") == "cut_points"
    assert res.product_yields_vol["cdu_resid"] > 0.25
    assert (
        res.product_properties["cdu_resid"]["sulfur_wt"]
        > res.product_properties["cdu_naphtha"]["sulfur_wt"]
    )


def test_cut_point_modes_catalog():
    modes = cdu_cut_point_modes()
    assert {m["id"] for m in modes} == {"cuts_light", "cuts_mid", "cuts_heavy"}
    assay = import_crude_from_assays_package("Cold_Lake_Blend")
    results = {
        m["id"]: solve_cdu_from_cut_points(assay, m["cut_points_c"], charge_kbd=50)
        for m in modes
    }
    assert results["cuts_light"].product_yields_vol["cdu_naphtha"] >= results["cuts_heavy"].product_yields_vol["cdu_naphtha"]


def test_allocate_straddle_is_linear():
    from pims_admm_llm.models.assay_swing import AssayCut

    cut = AssayCut(id="s", tbp_start_c=180, tbp_end_c=220, yield_vol=0.1, api=50.0)
    # naphtha EP mid of cut → half/half naph vs dist
    fr = allocate_cut_by_cut_points(
        cut, {"naphtha_ep_c": 200.0, "distillate_ep_c": 370.0, "gasoil_ep_c": 550.0}
    )
    assert abs(fr["cdu_naphtha"] - 0.5) < 1e-9
    assert abs(fr["cdu_distillate"] - 0.5) < 1e-9


def test_convenience_and_list():
    names = list_importable_assays()
    assert any("Cold" in n for n in names)
    d = cdu_yields_and_props_from_assay("Cold_Lake_Blend", charge_kbd=80.0)
    assert d["status"] == "Optimal"
    assert d["mass_balance"]["ok"] is True
    assert "naphtha_ep_c" in d["cut_points_c"]


def test_normalize_enforces_order():
    cp = normalize_cut_points(
        {"naphtha_ep_c": 400.0, "distillate_ep_c": 300.0, "gasoil_ep_c": 250.0}
    )
    assert cp["naphtha_ep_c"] < cp["distillate_ep_c"] < cp["gasoil_ep_c"]
