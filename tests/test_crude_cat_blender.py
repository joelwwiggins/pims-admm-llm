"""Crude→cat→blender case: tanks, specs, H2, fuel-gas BTU, mono/ADMM."""

from pims_admm_llm.models.crude_cat_blender import (
    compare_mono_admm,
    solve_crude_cat_blender,
    solve_crude_cat_blender_admm,
)


def test_mono_wti_mass_balance_and_quality():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=100.0)
    assert r.status == "Optimal"
    assert r.mass_balance["ok"] is True
    assert r.quality["ron_ok"] and r.quality["s_ok"]
    assert r.crude_kbd > 50.0
    assert r.products["gasoline"] > 1.0
    assert r.purchases["h2_kscf"] > 0.0
    assert r.products["fuel_gas_mmbtu"] > 0.0
    # tanks 7-day capacity
    assert r.tank["tank_go"]["cap"] == 7.0 * 100.0
    assert r.tank["tank_naph"]["end"] >= r.tank["tank_naph"]["start"] - 1e-6


def test_tanks_bypass_or_through():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=80.0)
    # either bypass or tank path used for gasoil to FCC
    go = r.tank["tank_go"]
    assert go["bypass"] + go["from_tank"] == r.streams["fcc_feed"] or abs(
        go["bypass"] + go["from_tank"] - r.streams["fcc_feed"]
    ) < 1e-4


def test_mono_admm_obj_close():
    c = compare_mono_admm(crude_name="WTI", max_crude_kbd=100.0)
    assert c["mass_balance_ok"]
    assert c["quality_ok"]
    assert c["obj_gap_rel"] < 0.02  # within 2%
    assert c["VERDICT"]["mono_obj"] > 0


def test_cold_lake_feasible_with_purchases():
    r = solve_crude_cat_blender(crude_name="Cold_Lake_Blend", max_crude_kbd=80.0)
    assert r.status == "Optimal"
    assert r.mass_balance["ok"]
    # heavy crude still makes gasoline via FCC + purchases
    assert r.products["gasoline"] > 0.0


def test_admm_path_label():
    a = solve_crude_cat_blender_admm(crude_name="WTI", max_crude_kbd=50.0)
    assert a.path.startswith("admm")
    assert "admm" in a.meta
