"""W5: mono LP + ADMM solve + obj gap + mass balance tests (crude→cat→blender)."""

from __future__ import annotations

from pims_admm_llm.models.crude_cat_blender import (
    H2_KSCF_PER_BBL_FCC,
    compare_mono_admm,
    solve_crude_cat_blender,
    solve_crude_cat_blender_admm,
)


def test_mono_wti_optimal_and_mass_balance():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=100.0)
    assert r.status == "Optimal"
    assert r.path == "mono"
    assert r.mass_balance["ok"] is True
    # every named check must pass (no hard-coded ok)
    for name, chk in r.mass_balance["checks"].items():
        assert chk["ok"] is True, f"{name} failed gap={chk['gap']}"
        assert chk["gap"] >= 0.0
    required = {
        "cdu_vs_crude",
        "naph_split",
        "go_split",
        "tank_naph",
        "tank_go",
        "fcc_feed",
        "fcc_products_vs_feed",
        "naph_use",
        "gasoline_blend",
        "h2_vs_fcc",
    }
    assert required.issubset(r.mass_balance["checks"].keys())
    assert r.crude_kbd > 50.0
    assert r.products["gasoline"] > 1.0
    assert r.purchases["h2_kscf"] > 0.0
    assert r.products["fuel_gas_mmbtu"] > 0.0
    # tanks 7-day capacity
    assert r.tank["tank_go"]["cap"] == 7.0 * 100.0
    assert r.tank["tank_naph"]["end"] >= r.tank["tank_naph"]["start"] - 1e-6
    assert r.quality["ron_ok"] and r.quality["s_ok"]


def test_mono_quality_specs():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=100.0)
    assert r.quality["gasoline_ron"] + 1e-5 >= r.quality["ron_min"]
    assert r.quality["gasoline_sulfur_wt"] <= r.quality["s_max"] + 1e-5
    # blend volume closes
    g = r.products["gasoline"]
    blend = (
        r.streams["bl_cdu_naph"]
        + r.streams["bl_fcc_naph"]
        + r.purchases["buy_naphtha"]
        + r.purchases["buy_alkylate"]
    )
    assert abs(g - blend) < 1e-3


def test_tanks_bypass_or_through():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=80.0)
    go = r.tank["tank_go"]
    assert abs(go["bypass"] + go["from_tank"] - r.streams["fcc_feed"]) < 1e-4
    naph = r.tank["tank_naph"]
    assert abs(naph["to_tank"] + naph["bypass"] - r.streams["cdu_naphtha"]) < 1e-4
    # inventory identity
    for t in (go, naph):
        assert abs(t["start"] + t["to_tank"] - t["from_tank"] - t["end"]) < 1e-4


def test_h2_and_fuel_gas_btu():
    r = solve_crude_cat_blender(crude_name="WTI", max_crude_kbd=100.0)
    assert abs(r.purchases["h2_kscf"] - H2_KSCF_PER_BBL_FCC * r.streams["fcc_feed"]) < 1e-6
    assert r.utilities["h2_kscf"] == r.purchases["h2_kscf"]
    assert r.utilities["fuel_gas_mmbtu"] == r.products["fuel_gas_mmbtu"]
    assert r.utilities["fuel_gas_revenue"] > 0.0
    assert r.utilities["h2_cost"] > 0.0


def test_mono_admm_obj_gap_and_mb():
    c = compare_mono_admm(crude_name="WTI", max_crude_kbd=100.0)
    assert c["mass_balance_ok"] is True
    assert c["quality_ok"] is True
    # MVP acceptance: gap within ~1–2%
    assert c["obj_gap_abs"] >= 0.0
    assert c["obj_gap_rel"] < 0.02
    assert c["VERDICT"]["mono_obj"] > 0
    assert c["VERDICT"]["admm_obj"] > 0
    assert abs(c["VERDICT"]["gap_rel"] - c["obj_gap_rel"]) < 1e-12
    assert c["VERDICT"]["mb_ok"] is True
    # path honesty labels
    assert c["mono"]["path"] == "mono"
    assert str(c["admm"]["path"]).startswith("admm")
    assert c["admm"]["meta"]["dual_recovery_path"] == "admm-blender-consensus+mono-recovery"
    assert "obj_gap_rel" in c["admm"]["meta"]["admm"]
    assert "residual_norm" in c["admm"]["meta"]["admm"]
    assert c["admm"]["meta"]["admm"]["iters_run"] >= 1


def test_admm_solve_mass_balance():
    a = solve_crude_cat_blender_admm(crude_name="WTI", max_crude_kbd=50.0, rho=1.2, max_iters=10)
    assert a.status == "Optimal"
    assert a.path.startswith("admm")
    assert a.mass_balance["ok"] is True
    for name, chk in a.mass_balance["checks"].items():
        assert chk["ok"] is True, f"admm {name} gap={chk['gap']}"
    assert a.meta["admm"]["rho"] == 1.2
    assert a.meta["admm"]["max_iters"] == 10
    assert a.meta["admm"]["obj_gap_rel"] < 0.02
    assert a.meta["dual_recovery_path"] == "admm-blender-consensus+mono-recovery"
    assert "lambda_bl_cdu_naph" in a.duals
    assert "residual_norm" in a.duals


def test_cold_lake_feasible_with_purchases():
    r = solve_crude_cat_blender(crude_name="Cold_Lake_Blend", max_crude_kbd=80.0)
    assert r.status == "Optimal"
    assert r.mass_balance["ok"]
    for chk in r.mass_balance["checks"].values():
        assert chk["ok"] is True
    # heavy crude still makes gasoline via FCC + purchases
    assert r.products["gasoline"] > 0.0


def test_admm_path_label():
    a = solve_crude_cat_blender_admm(crude_name="WTI", max_crude_kbd=50.0)
    assert a.path.startswith("admm")
    assert "admm" in a.meta


def test_no_purchases_still_feasible_or_explicit():
    """Allow purchases off: still Optimal if SR+FCC naphtha meet specs alone."""
    r = solve_crude_cat_blender(
        crude_name="WTI",
        max_crude_kbd=100.0,
        allow_purchases=False,
        gas_ron_min=80.0,  # relax RON so pure FCC+SR naph can clear without alkylate
        gas_s_max=0.05,
    )
    assert r.status == "Optimal"
    assert r.purchases["buy_naphtha"] == 0.0
    assert r.purchases["buy_alkylate"] == 0.0
    assert r.mass_balance["ok"] is True
