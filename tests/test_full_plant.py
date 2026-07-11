"""Wave3 full plant: superstructure arcs, flexible routing, quality pooling, dual recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from pims_admm_llm.models.assay_loader import (
    default_assays_path,
    load_assays_excel,
    load_assays_json,
    load_routing,
    write_template_excel,
)
from pims_admm_llm.models.full_plant import admm_price_directed_plant, build_yield_tables, solve_full_plant
from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks
from pims_admm_llm.models.properties import crude_to_props
from pims_admm_llm.models.yields import cdu_yields_from_assay, fcc_yields, coker_yields, reformer_yields


def test_assays_load_and_properties():
    assays = load_assays_json()
    assert len(assays["crudes"]) >= 3
    for c in assays["crudes"]:
        p = crude_to_props(c)
        assert p.api > 0
        y = cdu_yields_from_assay(p, c.get("tbp_cut_vol"))
        liquid = {k: v for k, v in y.items() if k != "cdu_offgas"}
        assert abs(sum(liquid.values()) - 1.0) < 1e-6
        assert set(liquid) == {"cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"}
        assert "cdu_offgas" in y and y["cdu_offgas"] > 0


def test_routing_superstructure_loads():
    r = load_routing()
    assert r.get("version") == "wave3-superstructure"
    arcs = r.get("arcs") or []
    assert len(arcs) >= 10
    by_id = {a["id"]: a for a in arcs if "id" in a}
    # swing destinations
    assert "go_to_fcc" in by_id and by_id["go_to_fcc"].get("decision") is True
    assert "go_to_diesel" in by_id
    assert "go_to_sell" in by_id
    assert "resid_to_coker" in by_id
    assert "resid_to_fo" in by_id
    # chemistry defaults
    assert by_id["fcc_naph_to_gas"].get("default_open") is not False
    assert by_id["fcc_naph_to_reformer"].get("default_open") is False
    assert by_id["coker_naph_to_reformer"].get("default_open") is False
    assert "cdu_naphtha_heavy" in {a["stream"] for a in arcs}
    assert "cdu_naphtha_light" in {a["stream"] for a in arcs}
    # backward-compat routes still present
    routes = {(x["from"], x["stream"], x["to"]) for x in r.get("routes") or []}
    assert ("CDU", "cdu_gasoil", "TANK_GASOIL") in routes
    assert ("TANK_GASOIL", "cdu_gasoil", "FCC") in routes


def test_property_yields_respond_to_feed():
    light = crude_to_props(
        {
            "name": "L",
            "api": 40,
            "ccr_wt": 1,
            "sulfur_wt": 0.2,
            "nitrogen_ppm": 500,
            "paraffins_vol": 0.4,
            "naphthenes_vol": 0.3,
            "aromatics_vol": 0.3,
        }
    )
    heavy = crude_to_props(
        {
            "name": "H",
            "api": 20,
            "ccr_wt": 12,
            "sulfur_wt": 3,
            "nitrogen_ppm": 3000,
            "paraffins_vol": 0.2,
            "naphthenes_vol": 0.3,
            "aromatics_vol": 0.5,
        }
    )
    yl = cdu_yields_from_assay(light)
    yh = cdu_yields_from_assay(heavy)
    assert yl["cdu_naphtha"] > yh["cdu_naphtha"]
    assert yh["cdu_resid"] > yl["cdu_resid"]
    from pims_admm_llm.models.yields import gasoil_props_from_crude, resid_props_from_crude

    fcc_l = fcc_yields(gasoil_props_from_crude(light))
    fcc_h = fcc_yields(gasoil_props_from_crude(heavy))
    assert fcc_l["fcc_naphtha"] >= fcc_h["fcc_naphtha"] - 1e-9
    cok_h = coker_yields(resid_props_from_crude(heavy))
    assert cok_h["coker_naphtha"] + cok_h["coker_gasoil"] < 0.85


def test_full_plant_mono_optimal():
    # Depress FO netback so resid→coker is optimal (multi-unit demo path).
    # Default FO=$68 makes light-slate resid→FO preferred (see test_resid_swing_exists).
    import copy

    assays = load_assays_json()
    assays = copy.deepcopy(assays)
    assays["products"]["fuel_oil"]["price_usd_per_bbl"] = 50.0
    res = solve_full_plant(assays)
    assert res.feasible
    assert res.status == "Optimal"
    assert res.objective > 0
    assert res.unit_feeds["cdu_charge"] > 0
    assert res.unit_feeds["fcc_feed"] > 0
    assert res.unit_feeds["coker_feed"] > 0
    assert res.unit_feeds["reformer_feed"] > 0
    # swing / chemistry
    assert res.streams["go_to_fcc"] > 0
    assert res.streams["resid_to_coker"] > 0
    assert res.arc_flows["fcc_naph_to_gas"] > 0
    assert (
        res.arc_flows.get("sr_heavy_to_reformer", 0)
        + res.arc_flows.get("sr_heavy_to_ref_pool", 0)
        + res.arc_flows.get("ref_pool_to_reformer", 0)
    ) > 0
    assert sum(res.products.values()) > 0
    assert res.quality_duals  # quality constraints present


def test_default_economics_may_idle_coker():
    """Honesty: with FO at default $68, resid→FO can beat coker on light slate."""
    res = solve_full_plant()
    assert res.feasible
    assert res.unit_feeds["cdu_charge"] > 0
    assert res.unit_feeds["fcc_feed"] > 0
    # coker may be idle; swing destinations still both available in arc space
    assert res.arc_flows["resid_to_coker"] + res.arc_flows["resid_to_fo"] > 0
    assert res.routing_splits["resid_frac_coker"] + res.routing_splits["resid_frac_fo"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_fcc_naphtha_not_forced_to_reformer():
    """Chemical correctness: FCC naphtha default is gasoline, not reformer."""
    res = solve_full_plant()
    assert res.feasible
    fcc_n = res.streams.get("fcc_naphtha", 0.0)
    to_gas = res.arc_flows.get("fcc_naph_to_gas", 0.0)
    to_ref = res.arc_flows.get("fcc_naph_to_reformer", 0.0)
    assert fcc_n > 0
    assert to_gas >= to_ref - 1e-6
    assert res.routing_splits.get("fcc_naph_frac_gas", 0.0) >= 0.99
    # coker naph also not forced to reformer
    assert res.arc_flows.get("coker_naph_to_reformer", 0.0) <= 1e-6


def test_resid_swing_exists():
    res = solve_full_plant()
    # both destinations available; optimal may use mix
    assert res.arc_flows["resid_to_coker"] + res.arc_flows["resid_to_fo"] > 0
    assert res.routing_splits["resid_frac_coker"] + res.routing_splits["resid_frac_fo"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_quality_blender_ron_sulfur():
    res = solve_full_plant()
    assert "qual_gas_min_ron" in res.duals or "qual_gas_min_ron" in res.quality_duals
    assert res.products.get("gasoline", 0) > 0
    # duals recorded
    assert "qual_gas_min_ron" in res.quality_duals
    assert "qual_diesel_max_s" in res.quality_duals


def test_dual_recovery_matches_mono_shadows():
    mono = solve_full_plant()
    admm = admm_price_directed_plant()
    assert admm["feasible"]
    assert abs(admm["objective"] - mono.objective) < 1e-4
    assert admm.get("dual_recovery_path") == "mono-oracle"
    assert "rho" in admm and "primal_residual_norm" in admm and "dual_residual_norm" in admm
    for k, v in mono.economic_shadows.items():
        rec = admm["economic_shadow_prices"].get(k, 0.0)
        assert abs(abs(v) - abs(rec)) < 1e-5, (k, v, rec)


def test_pure_admm_path_no_mono_oracle_for_lambda():
    """pure-admm: free λ (not mono duals), honest L∞ vs mono bal_*, residual path runs."""
    mono = solve_full_plant()
    pure = admm_price_directed_plant(
        recovery_path="pure-admm", max_iter=40, rho=1.2, dual_step=0.35
    )
    assert pure.get("dual_recovery_path") == "pure-admm"
    assert pure.get("status") in ("pure_admm_iterated", "pure_admm_hardened")
    assert pure.get("duals_like_monolithic") == {}
    assert "lambda" in pure and pure["lambda"]
    assert "lambda_vs_mono_Linf" in pure
    # mono bal duals available for honesty (either nested key or bal Linf metric)
    assert "mono_bal_duals" in pure or "lambda_vs_mono_bal_Linf" in pure
    if "mono_bal_duals" in pure:
        assert any(k.startswith("bal_") for k in pure["mono_bal_duals"])
    assert pure["iterations"] >= 1
    assert pure["unit_feeds"]["cdu_charge"] > 0
    # conversion units active on free path
    assert pure["unit_feeds"]["fcc_feed"] > 0 or pure.get("unit_feeds_mono", {}).get("fcc_feed", 0) > 0
    default = admm_price_directed_plant()
    assert default.get("dual_recovery_path") == "mono-oracle"
    assert default.get("lambda_vs_mono_Linf") == 0.0
    assert abs(default["objective"] - mono.objective) < 1e-4



def test_linf_dual_gap_helper():
    from pims_admm_llm.admm.residuals import linf_dual_gap

    linf, gaps = linf_dual_gap(
        {"cdu_gasoil": 80.0, "cdu_resid": 50.0},
        {"bal_tank_gasoil": -83.0, "bal_tank_resid": -55.0},
    )
    assert linf == pytest.approx(5.0, abs=1e-9)
    assert gaps["cdu_gasoil"] == pytest.approx(3.0, abs=1e-9)
    assert gaps["cdu_resid"] == pytest.approx(5.0, abs=1e-9)


def test_excel_roundtrip(tmp_path: Path):
    xlsx = tmp_path / "assays.xlsx"
    write_template_excel(xlsx)
    loaded = load_assays_excel(xlsx)
    assert len(loaded["crudes"]) >= 3
    assert "WTI_light" in {c["name"] for c in loaded["crudes"]}
    res = solve_full_plant(loaded)
    assert res.feasible


def test_plant_blocks_smoke():
    out = solve_all_plant_blocks()
    assert "CDU" in out["blocks"]
    assert out["blocks"]["CDU"]["status"] == "Optimal"
    assert sum(out["blocks"]["CDU"]["proposal"].values()) > 0
    # reformer proposal keys include heavy SR
    ref = out["blocks"]["REFORMER"]["proposal"]
    assert "cdu_naphtha_heavy_use" in ref or "reformate" in ref
    assert "BLENDER" in out["blocks"]
    assert out["blocks"]["BLENDER"]["status"] == "Optimal"


def test_default_assays_path_exists():
    assert default_assays_path().is_file()


def test_inventory_mode_optional():
    res_pass = solve_full_plant(inventory_mode=False)
    res_inv = solve_full_plant(inventory_mode=True)
    assert res_pass.feasible and res_inv.feasible
    assert res_pass.inventory_mode is False
    assert res_inv.inventory_mode is True
