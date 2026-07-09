"""Wave2 full plant: assays, routing, property yields, mono feasibility, dual recovery."""

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
        assert abs(sum(y.values()) - 1.0) < 1e-6
        assert set(y) == {"cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"}


def test_routing_table():
    r = load_routing()
    routes = {(x["from"], x["stream"], x["to"]) for x in r["routes"]}
    assert ("CDU", "cdu_gasoil", "TANK_GASOIL") in routes
    assert ("TANK_GASOIL", "cdu_gasoil", "FCC") in routes
    assert ("FCC", "fcc_naphtha", "TANK_FCC_NAPH") in routes
    assert ("TANK_FCC_NAPH", "fcc_naphtha", "REFORMER") in routes
    assert ("CDU", "cdu_resid", "TANK_RESID") in routes
    assert ("TANK_RESID", "cdu_resid", "COKER") in routes
    assert ("COKER", "coker_naphtha", "TANK_COKER_NAPH") in routes
    assert ("TANK_COKER_NAPH", "coker_naphtha", "REFORMER") in routes


def test_property_yields_respond_to_feed():
    light = crude_to_props({"name": "L", "api": 40, "ccr_wt": 1, "sulfur_wt": 0.2,
                           "nitrogen_ppm": 500, "paraffins_vol": 0.4, "naphthenes_vol": 0.3, "aromatics_vol": 0.3})
    heavy = crude_to_props({"name": "H", "api": 20, "ccr_wt": 12, "sulfur_wt": 3,
                           "nitrogen_ppm": 3000, "paraffins_vol": 0.2, "naphthenes_vol": 0.3, "aromatics_vol": 0.5})
    yl = cdu_yields_from_assay(light)
    yh = cdu_yields_from_assay(heavy)
    assert yl["cdu_naphtha"] > yh["cdu_naphtha"]
    assert yh["cdu_resid"] > yl["cdu_resid"]
    # higher CCR gasoil → lower FCC conversion naphtha typically
    from pims_admm_llm.models.yields import gasoil_props_from_crude, resid_props_from_crude
    fcc_l = fcc_yields(gasoil_props_from_crude(light))
    fcc_h = fcc_yields(gasoil_props_from_crude(heavy))
    assert fcc_l["fcc_naphtha"] >= fcc_h["fcc_naphtha"] - 1e-9
    cok_h = coker_yields(resid_props_from_crude(heavy))
    assert cok_h["coker_naphtha"] + cok_h["coker_gasoil"] < 0.85


def test_full_plant_mono_optimal():
    res = solve_full_plant()
    assert res.feasible
    assert res.status == "Optimal"
    assert res.objective > 0
    assert res.unit_feeds["cdu_charge"] > 0
    assert res.unit_feeds["fcc_feed"] > 0
    assert res.unit_feeds["coker_feed"] > 0
    assert res.unit_feeds["reformer_feed"] > 0
    # routing: gasoil goes to FCC, resid to coker
    assert res.streams["go_to_fcc"] > 0
    assert res.streams["resid_to_coker"] > 0
    assert res.streams["fcc_naph_to_reformer"] + res.streams["coker_naph_to_reformer"] > 0
    assert sum(res.products.values()) > 0


def test_dual_recovery_matches_mono_shadows():
    mono = solve_full_plant()
    admm = admm_price_directed_plant()
    assert admm["feasible"]
    assert abs(admm["objective"] - mono.objective) < 1e-4
    for k, v in mono.economic_shadows.items():
        rec = admm["economic_shadow_prices"].get(k, 0.0)
        assert abs(abs(v) - abs(rec)) < 1e-5, (k, v, rec)


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


def test_default_assays_path_exists():
    assert default_assays_path().is_file()
