"""Base-delta CDU/FCC submodels, stream compositions, auto-route, focused plant."""

from __future__ import annotations

import pytest

from pims_admm_llm.models.auto_route import (
    best_route,
    complete_missing_edges,
    connect_score,
    guess_route,
)
from pims_admm_llm.models.base_delta import (
    assert_every_product_has_exit,
    build_cdu_base_delta,
    build_fcc_base_delta,
    process_modes_cdu,
    process_modes_fcc,
    unit_submodels_cdu_fcc,
)
from pims_admm_llm.models.cdu_fcc import solve_cdu_fcc
from pims_admm_llm.models.stream_composition import get_stream


def test_cdu_every_product_has_exit():
    m = build_cdu_base_delta()
    assert_every_product_has_exit(m)
    assert set(m.products) == {e.stream for e in m.exits}
    y = m.evaluate()
    assert abs(sum(v for k, v in y.items() if k != "cdu_offgas") - 1.0) < 1e-6
    assert y["cdu_gasoil"] > 0.15


def test_fcc_every_product_has_exit_and_process_deltas():
    m = build_fcc_base_delta()
    assert_every_product_has_exit(m)
    y0 = m.evaluate(conditions={"riser_outlet_temp_f": 940.0})
    y1 = m.evaluate(conditions={"riser_outlet_temp_f": 1020.0})
    # Higher ROT → more naphtha / LPG (severity)
    assert y1["fcc_naphtha"] > y0["fcc_naphtha"]
    assert y1["fcc_lpg"] >= y0["fcc_lpg"] - 1e-9
    # Coke wt present and exit REGEN_HEAT
    assert any(e.stream == "fcc_coke" and e.default_sink == "REGEN_HEAT" for e in m.exits)


def test_process_modes_sos1_ready():
    cdu_modes = process_modes_cdu()
    fcc_modes = process_modes_fcc()
    assert {m["id"] for m in cdu_modes} == {"cuts_light", "cuts_mid", "cuts_heavy"}
    assert {m["id"] for m in fcc_modes} == {"rot_low", "rot_mid", "rot_high"}
    for m in fcc_modes:
        assert "riser_outlet_temp_f" in m["conditions"]
        assert set(m["yields"]) >= {"fcc_naphtha", "fcc_coke", "fcc_lpg"}
        assert "fcc_naphtha" in m["compositions"]


def test_stream_compositions_richer_than_ron_s():
    go = get_stream("cdu_gasoil")
    assert go.family == "gasoil"
    assert go.tbp_50_f >= 650
    assert go.api < 30
    naph = get_stream("fcc_naphtha")
    assert naph.ron >= 90
    assert naph.olefins_vol > 0.1


def test_auto_route_gasoil_to_fcc():
    g = best_route("cdu_gasoil")
    assert g.sink in ("FCC", "POOL_FCC")
    assert g.score >= 0.8


def test_auto_route_fcc_naphtha_not_reformer():
    ranked = guess_route("fcc_naphtha", top_k=8, min_score=0.05)
    sinks = [r.sink for r in ranked]
    assert sinks[0] in ("GASOLINE", "BLENDER", "HDT_NAPH")
    # reformer should not be top
    assert sinks[0] != "REFORMER"
    bad = connect_score("FCC", "REFORMER", stream="fcc_naphtha")
    assert bad["allowed"] is False or bad["score"] < 0.4


def test_auto_route_sr_heavy_prefers_reformer():
    g = best_route("cdu_naphtha_heavy")
    assert g.sink in ("REFORMER", "POOL_REFORMER", "GASOLINE", "BLENDER")
    # property boost should keep reformer competitive
    ranked = {r.sink: r.score for r in guess_route("cdu_naphtha_heavy", top_k=10, min_score=0.0)}
    assert ranked.get("REFORMER", 0) >= 0.7


def test_complete_missing_edges():
    produced = ["cdu_gasoil", "fcc_lpg", "fcc_coke"]
    edges = [{"stream": "cdu_gasoil", "to": "FCC"}]
    sug = complete_missing_edges(produced, edges)
    streams = {s["stream"] for s in sug}
    assert "fcc_lpg" in streams
    assert "fcc_coke" in streams
    assert "cdu_gasoil" not in streams  # already edged


def test_solve_cdu_fcc_optimal_and_exits():
    r = solve_cdu_fcc(max_crude_kbd=100.0)
    assert r.status == "Optimal"
    assert r.crude_kbd > 1.0
    assert r.streams["fcc_feed"] > 0.0
    assert r.streams["fcc_naphtha"] > 0.0
    # every product appears in exit maps
    for p in r.meta["products_cdu"]:
        assert p in r.cdu_exits
    for p in r.meta["products_fcc"]:
        assert p in r.fcc_exits
    assert r.cdu_exits["cdu_gasoil"] in ("FCC", "POOL_FCC")
    assert r.fcc_exits["fcc_coke"] == "REGEN_HEAT"
    assert r.process_conditions["FCC"]["riser_outlet_temp_f"] in (940.0, 980.0, 1020.0)
    # compositions present for products
    assert "fcc_naphtha" in r.compositions
    assert r.compositions["fcc_naphtha"]["ron"] > 80


def test_severity_mode_affects_objective_feasibility():
    low = solve_cdu_fcc(fix_fcc_mode="rot_low", max_crude_kbd=50)
    high = solve_cdu_fcc(fix_fcc_mode="rot_high", max_crude_kbd=50)
    assert low.status == "Optimal" and high.status == "Optimal"
    # High severity should produce at least as much naphtha yield coeff
    assert high.fcc_yields["fcc_naphtha"] > low.fcc_yields["fcc_naphtha"]


def test_unit_submodels_bundle():
    pack = unit_submodels_cdu_fcc()
    assert "CDU" in pack and "FCC" in pack
    assert pack["CDU"]["base_yields"]
    assert pack["FCC"]["deltas"]["fcc_naphtha"]["riser_outlet_temp_f"] != 0
