"""Base-delta CDU/FCC/COKER: compositions, auto-route, mass balances, cascade LP."""

from __future__ import annotations

from pims_admm_llm.models.auto_route import (
    best_route,
    complete_missing_edges,
    connect_score,
    guess_route,
)
from pims_admm_llm.models.base_delta import (
    assert_every_product_has_exit,
    auto_wire_edges_for_units,
    build_cdu_base_delta,
    build_coker_base_delta,
    build_fcc_base_delta,
    process_modes_cdu,
    process_modes_coker,
    process_modes_fcc,
    unit_submodels_cdu_fcc,
)
from pims_admm_llm.models.cdu_fcc import solve_cdu_fcc, solve_cdu_fcc_coker
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
    assert y1["fcc_naphtha"] > y0["fcc_naphtha"]
    assert y1["fcc_lpg"] >= y0["fcc_lpg"] - 1e-9
    assert any(e.stream == "fcc_coke" and e.default_sink == "REGEN_HEAT" for e in m.exits)


def test_coker_every_product_has_exit_and_recycle_delta():
    m = build_coker_base_delta()
    assert_every_product_has_exit(m)
    assert set(m.products) == {e.stream for e in m.exits}
    y_low = m.evaluate(conditions={"recycle_ratio": 0.05, "drum_outlet_temp_f": 930.0})
    y_high = m.evaluate(conditions={"recycle_ratio": 0.30, "drum_outlet_temp_f": 910.0})
    # Higher recycle → more coke, less gasoil (planning)
    assert y_high["coker_coke"] > y_low["coker_coke"]
    assert y_high["coker_gasoil"] < y_low["coker_gasoil"]
    assert any(e.stream == "coker_coke" and e.default_sink == "COKE_SALES" for e in m.exits)


def test_process_modes_sos1_ready():
    assert {m["id"] for m in process_modes_cdu()} == {"cuts_light", "cuts_mid", "cuts_heavy"}
    assert {m["id"] for m in process_modes_fcc()} == {"rot_low", "rot_mid", "rot_high"}
    assert {m["id"] for m in process_modes_coker()} == {"rec_low", "rec_mid", "rec_high"}
    for m in process_modes_coker():
        assert "recycle_ratio" in m["conditions"]
        assert "coker_naphtha" in m["yields"]
        assert "coker_naphtha" in m["compositions"]


def test_stream_compositions_richer_than_ron_s():
    go = get_stream("cdu_gasoil")
    assert go.family == "gasoil"
    assert go.tbp_50_f >= 650
    cn = get_stream("coker_naphtha")
    assert cn.olefins_vol > 0.2
    assert cn.sulfur_wt > 0.1


def test_auto_route_gasoil_to_fcc():
    g = best_route("cdu_gasoil")
    assert g.sink in ("FCC", "POOL_FCC")
    assert g.score >= 0.8


def test_auto_route_resid_to_coker():
    g = best_route("cdu_resid")
    assert g.sink in ("COKER", "POOL_COKER", "FO")
    ranked = {r.sink: r.score for r in guess_route("cdu_resid", top_k=8, min_score=0.0)}
    assert ranked.get("COKER", 0) >= 0.8 or ranked.get("POOL_COKER", 0) >= 0.8


def test_auto_route_fcc_naphtha_not_reformer():
    ranked = guess_route("fcc_naphtha", top_k=8, min_score=0.05)
    assert ranked[0].sink != "REFORMER"
    bad = connect_score("FCC", "REFORMER", stream="fcc_naphtha")
    assert bad["allowed"] is False or bad["score"] < 0.4


def test_auto_route_coker_naphtha_not_reformer():
    ranked = guess_route("coker_naphtha", top_k=8, min_score=0.05)
    assert ranked[0].sink in ("HDT_NAPH", "FO", "GASOLINE", "BLENDER")
    assert ranked[0].sink != "REFORMER"


def test_auto_wire_adds_coker_feed_and_products():
    edges = auto_wire_edges_for_units(["CDU", "FCC", "COKER"])
    streams = {e["stream"]: e for e in edges}
    assert streams["cdu_gasoil"]["to"] == "FCC"
    assert streams["cdu_resid"]["to"] == "COKER"
    assert "coker_naphtha" in streams
    assert "coker_coke" in streams
    # without coker, no resid→coker feed arc
    edges2 = auto_wire_edges_for_units(["CDU", "FCC"])
    assert not any(e["stream"] == "cdu_resid" and e["to"] == "COKER" for e in edges2)


def test_complete_missing_edges():
    produced = ["cdu_gasoil", "fcc_lpg", "fcc_coke"]
    edges = [{"stream": "cdu_gasoil", "to": "FCC"}]
    sug = complete_missing_edges(produced, edges)
    streams = {s["stream"] for s in sug}
    assert "fcc_lpg" in streams and "fcc_coke" in streams
    assert "cdu_gasoil" not in streams


def test_solve_cdu_fcc_optimal_mass_balance():
    r = solve_cdu_fcc(max_crude_kbd=100.0, enable_coker=False)
    assert r.status == "Optimal"
    assert r.crude_kbd > 1.0
    assert r.mass_balance["ok"] is True
    assert r.streams["fcc_naphtha"] > 0.0
    for p in r.meta["products_cdu"]:
        assert p in r.cdu_exits
    for p in r.meta["products_fcc"]:
        assert p in r.fcc_exits


def test_solve_with_coker_optimal_mass_balance():
    r = solve_cdu_fcc_coker(max_crude_kbd=100.0)
    assert r.status == "Optimal"
    assert "COKER" in r.enabled_units
    assert r.mass_balance["ok"] is True, r.mass_balance
    assert r.streams["coker_feed"] > 0.0 or r.streams.get("resid_to_fo", 0) > 0
    # resid fully allocated
    resid = r.streams["cdu_resid"]
    assert abs(r.streams["resid_to_coker"] + r.streams["resid_to_fo"] - resid) < 1e-3
    if r.streams["coker_feed"] > 1e-6:
        assert r.streams["coker_naphtha"] > 0.0
        assert r.streams["coker_coke"] > 0.0
        assert r.coker_exits["coker_coke"] == "COKE_SALES"
    assert r.process_conditions.get("COKER", {}).get("recycle_ratio") is not None


def test_coker_mode_affects_yields():
    low = solve_cdu_fcc(enable_coker=True, fix_coker_mode="rec_low", max_crude_kbd=50)
    high = solve_cdu_fcc(enable_coker=True, fix_coker_mode="rec_high", max_crude_kbd=50)
    assert low.status == "Optimal" and high.status == "Optimal"
    assert high.coker_yields["coker_coke"] > low.coker_yields["coker_coke"]


def test_unit_submodels_with_coker():
    pack = unit_submodels_cdu_fcc(include_coker=True)
    assert pack["enabled"] == ["CDU", "FCC", "COKER"]
    assert "COKER" in pack
    assert pack["COKER"]["base_yields"]["coker_gasoil"] > 0
