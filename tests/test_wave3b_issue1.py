"""Issue #1: pure ADMM λ path, graph→LP, inventory mode smoke."""

from __future__ import annotations

from pims_admm_llm.models.full_plant import admm_price_directed_plant, solve_full_plant
from pims_admm_llm.models.graph_solve import routing_from_graph, solve_from_graph


def test_pure_admm_path_labels_and_runs():
    out = admm_price_directed_plant(recovery_path="pure-admm", max_iter=15)
    assert out["dual_recovery_path"] == "pure-admm"
    assert "rho" in out and "primal_residual_norm" in out and "dual_residual_norm" in out
    assert "lambda_vs_mono_Linf" in out
    assert out["duals_like_monolithic"] == {}  # not mono-oracle
    assert out["lambda"]
    assert out["iterations"] >= 1


def test_mono_oracle_still_default():
    out = admm_price_directed_plant()
    assert out["dual_recovery_path"] == "mono-oracle"
    assert out["lambda_vs_mono_Linf"] == 0.0
    assert out["feasible"]
    assert abs(out["objective_gap_vs_mono"]) < 1e-6


def test_graph_routing_opens_fcc_path():
    nodes = [
        {"id": "1", "data": {"unitType": "CDU", "active": True}},
        {"id": "2", "data": {"unitType": "FCC", "active": True}},
        {"id": "3", "data": {"unitType": "BLENDER", "active": True}},
        {"id": "4", "data": {"unitType": "REFORMER", "active": True}},
        {"id": "5", "data": {"unitType": "COKER", "active": True}},
    ]
    edges = [
        {"source": "1", "target": "2"},  # CDU→FCC
        {"source": "2", "target": "3"},
        {"source": "1", "target": "4"},
        {"source": "4", "target": "3"},
        {"source": "1", "target": "5"},
        {"source": "5", "target": "3"},
    ]
    r = routing_from_graph(nodes, edges)
    arcs = {a["id"]: a for a in r["arcs"]}
    assert arcs["go_to_fcc"]["default_open"] is True
    # FCC→reformer should stay closed unless edge present
    assert arcs["fcc_naph_to_reformer"]["default_open"] is False


def test_solve_from_graph_real_lp():
    nodes = [
        {"id": "cdu", "data": {"unitType": "CDU", "active": True}},
        {"id": "fcc", "data": {"unitType": "FCC", "active": True}},
        {"id": "cok", "data": {"unitType": "COKER", "active": True}},
        {"id": "ref", "data": {"unitType": "REFORMER", "active": True}},
        {"id": "bl", "data": {"unitType": "BLENDER", "active": True}},
    ]
    edges = [
        {"source": "cdu", "target": "fcc"},
        {"source": "cdu", "target": "cok"},
        {"source": "cdu", "target": "ref"},
        {"source": "fcc", "target": "bl"},
        {"source": "cok", "target": "bl"},
        {"source": "ref", "target": "bl"},
    ]
    out = solve_from_graph(nodes, edges, recovery_path="mono-oracle", run_admm=True)
    assert out["ok"] is True
    assert out["feasible"] is True
    assert out["objective"] > 0
    assert out["admm"]["dual_recovery_path"] == "mono-oracle"
    assert out["unit_feeds"]["cdu_charge"] > 0


def test_inventory_mode_multi_period_smoke():
    res0 = solve_full_plant(inventory_mode=False)
    res1 = solve_full_plant(inventory_mode=True)
    assert res0.feasible and res1.feasible
    assert res0.inventory_mode is False
    assert res1.inventory_mode is True
    # with start inventory, ending heels allowed
    assert sum(res1.tank_end.values()) >= 0


def test_delta_base_quality_meta_present():
    res = solve_full_plant()
    assert res.feasible
    q = (res.meta or {}).get("quality") or {}
    assert q, "quality meta missing from FullPlantResult"
    # model label: delta_base or index
    model = str(q.get("model") or "").lower().replace("-", "_")
    assert "delta" in model or model in ("index", "linear")
    assert "ron_deltas" in q or "component_deltas" in q or "deltas" in q or "ron" in str(q).lower()


def test_graph_keeps_chemical_defaults_nonzero_obj():
    """Canvas without CDU→BLENDER still keeps default light-naph→gas open."""
    nodes = [
        {"id": "cdu", "data": {"unitType": "CDU", "active": True}},
        {"id": "fcc", "data": {"unitType": "FCC", "active": True}},
        {"id": "cok", "data": {"unitType": "COKER", "active": True}},
        {"id": "ref", "data": {"unitType": "REFORMER", "active": True}},
        {"id": "bl", "data": {"unitType": "BLENDER", "active": True}},
    ]
    edges = [
        {"source": "cdu", "target": "fcc"},
        {"source": "cdu", "target": "cok"},
        {"source": "cdu", "target": "ref"},
        {"source": "fcc", "target": "bl"},
        {"source": "ref", "target": "bl"},
    ]
    r = routing_from_graph(nodes, edges)
    arcs = {a["id"]: a for a in r["arcs"]}
    assert arcs["sr_light_to_gas"]["default_open"] is True
    assert arcs["fcc_naph_to_reformer"]["default_open"] is False
    out = solve_from_graph(nodes, edges, recovery_path="mono-oracle", run_admm=False)
    assert out["feasible"] is True
    assert out["objective"] > 0
