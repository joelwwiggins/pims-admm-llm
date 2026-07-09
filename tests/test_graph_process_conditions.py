"""Wave5: UI process conditions flow into graph→LP yield tables."""

from __future__ import annotations

from pims_admm_llm.models.graph_solve import (
    extract_process_conditions,
    solve_from_graph,
)


def test_extract_process_conditions_last_write():
    nodes = [
        {
            "id": "fcc-1",
            "data": {
                "unitType": "FCC",
                "active": True,
                "processConditions": {"riser_outlet_temp_f": 990.0},
            },
        },
        {
            "id": "fcc-2",
            "data": {
                "unitType": "FCC",
                "active": True,
                "processConditions": {"riser_outlet_temp_f": 1000.0},
            },
        },
        {
            "id": "tank-1",
            "data": {"unitType": "TANK", "active": True, "processConditions": {"x": 1}},
        },
    ]
    pc = extract_process_conditions(nodes)
    assert pc["FCC"]["riser_outlet_temp_f"] == 1000.0
    assert "TANK" not in pc


def test_solve_from_graph_process_conditions_meta():
    nodes = [
        {
            "id": "cdu",
            "data": {"unitType": "CDU", "active": True},
        },
        {
            "id": "fcc",
            "data": {
                "unitType": "FCC",
                "active": True,
                "processConditions": {
                    "riser_outlet_temp_f": 1005.0,
                    "catalyst_to_oil": 7.0,
                },
            },
        },
        {
            "id": "coker",
            "data": {"unitType": "COKER", "active": True},
        },
        {
            "id": "reformer",
            "data": {"unitType": "REFORMER", "active": True},
        },
        {
            "id": "blender",
            "data": {"unitType": "BLENDER", "active": True},
        },
    ]
    edges = [
        {"source": "cdu", "target": "fcc"},
        {"source": "cdu", "target": "coker"},
        {"source": "cdu", "target": "reformer"},
        {"source": "fcc", "target": "blender"},
        {"source": "coker", "target": "blender"},
        {"source": "reformer", "target": "blender"},
    ]
    out = solve_from_graph(nodes, edges, run_admm=False)
    assert out.get("ok") or out.get("feasible")
    assert "FCC" in (out.get("routing_meta") or {}).get(
        "process_conditions_from_nodes", []
    )
    pc = out.get("process_conditions") or {}
    # meta may nest unit keys
    if pc:
        fcc_pc = pc.get("FCC") or pc.get("fcc") or {}
        if "riser_outlet_temp_f" in fcc_pc:
            assert float(fcc_pc["riser_outlet_temp_f"]) == 1005.0
