"""TestClient coverage for FastAPI POST /api/graph (wave3 issue #1 wire-up)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from pims_admm_llm.models.graph_solve import (  # noqa: E402
    extract_active_units,
    routing_from_graph,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _default_nodes_edges():
    nodes = [
        {"id": "cdu-1", "data": {"unitType": "CDU", "active": True}},
        {"id": "fcc-1", "data": {"unitType": "FCC", "active": True}},
        {"id": "cok-1", "data": {"unitType": "COKER", "active": True}},
        {"id": "ref-1", "data": {"unitType": "REFORMER", "active": True}},
        {"id": "bl-1", "data": {"unitType": "BLENDER", "active": True}},
        {"id": "sell-1", "data": {"unitType": "SELL", "active": True}},
        {"id": "dead-fcc", "data": {"unitType": "FCC", "active": False}},
    ]
    edges = [
        {"source": "cdu-1", "target": "fcc-1"},
        {"source": "cdu-1", "target": "cok-1"},
        {"source": "cdu-1", "target": "ref-1"},
        {"source": "fcc-1", "target": "bl-1"},
        {"source": "cok-1", "target": "bl-1"},
        {"source": "ref-1", "target": "bl-1"},
        {"source": "bl-1", "target": "sell-1"},
    ]
    return nodes, edges


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["wave"] == "wave3"


def test_graph_stub_only(client: TestClient):
    nodes, edges = _default_nodes_edges()
    r = client.post(
        "/api/graph",
        json={"nodes": nodes, "edges": edges, "stub_only": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["admm_status"] == "stub"
    assert "clusters" in body
    process = next(c for c in body["clusters"] if c["id"] == "cluster_process")
    assert "dead-fcc" not in process["node_ids"]
    assert "cdu-1" in process["node_ids"]
    assert body.get("objective") is None
    assert body.get("dual_recovery_path") is None


def test_graph_real_solve_fields(client: TestClient):
    nodes, edges = _default_nodes_edges()
    r = client.post(
        "/api/graph",
        json={
            "nodes": nodes,
            "edges": edges,
            "recovery_path": "mono-oracle",
            "run_admm": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body.get("fallback") != "plant_import_failed"
    # Required contract fields
    assert body["objective"] is not None
    assert isinstance(body["unit_feeds"], dict) and body["unit_feeds"]
    assert isinstance(body["products"], dict) and body["products"]
    assert isinstance(body["routing_splits"], dict)
    assert isinstance(body["duals"], dict)
    assert body["rho"] is not None
    assert "primal" in body["residuals"] and "dual" in body["residuals"]
    assert body["dual_recovery_path"] == "mono-oracle"
    assert body["admm_status"] == "mono-oracle"
    assert body["residuals"]["primal"] is not None
    # Inactive node not in clusters
    process = next(c for c in body["clusters"] if c["id"] == "cluster_process")
    assert "dead-fcc" not in process["node_ids"]
    assert body["objective"] > 0


def test_inactive_nodes_skipped_in_routing():
    nodes, edges = _default_nodes_edges()
    units = extract_active_units(nodes)
    assert "FCC" in units  # active fcc-1
    routing = routing_from_graph(nodes, edges)
    assert routing.get("graph_driven") is True
    assert "dead-fcc" not in (routing.get("graph_units") or [])
    assert set(routing.get("graph_units") or []) >= {
        "CDU",
        "FCC",
        "COKER",
        "REFORMER",
        "BLENDER",
        "SELL",
    }


def test_graph_import_fallback(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """If plant import fails, respond with stub (not 500)."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name.startswith("pims_admm_llm") or name == "pims_admm_llm.models.graph_solve":
            raise ImportError("forced plant import failure for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    nodes, edges = _default_nodes_edges()
    r = client.post("/api/graph", json={"nodes": nodes, "edges": edges})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["admm_status"] == "stub"
    assert body.get("fallback") == "plant_import_failed"
    assert body.get("objective") is None
