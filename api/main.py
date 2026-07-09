"""
Minimal FastAPI backend for wave3 snap-together flowsheet UI.

Run from repo root:
  uvicorn api.main:app --reload --port 8008
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTING_PATH = REPO_ROOT / "data" / "routing.json"

app = FastAPI(
    title="pims-admm-llm flowsheet API",
    version="0.2.0-wave5",
    description="SvelteFlow snap-together routing UI + graph→LP/ADMM.",
)


def _cors_origins() -> list[str]:
    """CORS allowlist. Default local Vite + API. Override with CORS_ORIGINS=a,b or *."""
    raw = (os.environ.get("CORS_ORIGINS") or "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:8008",
        "http://localhost:8008",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    type: Optional[str] = None
    position: Optional[dict[str, float]] = None
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: Optional[str] = None
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class GraphPayload(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class ConnectPayload(BaseModel):
    """Edge attempt + optional port attributes for validation."""

    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    sourceType: Optional[str] = None
    targetType: Optional[str] = None
    stream: Optional[str] = None
    portAttrs: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Palette + soft connect rules
# ---------------------------------------------------------------------------

UNIT_PALETTE = [
    {"type": "CDU", "label": "CDU", "category": "process", "submodel": "lp"},
    {"type": "FCC", "label": "FCC", "category": "process", "submodel": "lp"},
    {"type": "COKER", "label": "Coker", "category": "process", "submodel": "lp"},
    {"type": "REFORMER", "label": "Reformer", "category": "process", "submodel": "lp"},
    {"type": "HDT_NAPH", "label": "HDT Naphtha", "category": "process", "submodel": "lp"},
    {"type": "BLENDER", "label": "Blender", "category": "process", "submodel": "lp"},
    {"type": "TANK", "label": "Tank", "category": "process", "submodel": "lp"},
    {"type": "SELL", "label": "Sell", "category": "process", "submodel": "lp"},
    {"type": "warehouse", "label": "Warehouse", "category": "supply_chain", "submodel": "lp"},
    {"type": "transport", "label": "Transport", "category": "supply_chain", "submodel": "lp"},
]

COMPAT: dict[str, set[str]] = {
    "CDU": {"TANK", "FCC", "COKER", "REFORMER", "HDT_NAPH", "BLENDER", "SELL", "warehouse", "transport"},
    "FCC": {"TANK", "BLENDER", "SELL", "HDT_NAPH", "REFORMER", "warehouse"},
    "COKER": {"TANK", "BLENDER", "SELL", "HDT_NAPH", "REFORMER", "warehouse"},
    "REFORMER": {"BLENDER", "SELL", "TANK"},
    "HDT_NAPH": {"BLENDER", "SELL", "TANK", "REFORMER"},
    "BLENDER": {"SELL", "warehouse", "transport"},
    "TANK": {"FCC", "COKER", "REFORMER", "HDT_NAPH", "BLENDER", "SELL", "warehouse", "transport"},
    "SELL": {"warehouse", "transport"},
    "warehouse": {"transport", "SELL", "BLENDER", "TANK"},
    "transport": {"warehouse", "SELL", "TANK", "BLENDER"},
}


def _normalize_unit_type(raw: Optional[str]) -> str:
    if not raw:
        return ""
    t = str(raw).strip()
    aliases = {
        "HYDROTREAT_NAPH": "HDT_NAPH",
        "HDT": "HDT_NAPH",
        "Tank": "TANK",
        "tank": "TANK",
    }
    if t in aliases:
        return aliases[t]
    if t.upper().startswith("TANK"):
        return "TANK"
    return t


def _load_routing() -> dict[str, Any]:
    if not ROUTING_PATH.is_file():
        return {"units": [], "arcs": [], "error": f"missing {ROUTING_PATH}"}
    with ROUTING_PATH.open() as f:
        return json.load(f)


def _stub_clusters(nodes: list[GraphNode]) -> list[dict[str, Any]]:
    """Group nodes into ADMM-style cluster stubs (process vs supply-chain)."""
    process: list[str] = []
    supply: list[str] = []
    other: list[str] = []
    for n in nodes:
        if not bool(n.data.get("active", True)):
            continue
        ut = _normalize_unit_type(
            n.data.get("unitType") or n.data.get("label") or n.type
        )
        if ut in ("warehouse", "transport"):
            supply.append(n.id)
        elif ut:
            process.append(n.id)
        else:
            other.append(n.id)
    clusters: list[dict[str, Any]] = []
    if process:
        clusters.append({"id": "cluster_process", "node_ids": process})
    if supply:
        clusters.append({"id": "cluster_supply_chain", "node_ids": supply})
    if other:
        clusters.append({"id": "cluster_other", "node_ids": other})
    if not clusters and nodes:
        clusters.append({"id": "cluster_all", "node_ids": [n.id for n in nodes]})
    return clusters


def _connect_score(
    source_type: str,
    target_type: str,
    stream: Optional[str],
) -> tuple[bool, float, str]:
    # Prefer property-based base-delta auto_route when stream is known
    if stream:
        try:
            from pims_admm_llm.models.auto_route import connect_score as prop_connect

            res = prop_connect(source_type, target_type, stream=stream)
            return bool(res["allowed"]), float(res["score"]), str(res["reason"])
        except Exception:
            pass

    if not source_type or not target_type:
        return True, 0.5, "types unknown — allowing connection (stub)"
    if source_type == target_type and source_type not in ("TANK", "warehouse", "transport"):
        return True, 0.35, f"same type {source_type} — allowed with low score (stub)"

    allowed_targets = COMPAT.get(source_type)
    if allowed_targets is None:
        return True, 0.4, f"no rule for source {source_type} — allow (stub)"
    if target_type not in allowed_targets:
        return False, 0.0, f"{source_type} → {target_type} not in stub compatibility table"

    score = 0.7
    reason = f"{source_type} → {target_type} allowed"
    if stream:
        score = min(1.0, score + 0.15)
        reason += f" (stream={stream})"
    return True, score, reason


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "pims-admm-llm-flowsheet", "wave": "wave3"}


@app.get("/api/routing")
def get_routing() -> dict[str, Any]:
    """Load data/routing.json arcs for palette defaults."""
    data = _load_routing()
    return {
        "ok": True,
        "version": data.get("version"),
        "units": data.get("units", []),
        "arcs": data.get("arcs", []),
        "palette": UNIT_PALETTE,
        "chemical_defaults": data.get("chemical_defaults", {}),
        "path": str(ROUTING_PATH.relative_to(REPO_ROOT)),
    }


class GraphSolvePayload(GraphPayload):
    """Graph plus optional solver flags (issue #1 wire-up)."""

    recovery_path: str = "mono-oracle"  # mono-oracle | pure-admm
    inventory_mode: Optional[bool] = None
    run_admm: bool = True
    stub_only: bool = False


def _stub_graph_response(
    payload: GraphSolvePayload,
    clusters: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "clusters": clusters,
        "node_count": len(payload.nodes),
        "edge_count": len(payload.edges),
        "admm_status": "stub",
        "message": (
            f"Accepted {len(payload.nodes)} nodes, {len(payload.edges)} edges; "
            f"{len(clusters)} cluster(s) stubbed. {reason}"
        ),
        "objective": None,
        "unit_feeds": {},
        "products": {},
        "routing_splits": {},
        "duals": {},
        "rho": None,
        "residuals": {"primal": None, "dual": None},
        "dual_recovery_path": None,
    }


@app.post("/api/graph")
def post_graph(payload: GraphSolvePayload) -> dict[str, Any]:
    """Accept {nodes, edges}; map to routing overlay; solve full plant LP + ADMM.

    Returns objective, unit_feeds, products, routing_splits, duals, rho,
    residuals, dual_recovery_path. Inactive nodes skipped in clusters.
    Falls back to stub if plant package import fails.
    """
    clusters = _stub_clusters(payload.nodes)
    if payload.stub_only:
        return _stub_graph_response(payload, clusters, reason="ADMM not run (stub_only).")

    # Real solve path
    try:
        import sys

        src = str(REPO_ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from pims_admm_llm.models.graph_solve import solve_from_graph

        nodes = [n.model_dump() for n in payload.nodes]
        edges = [e.model_dump() for e in payload.edges]
        result = solve_from_graph(
            nodes,
            edges,
            recovery_path=payload.recovery_path,
            inventory_mode=payload.inventory_mode,
            run_admm=payload.run_admm,
        )
        base = {
            "ok": True,
            "clusters": clusters,
            "node_count": len(payload.nodes),
            "edge_count": len(payload.edges),
        }
        base.update(result)
        base["clusters"] = clusters
        return base
    except ImportError as e:
        # Task contract: fallback stub if plant import fails
        out = _stub_graph_response(
            payload,
            clusters,
            reason=f"ADMM not run (plant import failed: {e}).",
        )
        out["fallback"] = "plant_import_failed"
        out["error"] = str(e)
        return out
    except Exception as e:
        base = {
            "ok": False,
            "clusters": clusters,
            "node_count": len(payload.nodes),
            "edge_count": len(payload.edges),
            "admm_status": "error",
            "message": f"graph solve failed: {type(e).__name__}: {e}",
            "error": str(e),
            "objective": None,
            "unit_feeds": {},
            "products": {},
            "routing_splits": {},
            "duals": {},
            "rho": None,
            "residuals": {"primal": None, "dual": None},
            "dual_recovery_path": None,
        }
        return base


@app.post("/api/connect")
def post_connect(payload: ConnectPayload) -> dict[str, Any]:
    """Validate an edge attempt; return {allowed, score, reason}."""
    src = _normalize_unit_type(payload.sourceType or payload.portAttrs.get("sourceType"))
    tgt = _normalize_unit_type(payload.targetType or payload.portAttrs.get("targetType"))
    stream = payload.stream or payload.portAttrs.get("stream")
    if payload.source == payload.target:
        return {
            "allowed": False,
            "score": 0.0,
            "reason": "self-loop rejected",
            "source": payload.source,
            "target": payload.target,
        }
    allowed, score, reason = _connect_score(src, tgt, stream)
    out: dict[str, Any] = {
        "allowed": allowed,
        "score": score,
        "reason": reason,
        "source": payload.source,
        "target": payload.target,
        "sourceType": src or None,
        "targetType": tgt or None,
        "stream": stream,
    }
    # Attach property-based guesses when stream known
    if stream:
        try:
            from pims_admm_llm.models.auto_route import connect_score as prop_connect

            prop = prop_connect(src or "", tgt or "", stream=stream)
            out["guesses"] = prop.get("guesses")
            out["best"] = prop.get("best")
        except Exception:
            pass
    return out


@app.get("/")
def root() -> dict[str, str]:
    return {
        "docs": "/docs",
        "health": "/health",
        "routing": "/api/routing",
        "graph": "POST /api/graph",
        "connect": "POST /api/connect",
    }
