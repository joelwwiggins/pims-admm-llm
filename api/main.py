"""
Wave3 UI API scaffold — graph → cluster stub + ADMM status stub.

Run from repo root:
  uvicorn api.main:app --reload --port 8000

Or:
  cd api && uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="pims-admm-llm UI API",
    description="Wave3 scaffold: accept SvelteFlow graph, return cluster + ADMM stubs",
    version="0.1.0-wave3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortAttrs(BaseModel):
    """Port / handle attributes used for connect validation."""

    stream: Optional[str] = None
    direction: Optional[str] = None  # "in" | "out"
    quality: Optional[dict[str, Any]] = None


class GraphNode(BaseModel):
    id: str
    type: Optional[str] = "unit"
    position: dict[str, float] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class GraphPayload(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ClusterStub(BaseModel):
    id: str
    unit_ids: list[str]
    submodel_types: list[str]


class AdmmStatusStub(BaseModel):
    status: str
    rho: float
    primal_residual: float
    dual_residual: float
    max_iter: int
    iteration: int
    message: str


class GraphResponse(BaseModel):
    ok: bool
    node_count: int
    edge_count: int
    active_units: list[str]
    clusters: list[ClusterStub]
    admm: AdmmStatusStub
    notes: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pims-admm-llm-ui-api"}


@app.get("/api/units")
def list_unit_palette() -> dict[str, Any]:
    """Palette of unit pieces for the UI dock (matches routing.json unit names)."""
    return {
        "units": [
            {
                "submodel": "CDU",
                "label": "CDU",
                "ports": {
                    "in": [{"id": "crude", "stream": "crude"}],
                    "out": [
                        {"id": "naphtha_light", "stream": "cdu_naphtha_light"},
                        {"id": "naphtha_heavy", "stream": "cdu_naphtha_heavy"},
                        {"id": "distillate", "stream": "cdu_distillate"},
                        {"id": "gasoil", "stream": "cdu_gasoil"},
                        {"id": "resid", "stream": "cdu_resid"},
                    ],
                },
            },
            {
                "submodel": "TANK",
                "label": "Tank",
                "ports": {
                    "in": [{"id": "in", "stream": "any"}],
                    "out": [{"id": "out", "stream": "any"}],
                },
            },
            {
                "submodel": "FCC",
                "label": "FCC",
                "ports": {
                    "in": [{"id": "feed", "stream": "gasoil"}],
                    "out": [
                        {"id": "naphtha", "stream": "fcc_naphtha"},
                        {"id": "lco", "stream": "fcc_lco"},
                        {"id": "slurry", "stream": "fcc_slurry"},
                    ],
                },
            },
            {
                "submodel": "COKER",
                "label": "Coker",
                "ports": {
                    "in": [{"id": "feed", "stream": "resid"}],
                    "out": [
                        {"id": "naphtha", "stream": "coker_naphtha"},
                        {"id": "gasoil", "stream": "coker_gasoil"},
                    ],
                },
            },
            {
                "submodel": "REFORMER",
                "label": "Reformer",
                "ports": {
                    "in": [{"id": "feed", "stream": "naphtha_heavy"}],
                    "out": [{"id": "reformate", "stream": "reformate"}],
                },
            },
            {
                "submodel": "HYDROTREAT_NAPH",
                "label": "Naphtha HDT",
                "ports": {
                    "in": [{"id": "feed", "stream": "naphtha"}],
                    "out": [{"id": "product", "stream": "naphtha_hdt"}],
                },
            },
            {
                "submodel": "BLENDER",
                "label": "Blender",
                "ports": {
                    "in": [
                        {"id": "comp_a", "stream": "any"},
                        {"id": "comp_b", "stream": "any"},
                        {"id": "comp_c", "stream": "any"},
                    ],
                    "out": [
                        {"id": "gasoline", "stream": "gasoline"},
                        {"id": "diesel", "stream": "diesel"},
                    ],
                },
            },
            {
                "submodel": "SELL",
                "label": "Sell / Product",
                "ports": {
                    "in": [{"id": "product", "stream": "any"}],
                    "out": [],
                },
            },
        ]
    }


def _cluster_stub(nodes: list[GraphNode]) -> list[ClusterStub]:
    """Group active units into ADMM-style clusters (stub: one cluster per submodel type)."""
    by_type: dict[str, list[str]] = {}
    for n in nodes:
        active = bool(n.data.get("active", True))
        if not active:
            continue
        sub = str(n.data.get("submodel") or n.data.get("label") or "UNKNOWN")
        by_type.setdefault(sub, []).append(n.id)
    clusters: list[ClusterStub] = []
    for i, (sub, ids) in enumerate(sorted(by_type.items())):
        clusters.append(
            ClusterStub(
                id=f"cluster_{i}_{sub.lower()}",
                unit_ids=ids,
                submodel_types=[sub],
            )
        )
    return clusters


@app.post("/api/graph", response_model=GraphResponse)
def submit_graph(payload: GraphPayload) -> GraphResponse:
    """
    Accept a SvelteFlow graph JSON and return:
      - cluster decomposition stub (by submodel type, active units only)
      - ADMM status stub (rho, primal/dual residual, max_iter)

    Real ADMM rebuild + solve will replace the stubs later.
    """
    active = [
        n.id
        for n in payload.nodes
        if bool(n.data.get("active", True))
    ]
    clusters = _cluster_stub(payload.nodes)
    notes = [
        "Wave3 scaffold: clusters grouped by submodel type (active only).",
        "ADMM fields are placeholders — wire to pims_admm_llm.admm in a later wave.",
        f"Received {len(payload.nodes)} nodes, {len(payload.edges)} edges.",
    ]
    if not payload.nodes:
        notes.append("Empty graph — drop units from the palette onto the canvas.")

    return GraphResponse(
        ok=True,
        node_count=len(payload.nodes),
        edge_count=len(payload.edges),
        active_units=active,
        clusters=clusters,
        admm=AdmmStatusStub(
            status="stub_idle",
            rho=1.0,
            primal_residual=0.0,
            dual_residual=0.0,
            max_iter=50,
            iteration=0,
            message="ADMM not run (scaffold). POST accepted; rebuild path TBD.",
        ),
        notes=notes,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
