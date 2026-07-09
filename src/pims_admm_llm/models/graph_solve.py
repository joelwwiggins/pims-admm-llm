"""Map SvelteFlow graph → routing overlay → full plant LP / ADMM.

Used by FastAPI ``POST /api/graph`` (issue #1).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .assay_loader import load_assays_json, load_routing
from .full_plant import admm_price_directed_plant, solve_full_plant


def _norm_unit(raw: Optional[str]) -> str:
    if not raw:
        return ""
    t = str(raw).strip()
    aliases = {
        "HYDROTREAT_NAPH": "HDT_NAPH",
        "HDT": "HDT_NAPH",
        "Tank": "TANK",
        "tank": "TANK",
        "Coker": "COKER",
        "coker": "COKER",
    }
    if t in aliases:
        return aliases[t]
    if t.upper().startswith("TANK"):
        # preserve specific tank ids when present (TANK_GASOIL); generic → TANK
        up = t.upper()
        return up if up != "TANK" else "TANK"
    up = t.upper()
    if up in ("CDU", "FCC", "COKER", "REFORMER", "BLENDER", "SELL", "HDT_NAPH"):
        return up
    return t


# Stream → process-unit producers (planning-grade; graph edge matching).
_STREAM_PRODUCERS: Dict[str, Set[str]] = {
    "cdu_gasoil": {"CDU"},
    "cdu_resid": {"CDU"},
    "cdu_naphtha": {"CDU"},
    "cdu_naphtha_light": {"CDU"},
    "cdu_naphtha_heavy": {"CDU"},
    "cdu_distillate": {"CDU"},
    "fcc_naphtha": {"FCC"},
    "fcc_lco": {"FCC"},
    "fcc_slurry": {"FCC"},
    "coker_naphtha": {"COKER"},
    "coker_gasoil": {"COKER"},
    "reformate": {"REFORMER"},
}


def _arc_endpoint_type(name: str) -> str:
    n = str(name)
    if n.upper().startswith("TANK"):
        return n.upper() if n.upper().startswith("TANK_") else "TANK"
    if n in ("HYDROTREAT_NAPH", "HDT_NAPH"):
        return "HDT_NAPH"
    return n


def _is_tank(name: str) -> bool:
    return str(name).upper().startswith("TANK")


def _is_passthrough(name: str) -> bool:
    n = str(name)
    return _is_tank(n) or n in (
        "SELL",
        "warehouse",
        "transport",
        "HDT_NAPH",
        "HYDROTREAT_NAPH",
    )


def extract_active_units(nodes: Iterable[dict]) -> Set[str]:
    units: Set[str] = set()
    for n in nodes:
        data = n.get("data") or {}
        if not bool(data.get("active", True)):
            continue
        ut = _norm_unit(data.get("unitType") or data.get("label") or n.get("type"))
        if ut:
            units.add(ut)
    return units


def extract_type_edges(nodes: Iterable[dict], edges: Iterable[dict]) -> Set[Tuple[str, str]]:
    id_to_type: Dict[str, str] = {}
    for n in nodes:
        data = n.get("data") or {}
        if not bool(data.get("active", True)):
            continue
        ut = _norm_unit(data.get("unitType") or data.get("label") or n.get("type"))
        if ut:
            id_to_type[str(n.get("id", ""))] = ut
    pairs: Set[Tuple[str, str]] = set()
    for e in edges:
        s = id_to_type.get(str(e.get("source", "")))
        t = id_to_type.get(str(e.get("target", "")))
        if s and t:
            pairs.add((s, t))
    # expand through tanks: A→TANK* and TANK*→B ⇒ A→B
    from_to_tank = {s for s, t in pairs if _is_tank(t)}
    tank_to = {t for s, t in pairs if _is_tank(s)}
    expanded = set(pairs)
    for a in from_to_tank:
        for b in tank_to:
            expanded.add((a, b))
    return expanded


def _edge_matches_arc(
    fr: str,
    to: str,
    stream: str,
    expanded: Set[Tuple[str, str]],
    process: Set[str],
) -> bool:
    """True if canvas edges support this decision arc (stream-aware).

    Critical: FCC→BLENDER must NOT open fcc_naph_to_reformer; only FCC→REFORMER does.
    """
    producers = set(_STREAM_PRODUCERS.get(stream, set()))

    # Direct non-tank unit endpoints (e.g. CDU→REFORMER for sr_heavy_to_reformer)
    if not _is_tank(fr):
        if (fr, to) in expanded:
            return True
        # unit → passthrough consumer (SELL/HDT)
        if _is_passthrough(to) and (fr, to) in expanded:
            return True
        return False

    # Tank-origin decision arcs: require producer→consumer for THIS stream only
    if not producers:
        return False
    if _is_passthrough(to) and to not in ("SELL", "HDT_NAPH", "HYDROTREAT_NAPH") and _is_tank(to):
        # tank→tank rare; require explicit producer edge into that tank class
        return any((p, to) in expanded for p in producers if p in process)

    for p in producers:
        if p not in process:
            continue
        if (p, to) in expanded:
            return True
    return False


def routing_from_graph(
    nodes: List[dict],
    edges: List[dict],
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Clone base routing; open/close decision arcs from the graph topology."""
    routing = copy.deepcopy(base or load_routing())
    units = extract_active_units(nodes)
    process = units - {"warehouse", "transport"}
    expanded = extract_type_edges(nodes, edges)

    if not process:
        routing["graph_driven"] = False
        return routing

    new_arcs = []
    for a in routing.get("arcs") or []:
        a = dict(a)
        fr = _arc_endpoint_type(a.get("from", ""))
        to = _arc_endpoint_type(a.get("to", ""))
        stream = str(a.get("stream") or "")

        # non-decision production arcs stay open if related process units exist
        if not a.get("decision", False):
            new_arcs.append(a)
            continue

        # endpoints must be relevant to canvas process units
        fr_need = None if _is_passthrough(fr) else fr
        to_need = None if _is_passthrough(to) else to
        if fr_need and fr_need not in process:
            a["default_open"] = False
            new_arcs.append(a)
            continue
        if to_need and to_need not in process:
            a["default_open"] = False
            new_arcs.append(a)
            continue

        if not expanded:
            # units only: keep chemical defaults
            new_arcs.append(a)
            continue

        # Hybrid edge policy:
        # - chemical defaults (base default_open=True) stay open when units present
        #   so missing COKER→BLENDER does not zero the whole plant
        # - non-default arcs (default_open=False, e.g. fcc_naph→reformer, go_to_sell)
        #   open only when stream-aware edges match
        if a.get("default_open", True):
            # optional: also allow explicit edge to reaffirm
            a["default_open"] = True
        else:
            a["default_open"] = _edge_matches_arc(fr, to, stream, expanded, process)
        new_arcs.append(a)

    routing["arcs"] = new_arcs
    routing["graph_units"] = sorted(process)
    routing["graph_edges"] = [list(p) for p in sorted(expanded)]
    routing["graph_driven"] = True
    return routing


def solve_from_graph(
    nodes: List[dict],
    edges: List[dict],
    *,
    recovery_path: str = "mono-oracle",
    inventory_mode: Optional[bool] = None,
    run_admm: bool = True,
) -> Dict[str, Any]:
    """Build routing from graph, solve mono LP, optional ADMM metrics.

    Top-level fields for API consumers (issue #1):
      objective, unit_feeds, products, routing_splits, duals,
      rho, residuals, dual_recovery_path
    """
    assays = load_assays_json()
    routing = routing_from_graph(nodes, edges)
    units = extract_active_units(nodes)
    process = units - {"warehouse", "transport"}

    mono = solve_full_plant(
        assays,
        routing=routing,
        inventory_mode=inventory_mode,
    )
    duals = {
        k: float(v)
        for k, v in (mono.duals or {}).items()
        if abs(float(v)) > 1e-12
    }
    out: Dict[str, Any] = {
        "ok": mono.feasible,
        "admm_status": "solved" if mono.feasible else "infeasible",
        "message": (
            f"graph-driven LP: {len(nodes)} nodes, {len(edges)} edges, "
            f"units={sorted(process)}"
        ),
        "feasible": mono.feasible,
        "status": mono.status,
        "objective": mono.objective,
        "unit_feeds": mono.unit_feeds,
        "products": mono.products,
        "routing_splits": mono.routing_splits,
        "duals": duals,
        "rho": None,
        "residuals": {"primal": None, "dual": None},
        "dual_recovery_path": None,
        "arc_flows": {k: v for k, v in mono.arc_flows.items() if abs(v) > 1e-9},
        "quality_duals": mono.quality_duals,
        "economic_shadows": mono.economic_shadows,
        "solve_time_s": mono.solve_time_s,
        "inventory_mode": mono.inventory_mode,
        "routing_meta": {
            "version": routing.get("version"),
            "graph_driven": routing.get("graph_driven", False),
            "graph_units": routing.get("graph_units"),
            "n_arcs_open": sum(
                1
                for a in routing.get("arcs") or []
                if a.get("decision") and a.get("default_open", True)
            ),
        },
    }

    if run_admm:
        admm = admm_price_directed_plant(
            assays,
            recovery_path=recovery_path,
            routing=routing,
        )
        path = admm.get("dual_recovery_path")
        rho = admm.get("rho")
        r_norm = admm.get("primal_residual_norm")
        s_norm = admm.get("dual_residual_norm")
        recovered = admm.get("duals_like_monolithic") or {}
        if recovered:
            duals = {k: float(v) for k, v in recovered.items() if abs(float(v)) > 1e-12}
            out["duals"] = duals
        out["rho"] = rho
        out["residuals"] = {"primal": r_norm, "dual": s_norm}
        out["dual_recovery_path"] = path
        out["admm"] = {
            "dual_recovery_path": path,
            "rho": rho,
            "max_iter": admm.get("max_iter"),
            "iterations": admm.get("iterations"),
            "primal_residual_norm": r_norm,
            "dual_residual_norm": s_norm,
            "objective": admm.get("objective"),
            "objective_gap_vs_mono": admm.get("objective_gap_vs_mono"),
            "lambda_vs_mono_Linf": admm.get("lambda_vs_mono_Linf"),
            "lambda": admm.get("lambda"),
            "economic_shadow_prices": admm.get("economic_shadow_prices"),
        }
        out["admm_status"] = path or "solved"
        out["message"] += (
            f"; ADMM path={path} "
            f"||r||={r_norm} "
            f"||s||={s_norm} "
            f"λ_vs_mono_L∞={admm.get('lambda_vs_mono_Linf')}"
        )

    return out
