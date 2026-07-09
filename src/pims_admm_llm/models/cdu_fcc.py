"""Focused crude tower → FCC plant using BASE/DELTA unit submodels.

Intentionally smaller than full_plant:
  crude → CDU (process modes) → every product exit
  cdu_gasoil → FCC (process modes) → every product exit

Missing flowsheet edges are filled by property-based auto_route.
Process conditions enter the LP as SOS1 mode binaries (severity / cut bands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pulp

from .auto_route import best_route, complete_missing_edges
from .base_delta import (
    build_cdu_base_delta,
    build_fcc_base_delta,
    process_modes_cdu,
    process_modes_fcc,
)
from .stream_composition import StreamComposition, get_stream


# Product values ($/bbl planning) — crude cost separate
DEFAULT_PRICES: Dict[str, float] = {
    "GASOLINE": 95.0,
    "DIESEL": 90.0,
    "FO": 55.0,
    "LPG": 45.0,
    "FUEL_GAS": 25.0,
    "REGEN_HEAT": 20.0,  # coke heat credit equiv
    "SELL": 70.0,
    "REFORMER": 85.0,  # intermediate value of heavy naphtha to reformer
    "HDT_NAPH": 88.0,
    "COKER": 60.0,
    "POOL_FCC": 75.0,
    "POOL_COKER": 58.0,
    "COKE_SALES": 15.0,
    "H2_GRID": 30.0,
    "BLENDER": 90.0,
}

CRUDE_COST = 70.0  # $/bbl


@dataclass
class CduFccResult:
    status: str
    objective: float
    crude_kbd: float
    cdu_mode: str
    fcc_mode: str
    cdu_yields: Dict[str, float]
    fcc_yields: Dict[str, float]
    cdu_exits: Dict[str, str]  # stream → sink
    fcc_exits: Dict[str, str]
    streams: Dict[str, float]
    compositions: Dict[str, Dict[str, Any]]
    process_conditions: Dict[str, Dict[str, Any]]
    auto_routes: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "crude_kbd": self.crude_kbd,
            "cdu_mode": self.cdu_mode,
            "fcc_mode": self.fcc_mode,
            "cdu_yields": dict(self.cdu_yields),
            "fcc_yields": dict(self.fcc_yields),
            "cdu_exits": dict(self.cdu_exits),
            "fcc_exits": dict(self.fcc_exits),
            "streams": dict(self.streams),
            "compositions": self.compositions,
            "process_conditions": self.process_conditions,
            "auto_routes": list(self.auto_routes),
            "meta": dict(self.meta),
        }


def _exit_map_for_model(exits: List[Any], compositions: Mapping[str, StreamComposition]) -> Dict[str, str]:
    """Prefer model ProductExit.default_sink; allow auto_route override check."""
    out: Dict[str, str] = {}
    for e in exits:
        stream = e.stream if hasattr(e, "stream") else e["stream"]
        default = e.default_sink if hasattr(e, "default_sink") else e["default_sink"]
        # Validate / refine with auto_route
        comp = compositions.get(stream) or get_stream(stream)
        guess = best_route(comp)
        # If default is close to best, keep default; if default is absurdly low, use guess
        from .auto_route import guess_route

        ranked = {g.sink: g.score for g in guess_route(comp, top_k=12, min_score=0.0)}
        if ranked.get(default, 0.0) >= 0.35:
            out[stream] = default
        else:
            out[stream] = guess.sink
    return out


def solve_cdu_fcc(
    *,
    crude_api: float = 30.0,
    crude_sulfur_wt: float = 1.0,
    crude_ccr_wt: float = 2.0,
    max_crude_kbd: float = 100.0,
    min_crude_kbd: float = 0.0,
    prices: Optional[Mapping[str, float]] = None,
    drawn_edges: Optional[List[Mapping[str, Any]]] = None,
    fix_cdu_mode: Optional[str] = None,
    fix_fcc_mode: Optional[str] = None,
    msg: bool = False,
) -> CduFccResult:
    """Solve crude → CDU → (exits) + gasoil → FCC → (exits) with process-mode SOS1."""
    price = dict(DEFAULT_PRICES)
    if prices:
        price.update(prices)

    crude_feed = {"api": crude_api, "sulfur_wt": crude_sulfur_wt, "ccr_wt": crude_ccr_wt}
    cdu_model = build_cdu_base_delta(reference_feed=crude_feed)
    cdu_modes = process_modes_cdu(cdu_model, crude_feed)

    # Gasoil composition for FCC feed depends lightly on crude
    go_comp = get_stream(
        "cdu_gasoil",
        {
            "api": max(15.0, crude_api - 8.0),
            "sulfur_wt": crude_sulfur_wt * 1.1,
            "ccr_wt": max(0.2, crude_ccr_wt * 0.25),
        },
    )
    go_feed = {
        "api": go_comp.api,
        "sulfur_wt": go_comp.sulfur_wt,
        "ccr_wt": go_comp.ccr_wt,
        "nitrogen_ppm": go_comp.nitrogen_ppm,
        "metals_ni_v_ppm": go_comp.metals_ni_v_ppm,
    }
    fcc_model = build_fcc_base_delta(reference_feed=go_feed)
    fcc_modes = process_modes_fcc(fcc_model, go_feed)

    # Exit maps from compositions at mid modes (refined by auto_route)
    cdu_mid = next(m for m in cdu_modes if m["id"] == "cuts_mid")
    fcc_mid = next(m for m in fcc_modes if m["id"] == "rot_mid")
    cdu_comps = {k: StreamComposition.from_mapping(k, v) for k, v in cdu_mid["compositions"].items()}
    fcc_comps = {k: StreamComposition.from_mapping(k, v) for k, v in fcc_mid["compositions"].items()}
    # Prefer user-drawn edges when present
    drawn = list(drawn_edges or [])
    auto = complete_missing_edges(
        list(cdu_model.products) + list(fcc_model.products),
        drawn,
        compositions={**cdu_comps, **fcc_comps},
    )
    cdu_exits = _exit_map_for_model(cdu_model.exits, cdu_comps)
    fcc_exits = _exit_map_for_model(fcc_model.exits, fcc_comps)
    for a in auto:
        s = a["stream"]
        if s in cdu_exits and not any(e.get("stream") == s for e in drawn):
            # already set by exit map; auto list documents the guess
            pass
        if s in fcc_exits and not any(e.get("stream") == s for e in drawn):
            pass
    # Apply drawn edges overrides
    for e in drawn:
        s = str(e.get("stream", ""))
        to = str(e.get("to", ""))
        if s in cdu_exits and to:
            cdu_exits[s] = to
        if s in fcc_exits and to:
            fcc_exits[s] = to

    # --- LP ---
    prob = pulp.LpProblem("cdu_fcc_base_delta", pulp.LpMaximize)
    crude = pulp.LpVariable("crude_kbd", lowBound=min_crude_kbd, upBound=max_crude_kbd)

    y_cdu = {
        m["id"]: pulp.LpVariable(f"cdu_mode_{m['id']}", cat="Binary") for m in cdu_modes
    }
    y_fcc = {
        m["id"]: pulp.LpVariable(f"fcc_mode_{m['id']}", cat="Binary") for m in fcc_modes
    }
    prob += pulp.lpSum(y_cdu.values()) == 1, "cdu_sos1"
    prob += pulp.lpSum(y_fcc.values()) == 1, "fcc_sos1"
    if fix_cdu_mode:
        if fix_cdu_mode not in y_cdu:
            raise ValueError(f"unknown cdu mode {fix_cdu_mode}")
        prob += y_cdu[fix_cdu_mode] == 1
    if fix_fcc_mode:
        if fix_fcc_mode not in y_fcc:
            raise ValueError(f"unknown fcc mode {fix_fcc_mode}")
        prob += y_fcc[fix_fcc_mode] == 1

    # CDU product rates: sum_m y_m * yield_{m,p} * crude
    # Linear: r_{m,p} <= M * y_m; r_{m,p} <= yield * crude; r_{m,p} >= yield*crude - M(1-y)
    M = max_crude_kbd * 1.2
    cdu_prod: Dict[str, pulp.LpVariable] = {}
    for p in cdu_model.products:
        cdu_prod[p] = pulp.LpVariable(f"cdu_prod_{p}", lowBound=0)
        # cdu_prod[p] = sum_m yield[m,p] * crude * y_m  → use mode-linked pieces
        pieces = []
        for m in cdu_modes:
            mid = m["id"]
            yld = float(m["yields"][p])
            r = pulp.LpVariable(f"cdu_{mid}_{p}", lowBound=0)
            # r <= M * y
            prob += r <= M * y_cdu[mid]
            # r <= yld * crude
            prob += r <= yld * crude
            # r >= yld * crude - M*(1-y)
            prob += r >= yld * crude - M * (1 - y_cdu[mid])
            pieces.append(r)
        prob += cdu_prod[p] == pulp.lpSum(pieces)

    # FCC feed = cdu_gasoil production (all gasoil to FCC in this focused plant)
    fcc_feed = cdu_prod["cdu_gasoil"]
    fcc_prod: Dict[str, pulp.LpVariable] = {}
    for p in fcc_model.products:
        fcc_prod[p] = pulp.LpVariable(f"fcc_prod_{p}", lowBound=0)
        pieces = []
        for m in fcc_modes:
            mid = m["id"]
            yld = float(m["yields"][p])
            r = pulp.LpVariable(f"fcc_{mid}_{p}", lowBound=0)
            prob += r <= M * y_fcc[mid]
            # r <= yld * fcc_feed
            prob += r <= yld * fcc_feed
            prob += r >= yld * fcc_feed - M * (1 - y_fcc[mid])
            pieces.append(r)
        prob += fcc_prod[p] == pulp.lpSum(pieces)

    # Objective: product credits − crude cost
    revenue_terms = []
    for p, var in cdu_prod.items():
        if p == "cdu_gasoil":
            # valued via FCC products, not double count; small FO option not used
            continue
        sink = cdu_exits[p]
        revenue_terms.append(price.get(sink, price["SELL"]) * var)
    for p, var in fcc_prod.items():
        sink = fcc_exits[p]
        revenue_terms.append(price.get(sink, price["SELL"]) * var)

    prob += pulp.lpSum(revenue_terms) - CRUDE_COST * crude

    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus.get(status_code, str(status_code))

    def _val(x: Any) -> float:
        v = pulp.value(x)
        if v is None:
            return 0.0
        return float(v)

    obj = _val(prob.objective)

    def _pick_mode(ys: Dict[str, pulp.LpVariable]) -> str:
        best, best_v = "", -1.0
        for k, v in ys.items():
            val = _val(v)
            if val > best_v:
                best, best_v = k, val
        return best

    cdu_mode = _pick_mode(y_cdu)
    fcc_mode = _pick_mode(y_fcc)
    cdu_mode_row = next(m for m in cdu_modes if m["id"] == cdu_mode)
    fcc_mode_row = next(m for m in fcc_modes if m["id"] == fcc_mode)

    streams: Dict[str, float] = {"crude": _val(crude)}
    for p, var in cdu_prod.items():
        streams[p] = _val(var)
    for p, var in fcc_prod.items():
        streams[p] = _val(var)
    streams["fcc_feed"] = _val(fcc_feed)

    # Compositions at chosen modes
    compositions = {
        **cdu_mode_row["compositions"],
        **fcc_mode_row["compositions"],
        "cdu_gasoil_feed_to_fcc": go_comp.to_dict(),
    }

    return CduFccResult(
        status=status,
        objective=obj,
        crude_kbd=streams["crude"],
        cdu_mode=cdu_mode,
        fcc_mode=fcc_mode,
        cdu_yields=dict(cdu_mode_row["yields"]),
        fcc_yields=dict(fcc_mode_row["yields"]),
        cdu_exits=cdu_exits,
        fcc_exits=fcc_exits,
        streams=streams,
        compositions=compositions,
        process_conditions={
            "CDU": dict(cdu_mode_row["conditions"]),
            "FCC": dict(fcc_mode_row["conditions"]),
        },
        auto_routes=auto,
        meta={
            "model": "cdu_fcc_base_delta",
            "crude_feed": crude_feed,
            "gasoil_feed": go_feed,
            "prices": price,
            "cdu_modes": [m["id"] for m in cdu_modes],
            "fcc_modes": [m["id"] for m in fcc_modes],
            "every_product_has_exit": True,
            "products_cdu": list(cdu_model.products),
            "products_fcc": list(fcc_model.products),
        },
    )
