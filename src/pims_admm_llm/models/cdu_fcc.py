"""Focused base-delta cascade: crude → CDU → FCC [→ COKER optional].

Enable units incrementally. Mass balances and product exits are checked before
adding more conversion units to the flowsheet.

  crude → CDU (cut modes) → every product exit
  cdu_gasoil → FCC (severity modes) → every product exit
  [if COKER] cdu_resid swing → COKER (recycle modes) | FO → coker products exit

Missing edges filled by auto_wire_edges_for_units / auto_route.
Process conditions enter as SOS1 mode binaries per unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import pulp

from .auto_route import best_route, complete_missing_edges, guess_route
from .base_delta import (
    auto_wire_edges_for_units,
    build_cdu_base_delta,
    build_coker_base_delta,
    build_fcc_base_delta,
    process_modes_cdu,
    process_modes_coker,
    process_modes_fcc,
)
from .stream_composition import StreamComposition, get_stream


DEFAULT_PRICES: Dict[str, float] = {
    "GASOLINE": 95.0,
    "DIESEL": 90.0,
    "FO": 55.0,
    "LPG": 45.0,
    "FUEL_GAS": 25.0,
    "REGEN_HEAT": 20.0,
    "SELL": 70.0,
    "REFORMER": 85.0,
    "HDT_NAPH": 88.0,
    "COKER": 60.0,
    "POOL_FCC": 75.0,
    "POOL_COKER": 58.0,
    "COKE_SALES": 15.0,
    "H2_GRID": 30.0,
    "BLENDER": 90.0,
    "FCC": 75.0,
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
    cdu_exits: Dict[str, str]
    fcc_exits: Dict[str, str]
    streams: Dict[str, float]
    compositions: Dict[str, Dict[str, Any]]
    process_conditions: Dict[str, Dict[str, Any]]
    auto_routes: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    # coker extensions (empty when disabled)
    coker_mode: str = ""
    coker_yields: Dict[str, float] = field(default_factory=dict)
    coker_exits: Dict[str, str] = field(default_factory=dict)
    mass_balance: Dict[str, Any] = field(default_factory=dict)
    enabled_units: List[str] = field(default_factory=lambda: ["CDU", "FCC"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "crude_kbd": self.crude_kbd,
            "enabled_units": list(self.enabled_units),
            "cdu_mode": self.cdu_mode,
            "fcc_mode": self.fcc_mode,
            "coker_mode": self.coker_mode,
            "cdu_yields": dict(self.cdu_yields),
            "fcc_yields": dict(self.fcc_yields),
            "coker_yields": dict(self.coker_yields),
            "cdu_exits": dict(self.cdu_exits),
            "fcc_exits": dict(self.fcc_exits),
            "coker_exits": dict(self.coker_exits),
            "streams": dict(self.streams),
            "compositions": self.compositions,
            "process_conditions": self.process_conditions,
            "auto_routes": list(self.auto_routes),
            "mass_balance": dict(self.mass_balance),
            "meta": dict(self.meta),
        }


def _exit_map_for_model(exits: List[Any], compositions: Mapping[str, StreamComposition]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in exits:
        stream = e.stream if hasattr(e, "stream") else e["stream"]
        default = e.default_sink if hasattr(e, "default_sink") else e["default_sink"]
        comp = compositions.get(stream) or get_stream(stream)
        ranked = {g.sink: g.score for g in guess_route(comp, top_k=12, min_score=0.0)}
        if ranked.get(default, 0.0) >= 0.35:
            out[stream] = default
        else:
            out[stream] = best_route(comp).sink
    return out


def _mode_linked_product(
    prob: pulp.LpProblem,
    name: str,
    feed: pulp.LpVariable,
    modes: List[Dict[str, Any]],
    y_mode: Dict[str, pulp.LpVariable],
    product: str,
    M: float,
) -> pulp.LpVariable:
    """prod = sum_m yield[m,p] * feed * y_m via big-M pieces."""
    total = pulp.LpVariable(f"{name}_{product}", lowBound=0)
    pieces = []
    for m in modes:
        mid = m["id"]
        yld = float(m["yields"][product])
        r = pulp.LpVariable(f"{name}_{mid}_{product}", lowBound=0)
        prob += r <= M * y_mode[mid]
        prob += r <= yld * feed
        prob += r >= yld * feed - M * (1 - y_mode[mid])
        pieces.append(r)
    prob += total == pulp.lpSum(pieces)
    return total


def _compute_mass_balance(
    streams: Mapping[str, float],
    *,
    coker_enabled: bool,
    cdu_yields: Mapping[str, float],
    fcc_yields: Mapping[str, float],
    coker_yields: Mapping[str, float],
) -> Dict[str, Any]:
    """Report balance residuals (planning vol basis; coke wt separate)."""
    crude = float(streams.get("crude", 0.0))
    cdu_liq = [
        "cdu_naphtha_light",
        "cdu_naphtha_heavy",
        "cdu_distillate",
        "cdu_gasoil",
        "cdu_resid",
    ]
    cdu_sum = sum(float(streams.get(p, 0.0)) for p in cdu_liq)
    cdu_gap = abs(cdu_sum - crude)

    fcc_feed = float(streams.get("fcc_feed", streams.get("cdu_gasoil", 0.0)))
    fcc_liq = ["fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry"]
    fcc_out = sum(float(streams.get(p, 0.0)) for p in fcc_liq)
    fcc_coke = float(streams.get("fcc_coke", 0.0))
    # liquids should ≈ fcc_feed * sum(liquid yields); coke wt on feed
    y_fcc_liq = sum(float(fcc_yields.get(p, 0.0)) for p in fcc_liq)
    y_fcc_coke = float(fcc_yields.get("fcc_coke", 0.0))
    fcc_liq_gap = abs(fcc_out - fcc_feed * y_fcc_liq)
    fcc_coke_gap = abs(fcc_coke - fcc_feed * y_fcc_coke)
    go_feed_gap = abs(fcc_feed - float(streams.get("cdu_gasoil", 0.0)))

    checks = {
        "cdu_liquid_vs_crude": {"value": cdu_sum, "target": crude, "abs_gap": cdu_gap, "ok": cdu_gap < 1e-3 * max(1.0, crude) + 1e-4},
        "fcc_feed_vs_gasoil": {"value": fcc_feed, "target": float(streams.get("cdu_gasoil", 0.0)), "abs_gap": go_feed_gap, "ok": go_feed_gap < 1e-3 * max(1.0, crude) + 1e-4},
        "fcc_liquids": {"value": fcc_out, "target": fcc_feed * y_fcc_liq, "abs_gap": fcc_liq_gap, "ok": fcc_liq_gap < 1e-2 * max(1.0, fcc_feed) + 1e-3},
        "fcc_coke_wt": {"value": fcc_coke, "target": fcc_feed * y_fcc_coke, "abs_gap": fcc_coke_gap, "ok": fcc_coke_gap < 1e-2 * max(1.0, fcc_feed) + 1e-3},
    }

    if coker_enabled:
        resid = float(streams.get("cdu_resid", 0.0))
        to_coker = float(streams.get("resid_to_coker", 0.0))
        to_fo = float(streams.get("resid_to_fo", 0.0))
        resid_split_gap = abs(to_coker + to_fo - resid)
        coker_feed = float(streams.get("coker_feed", to_coker))
        coker_liq = ["coker_dry_gas", "coker_lpg", "coker_naphtha", "coker_gasoil"]
        coker_out = sum(float(streams.get(p, 0.0)) for p in coker_liq)
        coker_coke = float(streams.get("coker_coke", 0.0))
        y_ck_liq = sum(float(coker_yields.get(p, 0.0)) for p in coker_liq)
        y_ck_coke = float(coker_yields.get("coker_coke", 0.0))
        ck_liq_gap = abs(coker_out - coker_feed * y_ck_liq)
        ck_coke_gap = abs(coker_coke - coker_feed * y_ck_coke)
        checks["resid_split"] = {
            "value": to_coker + to_fo,
            "target": resid,
            "abs_gap": resid_split_gap,
            "ok": resid_split_gap < 1e-3 * max(1.0, resid) + 1e-4,
        }
        checks["coker_feed"] = {
            "value": coker_feed,
            "target": to_coker,
            "abs_gap": abs(coker_feed - to_coker),
            "ok": abs(coker_feed - to_coker) < 1e-4,
        }
        checks["coker_liquids"] = {
            "value": coker_out,
            "target": coker_feed * y_ck_liq,
            "abs_gap": ck_liq_gap,
            "ok": ck_liq_gap < 1e-2 * max(1.0, coker_feed) + 1e-3,
        }
        checks["coker_coke_wt"] = {
            "value": coker_coke,
            "target": coker_feed * y_ck_coke,
            "abs_gap": ck_coke_gap,
            "ok": ck_coke_gap < 1e-2 * max(1.0, coker_feed) + 1e-3,
        }

    all_ok = all(c["ok"] for c in checks.values())
    return {"ok": all_ok, "checks": checks}


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
    fix_coker_mode: Optional[str] = None,
    enable_coker: bool = False,
    active_units: Optional[Sequence[str]] = None,
    msg: bool = False,
) -> CduFccResult:
    """Solve base-delta cascade. Pass enable_coker=True or active_units including COKER."""
    units: Set[str] = {"CDU", "FCC"}
    if active_units is not None:
        units = {u.upper() for u in active_units}
        units.add("CDU")  # tower always present
    if enable_coker:
        units.add("COKER")
    coker_on = "COKER" in units
    if "FCC" not in units:
        units.add("FCC")  # cascade MVP always has FCC for gasoil

    price = dict(DEFAULT_PRICES)
    if prices:
        price.update(prices)

    crude_feed = {"api": crude_api, "sulfur_wt": crude_sulfur_wt, "ccr_wt": crude_ccr_wt}
    cdu_model = build_cdu_base_delta(reference_feed=crude_feed)
    cdu_modes = process_modes_cdu(cdu_model, crude_feed)

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

    resid_comp = get_stream(
        "cdu_resid",
        {
            "api": max(5.0, crude_api - 18.0),
            "sulfur_wt": crude_sulfur_wt * 1.4,
            "ccr_wt": max(5.0, crude_ccr_wt * 2.2),
        },
    )
    resid_feed = {
        "api": resid_comp.api,
        "sulfur_wt": resid_comp.sulfur_wt,
        "ccr_wt": resid_comp.ccr_wt,
        "nitrogen_ppm": resid_comp.nitrogen_ppm,
        "asphaltenes_wt": 6.0,
    }
    coker_model = build_coker_base_delta(reference_feed=resid_feed) if coker_on else None
    coker_modes = process_modes_coker(coker_model, resid_feed) if coker_on and coker_model else []

    cdu_mid = next(m for m in cdu_modes if m["id"] == "cuts_mid")
    fcc_mid = next(m for m in fcc_modes if m["id"] == "rot_mid")
    cdu_comps = {k: StreamComposition.from_mapping(k, v) for k, v in cdu_mid["compositions"].items()}
    fcc_comps = {k: StreamComposition.from_mapping(k, v) for k, v in fcc_mid["compositions"].items()}
    coker_comps: Dict[str, StreamComposition] = {}
    if coker_on and coker_modes:
        ck_mid = next(m for m in coker_modes if m["id"] == "rec_mid")
        coker_comps = {k: StreamComposition.from_mapping(k, v) for k, v in ck_mid["compositions"].items()}

    drawn = list(drawn_edges or [])
    auto_edges = auto_wire_edges_for_units(sorted(units), drawn)
    produced = list(cdu_model.products) + list(fcc_model.products)
    if coker_on and coker_model:
        produced += list(coker_model.products)
    auto = complete_missing_edges(
        produced,
        drawn + auto_edges,
        compositions={**cdu_comps, **fcc_comps, **coker_comps},
    )
    # Prefer structural auto edges in auto_routes list
    auto_routes = auto_edges + auto

    cdu_exits = _exit_map_for_model(cdu_model.exits, cdu_comps)
    fcc_exits = _exit_map_for_model(fcc_model.exits, fcc_comps)
    coker_exits: Dict[str, str] = {}
    if coker_on and coker_model:
        coker_exits = _exit_map_for_model(coker_model.exits, coker_comps)
        cdu_exits["cdu_resid"] = "COKER"  # feed default when coker on; FO is swing var

    for e in drawn:
        s = str(e.get("stream", ""))
        to = str(e.get("to", ""))
        if not s or not to:
            continue
        if s in cdu_exits:
            cdu_exits[s] = to
        if s in fcc_exits:
            fcc_exits[s] = to
        if s in coker_exits:
            coker_exits[s] = to

    # --- LP ---
    prob = pulp.LpProblem("cdu_fcc_coker_base_delta", pulp.LpMaximize)
    crude = pulp.LpVariable("crude_kbd", lowBound=min_crude_kbd, upBound=max_crude_kbd)
    M = max_crude_kbd * 1.2

    y_cdu = {m["id"]: pulp.LpVariable(f"cdu_mode_{m['id']}", cat="Binary") for m in cdu_modes}
    y_fcc = {m["id"]: pulp.LpVariable(f"fcc_mode_{m['id']}", cat="Binary") for m in fcc_modes}
    prob += pulp.lpSum(y_cdu.values()) == 1, "cdu_sos1"
    prob += pulp.lpSum(y_fcc.values()) == 1, "fcc_sos1"
    if fix_cdu_mode:
        prob += y_cdu[fix_cdu_mode] == 1
    if fix_fcc_mode:
        prob += y_fcc[fix_fcc_mode] == 1

    cdu_prod = {
        p: _mode_linked_product(prob, "cdu", crude, cdu_modes, y_cdu, p, M)
        for p in cdu_model.products
    }

    fcc_feed = cdu_prod["cdu_gasoil"]
    fcc_prod = {
        p: _mode_linked_product(prob, "fcc", fcc_feed, fcc_modes, y_fcc, p, M)
        for p in fcc_model.products
    }

    coker_prod: Dict[str, pulp.LpVariable] = {}
    y_coker: Dict[str, pulp.LpVariable] = {}
    resid_to_coker = None
    resid_to_fo = None
    coker_feed_var = None
    if coker_on and coker_model:
        y_coker = {m["id"]: pulp.LpVariable(f"coker_mode_{m['id']}", cat="Binary") for m in coker_modes}
        prob += pulp.lpSum(y_coker.values()) == 1, "coker_sos1"
        if fix_coker_mode:
            if fix_coker_mode not in y_coker:
                raise ValueError(f"unknown coker mode {fix_coker_mode}")
            prob += y_coker[fix_coker_mode] == 1

        resid_to_coker = pulp.LpVariable("resid_to_coker", lowBound=0)
        resid_to_fo = pulp.LpVariable("resid_to_fo", lowBound=0)
        # resid balance
        prob += resid_to_coker + resid_to_fo == cdu_prod["cdu_resid"], "resid_split"
        coker_feed_var = resid_to_coker
        coker_prod = {
            p: _mode_linked_product(prob, "coker", coker_feed_var, coker_modes, y_coker, p, M)
            for p in coker_model.products
        }

    # Objective
    revenue_terms = []
    for p, var in cdu_prod.items():
        if p == "cdu_gasoil":
            continue  # valued via FCC
        if p == "cdu_resid" and coker_on:
            continue  # valued via coker products + FO swing
        sink = cdu_exits[p]
        revenue_terms.append(price.get(sink, price["SELL"]) * var)
    if coker_on and resid_to_fo is not None:
        revenue_terms.append(price["FO"] * resid_to_fo)
    for p, var in fcc_prod.items():
        sink = fcc_exits[p]
        revenue_terms.append(price.get(sink, price["SELL"]) * var)
    if coker_on:
        for p, var in coker_prod.items():
            sink = coker_exits[p]
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
    coker_mode = _pick_mode(y_coker) if y_coker else ""
    cdu_mode_row = next(m for m in cdu_modes if m["id"] == cdu_mode)
    fcc_mode_row = next(m for m in fcc_modes if m["id"] == fcc_mode)
    coker_mode_row = next((m for m in coker_modes if m["id"] == coker_mode), None)

    streams: Dict[str, float] = {"crude": _val(crude)}
    for p, var in cdu_prod.items():
        streams[p] = _val(var)
    for p, var in fcc_prod.items():
        streams[p] = _val(var)
    streams["fcc_feed"] = _val(fcc_feed)
    if coker_on and resid_to_coker is not None and resid_to_fo is not None:
        streams["resid_to_coker"] = _val(resid_to_coker)
        streams["resid_to_fo"] = _val(resid_to_fo)
        streams["coker_feed"] = _val(coker_feed_var) if coker_feed_var is not None else 0.0
        for p, var in coker_prod.items():
            streams[p] = _val(var)

    compositions = {
        **cdu_mode_row["compositions"],
        **fcc_mode_row["compositions"],
        "cdu_gasoil_feed_to_fcc": go_comp.to_dict(),
    }
    process_conditions: Dict[str, Dict[str, Any]] = {
        "CDU": dict(cdu_mode_row["conditions"]),
        "FCC": dict(fcc_mode_row["conditions"]),
    }
    coker_yields: Dict[str, float] = {}
    if coker_mode_row is not None:
        compositions.update(coker_mode_row["compositions"])
        compositions["cdu_resid_feed_to_coker"] = resid_comp.to_dict()
        process_conditions["COKER"] = dict(coker_mode_row["conditions"])
        coker_yields = dict(coker_mode_row["yields"])

    mb = _compute_mass_balance(
        streams,
        coker_enabled=coker_on,
        cdu_yields=dict(cdu_mode_row["yields"]),
        fcc_yields=dict(fcc_mode_row["yields"]),
        coker_yields=coker_yields,
    )

    return CduFccResult(
        status=status,
        objective=obj,
        crude_kbd=streams["crude"],
        cdu_mode=cdu_mode,
        fcc_mode=fcc_mode,
        coker_mode=coker_mode,
        cdu_yields=dict(cdu_mode_row["yields"]),
        fcc_yields=dict(fcc_mode_row["yields"]),
        coker_yields=coker_yields,
        cdu_exits=cdu_exits,
        fcc_exits=fcc_exits,
        coker_exits=coker_exits,
        streams=streams,
        compositions=compositions,
        process_conditions=process_conditions,
        auto_routes=auto_routes,
        mass_balance=mb,
        enabled_units=sorted(units),
        meta={
            "model": "cdu_fcc_coker_base_delta" if coker_on else "cdu_fcc_base_delta",
            "crude_feed": crude_feed,
            "gasoil_feed": go_feed,
            "resid_feed": resid_feed if coker_on else None,
            "prices": price,
            "cdu_modes": [m["id"] for m in cdu_modes],
            "fcc_modes": [m["id"] for m in fcc_modes],
            "coker_modes": [m["id"] for m in coker_modes] if coker_on else [],
            "every_product_has_exit": True,
            "products_cdu": list(cdu_model.products),
            "products_fcc": list(fcc_model.products),
            "products_coker": list(coker_model.products) if coker_model else [],
            "mass_balance_ok": mb["ok"],
        },
    )


def solve_cdu_fcc_coker(**kwargs: Any) -> CduFccResult:
    """Explicit alias: always enable coker."""
    kwargs = dict(kwargs)
    kwargs["enable_coker"] = True
    return solve_cdu_fcc(**kwargs)
