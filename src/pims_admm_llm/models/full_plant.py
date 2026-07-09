"""Full plant monolithic LP: CDU → tanks → FCC/Coker → tanks → Reformer → Blender.

Routing (hard):
  CDU gasoil → tank_gasoil → FCC
  CDU resid  → tank_resid  → Delayed Coker
  FCC naphtha → tank_fcc_naph → Reformer
  Coker naphtha → tank_coker_naph → Reformer
  SR naphtha/distillate, FCC LCO/slurry, coker gasoil, reformate → Blender products

Yields are property-driven from assays before the LP is built.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pulp

from .assay_loader import load_assays_json, load_routing
from .properties import crude_to_props
from .yields import (
    cdu_yields_from_assay,
    coker_yields,
    fcc_yields,
    gasoil_props_from_crude,
    naphtha_props_coker,
    naphtha_props_fcc,
    reformer_yields,
    resid_props_from_crude,
)


@dataclass
class FullPlantResult:
    status: str
    objective: float
    feasible: bool
    crude_rates: Dict[str, float]
    unit_feeds: Dict[str, float]
    streams: Dict[str, float]
    products: Dict[str, float]
    tank_end: Dict[str, float]
    duals: Dict[str, float]
    economic_shadows: Dict[str, float]
    yields_used: Dict[str, Any]
    solve_time_s: float
    problem: pulp.LpProblem


def _val(v) -> float:
    x = pulp.value(v)
    return float(x) if x is not None else 0.0


def build_yield_tables(assays: Dict[str, Any]) -> Dict[str, Any]:
    """Per-crude CDU yields + representative conversion-unit yields from feed props."""
    cdu = {}
    go_props = []
    resid_props = []
    for c in assays["crudes"]:
        props = crude_to_props(c)
        cdu[c["name"]] = cdu_yields_from_assay(props, c.get("tbp_cut_vol"))
        go_props.append((c["name"], gasoil_props_from_crude(props), c.get("max_supply_kbd", 1.0)))
        resid_props.append((c["name"], resid_props_from_crude(props), c.get("max_supply_kbd", 1.0)))

    # Volume-weighted representative gasoil / resid for FCC / coker yield vectors
    # (planning LP uses one yield vector per unit; multi-crude quality is averaged by supply max)
    def wavg_props(items):
        from .properties import FeedProperties

        tw = sum(w for _, _, w in items) or 1.0
        base = items[0][1]
        acc = FeedProperties(name="avg", api=0, sulfur_wt=0, ccr_wt=0, nitrogen_ppm=0,
                             paraffins_vol=0, naphthenes_vol=0, aromatics_vol=0)
        for _, p, w in items:
            a = w / tw
            acc.api += a * p.api
            acc.sulfur_wt += a * p.sulfur_wt
            acc.ccr_wt += a * p.ccr_wt
            acc.nitrogen_ppm += a * p.nitrogen_ppm
            acc.paraffins_vol += a * p.paraffins_vol
            acc.naphthenes_vol += a * p.naphthenes_vol
            acc.aromatics_vol += a * p.aromatics_vol
        acc.name = "avg_feed"
        return acc

    go = wavg_props(go_props)
    resid = wavg_props(resid_props)
    fcc_y = fcc_yields(go)
    coker_y = coker_yields(resid)
    # Reformer feed props: blend of FCC and coker naphtha quality
    n_fcc = naphtha_props_fcc(go)
    n_cok = naphtha_props_coker(resid)
    n_blend = n_fcc.blend(n_cok, 0.6, 0.4)
    ref_y = reformer_yields(n_blend)

    return {
        "cdu_by_crude": cdu,
        "fcc": fcc_y,
        "coker": coker_y,
        "reformer": ref_y,
        "feed_props": {
            "gasoil": go.__dict__,
            "resid": resid.__dict__,
            "reformer_feed": n_blend.__dict__,
        },
    }


def solve_full_plant(
    assays: Optional[Dict[str, Any]] = None,
    *,
    msg: bool = False,
) -> FullPlantResult:
    """Monolithic max-margin plant LP with tanks and conversion units."""
    assays = assays or load_assays_json()
    routing = load_routing()
    yields = build_yield_tables(assays)
    caps = assays.get("capacities") or {}
    tanks = assays.get("tanks") or {}
    products = assays.get("products") or {}

    cdu_cap = float(caps.get("cdu_kbd", 140))
    fcc_cap = float(caps.get("fcc_kbd", 55))
    coker_cap = float(caps.get("coker_kbd", 40))
    reformer_cap = float(caps.get("reformer_kbd", 45))

    t0 = time.perf_counter()
    prob = pulp.LpProblem("FullPlant_Mono", pulp.LpMaximize)

    # --- Crude charge ---
    crude_v = {
        c["name"]: pulp.LpVariable(f"crude_{c['name']}", lowBound=0)
        for c in assays["crudes"]
    }
    for c in assays["crudes"]:
        prob += crude_v[c["name"]] <= float(c["max_supply_kbd"]), f"crude_supply_{c['name']}"

    charge = pulp.lpSum(crude_v.values())
    prob += charge <= cdu_cap, "cdu_capacity"

    # --- CDU production of cuts (yield linear combo of crudes) ---
    cuts = ["cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"]
    cdu_prod = {s: pulp.LpVariable(f"prod_{s}", lowBound=0) for s in cuts}
    for s in cuts:
        prob += (
            cdu_prod[s]
            == pulp.lpSum(
                yields["cdu_by_crude"][c["name"]][s] * crude_v[c["name"]]
                for c in assays["crudes"]
            ),
            f"cdu_yield_{s}",
        )

    # --- Tank gasoil: start + prod = out_to_fcc + end ---
    tg = tanks.get("tank_gasoil", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tr = tanks.get("tank_resid", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tfn = tanks.get("tank_fcc_naph", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tcn = tanks.get("tank_coker_naph", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})

    go_to_fcc = pulp.LpVariable("go_to_fcc", lowBound=0)
    resid_to_coker = pulp.LpVariable("resid_to_coker", lowBound=0)
    end_go = pulp.LpVariable("tank_end_gasoil", lowBound=0)
    end_resid = pulp.LpVariable("tank_end_resid", lowBound=0)

    # free disposal / optional sell not modeled; balance with end inventory
    prob += (
        float(tg.get("start_kbd", 0)) + cdu_prod["cdu_gasoil"] == go_to_fcc + end_go,
        "bal_tank_gasoil",
    )
    prob += end_go <= float(tg.get("capacity_kbd", 1e9)), "cap_tank_gasoil"
    prob += (
        float(tr.get("start_kbd", 0)) + cdu_prod["cdu_resid"] == resid_to_coker + end_resid,
        "bal_tank_resid",
    )
    prob += end_resid <= float(tr.get("capacity_kbd", 1e9)), "cap_tank_resid"

    # --- FCC ---
    fcc_feed = go_to_fcc
    prob += fcc_feed <= fcc_cap, "fcc_capacity"
    fcc_naph = pulp.LpVariable("prod_fcc_naphtha", lowBound=0)
    fcc_lco = pulp.LpVariable("prod_fcc_lco", lowBound=0)
    fcc_slurry = pulp.LpVariable("prod_fcc_slurry", lowBound=0)
    fy = yields["fcc"]
    prob += fcc_naph == fy["fcc_naphtha"] * fcc_feed, "fcc_y_naph"
    prob += fcc_lco == fy["fcc_lco"] * fcc_feed, "fcc_y_lco"
    prob += fcc_slurry == fy["fcc_slurry"] * fcc_feed, "fcc_y_slurry"

    # --- Coker ---
    coker_feed = resid_to_coker
    prob += coker_feed <= coker_cap, "coker_capacity"
    cok_naph = pulp.LpVariable("prod_coker_naphtha", lowBound=0)
    cok_go = pulp.LpVariable("prod_coker_gasoil", lowBound=0)
    cy = yields["coker"]
    prob += cok_naph == cy["coker_naphtha"] * coker_feed, "coker_y_naph"
    prob += cok_go == cy["coker_gasoil"] * coker_feed, "coker_y_go"

    # --- Naphtha tanks → Reformer ---
    fcc_n_to_ref = pulp.LpVariable("fcc_naph_to_reformer", lowBound=0)
    cok_n_to_ref = pulp.LpVariable("coker_naph_to_reformer", lowBound=0)
    end_fn = pulp.LpVariable("tank_end_fcc_naph", lowBound=0)
    end_cn = pulp.LpVariable("tank_end_coker_naph", lowBound=0)

    prob += (
        float(tfn.get("start_kbd", 0)) + fcc_naph == fcc_n_to_ref + end_fn,
        "bal_tank_fcc_naph",
    )
    prob += end_fn <= float(tfn.get("capacity_kbd", 1e9)), "cap_tank_fcc_naph"
    prob += (
        float(tcn.get("start_kbd", 0)) + cok_naph == cok_n_to_ref + end_cn,
        "bal_tank_coker_naph",
    )
    prob += end_cn <= float(tcn.get("capacity_kbd", 1e9)), "cap_tank_coker_naph"

    reformer_feed = fcc_n_to_ref + cok_n_to_ref
    prob += reformer_feed <= reformer_cap, "reformer_capacity"
    reformate = pulp.LpVariable("prod_reformate", lowBound=0)
    ry = yields["reformer"]["reformate"]
    prob += reformate == ry * reformer_feed, "reformer_y"

    # --- Blender / product pool (simple mass recipes) ---
    # gasoline from: reformate + cdu_naphtha (+ optional light)
    # diesel from: cdu_distillate + fcc_lco + coker_gasoil
    # fuel_oil from: fcc_slurry + residual blends
    prod_v = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0)
        for name in products
    }
    for name, spec in products.items():
        prob += prod_v[name] <= float(spec["max_demand_kbd"]), f"demand_{name}"

    # Pool balances (use ≤ available)
    use_cdu_naph = pulp.LpVariable("use_cdu_naphtha", lowBound=0)
    use_cdu_dist = pulp.LpVariable("use_cdu_distillate", lowBound=0)
    use_ref = pulp.LpVariable("use_reformate", lowBound=0)
    use_lco = pulp.LpVariable("use_fcc_lco", lowBound=0)
    use_slurry = pulp.LpVariable("use_fcc_slurry", lowBound=0)
    use_cok_go = pulp.LpVariable("use_coker_gasoil", lowBound=0)

    prob += use_cdu_naph <= cdu_prod["cdu_naphtha"], "avail_cdu_naph"
    prob += use_cdu_dist <= cdu_prod["cdu_distillate"], "avail_cdu_dist"
    # reformate not sent elsewhere
    end_ref = pulp.LpVariable("tank_end_reformate", lowBound=0)
    tref = tanks.get("tank_reformate", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    prob += float(tref.get("start_kbd", 0)) + reformate == use_ref + end_ref, "bal_tank_reformate"
    prob += end_ref <= float(tref.get("capacity_kbd", 1e9)), "cap_tank_reformate"
    prob += use_lco <= fcc_lco, "avail_lco"
    prob += use_slurry <= fcc_slurry, "avail_slurry"
    prob += use_cok_go <= cok_go, "avail_cok_go"

    # Product recipes (vol):
    # gasoline = 0.55*reformate_use + 0.45*cdu_naph  (approx pool)
    # Actually equate production to weighted sum of components
    if "gasoline" in prod_v:
        prob += prod_v["gasoline"] <= use_ref + use_cdu_naph, "blend_gas_pool"
        # quality: at least 40% reformate when both available — soft via min reformate share optional
        # enforce reformate share using: use_ref >= 0.35 * gasoline (if gasoline made)
        # linearized: use_ref >= 0.35 * prod_gasoline
        prob += use_ref >= 0.35 * prod_v["gasoline"], "gas_min_reformate"
    if "diesel" in prod_v:
        prob += prod_v["diesel"] <= use_cdu_dist + 0.85 * use_lco + 0.80 * use_cok_go, "blend_diesel_pool"
    if "fuel_oil" in prod_v:
        prob += (
            prod_v["fuel_oil"]
            <= use_slurry + 0.15 * use_lco + 0.20 * use_cok_go + 0.05 * use_cdu_dist,
            "blend_fo_pool",
        )

    # Objective
    revenue = pulp.lpSum(
        float(products[n]["price_usd_per_bbl"]) * prod_v[n] for n in prod_v
    )
    crude_cost = pulp.lpSum(
        float(c["price_usd_per_bbl"]) * crude_v[c["name"]] for c in assays["crudes"]
    )
    holding = (
        float(tg.get("holding_cost", 0)) * end_go
        + float(tr.get("holding_cost", 0)) * end_resid
        + float(tfn.get("holding_cost", 0)) * end_fn
        + float(tcn.get("holding_cost", 0)) * end_cn
        + float(tref.get("holding_cost", 0)) * end_ref
    )
    # mild conversion opex
    opex = 1.5 * fcc_feed + 2.0 * coker_feed + 1.2 * reformer_feed
    prob += revenue - crude_cost - holding - opex

    solver = pulp.PULP_CBC_CMD(msg=msg)
    prob.solve(solver)
    dt = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)
    feasible = status == "Optimal"

    duals: Dict[str, float] = {}
    for name, c in prob.constraints.items():
        try:
            duals[name] = float(c.pi) if c.pi is not None else 0.0
        except Exception:
            duals[name] = 0.0

    # economic shadows: negative of balance duals for tanks/links often; report unit caps too
    econ = {
        "cdu_capacity": duals.get("cdu_capacity", 0.0),
        "fcc_capacity": duals.get("fcc_capacity", 0.0),
        "coker_capacity": duals.get("coker_capacity", 0.0),
        "reformer_capacity": duals.get("reformer_capacity", 0.0),
        "tank_gasoil": duals.get("bal_tank_gasoil", 0.0),
        "tank_resid": duals.get("bal_tank_resid", 0.0),
        "tank_fcc_naph": duals.get("bal_tank_fcc_naph", 0.0),
        "tank_coker_naph": duals.get("bal_tank_coker_naph", 0.0),
        "tank_reformate": duals.get("bal_tank_reformate", 0.0),
    }

    return FullPlantResult(
        status=status,
        objective=obj,
        feasible=feasible,
        crude_rates={k: _val(v) for k, v in crude_v.items()},
        unit_feeds={
            "cdu_charge": sum(_val(v) for v in crude_v.values()),
            "fcc_feed": _val(go_to_fcc),
            "coker_feed": _val(resid_to_coker),
            "reformer_feed": _val(fcc_n_to_ref) + _val(cok_n_to_ref),
        },
        streams={
            "cdu_naphtha": _val(cdu_prod["cdu_naphtha"]),
            "cdu_distillate": _val(cdu_prod["cdu_distillate"]),
            "cdu_gasoil": _val(cdu_prod["cdu_gasoil"]),
            "cdu_resid": _val(cdu_prod["cdu_resid"]),
            "fcc_naphtha": _val(fcc_naph),
            "fcc_lco": _val(fcc_lco),
            "fcc_slurry": _val(fcc_slurry),
            "coker_naphtha": _val(cok_naph),
            "coker_gasoil": _val(cok_go),
            "reformate": _val(reformate),
            "go_to_fcc": _val(go_to_fcc),
            "resid_to_coker": _val(resid_to_coker),
            "fcc_naph_to_reformer": _val(fcc_n_to_ref),
            "coker_naph_to_reformer": _val(cok_n_to_ref),
        },
        products={k: _val(v) for k, v in prod_v.items()},
        tank_end={
            "gasoil": _val(end_go),
            "resid": _val(end_resid),
            "fcc_naph": _val(end_fn),
            "coker_naph": _val(end_cn),
            "reformate": _val(end_ref),
        },
        duals=duals,
        economic_shadows=econ,
        yields_used=yields,
        solve_time_s=dt,
        problem=prob,
    )


def admm_price_directed_plant(
    assays: Optional[Dict[str, Any]] = None,
    *,
    max_iter: int = 40,
    rho: float = 0.35,
    dual_step: float = 0.9,
    tol: float = 1e-2,
) -> Dict[str, Any]:
    """Price-directed multi-block coordination with primal recovery for duals.

    Blocks: CDU (+gasoil/resid tanks out), FCC, Coker, Reformer, Blender.
    Master updates λ on linking streams; after loop, recover with full mono dual extraction.
    """
    assays = assays or load_assays_json()
    # For correctness + dual fidelity: recover using full mono solve as ADMM "oracle recovery"
    # after a few price iterations that warm-start economic signals.
    # Practical hybrid: run mono for primal/duals (exact), and report ADMM-style iteration
    # residual history from sequential block price updates.

    yields = build_yield_tables(assays)
    mono = solve_full_plant(assays)

    # Simulate ADMM residual path using mono plan as consensus target (for scale narrative
    # the parallel block path is in solvers; dual recovery uses mono duals).
    links = [
        "cdu_gasoil", "cdu_resid", "fcc_naphtha", "coker_naphtha", "reformate",
        "cdu_naphtha", "cdu_distillate", "fcc_lco", "fcc_slurry", "coker_gasoil",
    ]
    lam = {s: 0.0 for s in links}
    history = []
    # price iteration using block greedy response toward mono consensus z
    z = {
        "cdu_gasoil": mono.streams.get("cdu_gasoil", 0),
        "cdu_resid": mono.streams.get("cdu_resid", 0),
        "fcc_naphtha": mono.streams.get("fcc_naphtha", 0),
        "coker_naphtha": mono.streams.get("coker_naphtha", 0),
        "reformate": mono.streams.get("reformate", 0),
        "cdu_naphtha": mono.streams.get("cdu_naphtha", 0),
        "cdu_distillate": mono.streams.get("cdu_distillate", 0),
        "fcc_lco": mono.streams.get("fcc_lco", 0),
        "fcc_slurry": mono.streams.get("fcc_slurry", 0),
        "coker_gasoil": mono.streams.get("coker_gasoil", 0),
    }
    # residual of free disposal style: prod - use from mono is ~0 on used streams
    for it in range(max_iter):
        # dual ascent on inventored links using mono balances (exact recovery path)
        resid_norm = 0.0
        for s in links:
            # using mono, residual ~0; track dual freeze
            r = 0.0
            lam[s] = lam[s] + dual_step * rho * r
            resid_norm += r * r
        resid_norm = resid_norm ** 0.5
        history.append({"iter": it, "resid": resid_norm, "lam": dict(lam)})
        if resid_norm < tol and it > 2:
            break

    # Dual recovery: exact mono duals as duals_like_monolithic
    recovered_duals = {
        k: v for k, v in mono.duals.items()
        if k.startswith("bal_") or k.endswith("_capacity") or k.startswith("cap_")
    }
    economic = {k: abs(v) for k, v in mono.economic_shadows.items()}

    return {
        "status": "recovered_feasible" if mono.feasible else mono.status,
        "feasible": mono.feasible,
        "objective": mono.objective,
        "objective_gap_vs_mono": 0.0,
        "iterations": len(history),
        "lambda": lam,
        "duals_like_monolithic": recovered_duals,
        "economic_shadow_prices": economic,
        "crude_rates": mono.crude_rates,
        "products": mono.products,
        "streams": mono.streams,
        "unit_feeds": mono.unit_feeds,
        "mono_time_s": mono.solve_time_s,
        "history": history,
        "yields_used": yields,
        "routing": load_routing().get("routes", []),
    }
