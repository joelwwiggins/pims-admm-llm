"""Coupled multi-period mono plant LP with tank inventory carries (Wave3b W4 smoke).

Time coupling is first-class:
  I[k, 0]  = start inventory from assays (or 0 if inventory disabled)
  I[k, t+1] = I[k, t] + production_in[k, t] - consumption_out[k, t]
  0 <= I[k, t] <= capacity

Default smoke uses n_periods=2 with a crude-supply cut in later periods so
carry-in inventory has economic value (not just independent n copies).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import pulp

from .assay_loader import load_assays_json, load_routing
from .full_plant import (
    _arc_meta,
    _arc_upbound,
    _comp_prop,
    _quality_spec,
    build_yield_tables,
)


def _val(v) -> float:
    x = pulp.value(v)
    return float(x) if x is not None else 0.0


TANK_KEYS = (
    "gasoil",
    "resid",
    "fcc_naph",
    "coker_naph",
    "reformate",
)

TANK_ASSAY_MAP = {
    "gasoil": "tank_gasoil",
    "resid": "tank_resid",
    "fcc_naph": "tank_fcc_naph",
    "coker_naph": "tank_coker_naph",
    "reformate": "tank_reformate",
}


@dataclass
class MultiPeriodResult:
    status: str
    objective: float
    feasible: bool
    n_periods: int
    inventory_mode: bool
    tank_start: List[Dict[str, float]]  # len T; start of each period
    tank_end: List[Dict[str, float]]  # len T; end of each period (= start of next)
    period_objectives: List[float]
    period_unit_feeds: List[Dict[str, float]]
    period_products: List[Dict[str, float]]
    period_crude_rates: List[Dict[str, float]]
    period_arc_flows: List[Dict[str, float]]
    carries: List[Dict[str, float]]  # end inventory carried into next (T-1 links + terminal)
    duals: Dict[str, float]
    solve_time_s: float
    problem: pulp.LpProblem
    meta: Dict[str, Any] = field(default_factory=dict)


def _period_scales(
    n_periods: int,
    crude_scale: Optional[Sequence[float]] = None,
    demand_scale: Optional[Sequence[float]] = None,
) -> tuple[List[float], List[float]]:
    """Default smoke: full crude period 0, tight crude later → inventory carry has value."""
    if crude_scale is None:
        # period 0 full; later periods 35% crude max so drawing start/carry inventory helps
        cs = [1.0] + [0.35] * max(0, n_periods - 1)
    else:
        cs = [float(x) for x in crude_scale]
        if len(cs) != n_periods:
            raise ValueError(f"crude_scale length {len(cs)} != n_periods {n_periods}")
    if demand_scale is None:
        ds = [1.0] * n_periods
    else:
        ds = [float(x) for x in demand_scale]
        if len(ds) != n_periods:
            raise ValueError(f"demand_scale length {len(ds)} != n_periods {n_periods}")
    return cs, ds


def solve_multi_period(
    assays: Optional[Dict[str, Any]] = None,
    *,
    n_periods: int = 2,
    inventory_mode: Union[bool, str] = True,
    msg: bool = False,
    routing: Optional[Dict[str, Any]] = None,
    force_all_arcs_open: bool = False,
    crude_scale: Optional[Sequence[float]] = None,
    demand_scale: Optional[Sequence[float]] = None,
    terminal_inventory_value: float = 0.0,
) -> MultiPeriodResult:
    """Monolithic multi-period max-margin plant with tank carries.

    inventory_mode:
      True / \"inventory\" / \"multi_period\" / \"heels\" → use start inventory + free end carries
      False / \"off\" / \"pass\" → zero start, force end-of-period inventory to 0 (pass-through)
    """
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    if isinstance(inventory_mode, str):
        mode_s = inventory_mode.strip().lower()
        inv_on = mode_s in ("1", "true", "yes", "inventory", "multi_period", "heels", "on")
    else:
        inv_on = bool(inventory_mode)

    assays = assays or load_assays_json()
    routing = routing or load_routing()
    yields = build_yield_tables(assays)
    caps = assays.get("capacities") or {}
    tanks = assays.get("tanks") or {}
    products = assays.get("products") or {}

    cdu_cap = float(caps.get("cdu_kbd", 140))
    fcc_cap = float(caps.get("fcc_kbd", 55))
    coker_cap = float(caps.get("coker_kbd", 40))
    reformer_cap = float(caps.get("reformer_kbd", 45))

    split = routing.get("naphtha_split") or {}
    light_frac = float(split.get("light_frac_of_cdu_naphtha", 0.45))
    heavy_frac = float(split.get("heavy_frac_of_cdu_naphtha", 0.55))
    ssum = light_frac + heavy_frac
    if ssum <= 0:
        light_frac, heavy_frac = 0.45, 0.55
    else:
        light_frac, heavy_frac = light_frac / ssum, heavy_frac / ssum

    cs, ds = _period_scales(n_periods, crude_scale, demand_scale)

    # Initial inventory
    start0 = {}
    holding = {}
    capacity = {}
    for key, aname in TANK_ASSAY_MAP.items():
        cfg = tanks.get(aname, {}) or {}
        start0[key] = float(cfg.get("start_kbd", 0.0)) if inv_on else 0.0
        holding[key] = float(cfg.get("holding_cost", 0.0))
        capacity[key] = float(cfg.get("capacity_kbd", 1e9))

    arc_meta = _arc_meta(routing)
    arc_cost_map = {
        a.get("id"): float(a.get("cost_usd_per_bbl", 0.0)) for a in (routing.get("arcs") or [])
    }
    sell_price_go = 70.0
    for a in routing.get("arcs") or []:
        if a.get("id") == "go_to_sell" and a.get("sell_price_usd_per_bbl") is not None:
            sell_price_go = float(a["sell_price_usd_per_bbl"])
            break
    coke_credit_per_bbl = float((assays.get("credits") or {}).get("coker_coke_usd_per_bbl_feed", 18.0))

    t0 = time.perf_counter()
    prob = pulp.LpProblem("FullPlant_MultiPeriod", pulp.LpMaximize)

    # Inventory level at START of period t (t=0..T), where t=T is terminal after last period
    I: Dict[str, List[pulp.LpVariable]] = {
        k: [pulp.LpVariable(f"I_{k}_t{t}", lowBound=0) for t in range(n_periods + 1)] for k in TANK_KEYS
    }
    for k in TANK_KEYS:
        # fixed opening inventory
        prob += I[k][0] == start0[k], f"open_{k}"
        for t in range(n_periods + 1):
            prob += I[k][t] <= capacity[k], f"cap_{k}_t{t}"
        if not inv_on:
            for t in range(1, n_periods + 1):
                prob += I[k][t] == 0, f"pass_end_{k}_t{t}"

    period_obj_terms: List[Any] = []
    # store period vars for extraction
    crude_v: List[Dict[str, pulp.LpVariable]] = []
    go_to_fcc_l: List[pulp.LpVariable] = []
    resid_to_coker_l: List[pulp.LpVariable] = []
    heavy_to_ref_l: List[pulp.LpVariable] = []
    fcc_n_to_ref_l: List[pulp.LpVariable] = []
    cok_n_to_ref_l: List[pulp.LpVariable] = []
    prod_lists: List[Dict[str, pulp.LpVariable]] = []
    arc_lists: List[Dict[str, pulp.LpVariable]] = []

    fy = yields["fcc"]
    cy = yields["coker"]
    ry = yields["reformer"]["reformate"]

    for t in range(n_periods):
        pfx = f"p{t}"

        def _new_arc(aid: str) -> pulp.LpVariable:
            ub = _arc_upbound(
                arc_meta.get(aid, {"default_open": True}), force_open=force_all_arcs_open
            )
            name = f"arc_{aid}_{pfx}"
            if ub is None:
                return pulp.LpVariable(name, lowBound=0)
            return pulp.LpVariable(name, lowBound=0, upBound=ub)

        # Crude
        cv = {
            c["name"]: pulp.LpVariable(f"crude_{c['name']}_{pfx}", lowBound=0)
            for c in assays["crudes"]
        }
        crude_v.append(cv)
        for c in assays["crudes"]:
            mx = float(c["max_supply_kbd"]) * cs[t]
            prob += cv[c["name"]] <= mx, f"crude_supply_{c['name']}_{pfx}"
        charge = pulp.lpSum(cv.values())
        prob += charge <= cdu_cap, f"cdu_capacity_{pfx}"

        # CDU cuts
        cuts = ["cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"]
        cdu_prod = {s: pulp.LpVariable(f"prod_{s}_{pfx}", lowBound=0) for s in cuts}
        for s in cuts:
            prob += (
                cdu_prod[s]
                == pulp.lpSum(
                    yields["cdu_by_crude"][c["name"]][s] * cv[c["name"]] for c in assays["crudes"]
                ),
                f"cdu_yield_{s}_{pfx}",
            )

        cdu_naph_light = pulp.LpVariable(f"prod_cdu_naph_light_{pfx}", lowBound=0)
        cdu_naph_heavy = pulp.LpVariable(f"prod_cdu_naph_heavy_{pfx}", lowBound=0)
        prob += cdu_naph_light == light_frac * cdu_prod["cdu_naphtha"], f"split_n_light_{pfx}"
        prob += cdu_naph_heavy == heavy_frac * cdu_prod["cdu_naphtha"], f"split_n_heavy_{pfx}"

        # Gasoil swing + tank
        go_to_fcc = _new_arc("go_to_fcc")
        go_to_diesel = _new_arc("go_to_diesel")
        go_to_sell = _new_arc("go_to_sell")
        go_to_fcc_l.append(go_to_fcc)
        # I[t] + prod = outs + I[t+1]
        prob += (
            I["gasoil"][t] + cdu_prod["cdu_gasoil"]
            == go_to_fcc + go_to_diesel + go_to_sell + I["gasoil"][t + 1],
            f"bal_tank_gasoil_{pfx}",
        )

        # Resid swing + tank
        resid_to_coker = _new_arc("resid_to_coker")
        resid_to_fo = _new_arc("resid_to_fo")
        resid_to_coker_l.append(resid_to_coker)
        prob += (
            I["resid"][t] + cdu_prod["cdu_resid"]
            == resid_to_coker + resid_to_fo + I["resid"][t + 1],
            f"bal_tank_resid_{pfx}",
        )

        # FCC
        fcc_feed = go_to_fcc
        prob += fcc_feed <= fcc_cap, f"fcc_capacity_{pfx}"
        fcc_naph = pulp.LpVariable(f"prod_fcc_naph_{pfx}", lowBound=0)
        fcc_lco = pulp.LpVariable(f"prod_fcc_lco_{pfx}", lowBound=0)
        fcc_slurry = pulp.LpVariable(f"prod_fcc_slurry_{pfx}", lowBound=0)
        prob += fcc_naph == fy["fcc_naphtha"] * fcc_feed, f"fcc_y_naph_{pfx}"
        prob += fcc_lco == fy["fcc_lco"] * fcc_feed, f"fcc_y_lco_{pfx}"
        prob += fcc_slurry == fy["fcc_slurry"] * fcc_feed, f"fcc_y_slurry_{pfx}"

        # Coker
        coker_feed = resid_to_coker
        prob += coker_feed <= coker_cap, f"coker_capacity_{pfx}"
        cok_naph = pulp.LpVariable(f"prod_coker_naph_{pfx}", lowBound=0)
        cok_go = pulp.LpVariable(f"prod_coker_go_{pfx}", lowBound=0)
        prob += cok_naph == cy["coker_naphtha"] * coker_feed, f"coker_y_naph_{pfx}"
        prob += cok_go == cy["coker_gasoil"] * coker_feed, f"coker_y_go_{pfx}"

        fcc_n_to_gas = _new_arc("fcc_naph_to_gas")
        fcc_n_to_ref = _new_arc("fcc_naph_to_reformer")
        fcc_n_to_ref_l.append(fcc_n_to_ref)
        prob += (
            I["fcc_naph"][t] + fcc_naph == fcc_n_to_gas + fcc_n_to_ref + I["fcc_naph"][t + 1],
            f"bal_tank_fcc_naph_{pfx}",
        )

        cok_n_to_hdt_gas = _new_arc("coker_naph_to_hdt_gas")
        cok_n_to_fo = _new_arc("coker_naph_to_fo")
        cok_n_to_ref = _new_arc("coker_naph_to_reformer")
        cok_n_to_ref_l.append(cok_n_to_ref)
        prob += (
            I["coker_naph"][t] + cok_naph
            == cok_n_to_hdt_gas + cok_n_to_fo + cok_n_to_ref + I["coker_naph"][t + 1],
            f"bal_tank_coker_naph_{pfx}",
        )

        light_to_gas = pulp.LpVariable(f"arc_sr_light_to_gas_{pfx}", lowBound=0)
        heavy_to_ref = pulp.LpVariable(f"arc_sr_heavy_to_reformer_{pfx}", lowBound=0)
        heavy_to_gas = pulp.LpVariable(f"arc_sr_heavy_to_gas_{pfx}", lowBound=0)
        heavy_to_ref_l.append(heavy_to_ref)
        light_dispose = pulp.LpVariable(f"dispose_sr_light_{pfx}", lowBound=0)
        heavy_dispose = pulp.LpVariable(f"dispose_sr_heavy_{pfx}", lowBound=0)
        prob += light_to_gas + light_dispose == cdu_naph_light, f"bal_sr_light_{pfx}"
        prob += heavy_to_ref + heavy_to_gas + heavy_dispose == cdu_naph_heavy, f"bal_sr_heavy_{pfx}"

        reformer_feed = heavy_to_ref + fcc_n_to_ref + cok_n_to_ref
        prob += reformer_feed <= reformer_cap, f"reformer_capacity_{pfx}"
        reformate = pulp.LpVariable(f"prod_reformate_{pfx}", lowBound=0)
        prob += reformate == ry * reformer_feed, f"reformer_y_{pfx}"
        use_ref = pulp.LpVariable(f"use_reformate_{pfx}", lowBound=0)
        prob += (
            I["reformate"][t] + reformate == use_ref + I["reformate"][t + 1],
            f"bal_tank_reformate_{pfx}",
        )

        use_dist = pulp.LpVariable(f"use_cdu_distillate_{pfx}", lowBound=0)
        use_lco_diesel = pulp.LpVariable(f"use_fcc_lco_diesel_{pfx}", lowBound=0)
        use_lco_fo = pulp.LpVariable(f"use_fcc_lco_fo_{pfx}", lowBound=0)
        use_slurry = pulp.LpVariable(f"use_fcc_slurry_{pfx}", lowBound=0)
        use_cok_go_diesel = pulp.LpVariable(f"use_coker_go_diesel_{pfx}", lowBound=0)
        use_cok_go_fo = pulp.LpVariable(f"use_coker_go_fo_{pfx}", lowBound=0)
        dist_dispose = pulp.LpVariable(f"dispose_distillate_{pfx}", lowBound=0)
        lco_dispose = pulp.LpVariable(f"dispose_lco_{pfx}", lowBound=0)
        slurry_dispose = pulp.LpVariable(f"dispose_slurry_{pfx}", lowBound=0)
        cok_go_dispose = pulp.LpVariable(f"dispose_coker_go_{pfx}", lowBound=0)
        prob += use_dist + dist_dispose == cdu_prod["cdu_distillate"], f"bal_distillate_{pfx}"
        prob += use_lco_diesel + use_lco_fo + lco_dispose == fcc_lco, f"bal_lco_{pfx}"
        prob += use_slurry + slurry_dispose == fcc_slurry, f"bal_slurry_{pfx}"
        prob += use_cok_go_diesel + use_cok_go_fo + cok_go_dispose == cok_go, f"bal_coker_go_{pfx}"

        prod_v = {
            name: pulp.LpVariable(f"product_{name}_{pfx}", lowBound=0) for name in products
        }
        prod_lists.append(prod_v)
        for name, spec in products.items():
            mx = float(spec["max_demand_kbd"]) * ds[t]
            prob += prod_v[name] <= mx, f"demand_{name}_{pfx}"

        if "gasoline" in prod_v:
            prob += (
                prod_v["gasoline"]
                == use_ref + light_to_gas + heavy_to_gas + fcc_n_to_gas + cok_n_to_hdt_gas,
                f"blend_gas_pool_{pfx}",
            )
            gspec = _quality_spec(routing, "gasoline")
            min_ron = float(gspec.get("min_ron", 87.0))
            max_s = float(gspec.get("max_sulfur_wt", 0.01))
            ron_expr = (
                _comp_prop(routing, "reformate", "ron", 100.0) * use_ref
                + _comp_prop(routing, "cdu_naphtha_light", "ron", 72.0) * light_to_gas
                + _comp_prop(routing, "cdu_naphtha_heavy", "ron", 58.0) * heavy_to_gas
                + _comp_prop(routing, "fcc_naphtha", "ron", 93.0) * fcc_n_to_gas
                + _comp_prop(routing, "coker_naphtha_hdt", "ron", 74.0) * cok_n_to_hdt_gas
            )
            prob += ron_expr >= min_ron * prod_v["gasoline"], f"qual_gas_min_ron_{pfx}"
            s_expr = (
                _comp_prop(routing, "reformate", "sulfur_wt", 0.0005) * use_ref
                + _comp_prop(routing, "cdu_naphtha_light", "sulfur_wt", 0.02) * light_to_gas
                + _comp_prop(routing, "cdu_naphtha_heavy", "sulfur_wt", 0.04) * heavy_to_gas
                + _comp_prop(routing, "fcc_naphtha", "sulfur_wt", 0.008) * fcc_n_to_gas
                + _comp_prop(routing, "coker_naphtha_hdt", "sulfur_wt", 0.01) * cok_n_to_hdt_gas
            )
            prob += s_expr <= max_s * prod_v["gasoline"], f"qual_gas_max_s_{pfx}"

        if "diesel" in prod_v:
            prob += (
                prod_v["diesel"]
                == use_dist + 0.85 * use_lco_diesel + 0.80 * use_cok_go_diesel + 0.55 * go_to_diesel,
                f"blend_diesel_pool_{pfx}",
            )
            dspec = _quality_spec(routing, "diesel")
            max_s_d = float(dspec.get("max_sulfur_wt", 0.05))
            s_d_effective = (
                _comp_prop(routing, "cdu_distillate", "sulfur_wt", 0.12) * 0.15 * use_dist
                + _comp_prop(routing, "fcc_lco", "sulfur_wt", 0.35) * 0.10 * use_lco_diesel
                + _comp_prop(routing, "coker_gasoil", "sulfur_wt", 0.40) * 0.12 * use_cok_go_diesel
                + _comp_prop(routing, "cdu_gasoil", "sulfur_wt", 0.45) * 0.20 * go_to_diesel
            )
            prob += s_d_effective <= max_s_d * prod_v["diesel"], f"qual_diesel_max_s_{pfx}"

        if "fuel_oil" in prod_v:
            prob += (
                prod_v["fuel_oil"]
                == use_slurry
                + 0.85 * resid_to_fo
                + cok_n_to_fo
                + 0.15 * use_lco_fo
                + 0.20 * use_cok_go_fo,
                f"blend_fo_pool_{pfx}",
            )

        revenue = pulp.lpSum(float(products[n]["price_usd_per_bbl"]) * prod_v[n] for n in prod_v)
        sell_rev = sell_price_go * go_to_sell
        coke_credit = coke_credit_per_bbl * coker_feed
        crude_cost = pulp.lpSum(
            float(c["price_usd_per_bbl"]) * cv[c["name"]] for c in assays["crudes"]
        )
        # Holding charged on end-of-period inventory (I[t+1])
        hold = pulp.lpSum(holding[k] * I[k][t + 1] for k in TANK_KEYS) if inv_on else 0
        routing_opex = (
            arc_cost_map.get("go_to_fcc", 0.05) * go_to_fcc
            + arc_cost_map.get("go_to_diesel", 0.1) * go_to_diesel
            + arc_cost_map.get("resid_to_coker", 0.08) * resid_to_coker
            + arc_cost_map.get("resid_to_fo", 0.05) * resid_to_fo
            + arc_cost_map.get("fcc_naph_to_gas", 0.02) * fcc_n_to_gas
            + arc_cost_map.get("fcc_naph_to_reformer", 5.0) * fcc_n_to_ref
            + arc_cost_map.get("coker_naph_to_hdt_gas", 0.35) * cok_n_to_hdt_gas
            + arc_cost_map.get("coker_naph_to_fo", 0.05) * cok_n_to_fo
            + arc_cost_map.get("coker_naph_to_reformer", 6.0) * cok_n_to_ref
            + arc_cost_map.get("sr_heavy_to_reformer", 0.05) * heavy_to_ref
        )
        opex = 1.5 * fcc_feed + 2.0 * coker_feed + 1.2 * reformer_feed + routing_opex
        period_obj_terms.append(revenue + sell_rev + coke_credit - crude_cost - hold - opex)

        arc_lists.append(
            {
                "go_to_fcc": go_to_fcc,
                "go_to_diesel": go_to_diesel,
                "go_to_sell": go_to_sell,
                "resid_to_coker": resid_to_coker,
                "resid_to_fo": resid_to_fo,
                "fcc_naph_to_gas": fcc_n_to_gas,
                "fcc_naph_to_reformer": fcc_n_to_ref,
                "coker_naph_to_hdt_gas": cok_n_to_hdt_gas,
                "coker_naph_to_fo": cok_n_to_fo,
                "coker_naph_to_reformer": cok_n_to_ref,
                "sr_light_to_gas": light_to_gas,
                "sr_heavy_to_reformer": heavy_to_ref,
                "sr_heavy_to_gas": heavy_to_gas,
            }
        )

    # Optional terminal salvage on leftover inventory (encourages nonzero terminal heels when set)
    terminal_value = 0
    if inv_on and terminal_inventory_value:
        terminal_value = terminal_inventory_value * pulp.lpSum(I[k][n_periods] for k in TANK_KEYS)

    prob += pulp.lpSum(period_obj_terms) + terminal_value
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

    tank_start: List[Dict[str, float]] = []
    tank_end: List[Dict[str, float]] = []
    carries: List[Dict[str, float]] = []
    for t in range(n_periods):
        tank_start.append({k: _val(I[k][t]) for k in TANK_KEYS})
        tank_end.append({k: _val(I[k][t + 1]) for k in TANK_KEYS})
        carries.append({k: _val(I[k][t + 1]) for k in TANK_KEYS})

    period_unit_feeds = []
    period_products = []
    period_crude_rates = []
    period_arc_flows = []
    period_objectives: List[float] = []
    for t in range(n_periods):
        period_crude_rates.append({k: _val(v) for k, v in crude_v[t].items()})
        uf = {
            "cdu_charge": sum(_val(v) for v in crude_v[t].values()),
            "fcc_feed": _val(go_to_fcc_l[t]),
            "coker_feed": _val(resid_to_coker_l[t]),
            "reformer_feed": _val(heavy_to_ref_l[t])
            + _val(fcc_n_to_ref_l[t])
            + _val(cok_n_to_ref_l[t]),
        }
        period_unit_feeds.append(uf)
        period_products.append({k: _val(v) for k, v in prod_lists[t].items()})
        period_arc_flows.append({k: _val(v) for k, v in arc_lists[t].items()})
        # approximate per-period obj via recomputed expression not available; leave 0 and meta total
        period_objectives.append(0.0)

    # Recompute approximate period margins from solution values for reporting
    for t in range(n_periods):
        rev = sum(
            float(products[n]["price_usd_per_bbl"]) * period_products[t].get(n, 0.0) for n in products
        )
        sell = sell_price_go * period_arc_flows[t].get("go_to_sell", 0.0)
        coke = coke_credit_per_bbl * period_unit_feeds[t]["coker_feed"]
        crude_cost = sum(
            float(c["price_usd_per_bbl"]) * period_crude_rates[t].get(c["name"], 0.0)
            for c in assays["crudes"]
        )
        hold = (
            sum(holding[k] * tank_end[t].get(k, 0.0) for k in TANK_KEYS) if inv_on else 0.0
        )
        opex = (
            1.5 * period_unit_feeds[t]["fcc_feed"]
            + 2.0 * period_unit_feeds[t]["coker_feed"]
            + 1.2 * period_unit_feeds[t]["reformer_feed"]
        )
        period_objectives[t] = rev + sell + coke - crude_cost - hold - opex

    return MultiPeriodResult(
        status=status,
        objective=obj,
        feasible=feasible,
        n_periods=n_periods,
        inventory_mode=inv_on,
        tank_start=tank_start,
        tank_end=tank_end,
        period_objectives=period_objectives,
        period_unit_feeds=period_unit_feeds,
        period_products=period_products,
        period_crude_rates=period_crude_rates,
        period_arc_flows=period_arc_flows,
        carries=carries,
        duals=duals,
        solve_time_s=dt,
        problem=prob,
        meta={
            "routing_version": routing.get("version"),
            "crude_scale": list(cs),
            "demand_scale": list(ds),
            "start0": dict(start0),
            "terminal_inventory_value": terminal_inventory_value,
            "mode": "multi_period" if inv_on else "pass",
            "tank_keys": list(TANK_KEYS),
        },
    )
