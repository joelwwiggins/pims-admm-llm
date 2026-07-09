"""Simplified 2-block monolithic LP for dual-matching tests (no inventory/utilities).

  max  revenue - crude_cost
  s.t. prod_i - use_i = 0          (linking / balance)
       yields, CDU capacity, demand bounds, crude supply
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pulp

from pims_admm_llm.models.data import RefineryData


@dataclass
class SimpleMonoResult:
    status: str
    objective: float
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    duals: Dict[str, float]
    solve_time_s: float


def solve_simple_monolithic(data: RefineryData, msg: bool = False) -> SimpleMonoResult:
    import time

    prob = pulp.LpProblem("Refinery_Simple_Mono", pulp.LpMaximize)
    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", lowBound=0, upBound=c.max_supply_kbd)
        for c in data.crudes
    }
    inter_prod = {i: pulp.LpVariable(f"prod_{i}", lowBound=0) for i in data.intermediates}
    inter_use = {i: pulp.LpVariable(f"use_{i}", lowBound=0) for i in data.intermediates}
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=spec.max_demand_kbd)
        for name, spec in data.products.items()
    }

    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    cost = pulp.lpSum(c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes)
    prob += revenue - cost, "margin"

    prob += (
        pulp.lpSum(crude_vars[c.name] for c in data.crudes) <= data.cdu_capacity_kbd,
        "cdu_capacity",
    )
    for i in data.intermediates:
        prob += (
            inter_prod[i]
            == pulp.lpSum(c.yields.get(i, 0.0) * crude_vars[c.name] for c in data.crudes),
            f"yield_{i}",
        )
        # equality balance for clean dual match with ADMM
        prob += inter_prod[i] - inter_use[i] == 0, f"balance_{i}"

    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p] for p in data.blend_recipes
        )
        prob += inter_use[i] >= rhs, f"blend_use_{i}"

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=60))
    t1 = time.perf_counter()

    duals: Dict[str, float] = {}
    for name, constraint in prob.constraints.items():
        try:
            duals[name] = float(constraint.pi)
        except Exception:
            duals[name] = 0.0

    crude_rates = {}
    product_rates = {}
    intermediate_prod = {}
    intermediate_use = {}
    for v in prob.variables():
        val = float(v.varValue or 0.0)
        if v.name.startswith("crude_"):
            crude_rates[v.name.replace("crude_", "", 1)] = val
        elif v.name.startswith("product_"):
            product_rates[v.name.replace("product_", "", 1)] = val
        elif v.name.startswith("prod_"):
            intermediate_prod[v.name.replace("prod_", "", 1)] = val
        elif v.name.startswith("use_"):
            intermediate_use[v.name.replace("use_", "", 1)] = val

    return SimpleMonoResult(
        status=pulp.LpStatus[prob.status],
        objective=float(pulp.value(prob.objective) or 0.0),
        crude_rates=crude_rates,
        product_rates=product_rates,
        intermediate_prod=intermediate_prod,
        intermediate_use=intermediate_use,
        duals=duals,
        solve_time_s=t1 - t0,
    )
