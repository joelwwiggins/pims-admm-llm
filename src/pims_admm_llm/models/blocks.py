"""Block-angular and monolithic LP builders for the toy refinery.

Blocks:
  - CDU: crude selection + distillation yields → intermediate production
  - Blender: intermediate consumption → finished products

Linking variables (consensus in ADMM): intermediate flows (kbd)
  naphtha, distillate, gasoil, residue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp

from .data import RefineryData


class BlockNames:
    CDU = "CDU"
    BLENDER = "Blender"


@dataclass
class MonolithicResult:
    status: str
    objective: float
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    duals: Dict[str, float]  # shadow prices on key constraints
    solve_time_s: float
    problem: pulp.LpProblem


def build_monolithic_lp(data: RefineryData) -> pulp.LpProblem:
    """Full planning LP (PIMS-style toy): max margin subject to balances & capacities."""
    prob = pulp.LpProblem("Refinery_Monolithic", pulp.LpMaximize)

    # Crude purchase / charge rates
    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", lowBound=0, upBound=c.max_supply_kbd)
        for c in data.crudes
    }

    # Intermediate production (from CDU) and use (in blender)
    inter_prod = {
        i: pulp.LpVariable(f"prod_{i}", lowBound=0) for i in data.intermediates
    }
    inter_use = {
        i: pulp.LpVariable(f"use_{i}", lowBound=0) for i in data.intermediates
    }

    # Finished products
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=spec.max_demand_kbd)
        for name, spec in data.products.items()
    }

    # Objective: product revenue - crude cost
    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    cost = pulp.lpSum(c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes)
    prob += revenue - cost, "margin"

    # CDU capacity
    total_crude = pulp.lpSum(crude_vars[c.name] for c in data.crudes)
    prob += total_crude <= data.cdu_capacity_kbd, "cdu_capacity"

    # Yield: intermediate production from crude slate
    for i in data.intermediates:
        prob += (
            inter_prod[i]
            == pulp.lpSum(c.yields.get(i, 0.0) * crude_vars[c.name] for c in data.crudes),
            f"yield_{i}",
        )

    # Material balance on intermediates (linking)
    for i in data.intermediates:
        prob += inter_prod[i] >= inter_use[i], f"balance_{i}"

    # Blender recipes (simplified fixed-fraction recipes, scalable)
    for prod_name, recipe in data.blend_recipes.items():
        # product rate constrained by recipe components
        # sum of component fractions should be ~1; enforce use = recipe * product
        for comp, frac in recipe.items():
            if comp not in inter_use:
                continue
            # Accumulate: each product pulls its share; we add soft equality via constraints
            # Use one aggregated constraint set:
            # For each intermediate, use_i >= sum_p recipe[p][i] * product_p
            pass

    # Aggregate intermediate use from recipes
    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p]
            for p in data.blend_recipes
        )
        prob += inter_use[i] >= rhs, f"blend_use_{i}"

    # Optional: force products to consume only their recipe (already above)
    # Product production limited by tightest component is implicit via balances.

    return prob


def extract_monolithic_solution(
    prob: pulp.LpProblem, data: RefineryData, solve_time_s: float
) -> MonolithicResult:
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)

    crude_rates = {}
    product_rates = {}
    intermediate_prod = {}
    intermediate_use = {}
    for v in prob.variables():
        val = float(v.varValue or 0.0)
        if v.name.startswith("crude_"):
            crude_rates[v.name.replace("crude_", "")] = val
        elif v.name.startswith("product_"):
            product_rates[v.name.replace("product_", "")] = val
        elif v.name.startswith("prod_"):
            intermediate_prod[v.name.replace("prod_", "")] = val
        elif v.name.startswith("use_"):
            intermediate_use[v.name.replace("use_", "")] = val

    duals: Dict[str, float] = {}
    for name, constraint in prob.constraints.items():
        try:
            duals[name] = float(constraint.pi)
        except Exception:
            duals[name] = 0.0

    return MonolithicResult(
        status=status,
        objective=obj,
        crude_rates=crude_rates,
        product_rates=product_rates,
        intermediate_prod=intermediate_prod,
        intermediate_use=intermediate_use,
        duals=duals,
        solve_time_s=solve_time_s,
        problem=prob,
    )


def solve_monolithic(data: RefineryData, msg: bool = False) -> MonolithicResult:
    import time

    prob = build_monolithic_lp(data)
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=60))
    t1 = time.perf_counter()
    return extract_monolithic_solution(prob, data, t1 - t0)
