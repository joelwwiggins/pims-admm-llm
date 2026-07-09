"""Primal recovery + dual extraction after ADMM block solves.

On degenerate LP faces the last ADMM blender iterate need not be unique.
Given intermediate production from the CDU (or consensus z), solve the exact
blender LP and read duals on intermediate availability — these match the
monolithic yield/balance shadow prices at optimality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pulp

from pims_admm_llm.models.data import RefineryData


@dataclass
class RecoveryResult:
    product_rates: Dict[str, float]
    intermediate_use: Dict[str, float]
    shadow_prices: Dict[str, float]  # economic $/bbl of each intermediate
    duals_like_monolithic: Dict[str, float]  # balance_* sign (maximize-form)
    objective_revenue: float
    status: str


def recover_blender_with_duals(
    data: RefineryData,
    intermediate_prod: Mapping[str, float],
) -> RecoveryResult:
    """max revenue s.t. recipe use <= available intermediate production."""
    prob = pulp.LpProblem("ADMM_Primal_Recovery_Blender", pulp.LpMaximize)
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=spec.max_demand_kbd)
        for name, spec in data.products.items()
    }
    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    prob += revenue, "revenue"

    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p] for p in data.blend_recipes
        )
        avail = float(intermediate_prod.get(i, 0.0))
        prob += rhs <= avail + 1e-9, f"avail_{i}"

    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    status = pulp.LpStatus.get(status_code, str(status_code))

    product_rates = {
        n: float(prod_vars[n].varValue or 0.0) for n in prod_vars
    }
    intermediate_use = {
        i: float(
            sum(
                data.blend_recipes[p].get(i, 0.0) * product_rates[p]
                for p in data.blend_recipes
            )
        )
        for i in data.intermediates
    }
    shadow: Dict[str, float] = {}
    duals_like: Dict[str, float] = {}
    for i in data.intermediates:
        try:
            pi = float(prob.constraints[f"avail_{i}"].pi)
        except Exception:
            pi = 0.0
        # PuLP maximize, constraint use <= avail: pi >= 0 is marginal value of avail
        shadow[i] = pi
        duals_like[f"balance_{i}"] = -pi  # match mono equality/balance maximize dual sign

    return RecoveryResult(
        product_rates=product_rates,
        intermediate_use=intermediate_use,
        shadow_prices=shadow,
        duals_like_monolithic=duals_like,
        objective_revenue=float(pulp.value(prob.objective) or 0.0),
        status=status,
    )


def economic_objective(
    data: RefineryData,
    crude_rates: Mapping[str, float],
    product_rates: Mapping[str, float],
) -> float:
    rev = sum(
        data.products[n].price_usd_per_bbl * product_rates.get(n, 0.0)
        for n in data.products
    )
    cost = sum(c.price_usd_per_bbl * crude_rates.get(c.name, 0.0) for c in data.crudes)
    return float(rev - cost)
