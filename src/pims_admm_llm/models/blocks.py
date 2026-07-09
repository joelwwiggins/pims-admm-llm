"""Block-angular and monolithic LP builders for the toy refinery.

Blocks (PuLP subproblems):
  - CDU:        crude selection + distillation yields → intermediate production
                + utility draw from charge rate
  - Inventory:  tank farm balances / capacity / holding cost on intermediates
  - Blender:    intermediate consumption → finished products + utility draw
  - Utilities:  supply shared utilities up to capacity at unit cost

Linking variables (block-angular structure):
  - Intermediate streams (kbd): naphtha, distillate, gasoil, residue
      CDU produces → Inventory buffers → Blender consumes
  - Utilities: fuel_gas, steam, power
      Utilities supplies → CDU + Blender demand

ADMM dual prices λ can be injected into each subproblem objective so workers
3–4 can coordinate without rebuilding the full model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import pulp

from .data import RefineryData


class BlockNames:
    CDU = "CDU"
    INVENTORY = "Inventory"
    BLENDER = "Blender"
    UTILITIES = "Utilities"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.CDU, cls.INVENTORY, cls.BLENDER, cls.UTILITIES]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MonolithicResult:
    status: str
    objective: float
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    inventory_end: Dict[str, float]
    utility_supply: Dict[str, float]
    utility_use_cdu: Dict[str, float]
    utility_use_blend: Dict[str, float]
    duals: Dict[str, float]
    solve_time_s: float
    problem: pulp.LpProblem


@dataclass
class SubproblemResult:
    block: str
    status: str
    objective: float
    primals: Dict[str, float]
    linking_intermediates: Dict[str, float]
    linking_utilities: Dict[str, float]
    solve_time_s: float
    problem: pulp.LpProblem
    message: str = ""


# ---------------------------------------------------------------------------
# Monolithic LP (PIMS-style full model)
# ---------------------------------------------------------------------------


def build_monolithic_lp(
    data: RefineryData,
    *,
    include_inventory: bool = True,
    include_utilities: bool = True,
) -> pulp.LpProblem:
    """Full planning LP (PIMS-style toy): max margin subject to balances & capacities.

    Explicit named constraints (not only variable bounds) for clean dual extraction:
      - cdu_capacity
      - crude_supply_<name>
      - product_demand_<name>
      - tank_<intermediate>  (ending inventory ≤ tank capacity)
      - inv_balance_<intermediate>
      - yield_*, blend_use_*, balance_*
      - utility_cap_<name> (if utilities defined)
    """
    prob = pulp.LpProblem("Refinery_Monolithic", pulp.LpMaximize)

    # Crude purchase / charge rates — supply dual via named constraint
    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", lowBound=0, upBound=None)
        for c in data.crudes
    }

    # Intermediate production (from CDU) and use (in blender)
    inter_prod = {
        i: pulp.LpVariable(f"prod_{i}", lowBound=0) for i in data.intermediates
    }
    inter_use = {
        i: pulp.LpVariable(f"use_{i}", lowBound=0) for i in data.intermediates
    }

    # Ending inventory (tank farm) when inventory specs exist
    inv_end = {}
    if include_inventory:
        for i in data.intermediates:
            inv_end[i] = pulp.LpVariable(f"inv_end_{i}", lowBound=0, upBound=None)

    # Finished products — demand dual via named constraint
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=None)
        for name, spec in data.products.items()
    }

    # Objective: product revenue - crude cost - holding - utility cost
    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    cost = pulp.lpSum(c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes)
    holding = 0
    if include_inventory and inv_end:
        holding = pulp.lpSum(
            [
                (
                    (data.inventory or {}).get(i).holding_cost_usd_per_bbl
                    if (data.inventory or {}).get(i)
                    else 0.0
                )
                * inv_end[i]
                for i in data.intermediates
            ]
        )

    util_use_expr = {uname: 0 for uname in getattr(data, "utility_names", []) or []}
    if include_utilities and getattr(data, "utilities", None):
        for c in data.crudes:
            for uname, rate in (c.utility_use or {}).items():
                if uname in util_use_expr:
                    util_use_expr[uname] += rate * crude_vars[c.name]
        for pname, pspec in data.products.items():
            for uname, rate in (pspec.utility_use or {}).items():
                if uname in util_use_expr:
                    util_use_expr[uname] += rate * prod_vars[pname]
    util_cost = 0
    if include_utilities and getattr(data, "utilities", None):
        util_cost = pulp.lpSum(
            data.utilities[uname].cost_usd_per_unit * util_use_expr[uname]
            for uname in util_use_expr
            if uname in data.utilities
        )

    prob += revenue - cost - holding - util_cost, "margin"

    # Crude supply caps (named → dual = crude flexibility value)
    for c in data.crudes:
        prob += crude_vars[c.name] <= c.max_supply_kbd, f"crude_supply_{c.name}"

    # Product demand caps
    for name, spec in data.products.items():
        prob += prod_vars[name] <= spec.max_demand_kbd, f"product_demand_{name}"

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

    # Inventory balance: start + prod = use + end  (PIMS single-period tank)
    if include_inventory and inv_end:
        for i in data.intermediates:
            inv = (data.inventory or {}).get(i)
            start = float(inv.start_kbd) if inv is not None else 0.0
            prob += (
                start + inter_prod[i] - inter_use[i] - inv_end[i] == 0,
                f"inv_balance_{i}",
            )
            cap = float(inv.capacity_kbd) if inv is not None else 1e6
            prob += inv_end[i] <= cap, f"tank_{i}"
            # linking residual form
            prob += start + inter_prod[i] >= inter_use[i], f"balance_{i}"
    else:
        for i in data.intermediates:
            prob += inter_prod[i] >= inter_use[i], f"balance_{i}"

    # Aggregate intermediate use from recipes
    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p]
            for p in data.blend_recipes
        )
        prob += inter_use[i] >= rhs, f"blend_use_{i}"

    # Utility capacity duals
    if include_utilities and getattr(data, "utilities", None):
        for uname, uspec in data.utilities.items():
            expr = util_use_expr.get(uname)
            if expr is not None and expr != 0:
                prob += expr <= uspec.capacity, f"utility_cap_{uname}"

    return prob


def extract_monolithic_solution(
    prob: pulp.LpProblem, data: RefineryData, solve_time_s: float
) -> MonolithicResult:
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)

    crude_rates: Dict[str, float] = {}
    product_rates: Dict[str, float] = {}
    intermediate_prod: Dict[str, float] = {}
    intermediate_use: Dict[str, float] = {}
    inventory_end: Dict[str, float] = {}
    utility_supply: Dict[str, float] = {}
    utility_use_cdu: Dict[str, float] = {}
    utility_use_blend: Dict[str, float] = {}

    for v in prob.variables():
        val = float(v.varValue or 0.0)
        n = v.name
        if n.startswith("crude_"):
            crude_rates[n.replace("crude_", "", 1)] = val
        elif n.startswith("product_"):
            product_rates[n.replace("product_", "", 1)] = val
        elif n.startswith("prod_"):
            intermediate_prod[n.replace("prod_", "", 1)] = val
        elif n.startswith("use_"):
            intermediate_use[n.replace("use_", "", 1)] = val
        elif n.startswith("inv_end_"):
            inventory_end[n.replace("inv_end_", "", 1)] = val
        elif n.startswith("util_supply_"):
            utility_supply[n.replace("util_supply_", "", 1)] = val
        elif n.startswith("util_use_cdu_"):
            utility_use_cdu[n.replace("util_use_cdu_", "", 1)] = val
        elif n.startswith("util_use_blend_"):
            utility_use_blend[n.replace("util_use_blend_", "", 1)] = val

    duals: Dict[str, float] = {}
    for name, constraint in prob.constraints.items():
        try:
            duals[name] = float(constraint.pi)
        except Exception:
            duals[name] = 0.0

    # Derive utility draws from intensities when model uses expression form
    if not utility_use_cdu and data.utility_names:
        for u in data.utility_names:
            utility_use_cdu[u] = sum(
                c.utility_use.get(u, 0.0) * crude_rates.get(c.name, 0.0)
                for c in data.crudes
            )
            utility_use_blend[u] = sum(
                data.products[p].utility_use.get(u, 0.0) * product_rates.get(p, 0.0)
                for p in data.products
            )
            utility_supply[u] = utility_use_cdu[u] + utility_use_blend[u]

    return MonolithicResult(
        status=status,
        objective=obj,
        crude_rates=crude_rates,
        product_rates=product_rates,
        intermediate_prod=intermediate_prod,
        intermediate_use=intermediate_use,
        inventory_end=inventory_end,
        utility_supply=utility_supply,
        utility_use_cdu=utility_use_cdu,
        utility_use_blend=utility_use_blend,
        duals=duals,
        solve_time_s=solve_time_s,
        problem=prob,
    )


def solve_monolithic(
    data: RefineryData,
    msg: bool = False,
    *,
    include_inventory: bool = True,
    include_utilities: bool = True,
    time_limit: int = 60,
) -> MonolithicResult:
    prob = build_monolithic_lp(
        data,
        include_inventory=include_inventory,
        include_utilities=include_utilities,
    )
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=time_limit))
    t1 = time.perf_counter()
    return extract_monolithic_solution(prob, data, t1 - t0)


# ---------------------------------------------------------------------------
# PuLP subproblems (ADMM-ready)
# ---------------------------------------------------------------------------


def build_cdu_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_intermediates: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """CDU block: crude selection + yields → intermediate production.

    Objective (max):
        - crude_cost
        + Σ λ_i * prod_i          (sell intermediates at master prices)
        - Σ μ_u * util_use_u      (pay utility prices)
        - ρ Σ |prod_i - z_i|      (optional L1 ADMM consensus penalty)
    """
    intermediate_prices = intermediate_prices or {}
    utility_prices = utility_prices or {}
    consensus_intermediates = consensus_intermediates or {}

    prob = pulp.LpProblem("Block_CDU", pulp.LpMaximize)

    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", lowBound=0, upBound=c.max_supply_kbd)
        for c in data.crudes
    }
    inter_prod = {
        i: pulp.LpVariable(f"prod_{i}", lowBound=0) for i in data.intermediates
    }
    util_use = {
        u: pulp.LpVariable(f"util_use_cdu_{u}", lowBound=0) for u in data.utility_names
    }

    # L1 deviation for ADMM
    d_plus = {
        i: pulp.LpVariable(f"cdu_dplus_{i}", lowBound=0) for i in data.intermediates
    }
    d_minus = {
        i: pulp.LpVariable(f"cdu_dminus_{i}", lowBound=0) for i in data.intermediates
    }

    crude_cost = pulp.lpSum(
        c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes
    )
    price_term = pulp.lpSum(
        float(intermediate_prices.get(i, 0.0)) * inter_prod[i]
        for i in data.intermediates
    )
    util_term = pulp.lpSum(
        float(utility_prices.get(u, 0.0)) * util_use[u] for u in data.utility_names
    )
    penalty = pulp.lpSum(
        float(rho) * (d_plus[i] + d_minus[i]) for i in data.intermediates
    )
    prob += price_term - crude_cost - util_term - penalty, "cdu_augmented_obj"

    prob += (
        pulp.lpSum(crude_vars[c.name] for c in data.crudes) <= data.cdu_capacity_kbd,
        "cdu_capacity",
    )
    for i in data.intermediates:
        prob += (
            inter_prod[i]
            == pulp.lpSum(
                c.yields.get(i, 0.0) * crude_vars[c.name] for c in data.crudes
            ),
            f"yield_{i}",
        )
    for u in data.utility_names:
        prob += (
            util_use[u]
            == pulp.lpSum(
                c.utility_use.get(u, 0.0) * crude_vars[c.name] for c in data.crudes
            ),
            f"util_cdu_def_{u}",
        )
    for i in data.intermediates:
        z_i = float(consensus_intermediates.get(i, 0.0))
        prob += inter_prod[i] - z_i == d_plus[i] - d_minus[i], f"cdu_dev_{i}"

    return prob


def build_inventory_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    consensus_in: Optional[Mapping[str, float]] = None,
    consensus_out: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """Tank-farm block: start + inflow = end + outflow; capacity; holding cost.

    Linking:
      inflow_i  ≈ CDU production (price paid to inflow / sold outflow)
      outflow_i ≈ Blender use

    Objective (max):
        - holding_cost
        + Σ λ_out_i * outflow_i - Σ λ_in_i * inflow_i
        - ρ penalties on consensus deviations
    """
    intermediate_prices = intermediate_prices or {}
    consensus_in = consensus_in or {}
    consensus_out = consensus_out or {}

    # Allow separate in/out prices; default to single λ map
    # Convention: positive λ is intermediate value ($/bbl). Tank pays λ for
    # inflow (buys from CDU) and receives λ for outflow (sells to blender).
    lam = intermediate_prices

    prob = pulp.LpProblem("Block_Inventory", pulp.LpMaximize)

    inflow = {
        i: pulp.LpVariable(f"inv_in_{i}", lowBound=0) for i in data.intermediates
    }
    outflow = {
        i: pulp.LpVariable(f"inv_out_{i}", lowBound=0) for i in data.intermediates
    }
    inv_end = {}
    for i in data.intermediates:
        inv = data.inventory[i]
        inv_end[i] = pulp.LpVariable(
            f"inv_end_{i}", lowBound=0, upBound=inv.capacity_kbd
        )

    d_in_p = {
        i: pulp.LpVariable(f"inv_din_p_{i}", lowBound=0) for i in data.intermediates
    }
    d_in_m = {
        i: pulp.LpVariable(f"inv_din_m_{i}", lowBound=0) for i in data.intermediates
    }
    d_out_p = {
        i: pulp.LpVariable(f"inv_dout_p_{i}", lowBound=0) for i in data.intermediates
    }
    d_out_m = {
        i: pulp.LpVariable(f"inv_dout_m_{i}", lowBound=0) for i in data.intermediates
    }

    hold = pulp.lpSum(
        data.inventory[i].holding_cost_usd_per_bbl * inv_end[i]
        for i in data.intermediates
    )
    # sell outflow at λ, buy inflow at λ
    trade = pulp.lpSum(
        float(lam.get(i, 0.0)) * (outflow[i] - inflow[i]) for i in data.intermediates
    )
    penalty = pulp.lpSum(
        float(rho)
        * (d_in_p[i] + d_in_m[i] + d_out_p[i] + d_out_m[i])
        for i in data.intermediates
    )
    prob += trade - hold - penalty, "inventory_augmented_obj"

    for i in data.intermediates:
        start = data.inventory[i].start_kbd
        prob += start + inflow[i] == inv_end[i] + outflow[i], f"inv_balance_{i}"
        z_in = float(consensus_in.get(i, 0.0))
        z_out = float(consensus_out.get(i, 0.0))
        prob += inflow[i] - z_in == d_in_p[i] - d_in_m[i], f"inv_dev_in_{i}"
        prob += outflow[i] - z_out == d_out_p[i] - d_out_m[i], f"inv_dev_out_{i}"

    return prob


def build_blender_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_intermediates: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """Blender block: consume intermediates → finished products.

    Objective (max):
        + product revenue
        - Σ λ_i * use_i
        - Σ μ_u * util_use_u
        - ρ Σ |use_i - z_i|
    """
    intermediate_prices = intermediate_prices or {}
    utility_prices = utility_prices or {}
    consensus_intermediates = consensus_intermediates or {}

    prob = pulp.LpProblem("Block_Blender", pulp.LpMaximize)

    inter_use = {
        i: pulp.LpVariable(f"use_{i}", lowBound=0) for i in data.intermediates
    }
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=spec.max_demand_kbd)
        for name, spec in data.products.items()
    }
    util_use = {
        u: pulp.LpVariable(f"util_use_blend_{u}", lowBound=0)
        for u in data.utility_names
    }
    d_plus = {
        i: pulp.LpVariable(f"bl_dplus_{i}", lowBound=0) for i in data.intermediates
    }
    d_minus = {
        i: pulp.LpVariable(f"bl_dminus_{i}", lowBound=0) for i in data.intermediates
    }

    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    feed_cost = pulp.lpSum(
        float(intermediate_prices.get(i, 0.0)) * inter_use[i]
        for i in data.intermediates
    )
    util_term = pulp.lpSum(
        float(utility_prices.get(u, 0.0)) * util_use[u] for u in data.utility_names
    )
    penalty = pulp.lpSum(
        float(rho) * (d_plus[i] + d_minus[i]) for i in data.intermediates
    )
    prob += revenue - feed_cost - util_term - penalty, "blender_augmented_obj"

    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p]
            for p in data.blend_recipes
        )
        prob += inter_use[i] >= rhs, f"blend_use_{i}"

    for u in data.utility_names:
        prob += (
            util_use[u]
            == pulp.lpSum(
                data.products[p].utility_use.get(u, 0.0) * prod_vars[p]
                for p in prod_vars
            ),
            f"util_blend_def_{u}",
        )

    for i in data.intermediates:
        z_i = float(consensus_intermediates.get(i, 0.0))
        prob += inter_use[i] - z_i == d_plus[i] - d_minus[i], f"bl_dev_{i}"

    return prob


def build_utilities_subproblem(
    data: RefineryData,
    *,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_demand: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """Utilities block: supply fuel_gas/steam/power up to capacity.

    Objective (max):
        + Σ μ_u * supply_u   (sell utilities at master prices)
        - Σ cost_u * supply_u
        - ρ Σ |supply_u - z_u|
    """
    utility_prices = utility_prices or {}
    consensus_demand = consensus_demand or {}

    prob = pulp.LpProblem("Block_Utilities", pulp.LpMaximize)

    supply = {
        u: pulp.LpVariable(
            f"util_supply_{u}",
            lowBound=0,
            upBound=data.utilities[u].capacity,
        )
        for u in data.utility_names
    }
    d_plus = {
        u: pulp.LpVariable(f"util_dplus_{u}", lowBound=0) for u in data.utility_names
    }
    d_minus = {
        u: pulp.LpVariable(f"util_dminus_{u}", lowBound=0) for u in data.utility_names
    }

    revenue = pulp.lpSum(
        float(utility_prices.get(u, 0.0)) * supply[u] for u in data.utility_names
    )
    cost = pulp.lpSum(
        data.utilities[u].cost_usd_per_unit * supply[u] for u in data.utility_names
    )
    penalty = pulp.lpSum(
        float(rho) * (d_plus[u] + d_minus[u]) for u in data.utility_names
    )
    prob += revenue - cost - penalty, "utilities_augmented_obj"

    for u in data.utility_names:
        z_u = float(consensus_demand.get(u, 0.0))
        # soft consensus: supply tracks demanded z
        prob += supply[u] - z_u == d_plus[u] - d_minus[u], f"util_dev_{u}"

    return prob


def _solve_pulp(
    prob: pulp.LpProblem,
    *,
    msg: bool = False,
    time_limit: int = 30,
    warm_start: Optional[Mapping[str, float]] = None,
) -> tuple[str, float, Dict[str, float], float]:
    if warm_start:
        for v in prob.variables():
            if v.name in warm_start:
                v.setInitialValue(float(warm_start[v.name]))
    solver = pulp.PULP_CBC_CMD(
        msg=int(msg),
        timeLimit=time_limit,
        warmStart=bool(warm_start),
    )
    t0 = time.perf_counter()
    status_code = prob.solve(solver)
    t1 = time.perf_counter()
    status = pulp.LpStatus.get(status_code, str(status_code))
    obj = float(pulp.value(prob.objective) or 0.0)
    primals = {v.name: float(v.varValue or 0.0) for v in prob.variables()}
    return status, obj, primals, t1 - t0


def _prefix_map(primals: Mapping[str, float], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in primals.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = float(v)
    return out


def solve_cdu_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_intermediates: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
    warm_start: Optional[Mapping[str, float]] = None,
    msg: bool = False,
) -> SubproblemResult:
    prob = build_cdu_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        consensus_intermediates=consensus_intermediates,
        rho=rho,
    )
    status, obj, primals, dt = _solve_pulp(prob, msg=msg, warm_start=warm_start)
    return SubproblemResult(
        block=BlockNames.CDU,
        status=status,
        objective=obj,
        primals=primals,
        linking_intermediates=_prefix_map(primals, "prod_"),
        linking_utilities=_prefix_map(primals, "util_use_cdu_"),
        solve_time_s=dt,
        problem=prob,
    )


def solve_inventory_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    consensus_in: Optional[Mapping[str, float]] = None,
    consensus_out: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
    warm_start: Optional[Mapping[str, float]] = None,
    msg: bool = False,
) -> SubproblemResult:
    prob = build_inventory_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        consensus_in=consensus_in,
        consensus_out=consensus_out,
        rho=rho,
    )
    status, obj, primals, dt = _solve_pulp(prob, msg=msg, warm_start=warm_start)
    # report outflow as primary linking (to blender)
    return SubproblemResult(
        block=BlockNames.INVENTORY,
        status=status,
        objective=obj,
        primals=primals,
        linking_intermediates=_prefix_map(primals, "inv_out_"),
        linking_utilities={},
        solve_time_s=dt,
        problem=prob,
    )


def solve_blender_subproblem(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_intermediates: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
    warm_start: Optional[Mapping[str, float]] = None,
    msg: bool = False,
) -> SubproblemResult:
    prob = build_blender_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        consensus_intermediates=consensus_intermediates,
        rho=rho,
    )
    status, obj, primals, dt = _solve_pulp(prob, msg=msg, warm_start=warm_start)
    return SubproblemResult(
        block=BlockNames.BLENDER,
        status=status,
        objective=obj,
        primals=primals,
        linking_intermediates=_prefix_map(primals, "use_"),
        linking_utilities=_prefix_map(primals, "util_use_blend_"),
        solve_time_s=dt,
        problem=prob,
    )


def solve_utilities_subproblem(
    data: RefineryData,
    *,
    utility_prices: Optional[Mapping[str, float]] = None,
    consensus_demand: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
    warm_start: Optional[Mapping[str, float]] = None,
    msg: bool = False,
) -> SubproblemResult:
    prob = build_utilities_subproblem(
        data,
        utility_prices=utility_prices,
        consensus_demand=consensus_demand,
        rho=rho,
    )
    status, obj, primals, dt = _solve_pulp(prob, msg=msg, warm_start=warm_start)
    return SubproblemResult(
        block=BlockNames.UTILITIES,
        status=status,
        objective=obj,
        primals=primals,
        linking_intermediates={},
        linking_utilities=_prefix_map(primals, "util_supply_"),
        solve_time_s=dt,
        problem=prob,
    )


def solve_all_subproblems(
    data: RefineryData,
    *,
    intermediate_prices: Optional[Mapping[str, float]] = None,
    utility_prices: Optional[Mapping[str, float]] = None,
    rho: float = 0.0,
    msg: bool = False,
) -> Dict[str, SubproblemResult]:
    """Solve all four PuLP blocks independently at given dual prices (parallelizable)."""
    intermediate_prices = intermediate_prices or {i: 0.0 for i in data.intermediates}
    utility_prices = utility_prices or {u: 0.0 for u in data.utility_names}

    cdu = solve_cdu_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        rho=rho,
        msg=msg,
    )
    inv = solve_inventory_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        consensus_in=cdu.linking_intermediates,
        consensus_out={i: 0.0 for i in data.intermediates},
        rho=rho,
        msg=msg,
    )
    blend = solve_blender_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        consensus_intermediates=inv.linking_intermediates,
        rho=rho,
        msg=msg,
    )
    # total utility demand from CDU + blender
    demand = {
        u: cdu.linking_utilities.get(u, 0.0) + blend.linking_utilities.get(u, 0.0)
        for u in data.utility_names
    }
    util = solve_utilities_subproblem(
        data,
        utility_prices=utility_prices,
        consensus_demand=demand,
        rho=rho,
        msg=msg,
    )
    return {
        BlockNames.CDU: cdu,
        BlockNames.INVENTORY: inv,
        BlockNames.BLENDER: blend,
        BlockNames.UTILITIES: util,
    }


def describe_block_angular_structure(data: RefineryData) -> Dict[str, object]:
    """Machine-readable map of blocks, local vars, and linking constraints."""
    return {
        "blocks": BlockNames.all(),
        "intermediates": list(data.intermediates),
        "utilities": list(data.utility_names),
        "linking_constraints": {
            "inventory_balance": [
                f"start_{i} + prod_{i} = inv_end_{i} + use_{i}"
                for i in data.intermediates
            ],
            "utility_balance": [
                f"supply_{u} >= use_cdu_{u} + use_blend_{u}"
                for u in data.utility_names
            ],
        },
        "local_constraints": {
            BlockNames.CDU: ["cdu_capacity", "yields", "utility_draw_from_charge"],
            BlockNames.INVENTORY: ["tank_capacity", "material_balance", "holding_cost"],
            BlockNames.BLENDER: ["blend_recipes", "product_demand", "utility_draw"],
            BlockNames.UTILITIES: ["utility_capacity", "utility_cost"],
        },
        "crudes": [c.name for c in data.crudes],
        "products": list(data.products.keys()),
    }
