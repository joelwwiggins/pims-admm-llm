"""Per-block LP subproblems for ADMM / DW-style coordination.

Each block is a standalone PuLP problem. Master sends dual prices λ (and
optional consensus z, ρ) for linking intermediate streams. Warm-start values
from the previous iteration are applied when available.

Quadratic ADMM term (ρ/2)||x-z||² is approximated with an LP-friendly L1
penalty ρ·|x-z| so we stay in CBC/PuLP (pure LP).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pulp

from pims_admm_llm.models.data import RefineryData

LINKING_STREAMS = ("naphtha", "distillate", "gasoil", "residue")


@dataclass
class BlockSolveRequest:
    """Everything needed to solve one block (picklable for process pool)."""

    block_name: str
    prices: Dict[str, float] = field(default_factory=dict)  # λ on linking streams
    consensus: Dict[str, float] = field(default_factory=dict)  # z
    rho: float = 0.0  # L1 ADMM penalty weight
    warm_start: Dict[str, float] = field(default_factory=dict)
    time_limit_s: float = 30.0
    msg: bool = False
    # Optional payload overrides (for process workers that re-load data)
    data_path: Optional[str] = None


@dataclass
class BlockSolveResult:
    block_name: str
    status: str
    local_objective: float
    linking_flows: Dict[str, float]  # x on linking streams (prod or use side)
    primal: Dict[str, float]
    solve_time_s: float
    warm_started: bool
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_name": self.block_name,
            "status": self.status,
            "local_objective": self.local_objective,
            "linking_flows": dict(self.linking_flows),
            "primal": dict(self.primal),
            "solve_time_s": self.solve_time_s,
            "warm_started": self.warm_started,
            "message": self.message,
        }


def _apply_warm_start(prob: pulp.LpProblem, warm_start: Mapping[str, float]) -> int:
    """Set initial values on variables; returns count of vars warm-started."""
    if not warm_start:
        return 0
    n = 0
    for v in prob.variables():
        if v.name in warm_start:
            val = float(warm_start[v.name])
            v.setInitialValue(val)
            n += 1
    return n


def _extract_primal(prob: pulp.LpProblem) -> Dict[str, float]:
    return {v.name: float(v.varValue or 0.0) for v in prob.variables()}


def build_cdu_block(
    data: RefineryData,
    prices: Mapping[str, float] | None = None,
    consensus: Mapping[str, float] | None = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """CDU block: choose crude slate, produce intermediates.

    Objective (maximize):
        - sum crude_cost
        + sum_i λ_i * prod_i
        - rho * |prod_i - z_i|   (optional L1 ADMM penalty)
    """
    prices = prices or {}
    consensus = consensus or {}
    prob = pulp.LpProblem("Block_CDU", pulp.LpMaximize)

    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", lowBound=0, upBound=c.max_supply_kbd)
        for c in data.crudes
    }
    inter_prod = {
        i: pulp.LpVariable(f"prod_{i}", lowBound=0) for i in data.intermediates
    }

    # Deviation vars for L1 ADMM: |prod - z| = d_plus + d_minus
    d_plus = {
        i: pulp.LpVariable(f"cdu_dplus_{i}", lowBound=0) for i in data.intermediates
    }
    d_minus = {
        i: pulp.LpVariable(f"cdu_dminus_{i}", lowBound=0) for i in data.intermediates
    }

    crude_cost = pulp.lpSum(
        c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes
    )
    # Selling intermediates to master at λ (shadow prices)
    price_term = pulp.lpSum(
        float(prices.get(i, 0.0)) * inter_prod[i] for i in data.intermediates
    )
    penalty = pulp.lpSum(
        float(rho) * (d_plus[i] + d_minus[i]) for i in data.intermediates
    )
    # Local "margin" contribution: value of intermediates sold at λ minus crude cost
    prob += price_term - crude_cost - penalty, "cdu_augmented_obj"

    # Capacity
    prob += (
        pulp.lpSum(crude_vars[c.name] for c in data.crudes) <= data.cdu_capacity_kbd,
        "cdu_capacity",
    )

    # Yields
    for i in data.intermediates:
        prob += (
            inter_prod[i]
            == pulp.lpSum(
                c.yields.get(i, 0.0) * crude_vars[c.name] for c in data.crudes
            ),
            f"yield_{i}",
        )

    # L1 deviation: prod - z = d_plus - d_minus
    for i in data.intermediates:
        z_i = float(consensus.get(i, 0.0))
        prob += inter_prod[i] - z_i == d_plus[i] - d_minus[i], f"cdu_dev_{i}"

    return prob


def build_blender_block(
    data: RefineryData,
    prices: Mapping[str, float] | None = None,
    consensus: Mapping[str, float] | None = None,
    rho: float = 0.0,
) -> pulp.LpProblem:
    """Blender block: consume intermediates, make products.

    Objective (maximize):
        + product revenue
        - sum_i λ_i * use_i     (pay master prices for intermediates)
        - rho * |use_i - z_i|
    """
    prices = prices or {}
    consensus = consensus or {}
    prob = pulp.LpProblem("Block_Blender", pulp.LpMaximize)

    inter_use = {
        i: pulp.LpVariable(f"use_{i}", lowBound=0) for i in data.intermediates
    }
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", lowBound=0, upBound=spec.max_demand_kbd)
        for name, spec in data.products.items()
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
    price_term = pulp.lpSum(
        float(prices.get(i, 0.0)) * inter_use[i] for i in data.intermediates
    )
    penalty = pulp.lpSum(
        float(rho) * (d_plus[i] + d_minus[i]) for i in data.intermediates
    )
    # Buy intermediates at λ, sell products
    prob += revenue - price_term - penalty, "blender_augmented_obj"

    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p]
            for p in data.blend_recipes
        )
        prob += inter_use[i] >= rhs, f"blend_use_{i}"

    for i in data.intermediates:
        z_i = float(consensus.get(i, 0.0))
        prob += inter_use[i] - z_i == d_plus[i] - d_minus[i], f"bl_dev_{i}"

    return prob


def _linking_from_primal(block_name: str, primal: Mapping[str, float]) -> Dict[str, float]:
    flows: Dict[str, float] = {}
    prefix = "prod_" if block_name.upper() == "CDU" else "use_"
    for stream in LINKING_STREAMS:
        key = f"{prefix}{stream}"
        if key in primal:
            flows[stream] = float(primal[key])
        else:
            # fallback scan
            for k, v in primal.items():
                if k.endswith(stream) and (k.startswith("prod_") or k.startswith("use_")):
                    flows[stream] = float(v)
                    break
            else:
                flows[stream] = 0.0
    return flows


def solve_block(
    data: RefineryData,
    request: BlockSolveRequest,
) -> BlockSolveResult:
    """Build + solve one block with prices / consensus / warm-start."""
    name = request.block_name
    upper = name.upper()
    if upper == "CDU":
        prob = build_cdu_block(
            data, request.prices, request.consensus, request.rho
        )
    elif upper in ("BLENDER", "BLEND"):
        name = "Blender"
        prob = build_blender_block(
            data, request.prices, request.consensus, request.rho
        )
    else:
        return BlockSolveResult(
            block_name=name,
            status="Error",
            local_objective=0.0,
            linking_flows={},
            primal={},
            solve_time_s=0.0,
            warm_started=False,
            message=f"Unknown block: {request.block_name}",
        )

    n_warm = _apply_warm_start(prob, request.warm_start)
    warm = n_warm > 0

    solver = pulp.PULP_CBC_CMD(
        msg=int(request.msg),
        timeLimit=request.time_limit_s,
        warmStart=warm,
        options=["sec", str(int(request.time_limit_s))],
    )

    t0 = time.perf_counter()
    status_code = prob.solve(solver)
    t1 = time.perf_counter()

    status = pulp.LpStatus.get(status_code, str(status_code))
    obj = float(pulp.value(prob.objective) or 0.0)
    primal = _extract_primal(prob)
    linking = _linking_from_primal(name, primal)

    return BlockSolveResult(
        block_name=name,
        status=status,
        local_objective=obj,
        linking_flows=linking,
        primal=primal,
        solve_time_s=t1 - t0,
        warm_started=warm,
        message=f"warm_vars={n_warm}",
    )


def default_block_names() -> List[str]:
    return ["CDU", "Blender"]
