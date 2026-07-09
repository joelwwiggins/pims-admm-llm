"""Per-block augmented-Lagrangian subproblems.

Backends
--------
* **scipy_l2** — classical quadratic ADMM (SLSQP). Proximal term + consensus z
  for uniqueness on degenerate LP faces.
* **pulp_l1** — Worker-2 PuLP multi-block builders with L1 consensus penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from pims_admm_llm.models.data import RefineryData


@dataclass
class BlockSolution:
    block: str
    status: str
    local_obj: float
    augmented_obj: float
    linking: Dict[str, float]
    linking_utilities: Dict[str, float]
    primals: Dict[str, float]
    message: str = ""
    solve_time_s: float = 0.0


def _arr(names: Sequence[str], d: Mapping[str, float], default: float = 0.0) -> np.ndarray:
    return np.array([float(d.get(n, default)) for n in names], dtype=float)


# ---------------------------------------------------------------------------
# PuLP / L1 backend
# ---------------------------------------------------------------------------


def solve_blocks_pulp(
    data: RefineryData,
    *,
    intermediate_prices: Mapping[str, float],
    utility_prices: Mapping[str, float],
    z_intermediates: Mapping[str, float],
    z_utilities: Mapping[str, float],
    rho: float,
) -> Dict[str, BlockSolution]:
    from pims_admm_llm.models.blocks import (
        BlockNames,
        solve_blender_subproblem,
        solve_cdu_subproblem,
        solve_inventory_subproblem,
        solve_utilities_subproblem,
    )

    cdu = solve_cdu_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        consensus_intermediates=z_intermediates,
        rho=rho,
    )
    inv = solve_inventory_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        consensus_in=cdu.linking_intermediates,
        consensus_out=z_intermediates,
        rho=rho,
    )
    blend = solve_blender_subproblem(
        data,
        intermediate_prices=intermediate_prices,
        utility_prices=utility_prices,
        consensus_intermediates=inv.linking_intermediates,
        rho=rho,
    )
    demand = {
        u: cdu.linking_utilities.get(u, 0.0) + blend.linking_utilities.get(u, 0.0)
        for u in data.utility_names
    }
    util = solve_utilities_subproblem(
        data,
        utility_prices=utility_prices,
        consensus_demand=demand
        if any(abs(v) > 1e-12 for v in demand.values())
        else z_utilities,
        rho=rho,
    )

    def wrap(res) -> BlockSolution:
        return BlockSolution(
            block=res.block,
            status=res.status,
            local_obj=float(res.objective),
            augmented_obj=float(res.objective),
            linking=dict(res.linking_intermediates),
            linking_utilities=dict(res.linking_utilities),
            primals=dict(res.primals),
            solve_time_s=float(res.solve_time_s),
        )

    return {
        BlockNames.CDU: wrap(cdu),
        BlockNames.INVENTORY: wrap(inv),
        BlockNames.BLENDER: wrap(blend),
        BlockNames.UTILITIES: wrap(util),
        "_utility_demand": BlockSolution(
            block="_utility_demand",
            status="aggregate",
            local_obj=0.0,
            augmented_obj=0.0,
            linking={},
            linking_utilities=demand,
            primals=demand,
        ),
    }


# ---------------------------------------------------------------------------
# SciPy L2 QP backend
# ---------------------------------------------------------------------------


def _cdu_matrices(data: RefineryData):
    crudes = data.crudes
    intermediates = list(data.intermediates)
    n_c, n_i = len(crudes), len(intermediates)
    Y = np.zeros((n_c, n_i))
    prices = np.zeros(n_c)
    ub = np.zeros(n_c)
    for i, c in enumerate(crudes):
        prices[i] = c.price_usd_per_bbl
        ub[i] = c.max_supply_kbd
        for j, name in enumerate(intermediates):
            Y[i, j] = c.yields.get(name, 0.0)
    return intermediates, crudes, Y, prices, ub


def _blend_matrices(data: RefineryData):
    products = list(data.products.keys())
    intermediates = list(data.intermediates)
    n_p, n_i = len(products), len(intermediates)
    R = np.zeros((n_p, n_i))
    prices = np.zeros(n_p)
    ub = np.zeros(n_p)
    for i, name in enumerate(products):
        prices[i] = data.products[name].price_usd_per_bbl
        ub[i] = data.products[name].max_demand_kbd
        for j, iname in enumerate(intermediates):
            R[i, j] = data.blend_recipes.get(name, {}).get(iname, 0.0)
    return intermediates, products, R, prices, ub


def solve_cdu_block_qp(
    data: RefineryData,
    lam: Mapping[str, float],
    *,
    z: Mapping[str, float],
    rho: float,
    x0: Optional[Mapping[str, float]] = None,
    prox_center: Optional[Mapping[str, float]] = None,
    prox_weight: float = 0.0,
) -> BlockSolution:
    """CDU: min cost + λᵀprod + (ρ/2)||prod - z||² + (γ/2)||x - x̄||²."""
    intermediates, crudes, Y, prices, ub = _cdu_matrices(data)
    n_c = len(crudes)
    lam_v = _arr(intermediates, lam)
    z_v = _arr(intermediates, z)
    gamma = float(prox_weight)
    if prox_center is not None:
        xbar = np.array([float(prox_center.get(c.name, 0.0)) for c in crudes], dtype=float)
    else:
        xbar = np.zeros(n_c)

    if x0 is not None:
        x_init = np.array([float(x0.get(c.name, 0.0)) for c in crudes], dtype=float)
    else:
        x_init = np.minimum(ub, data.cdu_capacity_kbd / max(n_c, 1) * 0.5)

    def prod_of(x):
        return Y.T @ x

    def fun(x):
        prod = prod_of(x)
        val = float(prices @ x + lam_v @ prod + 0.5 * rho * np.sum((prod - z_v) ** 2))
        if gamma > 0:
            val += 0.5 * gamma * float(np.sum((x - xbar) ** 2))
        return val

    def jac(x):
        prod = prod_of(x)
        g = prices + Y @ lam_v + rho * (Y @ (prod - z_v))
        if gamma > 0:
            g = g + gamma * (x - xbar)
        return g

    cons = [
        {
            "type": "ineq",
            "fun": lambda x: data.cdu_capacity_kbd - float(np.sum(x)),
            "jac": lambda x: -np.ones(n_c),
        }
    ]
    bounds = [(0.0, float(ub[i])) for i in range(n_c)]
    res = minimize(
        fun,
        x_init,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 1000, "disp": False},
    )
    xv = np.clip(res.x, 0.0, ub)
    s = float(np.sum(xv))
    if s > data.cdu_capacity_kbd + 1e-9:
        xv *= data.cdu_capacity_kbd / s
    prod_v = prod_of(xv)
    return BlockSolution(
        block="CDU",
        status="optimal" if res.success else f"solver_{res.message}",
        local_obj=float(-prices @ xv),
        augmented_obj=float(fun(xv)),
        linking={intermediates[j]: float(prod_v[j]) for j in range(len(intermediates))},
        linking_utilities={},
        primals={c.name: float(xv[i]) for i, c in enumerate(crudes)},
        message=str(res.message),
    )


def solve_blender_block_qp(
    data: RefineryData,
    lam: Mapping[str, float],
    *,
    z: Mapping[str, float],
    rho: float,
    x0: Optional[Mapping[str, float]] = None,
    prox_center: Optional[Mapping[str, float]] = None,
    prox_weight: float = 0.0,
) -> BlockSolution:
    """Blender: min -revenue - λᵀuse + (ρ/2)||use - z||² + (γ/2)||y - ȳ||².

    Note: λ is the *same* dual as CDU's; consumer incidence is −use in residual
    prod−use, so the linear dual term on use is −λ (implemented here as −λᵀuse
    in the min objective). Callers pass the balance dual λ; z is the consensus
    target for use.
    """
    intermediates, products, R, prices, ub = _blend_matrices(data)
    n_p = len(products)
    lam_v = _arr(intermediates, lam)
    z_v = _arr(intermediates, z)
    gamma = float(prox_weight)
    if prox_center is not None:
        ybar = np.array([float(prox_center.get(p, 0.0)) for p in products], dtype=float)
    else:
        ybar = np.zeros(n_p)

    if x0 is not None:
        y_init = np.array([float(x0.get(p, 0.0)) for p in products], dtype=float)
    else:
        y_init = ub * 0.3

    def use_of(y):
        return R.T @ y

    def fun(y):
        use = use_of(y)
        # min -rev - λ·use + (ρ/2)||use - z||²
        val = float(-prices @ y - lam_v @ use + 0.5 * rho * np.sum((use - z_v) ** 2))
        if gamma > 0:
            val += 0.5 * gamma * float(np.sum((y - ybar) ** 2))
        return val

    def jac(y):
        use = use_of(y)
        g = -prices - R @ lam_v + rho * (R @ (use - z_v))
        if gamma > 0:
            g = g + gamma * (y - ybar)
        return g

    bounds = [(0.0, float(ub[i])) for i in range(n_p)]
    res = minimize(
        fun,
        y_init,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        options={"ftol": 1e-14, "maxiter": 1000, "disp": False},
    )
    yv = np.clip(res.x, 0.0, ub)
    use_v = use_of(yv)
    return BlockSolution(
        block="Blender",
        status="optimal" if res.success else f"solver_{res.message}",
        local_obj=float(prices @ yv),
        augmented_obj=float(fun(yv)),
        linking={intermediates[j]: float(use_v[j]) for j in range(len(intermediates))},
        linking_utilities={},
        primals={products[i]: float(yv[i]) for i in range(n_p)},
        message=str(res.message),
    )


# Back-compat: balance form (use_fixed / prod_fixed)
def solve_cdu_block(data, lam, *, use_fixed=None, z=None, rho=1.0, x0=None, **kw):
    target = z if z is not None else (use_fixed or {})
    return solve_cdu_block_qp(data, lam, z=target, rho=rho, x0=x0, **kw)


def solve_blender_block(data, lam, *, prod_fixed=None, z=None, rho=1.0, x0=None, **kw):
    target = z if z is not None else (prod_fixed or {})
    return solve_blender_block_qp(data, lam, z=target, rho=rho, x0=x0, **kw)


def solve_cdu_consensus(data, lam, z, rho, x0=None, **kw):
    return solve_cdu_block_qp(data, lam, z=z, rho=rho, x0=x0, **kw)


def solve_blender_consensus(data, lam, z, rho, x0=None, **kw):
    # for consensus, blender uses dual −λ on residual use−z if residual is prod−use
    # with separate dual copies; here we keep same interface as solve_blender_block_qp
    return solve_blender_block_qp(data, lam, z=z, rho=rho, x0=x0, **kw)
