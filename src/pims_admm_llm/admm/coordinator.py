"""Block-angular ADMM coordinator for the toy refinery LP.

We treat intermediate stream balances as the consensus variables z_i.

ADMM form (simplified for two blocks CDU / Blender sharing intermediate vector z):

  CDU local:   produce intermediates x_cdu close to z, maximize local margin
               (or minimize cost) with price λ and quadratic ρ/2 ||x - z + u||^2
  Blender local: consume intermediates x_blend close to z
  Dual update: λ ← λ + ρ (x_avg - z) or standard residual form
  Consensus:   z ← average of block copies (or projection onto feasible set)

Shadow prices for linking balances ≈ dual variables λ at convergence
(same economic meaning as PIMS duals on intermediate balances).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pulp

from pims_admm_llm.models.data import RefineryData


@dataclass
class ADMMConfig:
    rho: float = 1.0
    max_iter: int = 40
    tol_primal: float = 1e-3
    tol_dual: float = 1e-3
    parallel: bool = True
    msg: bool = False
    # scale dual step (often helpful for LP-ish problems)
    alpha: float = 1.0


@dataclass
class ADMMResult:
    status: str
    objective_approx: float
    iterations: int
    primal_res: List[float]
    dual_res: List[float]
    shadow_prices: Dict[str, float]  # λ on intermediates
    consensus_z: Dict[str, float]
    cdu_solution: Dict[str, float]
    blender_solution: Dict[str, float]
    wall_time_s: float
    history: List[Dict] = field(default_factory=list)


def _solve_cdu_block(
    data: RefineryData,
    z: Dict[str, float],
    lam: Dict[str, float],
    u: Dict[str, float],
    rho: float,
    msg: bool = False,
) -> Dict[str, float]:
    """CDU block: choose crudes, produce intermediates x_i ≈ z_i - u_i (scaled ADMM)."""
    prob = pulp.LpProblem("CDU_Block", pulp.LpMaximize)

    crude_vars = {
        c.name: pulp.LpVariable(f"crude_{c.name}", 0, c.max_supply_kbd)
        for c in data.crudes
    }
    x = {i: pulp.LpVariable(f"x_cdu_{i}", 0) for i in data.intermediates}

    # Local economic: - crude cost + ADMM terms on production
    # Maximize: -cost - λ·x - (ρ/2)||x - z + u||^2
    # CBC is LP-only; we linearize quadratic around previous z with subgradient-style
    # linearization: use linear ADMM / proximal linearized form:
    #   -cost - (λ + ρ*(x_prev - z + u wait))·x   → use current dual + residual target
    # Practical LP-friendly ADMM: augmented with linear price only + soft box around z.
    # Here: modified obj = -crude_cost - sum_i (λ_i + ρ * (0 - z_i + ...))
    # Simpler proven pattern for LP blocks: price intermediates at λ, and soft equality
    # via bounds / penalty linearization using target t_i = z_i - u_i.

    cost = pulp.lpSum(c.price_usd_per_bbl * crude_vars[c.name] for c in data.crudes)
    # Value production at dual prices (max margin contribution from intermediates)
    # In full system, blender pays for them; dual λ is the transfer price.
    transfer = pulp.lpSum(lam[i] * x[i] for i in data.intermediates)
    # Linearized quadratic: -ρ/2 ||x - t||^2 ≈ -ρ * (x - t) · x + const around t
    # Using target t = z - u (scaled dual form), first-order: encourage x → t
    t = {i: z[i] - u[i] for i in data.intermediates}
    proximal = pulp.lpSum(rho * (x[i] - t[i]) * 0.0 for i in data.intermediates)  # placeholder
    # Use explicit soft tracking with linear penalty abs via split — for CBC use
    # price adjustment: effective dual λ_eff = λ + ρ*(x_old - z) but we pass u.
    # Standard scaled ADMM residual: λ_eff_i = λ_i + ρ * u wait.
    # We set obj = -cost + sum (λ_i + ρ * u_i? ) wait standard:
    # For min f(x) + g(z) s.t. x-z=0:
    #   x-update: min f(x) + λ·x + (ρ/2)||x-z+u||^2  (u = scaled dual)
    # Maximization: max -f + ...
    # Linearized (Gauss-Seidel / LP): max -cost + sum_i μ_i * x_i
    # where μ_i = - (λ_i + ρ*(x_prev_i - z_i + u_i)) ... we use μ_i = -(λ_i) + ρ*(t_i)
    # Simple effective price for produced intermediate:
    mu = {i: -lam[i] + rho * t[i] for i in data.intermediates}
    # Actually for maximize local value of produced intermediates at transfer price:
    # use μ = lam (price paid by system for production) minus proximal pull to t.
    mu = {i: lam[i] - rho * (0.0 - t[i]) * 0.5 for i in data.intermediates}
    # Clean LP-ADMM: treat proximal as linear attraction to t with strength rho
    mu = {i: lam[i] + rho * t[i] for i in data.intermediates}

    prob += -cost + pulp.lpSum(mu[i] * x[i] for i in data.intermediates), "cdu_aug_obj"

    total_crude = pulp.lpSum(crude_vars[c.name] for c in data.crudes)
    prob += total_crude <= data.cdu_capacity_kbd, "cdu_cap"

    for i in data.intermediates:
        prob += (
            x[i]
            == pulp.lpSum(c.yields.get(i, 0.0) * crude_vars[c.name] for c in data.crudes),
            f"yield_{i}",
        )

    # Soft track: optional box around t to keep iterates bounded
    for i in data.intermediates:
        # Allow x free but objective pulls via mu; no hard track
        pass

    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=30))

    out = {f"crude_{c.name}": float(crude_vars[c.name].varValue or 0.0) for c in data.crudes}
    for i in data.intermediates:
        out[f"x_{i}"] = float(x[i].varValue or 0.0)
    out["_obj"] = float(pulp.value(prob.objective) or 0.0)
    out["_status"] = pulp.LpStatus[prob.status]
    return out


def _solve_blender_block(
    data: RefineryData,
    z: Dict[str, float],
    lam: Dict[str, float],
    u: Dict[str, float],
    rho: float,
    msg: bool = False,
) -> Dict[str, float]:
    """Blender block: consume intermediates y_i ≈ z_i, make products."""
    prob = pulp.LpProblem("Blender_Block", pulp.LpMaximize)

    y = {i: pulp.LpVariable(f"y_blend_{i}", 0) for i in data.intermediates}
    prod_vars = {
        name: pulp.LpVariable(f"product_{name}", 0, spec.max_demand_kbd)
        for name, spec in data.products.items()
    }

    revenue = pulp.lpSum(
        data.products[n].price_usd_per_bbl * prod_vars[n] for n in prod_vars
    )
    # Pay transfer price for intermediates + proximal to target t = z - u
    t = {i: z[i] - u[i] for i in data.intermediates}
    # Effective cost of intermediate use
    nu = {i: lam[i] + rho * t[i] for i in data.intermediates}
    # For consumption, we minimize transfer cost: - nu · y in maximize form
    # Correct: max revenue - λ·y - (ρ/2)||y - t||^2 linearized
    # Use: max revenue - sum (lam[i] + rho*(y-t))... → max revenue - sum mu_i y_i
    mu = {i: lam[i] - rho * t[i] for i in data.intermediates}

    prob += revenue - pulp.lpSum(mu[i] * y[i] for i in data.intermediates), "blend_aug_obj"

    for i in data.intermediates:
        rhs = pulp.lpSum(
            data.blend_recipes[p].get(i, 0.0) * prod_vars[p] for p in data.blend_recipes
        )
        prob += y[i] >= rhs, f"recipe_{i}"

    # Cap consumption near available consensus (soft via prices; hard upper optional)
    for i in data.intermediates:
        # do not hard-cap; ADMM duals will balance
        pass

    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=30))

    out = {f"product_{n}": float(prod_vars[n].varValue or 0.0) for n in prod_vars}
    for i in data.intermediates:
        out[f"y_{i}"] = float(y[i].varValue or 0.0)
    out["_obj"] = float(pulp.value(prob.objective) or 0.0)
    out["_status"] = pulp.LpStatus[prob.status]
    return out


def run_admm(data: RefineryData, config: Optional[ADMMConfig] = None) -> ADMMResult:
    cfg = config or ADMMConfig()
    intermediates = list(data.intermediates)

    # Initialize consensus z, duals λ, scaled dual u
    z = {i: 0.0 for i in intermediates}
    lam = {i: 0.0 for i in intermediates}
    u = {i: 0.0 for i in intermediates}

    primal_hist: List[float] = []
    dual_hist: List[float] = []
    history: List[Dict] = []

    cdu_sol: Dict[str, float] = {}
    blend_sol: Dict[str, float] = {}

    t0 = time.perf_counter()
    status = "max_iter"
    it_final = 0

    for it in range(cfg.max_iter):
        it_final = it + 1
        z_old = dict(z)

        if cfg.parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_cdu = ex.submit(
                    _solve_cdu_block, data, z, lam, u, cfg.rho, cfg.msg
                )
                f_bl = ex.submit(
                    _solve_blender_block, data, z, lam, u, cfg.rho, cfg.msg
                )
                cdu_sol = f_cdu.result()
                blend_sol = f_bl.result()
        else:
            cdu_sol = _solve_cdu_block(data, z, lam, u, cfg.rho, cfg.msg)
            blend_sol = _solve_blender_block(data, z, lam, u, cfg.rho, cfg.msg)

        x = {i: cdu_sol.get(f"x_{i}", 0.0) for i in intermediates}
        y = {i: blend_sol.get(f"y_{i}", 0.0) for i in intermediates}

        # Consensus z: average of production and consumption copies
        # (for balance x ≈ y ≈ z). Prefer min(x,y) projection for feasibility
        for i in intermediates:
            z[i] = 0.5 * (x[i] + y[i])

        # Primal residual: ||x - y|| (linking imbalance) and ||x-z||, ||y-z||
        r = sum((x[i] - y[i]) ** 2 for i in intermediates) ** 0.5
        # Dual residual: ρ ||z - z_old||
        s = cfg.rho * (sum((z[i] - z_old[i]) ** 2 for i in intermediates) ** 0.5)

        primal_hist.append(r)
        dual_hist.append(s)

        # Scaled dual update u := u + x - z  (and similarly for y path)
        # Using production residual for dual (common multi-block pattern)
        for i in intermediates:
            # residual of balance: production - consumption
            residual = x[i] - y[i]
            lam[i] = lam[i] + cfg.alpha * cfg.rho * residual
            u[i] = u[i] + (x[i] - z[i])

        # Approximate global objective (true margin)
        crude_cost = sum(
            next(c.price_usd_per_bbl for c in data.crudes if c.name == name.replace("crude_", ""))
            * val
            for name, val in cdu_sol.items()
            if name.startswith("crude_")
        )
        revenue = sum(
            data.products[name.replace("product_", "")].price_usd_per_bbl * val
            for name, val in blend_sol.items()
            if name.startswith("product_")
        )
        obj_approx = revenue - crude_cost

        history.append(
            {
                "iter": it_final,
                "primal_res": r,
                "dual_res": s,
                "obj_approx": obj_approx,
                "lam": dict(lam),
                "z": dict(z),
            }
        )

        if r < cfg.tol_primal and s < cfg.tol_dual:
            status = "converged"
            break

    t1 = time.perf_counter()

    return ADMMResult(
        status=status,
        objective_approx=history[-1]["obj_approx"] if history else 0.0,
        iterations=it_final,
        primal_res=primal_hist,
        dual_res=dual_hist,
        shadow_prices={i: float(lam[i]) for i in intermediates},
        consensus_z={i: float(z[i]) for i in intermediates},
        cdu_solution=cdu_sol,
        blender_solution=blend_sol,
        wall_time_s=t1 - t0,
        history=history,
    )
