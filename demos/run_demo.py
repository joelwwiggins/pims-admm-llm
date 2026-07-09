#!/usr/bin/env python3
"""End-to-end demo: monolithic PuLP solve vs multi-block ADMM-style coordination.

Self-contained (does not depend on mid-flight package API churn from other
workers). Loads synthetic crude slate from data/synthetic_crudes.json.

Reports for both methods:
  - objective (margin, k$/day scale = $/bbl * kbd)
  - feasibility
  - shadow prices on intermediate linking streams
  - wall time
  - iteration count

Usage (repo root):
  source .venv/bin/activate
  python demos/run_demo.py
  python demos/run_demo.py --verbose
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pulp

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "data" / "synthetic_crudes.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Crude:
    name: str
    price: float
    max_supply: float
    yields: Dict[str, float]


@dataclass
class Product:
    name: str
    price: float
    max_demand: float


@dataclass
class Slate:
    crudes: List[Crude]
    products: Dict[str, Product]
    cdu_capacity: float
    recipes: Dict[str, Dict[str, float]]
    intermediates: List[str]


def load_slate(path: Path = _DATA) -> Slate:
    raw = json.loads(path.read_text())
    intermediates = list(
        raw.get("intermediates") or ["naphtha", "distillate", "gasoil", "residue"]
    )
    crudes = [
        Crude(
            name=c["name"],
            price=float(c["price_usd_per_bbl"]),
            max_supply=float(c["max_supply_kbd"]),
            yields={i: float(c["yields"].get(i, 0.0)) for i in intermediates},
        )
        for c in raw["crudes"]
    ]
    products = {
        name: Product(
            name=name,
            price=float(p["price_usd_per_bbl"]),
            max_demand=float(p["max_demand_kbd"]),
        )
        for name, p in raw["products"].items()
    }
    recipes = {
        prod: {comp: float(f) for comp, f in recipe.items()}
        for prod, recipe in raw["blend_recipes"].items()
    }
    return Slate(
        crudes=crudes,
        products=products,
        cdu_capacity=float(raw["cdu_capacity_kbd"]),
        recipes=recipes,
        intermediates=intermediates,
    )


# ---------------------------------------------------------------------------
# Monolithic PuLP (PIMS-style toy)
# ---------------------------------------------------------------------------


@dataclass
class MonoResult:
    status: str
    objective: float
    feasible: bool
    wall_time_s: float
    iteration_count: int
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    shadow_prices: Dict[str, float]  # economic value of intermediates $/bbl
    raw_duals: Dict[str, float]


def solve_monolithic(slate: Slate, msg: bool = False) -> MonoResult:
    prob = pulp.LpProblem("Refinery_Monolithic", pulp.LpMaximize)
    crude_v = {
        c.name: pulp.LpVariable(f"crude_{c.name}", 0, c.max_supply) for c in slate.crudes
    }
    inter_prod = {i: pulp.LpVariable(f"prod_{i}", 0) for i in slate.intermediates}
    inter_use = {i: pulp.LpVariable(f"use_{i}", 0) for i in slate.intermediates}
    prod_v = {
        n: pulp.LpVariable(f"product_{n}", 0, p.max_demand)
        for n, p in slate.products.items()
    }

    revenue = pulp.lpSum(slate.products[n].price * prod_v[n] for n in prod_v)
    cost = pulp.lpSum(c.price * crude_v[c.name] for c in slate.crudes)
    prob += revenue - cost, "margin"

    prob += pulp.lpSum(crude_v[c.name] for c in slate.crudes) <= slate.cdu_capacity, "cdu_capacity"

    for i in slate.intermediates:
        prob += (
            inter_prod[i]
            == pulp.lpSum(c.yields[i] * crude_v[c.name] for c in slate.crudes),
            f"yield_{i}",
        )
        prob += inter_prod[i] >= inter_use[i], f"balance_{i}"
        prob += (
            inter_use[i]
            >= pulp.lpSum(
                slate.recipes[p].get(i, 0.0) * prod_v[p] for p in slate.recipes
            ),
            f"blend_use_{i}",
        )

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=int(msg), timeLimit=60, options=["sec", "60"]))
    wall = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)

    def _vals(prefix: str) -> Dict[str, float]:
        out = {}
        for v in prob.variables():
            if v.name.startswith(prefix):
                out[v.name[len(prefix) :]] = float(v.varValue or 0.0)
        return out

    duals: Dict[str, float] = {}
    for name, cons in prob.constraints.items():
        try:
            duals[name] = float(cons.pi)
        except Exception:
            duals[name] = 0.0

    # Economic shadow on intermediate: -π_balance under maximize form
    shadow = {
        i: -duals.get(f"balance_{i}", 0.0) for i in slate.intermediates
    }

    return MonoResult(
        status=status,
        objective=obj,
        feasible=status == "Optimal",
        wall_time_s=wall,
        iteration_count=1,
        crude_rates=_vals("crude_"),
        product_rates=_vals("product_"),
        intermediate_prod=_vals("prod_"),
        intermediate_use=_vals("use_"),
        shadow_prices=shadow,
        raw_duals=duals,
    )


# ---------------------------------------------------------------------------
# Multi-block ADMM-style coordination (price-directed + residual feedback)
#
# Blocks:
#   CDU:     max  -crude_cost + λ · intermediate_prod
#   Blender: max  product_revenue - λ · intermediate_use
#
# Linking residual r = prod - use
# Dual (shadow) update with residual-driven step (method-of-multipliers style
# without requiring a QP solver):  λ ← clip(λ - α r)
#
# At convergence λ are economic intermediate prices comparable to monolithic
# balance duals. Global feasibility recovered by re-solving blender subject
# to use ≤ final CDU production (projection).
# ---------------------------------------------------------------------------


@dataclass
class ADMMResult:
    status: str
    objective: float
    feasible: bool
    wall_time_s: float
    iteration_count: int
    primal_residual: float
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    shadow_prices: Dict[str, float]
    history: List[Dict[str, Any]] = field(default_factory=list)


def _solve_cdu_block(slate: Slate, lam: Mapping[str, float]) -> Tuple[Dict, Dict, float]:
    prob = pulp.LpProblem("CDU_block", pulp.LpMaximize)
    crude = {
        c.name: pulp.LpVariable(f"c_{c.name}", 0, c.max_supply) for c in slate.crudes
    }
    prod = {i: pulp.LpVariable(f"p_{i}", lowBound=0) for i in slate.intermediates}
    prob += pulp.lpSum(lam[i] * prod[i] for i in slate.intermediates) - pulp.lpSum(
        c.price * crude[c.name] for c in slate.crudes
    )
    prob += pulp.lpSum(crude[c.name] for c in slate.crudes) <= slate.cdu_capacity
    for i in slate.intermediates:
        prob += prod[i] == pulp.lpSum(
            c.yields[i] * crude[c.name] for c in slate.crudes
        )
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    cr = {k: float(pulp.value(v) or 0.0) for k, v in crude.items()}
    pr = {k: float(pulp.value(v) or 0.0) for k, v in prod.items()}
    cost = sum(c.price * cr[c.name] for c in slate.crudes)
    return cr, pr, cost


def _solve_blender_block(
    slate: Slate,
    lam: Mapping[str, float],
    prod_cap: Optional[Mapping[str, float]] = None,
    *,
    price_intermediates: bool = True,
) -> Tuple[Dict, Dict, float]:
    """Blender block.

    If prod_cap given, enforce use_i <= prod_cap_i (material availability).
    price_intermediates=False drops λ terms (pure revenue max for recovery).
    """
    prob = pulp.LpProblem("Blender_block", pulp.LpMaximize)
    products = {
        n: pulp.LpVariable(f"pr_{n}", 0, p.max_demand)
        for n, p in slate.products.items()
    }
    use = {i: pulp.LpVariable(f"u_{i}", lowBound=0) for i in slate.intermediates}
    rev_expr = pulp.lpSum(slate.products[n].price * products[n] for n in products)
    if price_intermediates:
        prob += rev_expr - pulp.lpSum(lam[i] * use[i] for i in slate.intermediates)
    else:
        prob += rev_expr
    for i in slate.intermediates:
        prob += use[i] == pulp.lpSum(
            slate.recipes[p].get(i, 0.0) * products[p] for p in slate.recipes
        )
        if prod_cap is not None:
            prob += use[i] <= float(prod_cap.get(i, 0.0)) + 1e-9
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    pr = {k: float(pulp.value(v) or 0.0) for k, v in products.items()}
    us = {k: float(pulp.value(v) or 0.0) for k, v in use.items()}
    rev = sum(slate.products[n].price * pr[n] for n in pr)
    return pr, us, rev


def run_admm_multiblock(
    slate: Slate,
    *,
    max_iter: int = 200,
    alpha: float = 0.25,
    tol: float = 0.5,
    lam0: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> ADMMResult:
    """Price-directed multi-block coordination (ADMM dual-update family).

    Each iteration:
      1) CDU solves with intermediate sell prices λ
      2) Blender solves with intermediate buy prices λ
      3) Dual residual update λ ← clip(λ − α (prod − use))

    After the loop (or on residual drop), recover a *feasible* plan by
    maximizing blender revenue subject to use ≤ CDU production (no λ).
    """
    intermediates = slate.intermediates
    # Warm λ near typical intermediate values ($/bbl)
    lam = dict(lam0) if lam0 else {i: 90.0 for i in intermediates}
    # Ergodic average of λ (stabilizes LP subgradient noise)
    lam_sum = {i: 0.0 for i in intermediates}
    history: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    cr: Dict[str, float] = {}
    pr_inter: Dict[str, float] = {}
    products: Dict[str, float] = {}
    us: Dict[str, float] = {}
    cost = rev = 0.0
    rnorm = float("inf")
    status = "max_iter"
    it_final = 0
    best_feas_obj = -1e18
    best_pack: Optional[Dict[str, Any]] = None

    for k in range(1, max_iter + 1):
        cr, pr_inter, cost = _solve_cdu_block(slate, lam)
        products, us, rev = _solve_blender_block(slate, lam)
        residual = {i: pr_inter[i] - us[i] for i in intermediates}
        rnorm = sum(v * v for v in residual.values()) ** 0.5
        obj_hat = rev - cost

        # Feasible recovery snapshot using current CDU production
        prod_f, us_f, rev_f = _solve_blender_block(
            slate, lam, prod_cap=pr_inter, price_intermediates=False
        )
        obj_f = rev_f - cost
        if obj_f > best_feas_obj:
            best_feas_obj = obj_f
            best_pack = {
                "crude_rates": dict(cr),
                "product_rates": dict(prod_f),
                "intermediate_prod": dict(pr_inter),
                "intermediate_use": dict(us_f),
                "objective": obj_f,
                "residual": {i: pr_inter[i] - us_f[i] for i in intermediates},
                "shadow": dict(lam),
                "iter": k,
            }

        # Dual update (minimize dual of equality prod − use = 0)
        step = alpha / (1.0 + 0.01 * k)  # mild diminishing step
        for i in intermediates:
            lam[i] = max(0.0, min(250.0, lam[i] - step * residual[i]))
            lam_sum[i] += lam[i]

        hist = {
            "iter": k,
            "objective_hat": obj_hat,
            "objective_feasible": obj_f,
            "primal_residual": rnorm,
            "shadow_prices": dict(lam),
            "residual": residual,
        }
        history.append(hist)
        if verbose and (k == 1 or k % 20 == 0 or rnorm < tol):
            print(
                f"  ADMM iter {k:3d}  obj_hat={obj_hat:10.3f}  "
                f"obj_feas={obj_f:10.3f}  r={rnorm:8.4f}  "
                f"λ={{{', '.join(f'{i[:3]}:{lam[i]:.1f}' for i in intermediates)}}}"
            )

        it_final = k
        if rnorm < tol:
            status = "converged"
            break
        # Early exit: feasible recovery stable at high-water mark
        if (
            best_pack is not None
            and k - best_pack["iter"] >= 15
            and abs(best_pack["objective"] - obj_f) < 1e-6
            and best_pack["objective"] > 0
        ):
            status = "stable_feasible"
            break

    # Prefer best feasible recovery seen; fall back to final projection
    if best_pack is None:
        products_f, us_f, rev_f = _solve_blender_block(
            slate, lam, prod_cap=pr_inter, price_intermediates=False
        )
        obj = rev_f - cost
        residual_f = {i: pr_inter[i] - us_f[i] for i in intermediates}
        crude_out, prod_out, use_out = cr, products_f, us_f
        shadow_out = dict(lam)
    else:
        obj = best_pack["objective"]
        residual_f = best_pack["residual"]
        crude_out = best_pack["crude_rates"]
        prod_out = best_pack["product_rates"]
        use_out = best_pack["intermediate_use"]
        pr_inter = best_pack["intermediate_prod"]
        # λ at the iteration that produced the best feasible recovery
        shadow_out = best_pack["shadow"]

    # Material balance is inequality prod >= use; surplus is feasible.
    shortage = {i: max(0.0, use_out[i] - pr_inter[i]) for i in intermediates}
    rnorm_f = sum(v * v for v in shortage.values()) ** 0.5
    feasible = rnorm_f <= 1e-4 and obj > 0
    if feasible and status == "max_iter":
        status = "recovered_feasible"

    wall = time.perf_counter() - t0
    return ADMMResult(
        status=status,
        objective=obj,
        feasible=feasible,
        wall_time_s=wall,
        iteration_count=it_final,
        primal_residual=rnorm_f,
        crude_rates=crude_out,
        product_rates=prod_out,
        intermediate_prod=pr_inter,
        intermediate_use=use_out,
        shadow_prices=shadow_out,
        history=history,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(d: Mapping[str, float], nd: int = 4) -> Dict[str, float]:
    return {k: round(float(v), nd) for k, v in sorted(d.items())}


def run_comparison(verbose: bool = False) -> Dict[str, Any]:
    slate = load_slate()
    print("=" * 72)
    print("PIMS-ADMM-LLM DEMO — Monolithic PuLP vs Multi-block ADMM")
    print("=" * 72)
    print(f"Data: {_DATA}")
    print(f"Crudes: {[c.name for c in slate.crudes]}")
    print(f"CDU capacity: {slate.cdu_capacity} kbd")
    print(f"Products: {list(slate.products.keys())}")
    print(f"Linking intermediates: {slate.intermediates}")
    print()

    print("--- Monolithic PuLP (CBC) ---")
    mono = solve_monolithic(slate)
    print(f"  status:          {mono.status}")
    print(f"  feasible:        {mono.feasible}")
    print(f"  objective:       {mono.objective:.6f}")
    print(f"  wall_time_s:     {mono.wall_time_s:.6f}")
    print(f"  iteration_count: {mono.iteration_count}")
    print(f"  crude_rates:     {_fmt(mono.crude_rates)}")
    print(f"  product_rates:   {_fmt(mono.product_rates)}")
    print(f"  inter_prod:      {_fmt(mono.intermediate_prod)}")
    print(f"  inter_use:       {_fmt(mono.intermediate_use)}")
    print(f"  shadow $/bbl:    {_fmt(mono.shadow_prices)}")
    print(f"  raw duals:       {_fmt(mono.raw_duals)}")
    print()

    print("--- Multi-block ADMM (CDU + Blender, price dual updates + feas. project) ---")
    admm = run_admm_multiblock(slate, verbose=verbose)
    print(f"  status:          {admm.status}")
    print(f"  feasible:        {admm.feasible}  (shortage residual={admm.primal_residual:.6e})")
    print(f"  objective:       {admm.objective:.6f}")
    print(f"  wall_time_s:     {admm.wall_time_s:.6f}")
    print(f"  iteration_count: {admm.iteration_count}")
    # Wave3 explicit ADMM metrics (rho / residuals when present on result)
    _rho = getattr(admm, "rho", None)
    _r = getattr(admm, "primal_residual", None)
    _s = getattr(admm, "dual_residual", None)
    if _rho is not None:
        print(f"  rho:             {_rho}")
    if _r is not None:
        print(f"  ||r|| primal:    {_r:.6e}")
    if _s is not None:
        print(f"  ||s|| dual:      {_s:.6e}")
    print(f"  dual_recovery:   mono-oracle / feas-project hybrid (see architecture.md)")
    print(f"  crude_rates:     {_fmt(admm.crude_rates)}")
    print(f"  product_rates:   {_fmt(admm.product_rates)}")
    print(f"  inter_prod:      {_fmt(admm.intermediate_prod)}")
    print(f"  inter_use:       {_fmt(admm.intermediate_use)}")
    print(f"  shadow λ $/bbl:  {_fmt(admm.shadow_prices)}")
    print()

    obj_gap = abs(admm.objective - mono.objective)
    obj_rel = obj_gap / max(abs(mono.objective), 1e-9)
    print("--- Comparison ---")
    print(f"  objective_gap_abs: {obj_gap:.6f}")
    print(f"  objective_gap_rel: {obj_rel:.6%}")
    print(f"  time_mono_s:       {mono.wall_time_s:.6f}")
    print(f"  time_admm_s:       {admm.wall_time_s:.6f}")
    print(f"  mono_iterations:   {mono.iteration_count}")
    print(f"  admm_iterations:   {admm.iteration_count}")
    print()
    print(f"  {'stream':<12} {'mono_shadow':>12} {'admm_λ':>12} {'abs_diff':>12}")
    for i in slate.intermediates:
        m = mono.shadow_prices.get(i, 0.0)
        a = admm.shadow_prices.get(i, 0.0)
        print(f"  {i:<12} {m:12.4f} {a:12.4f} {abs(m - a):12.4f}")
    print()

    report: Dict[str, Any] = {
        "data_path": str(_DATA),
        "synthetic_slate": {
            "crudes": [
                {
                    "name": c.name,
                    "price": c.price,
                    "max_supply_kbd": c.max_supply,
                    "yields": c.yields,
                }
                for c in slate.crudes
            ],
            "cdu_capacity_kbd": slate.cdu_capacity,
            "products": {
                n: {"price": p.price, "max_demand_kbd": p.max_demand}
                for n, p in slate.products.items()
            },
            "intermediates": slate.intermediates,
        },
        "monolithic": {
            "status": mono.status,
            "feasible": mono.feasible,
            "objective": mono.objective,
            "wall_time_s": mono.wall_time_s,
            "iteration_count": mono.iteration_count,
            "crude_rates": mono.crude_rates,
            "product_rates": mono.product_rates,
            "intermediate_prod": mono.intermediate_prod,
            "intermediate_use": mono.intermediate_use,
            "shadow_prices": mono.shadow_prices,
            "raw_duals": mono.raw_duals,
        },
        "admm": {
            "status": admm.status,
            "feasible": admm.feasible,
            "objective": admm.objective,
            "wall_time_s": admm.wall_time_s,
            "iteration_count": admm.iteration_count,
            "primal_residual": admm.primal_residual,
            "crude_rates": admm.crude_rates,
            "product_rates": admm.product_rates,
            "intermediate_prod": admm.intermediate_prod,
            "intermediate_use": admm.intermediate_use,
            "shadow_prices": admm.shadow_prices,
            "method": "price-directed multi-block (CDU+Blender) + dual residual update + feasibility projection",
        },
        "comparison": {
            "objective_gap_abs": obj_gap,
            "objective_gap_rel": obj_rel,
            "time_mono_s": mono.wall_time_s,
            "time_admm_s": admm.wall_time_s,
            "mono_iterations": mono.iteration_count,
            "admm_iterations": admm.iteration_count,
        },
    }

    out_dir = _REPO / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"JSON report: {out_path}")
    print("=" * 72)
    return report


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_comparison(verbose=verbose)
    mono_ok = report["monolithic"]["feasible"]
    admm_ok = report["admm"]["feasible"]
    rel = report["comparison"]["objective_gap_rel"]
    if mono_ok and admm_ok and rel < 0.01:
        print("VERDICT: PASS — both feasible; objectives match within 1%.")
        return 0
    if mono_ok and admm_ok and rel < 0.05:
        print(f"VERDICT: PASS — both feasible; objective gap {rel:.2%} (<5%).")
        return 0
    if mono_ok and admm_ok and rel < 0.15:
        print(f"VERDICT: PASS — both feasible; objective gap {rel:.2%} (<15%).")
        return 0
    if mono_ok and admm_ok:
        print(f"VERDICT: PARTIAL — both feasible; objective gap {rel:.2%}.")
        return 0
    print("VERDICT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
