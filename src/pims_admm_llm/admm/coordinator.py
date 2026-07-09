"""ADMM coordinator: dual updates, consensus, convergence, shadow prices.

Backends
--------
qp_l2
    Classical 2-block (CDU / Blender) ADMM with quadratic augmented Lagrangian
    (SciPy SLSQP). After the block loop, *primal recovery* solves the exact
    blender LP given CDU intermediate production and extracts duals that match
    monolithic balance/yield shadow prices at optimality.

pulp_l1
    Multi-block coordination using Worker-2 PuLP subproblems (CDU, Inventory,
    Blender, Utilities) with L1 consensus penalties and dual price injection.

Dual variables λ (online ADMM path) and recovered shadow prices (from the
recovery LP) are both exposed. Economic interpretation: recovered prices are
$/bbl marginal value of each intermediate for make-buy-sell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from pims_admm_llm.models.data import RefineryData

from .recovery import economic_objective, recover_blender_with_duals
from .residuals import residual_norms, converged
from .subproblems import solve_blender_block_qp, solve_blocks_pulp, solve_cdu_block_qp


Backend = Literal["qp_l2", "pulp_l1"]
Mode = Literal["balance", "consensus"]


@dataclass
class ADMMConfig:
    rho: float = 3.0
    max_iter: int = 80
    abs_tol: float = 1e-3
    rel_tol: float = 1e-3
    backend: Backend = "qp_l2"
    mode: Mode = "balance"
    adaptive_rho: bool = False
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    rho_min: float = 0.5
    rho_max: float = 50.0
    alpha: float = 1.0
    verbose: bool = False
    min_activity: float = 1.0
    dual_step: float = 0.5  # damping β on dual update
    prox_weight: float = 0.0
    # stop when crude slate stable this many iters (residual may plateau on LP faces)
    stable_crude_iters: int = 15
    recover_primal: bool = True


@dataclass
class ADMMState:
    iteration: int
    lam: Dict[str, float]
    mu: Dict[str, float]
    z: Dict[str, float]
    z_util: Dict[str, float]
    prod: Dict[str, float]
    use: Dict[str, float]
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    r_norm: float
    s_norm: float
    residual: Dict[str, float]
    rho: float
    objective_hat: float
    block_status: Dict[str, str] = field(default_factory=dict)


@dataclass
class ADMMResult:
    status: str
    converged: bool
    iterations: int
    objective: float
    shadow_prices: Dict[str, float]
    utility_shadow_prices: Dict[str, float]
    economic_shadow_prices: Dict[str, float]
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    residual: Dict[str, float]
    r_norm: float
    s_norm: float
    rho: float
    history: List[ADMMState]
    solve_time_s: float
    backend: str
    mode: str
    duals_like_monolithic: Dict[str, float] = field(default_factory=dict)
    online_duals: Dict[str, float] = field(default_factory=dict)
    recovered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": self.objective,
            "shadow_prices": self.shadow_prices,
            "utility_shadow_prices": self.utility_shadow_prices,
            "economic_shadow_prices": self.economic_shadow_prices,
            "crude_rates": self.crude_rates,
            "product_rates": self.product_rates,
            "intermediate_prod": self.intermediate_prod,
            "intermediate_use": self.intermediate_use,
            "residual": self.residual,
            "r_norm": self.r_norm,
            "s_norm": self.s_norm,
            "rho": self.rho,
            "solve_time_s": self.solve_time_s,
            "backend": self.backend,
            "mode": self.mode,
            "duals_like_monolithic": self.duals_like_monolithic,
            "online_duals": self.online_duals,
            "recovered": self.recovered,
        }


class ADMMCoordinator:
    """Alternating Direction Method of Multipliers for the toy refinery LP."""

    def __init__(self, data: RefineryData, config: Optional[ADMMConfig] = None):
        self.data = data
        self.config = config or ADMMConfig()
        self.linking_names: List[str] = list(data.intermediates)
        self.util_names: List[str] = list(getattr(data, "utility_names", []) or [])
        self.lam: Dict[str, float] = {n: 0.0 for n in self.linking_names}
        self.mu: Dict[str, float] = {n: 0.0 for n in self.util_names}
        self.z: Dict[str, float] = {n: 0.0 for n in self.linking_names}
        self.z_util: Dict[str, float] = {n: 0.0 for n in self.util_names}
        self.rho = float(self.config.rho)
        self.history: List[ADMMState] = []
        self._crude_warm: Dict[str, float] = {}
        self._prod_warm: Dict[str, float] = {}
        self._last_use: Dict[str, float] = {n: 5.0 for n in self.linking_names}
        self._stable = 0
        self._last_crude_key = None

    def reset(self) -> None:
        self.lam = {n: 0.0 for n in self.linking_names}
        self.mu = {n: 0.0 for n in self.util_names}
        self.z = {n: 5.0 for n in self.linking_names}
        self.z_util = {n: 0.5 for n in self.util_names}
        self.rho = float(self.config.rho)
        self.history.clear()
        self._crude_warm.clear()
        self._prod_warm.clear()
        self._last_use = {n: 5.0 for n in self.linking_names}
        self._stable = 0
        self._last_crude_key = None

    def _step_qp(self, iteration: int) -> ADMMState:
        cfg = self.config
        z_old = dict(self.z)
        beta = cfg.dual_step

        # Gauss–Seidel balance ADMM:
        # CDU uses previous blender use as z; blender uses fresh CDU prod as z.
        cdu = solve_cdu_block_qp(
            self.data,
            self.lam,
            z=self._last_use,
            rho=self.rho,
            x0=self._crude_warm or None,
            prox_center=self._crude_warm or None,
            prox_weight=cfg.prox_weight,
        )
        prod = dict(cdu.linking)
        blender = solve_blender_block_qp(
            self.data,
            self.lam,
            z=prod,
            rho=self.rho,
            x0=self._prod_warm or None,
            prox_center=self._prod_warm or None,
            prox_weight=cfg.prox_weight,
        )
        use = dict(blender.linking)
        z_new = {n: 0.5 * (prod[n] + use[n]) for n in self.linking_names}

        r_norm, s_norm, residual = residual_norms(
            self.linking_names, prod, use, z_new, z_old, self.rho
        )
        for n in self.linking_names:
            self.lam[n] = self.lam[n] + beta * self.rho * residual[n]

        obj_hat = float(cdu.local_obj + blender.local_obj)
        state = ADMMState(
            iteration=iteration,
            lam=dict(self.lam),
            mu=dict(self.mu),
            z=z_new,
            z_util=dict(self.z_util),
            prod=prod,
            use=use,
            crude_rates=dict(cdu.primals),
            product_rates=dict(blender.primals),
            r_norm=r_norm,
            s_norm=s_norm,
            residual=residual,
            rho=self.rho,
            objective_hat=obj_hat,
            block_status={"CDU": cdu.status, "Blender": blender.status},
        )
        self.z = z_new
        self._last_use = use
        self._crude_warm = dict(cdu.primals)
        self._prod_warm = dict(blender.primals)

        crude_key = tuple(round(cdu.primals.get(c.name, 0.0), 2) for c in self.data.crudes)
        if crude_key == self._last_crude_key and sum(cdu.primals.values()) >= cfg.min_activity:
            self._stable += 1
        else:
            self._stable = 0
        self._last_crude_key = crude_key

        self.history.append(state)
        if cfg.verbose:
            sp = ", ".join(f"{k}:{v:.2f}" for k, v in self.lam.items())
            print(
                f"iter={iteration:3d} r={r_norm:.4e} s={s_norm:.4e} "
                f"obj~={obj_hat:.2f} rho={self.rho:.3g} stable={self._stable} λ={{{sp}}}"
            )
        return state

    def _step_pulp(self, iteration: int) -> ADMMState:
        cfg = self.config
        z_old = dict(self.z)
        blocks = solve_blocks_pulp(
            self.data,
            intermediate_prices=self.lam,
            utility_prices=self.mu,
            z_intermediates=self.z,
            z_utilities=self.z_util,
            rho=self.rho,
        )
        cdu = blocks["CDU"]
        inv = blocks["Inventory"]
        blend = blocks["Blender"]
        util = blocks["Utilities"]
        demand = blocks["_utility_demand"].linking_utilities

        prod = dict(cdu.linking)
        use = dict(blend.linking)
        inv_in = {
            n: float(inv.primals.get(f"inv_in_{n}", 0.0)) for n in self.linking_names
        }
        inv_out = dict(inv.linking)
        residual = {n: prod.get(n, 0.0) - use.get(n, 0.0) for n in self.linking_names}
        r_chain = {
            n: (prod.get(n, 0.0) - inv_in.get(n, 0.0))
            + (inv_out.get(n, 0.0) - use.get(n, 0.0))
            for n in self.linking_names
        }
        r_norm = float(np.linalg.norm([residual[n] for n in self.linking_names]))
        r_util = {
            u: demand.get(u, 0.0) - util.linking_utilities.get(u, 0.0)
            for u in self.util_names
        }
        if self.util_names:
            r_norm = float(
                np.sqrt(
                    r_norm**2
                    + float(np.linalg.norm([r_util[u] for u in self.util_names])) ** 2
                )
            )
        z_new = {n: 0.5 * (prod.get(n, 0.0) + use.get(n, 0.0)) for n in self.linking_names}
        z_util_new = {
            u: 0.5 * (demand.get(u, 0.0) + util.linking_utilities.get(u, 0.0))
            for u in self.util_names
        }
        s_norm = float(
            self.rho
            * np.linalg.norm([z_new[n] - z_old.get(n, 0.0) for n in self.linking_names])
        )
        step = self.rho * cfg.dual_step
        for n in self.linking_names:
            self.lam[n] = self.lam[n] - step * r_chain[n]
        for u in self.util_names:
            self.mu[u] = self.mu[u] + step * r_util[u]

        crude_rates = {
            k.replace("crude_", "", 1): v
            for k, v in cdu.primals.items()
            if k.startswith("crude_")
        }
        product_rates = {
            k.replace("product_", "", 1): v
            for k, v in blend.primals.items()
            if k.startswith("product_")
        }
        obj_hat = float(cdu.local_obj + blend.local_obj + inv.local_obj + util.local_obj)
        state = ADMMState(
            iteration=iteration,
            lam=dict(self.lam),
            mu=dict(self.mu),
            z=z_new,
            z_util=z_util_new,
            prod=prod,
            use=use,
            crude_rates=crude_rates,
            product_rates=product_rates,
            r_norm=r_norm,
            s_norm=s_norm,
            residual=residual,
            rho=self.rho,
            objective_hat=obj_hat,
            block_status={
                "CDU": cdu.status,
                "Inventory": inv.status,
                "Blender": blend.status,
                "Utilities": util.status,
            },
        )
        self.z = z_new
        self.z_util = z_util_new
        crude_key = tuple(round(crude_rates.get(c.name, 0.0), 2) for c in self.data.crudes)
        if crude_key == self._last_crude_key and sum(crude_rates.values()) >= cfg.min_activity:
            self._stable += 1
        else:
            self._stable = 0
        self._last_crude_key = crude_key
        self.history.append(state)
        if cfg.verbose:
            sp = ", ".join(f"{k}:{v:.2f}" for k, v in self.lam.items())
            print(
                f"iter={iteration:3d} r={r_norm:.4e} s={s_norm:.4e} "
                f"obj~={obj_hat:.2f} rho={self.rho:.3g} λ={{{sp}}}"
            )
        return state

    def step(self, iteration: int) -> ADMMState:
        if self.config.backend == "pulp_l1":
            return self._step_pulp(iteration)
        return self._step_qp(iteration)

    def run(self) -> ADMMResult:
        self.reset()
        if self.config.backend == "pulp_l1":
            # economic prices seed
            self.lam = {n: 50.0 for n in self.linking_names}
            self.mu = {n: 10.0 for n in self.util_names}

        t0 = time.perf_counter()
        cfg = self.config
        last: Optional[ADMMState] = None
        done = False

        for k in range(cfg.max_iter):
            last = self.step(k)
            scale = max(
                float(np.linalg.norm([last.prod[n] for n in self.linking_names])),
                float(np.linalg.norm([last.use[n] for n in self.linking_names])),
                1.0,
            )
            if converged(last.r_norm, last.s_norm, cfg.abs_tol, cfg.rel_tol, scale):
                if sum(last.crude_rates.values()) >= cfg.min_activity:
                    done = True
                    break
            if self._stable >= cfg.stable_crude_iters and sum(last.crude_rates.values()) >= cfg.min_activity:
                # LP-face residual plateau with stable crude → recovery step
                done = True
                break

        assert last is not None
        t1 = time.perf_counter()

        crude = dict(last.crude_rates)
        inter_prod = dict(last.prod)
        online_duals = dict(self.lam)
        util_shadow = dict(self.mu)
        recovered = False

        if cfg.recover_primal and cfg.backend == "qp_l2":
            rec = recover_blender_with_duals(self.data, inter_prod)
            products = rec.product_rates
            inter_use = rec.intermediate_use
            residual = {n: inter_prod[n] - inter_use[n] for n in self.linking_names}
            r_norm = float(np.linalg.norm([residual[n] for n in self.linking_names]))
            objective = economic_objective(self.data, crude, products)
            # Recovered duals match mono at optimality
            economic = dict(rec.shadow_prices)
            duals_like = dict(rec.duals_like_monolithic)
            shadow = {n: duals_like.get(f"balance_{n}", -economic[n]) for n in self.linking_names}
            recovered = True
            if r_norm <= cfg.abs_tol + cfg.rel_tol * max(
                float(np.linalg.norm([inter_prod[n] for n in self.linking_names])), 1.0
            ):
                done = True
        else:
            products = dict(last.product_rates)
            inter_use = dict(last.use)
            residual = dict(last.residual)
            r_norm = last.r_norm
            objective = economic_objective(self.data, crude, products)
            if cfg.backend == "qp_l2":
                shadow = dict(online_duals)
                economic = {n: float(-shadow[n]) for n in self.linking_names}
                duals_like = {f"balance_{n}": float(shadow[n]) for n in self.linking_names}
            else:
                shadow = dict(online_duals)
                economic = dict(shadow)
                duals_like = {f"balance_{n}": float(-shadow[n]) for n in self.linking_names}

        duals_like.update({f"admm_lambda_{n}": float(online_duals[n]) for n in self.linking_names})
        duals_like.update({f"economic_{n}": float(economic[n]) for n in self.linking_names})
        for u, v in util_shadow.items():
            duals_like[f"utility_price_{u}"] = float(v)

        return ADMMResult(
            status="converged" if done else "max_iter",
            converged=done,
            iterations=last.iteration + 1,
            objective=objective,
            shadow_prices={n: float(shadow[n]) for n in self.linking_names},
            utility_shadow_prices={u: float(util_shadow[u]) for u in self.util_names},
            economic_shadow_prices={n: float(economic[n]) for n in self.linking_names},
            crude_rates=crude,
            product_rates=products,
            intermediate_prod=inter_prod,
            intermediate_use=inter_use,
            residual=residual,
            r_norm=r_norm,
            s_norm=last.s_norm,
            rho=self.rho,
            history=self.history,
            solve_time_s=t1 - t0,
            backend=cfg.backend,
            mode=cfg.mode,
            duals_like_monolithic=duals_like,
            online_duals={n: float(online_duals[n]) for n in self.linking_names},
            recovered=recovered,
        )


def run_admm(
    data: RefineryData,
    config: Optional[ADMMConfig] = None,
) -> ADMMResult:
    return ADMMCoordinator(data, config).run()
