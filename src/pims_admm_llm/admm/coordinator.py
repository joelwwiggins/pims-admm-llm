"""ADMM coordinator: dual updates, consensus, convergence, shadow prices.

Backends
--------
* ``backend="qp_l2"`` — classical 2-block (CDU, Blender) balance ADMM with
  quadratic augmented Lagrangian (CVXPY/OSQP). Duals λ match simplified
  monolithic balance duals at convergence.
* ``backend="pulp_l1"`` — multi-block (CDU, Inventory, Blender, Utilities)
  using Worker-2 PuLP subproblems with L1 consensus penalties. Dual prices
  λ (intermediates) and μ (utilities) are extractable shadow prices.

Economic reading
----------------
At convergence, intermediate shadow price ($/bbl) is the absolute value of
the dual on material balance. For the QP min-form:

  λ_ADMM[i] ≈ − dual_monolithic[balance_i]   (PuLP maximize-form)

so make-buy-sell tables use ``shadow_prices`` directly as positive values
when reported via ``economic_shadow_prices``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

import numpy as np

from pims_admm_llm.models.data import RefineryData

from .residuals import residual_norms, converged
from .subproblems import (
    BlockSolution,
    solve_blender_block_qp,
    solve_blocks_pulp,
    solve_cdu_block_qp,
)


Backend = Literal["qp_l2", "pulp_l1"]
Mode = Literal["balance", "consensus"]


@dataclass
class ADMMConfig:
    rho: float = 2.0
    max_iter: int = 200
    abs_tol: float = 1e-3
    rel_tol: float = 1e-3
    backend: Backend = "qp_l2"
    mode: Mode = "balance"  # used by qp_l2
    adaptive_rho: bool = True
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    rho_min: float = 0.05
    rho_max: float = 50.0
    alpha: float = 1.0  # over-relaxation; 1 = classic
    verbose: bool = False
    min_activity: float = 1e-2
    # dual step size scale for pulp_l1 (subgradient-like on residual)
    dual_step: float = 1.0


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
        self._last_prod: Dict[str, float] = {n: 0.0 for n in self.linking_names}
        self._last_use: Dict[str, float] = {n: 0.0 for n in self.linking_names}

    def reset(self) -> None:
        self.lam = {n: 0.0 for n in self.linking_names}
        self.mu = {n: 0.0 for n in self.util_names}
        # Seed consensus away from pure zero to avoid early dual collapse
        self.z = {n: 5.0 for n in self.linking_names}
        self.z_util = {n: 0.5 for n in self.util_names}
        self.rho = float(self.config.rho)
        self.history.clear()
        self._crude_warm.clear()
        self._prod_warm.clear()
        self._last_prod = {n: 0.0 for n in self.linking_names}
        self._last_use = {n: 5.0 for n in self.linking_names}

    # ------------------------------------------------------------------ qp_l2
    def _step_qp(self, iteration: int) -> ADMMState:
        cfg = self.config
        z_old = dict(self.z)

        cdu = solve_cdu_block_qp(
            self.data,
            self.lam,
            use_fixed=self._last_use,
            rho=self.rho,
            x0=self._crude_warm or None,
        )
        prod = dict(cdu.linking)
        if cfg.alpha != 1.0:
            prod_hat = {
                n: cfg.alpha * prod[n] + (1.0 - cfg.alpha) * self._last_use[n]
                for n in self.linking_names
            }
        else:
            prod_hat = prod

        blender = solve_blender_block_qp(
            self.data,
            self.lam,
            prod_fixed=prod_hat,
            rho=self.rho,
            x0=self._prod_warm or None,
        )
        use = dict(blender.linking)
        z_new = {n: 0.5 * (prod[n] + use[n]) for n in self.linking_names}

        r_norm, _, residual = residual_norms(
            self.linking_names, prod, use, z_new, z_old, self.rho
        )
        delta = np.array(
            [
                (prod[n] + use[n]) - (self._last_prod[n] + self._last_use[n])
                for n in self.linking_names
            ],
            dtype=float,
        )
        s_norm = float(self.rho * np.linalg.norm(delta) / 2.0)

        # Dual update on over-relaxed residual
        for n in self.linking_names:
            self.lam[n] = self.lam[n] + self.rho * (prod_hat[n] - use[n])

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
        self._last_prod = prod
        self._last_use = use
        self._crude_warm = dict(cdu.primals)
        self._prod_warm = dict(blender.primals)
        self._adapt_rho(r_norm, s_norm)
        state.lam = dict(self.lam)
        state.rho = self.rho
        self.history.append(state)
        if cfg.verbose:
            sp = ", ".join(f"{k}:{v:.2f}" for k, v in self.lam.items())
            print(
                f"iter={iteration:3d} r={r_norm:.4e} s={s_norm:.4e} "
                f"obj~={obj_hat:.2f} rho={self.rho:.3g} λ={{{sp}}}"
            )
        return state

    # ---------------------------------------------------------------- pulp_l1
    def _step_pulp(self, iteration: int) -> ADMMState:
        cfg = self.config
        z_old = dict(self.z)

        # Economic prices for Worker-2 subproblems: positive intermediate value
        # Worker-2 CDU sells at λ_price, blender buys at λ_price.
        # Our lam is min-form dual; economic price = -lam when using QP sign.
        # For pulp_l1 we store economic prices directly in self.lam (positive).
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
        # Inventory outflow should match blender use; inflow match CDU prod
        inv_out = dict(inv.linking)
        inv_in = {
            n: float(inv.primals.get(f"inv_in_{n}", 0.0)) for n in self.linking_names
        }

        # Residuals: CDU prod vs inv in; inv out vs blender use; util supply vs demand
        r_mat = {
            n: (prod.get(n, 0.0) - inv_in.get(n, 0.0))
            + (inv_out.get(n, 0.0) - use.get(n, 0.0))
            for n in self.linking_names
        }
        # Also direct prod - use residual (overall material)
        residual = {n: prod.get(n, 0.0) - use.get(n, 0.0) for n in self.linking_names}
        # Prefer inv chain residual for dual updates when inventory present
        residual_for_dual = r_mat

        r_norm = float(np.linalg.norm([residual[n] for n in self.linking_names]))
        r_util = {
            u: demand.get(u, 0.0) - util.linking_utilities.get(u, 0.0)
            for u in self.util_names
        }
        r_util_norm = float(np.linalg.norm([r_util[u] for u in self.util_names])) if self.util_names else 0.0
        r_norm = float(np.sqrt(r_norm**2 + r_util_norm**2))

        z_new = {n: 0.5 * (prod.get(n, 0.0) + use.get(n, 0.0)) for n in self.linking_names}
        z_util_new = {
            u: 0.5 * (demand.get(u, 0.0) + util.linking_utilities.get(u, 0.0))
            for u in self.util_names
        }
        s_norm = float(
            self.rho
            * np.linalg.norm([z_new[n] - z_old.get(n, 0.0) for n in self.linking_names])
        )

        # Dual ascent on imbalance (subgradient / linearized ADMM dual step)
        step = self.rho * cfg.dual_step
        for n in self.linking_names:
            # If prod > use, intermediate is long → lower its price
            self.lam[n] = self.lam[n] - step * residual_for_dual[n]
        for u in self.util_names:
            # If demand > supply, raise utility price
            self.mu[u] = self.mu[u] + step * r_util[u]

        # Crude / product extraction from primals
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
        self._last_prod = prod
        self._last_use = use
        self._adapt_rho(r_norm, s_norm)
        state.lam = dict(self.lam)
        state.mu = dict(self.mu)
        state.rho = self.rho
        self.history.append(state)
        if cfg.verbose:
            sp = ", ".join(f"{k}:{v:.2f}" for k, v in self.lam.items())
            print(
                f"iter={iteration:3d} r={r_norm:.4e} s={s_norm:.4e} "
                f"obj~={obj_hat:.2f} rho={self.rho:.3g} λ={{{sp}}}"
            )
        return state

    def _adapt_rho(self, r_norm: float, s_norm: float) -> None:
        if not self.config.adaptive_rho:
            return
        old = self.rho
        if r_norm > self.config.mu * max(s_norm, 1e-12):
            self.rho = min(self.rho * self.config.tau_incr, self.config.rho_max)
        elif s_norm > self.config.mu * max(r_norm, 1e-12):
            self.rho = max(self.rho / self.config.tau_decr, self.config.rho_min)
        if abs(self.rho - old) > 1e-15 and self.config.backend == "qp_l2":
            scale = old / self.rho
            self.lam = {n: self.lam[n] * scale for n in self.linking_names}

    def step(self, iteration: int) -> ADMMState:
        if self.config.backend == "pulp_l1":
            return self._step_pulp(iteration)
        return self._step_qp(iteration)

    def _activity(self, state: ADMMState) -> float:
        return float(
            sum(abs(v) for v in state.crude_rates.values())
            + sum(abs(v) for v in state.product_rates.values())
        )

    def run(self) -> ADMMResult:
        self.reset()
        # pulp_l1 stores economic prices; seed from rough mid prices
        if self.config.backend == "pulp_l1":
            self.lam = {n: 50.0 for n in self.linking_names}
            self.mu = {n: 10.0 for n in self.util_names}

        t0 = time.perf_counter()
        cfg = self.config
        last: Optional[ADMMState] = None
        done = False
        best: Optional[ADMMState] = None
        best_score = float("inf")

        for k in range(cfg.max_iter):
            last = self.step(k)
            scale = max(
                float(np.linalg.norm([last.prod[n] for n in self.linking_names])),
                float(np.linalg.norm([last.use[n] for n in self.linking_names])),
                1.0,
            )
            score = last.r_norm + 0.1 * last.s_norm
            if self._activity(last) >= cfg.min_activity and score < best_score:
                best_score = score
                best = last
            if converged(last.r_norm, last.s_norm, cfg.abs_tol, cfg.rel_tol, scale):
                if self._activity(last) >= cfg.min_activity:
                    done = True
                    break

        if last is None:
            raise RuntimeError("ADMM produced no iterations")
        if not done and best is not None:
            last = best

        t1 = time.perf_counter()
        shadow = {n: float(last.lam[n]) for n in self.linking_names}
        util_shadow = {n: float(last.mu[n]) for n in self.util_names}

        if cfg.backend == "qp_l2":
            # min-form λ; economic value and PuLP maximize balance dual mapping
            economic = {n: float(-shadow[n]) for n in self.linking_names}
            duals_like = {f"balance_{n}": float(shadow[n]) for n in self.linking_names}
            duals_like.update({f"admm_lambda_{n}": shadow[n] for n in self.linking_names})
        else:
            # pulp_l1 already stores economic prices
            economic = dict(shadow)
            duals_like = {f"balance_{n}": float(-shadow[n]) for n in self.linking_names}
            duals_like.update({f"admm_lambda_{n}": shadow[n] for n in self.linking_names})
            for u, v in util_shadow.items():
                duals_like[f"utility_cap_{u}"] = float(-v)

        status = "converged" if done else "max_iter"
        return ADMMResult(
            status=status,
            converged=done,
            iterations=last.iteration + 1,
            objective=last.objective_hat,
            shadow_prices=shadow,
            utility_shadow_prices=util_shadow,
            economic_shadow_prices=economic,
            crude_rates=last.crude_rates,
            product_rates=last.product_rates,
            intermediate_prod=last.prod,
            intermediate_use=last.use,
            residual=last.residual,
            r_norm=last.r_norm,
            s_norm=last.s_norm,
            rho=self.rho,
            history=self.history,
            solve_time_s=t1 - t0,
            backend=cfg.backend,
            mode=cfg.mode,
            duals_like_monolithic=duals_like,
        )


def run_admm(
    data: RefineryData,
    config: Optional[ADMMConfig] = None,
) -> ADMMResult:
    return ADMMCoordinator(data, config).run()
