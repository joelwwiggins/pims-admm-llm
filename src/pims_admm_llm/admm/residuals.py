"""Primal/dual residuals and convergence tests for consensus / balance ADMM."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def _vec(names: Sequence[str], d: Mapping[str, float]) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in names], dtype=float)


def primal_residual(
    linking_names: Sequence[str],
    prod: Mapping[str, float],
    use: Mapping[str, float],
) -> Dict[str, float]:
    """r = prod - use (material imbalance on intermediates)."""
    return {n: float(prod.get(n, 0.0) - use.get(n, 0.0)) for n in linking_names}


def dual_residual(
    linking_names: Sequence[str],
    z_new: Mapping[str, float],
    z_old: Mapping[str, float],
    rho: float,
) -> tuple[Dict[str, float], float]:
    """s = rho * (z^{k+1} - z^k); returns (per-name dict, ||s||_2)."""
    s = {n: float(rho * (z_new.get(n, 0.0) - z_old.get(n, 0.0))) for n in linking_names}
    norm = float(np.linalg.norm(_vec(linking_names, s)))
    return s, norm


def residual_norms(
    linking_names: Sequence[str],
    prod: Mapping[str, float],
    use: Mapping[str, float],
    z_new: Mapping[str, float],
    z_old: Mapping[str, float],
    rho: float,
) -> tuple[float, float, Dict[str, float]]:
    r = primal_residual(linking_names, prod, use)
    r_norm = float(np.linalg.norm(_vec(linking_names, r)))
    _, s_norm = dual_residual(linking_names, z_new, z_old, rho)
    return r_norm, s_norm, r


def converged(
    r_norm: float,
    s_norm: float,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
    scale: float = 1.0,
) -> bool:
    """Standard ADMM stopping: ||r|| <= eps_pri and ||s|| <= eps_dual.

    eps = abs_tol + rel_tol * scale (scale typically max(||prod||, ||use||, ||z||)).
    """
    eps = abs_tol + rel_tol * max(scale, 1.0)
    return r_norm <= eps and s_norm <= eps


def linf_dual_gap(
    lambda_prices: Mapping[str, float],
    mono_bal_duals: Mapping[str, float],
    stream_to_bal: Mapping[str, str] | None = None,
) -> tuple[float, Dict[str, float]]:
    """L∞ | |λ| − |mono bal dual| | on mapped linking streams (honest report).

    Returns (L∞, per-stream abs gaps). Does **not** claim dual recovery.
    """
    default_map = {
        "cdu_gasoil": "bal_tank_gasoil",
        "cdu_resid": "bal_tank_resid",
        "fcc_naphtha": "bal_tank_fcc_naph",
        "coker_naphtha": "bal_tank_coker_naph",
        "reformate": "bal_tank_reformate",
        "cdu_naphtha_heavy": "bal_sr_heavy",
        "cdu_naphtha_light": "bal_sr_light",
        "cdu_distillate": "bal_distillate",
        "fcc_lco": "bal_lco",
        "fcc_slurry": "bal_slurry",
        "coker_gasoil": "bal_coker_go",
    }
    mapping = dict(default_map)
    if stream_to_bal:
        mapping.update(stream_to_bal)
    gaps: Dict[str, float] = {}
    linf = 0.0
    for stream, bal_key in mapping.items():
        if stream not in lambda_prices and bal_key not in mono_bal_duals:
            continue
        lam = abs(float(lambda_prices.get(stream, 0.0)))
        mono = abs(float(mono_bal_duals.get(bal_key, 0.0)))
        g = abs(lam - mono)
        gaps[stream] = g
        linf = max(linf, g)
    return float(linf), gaps
