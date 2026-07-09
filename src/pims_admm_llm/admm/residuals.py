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
