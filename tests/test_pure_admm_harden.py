"""Hardened pure-ADMM residual / λ honesty tests."""

from __future__ import annotations

from pims_admm_llm.admm.pure_plant_admm import run_pure_plant_admm
from pims_admm_llm.models.full_plant import admm_price_directed_plant


def test_pure_admm_residual_improves_and_shortage_bounded():
    out = run_pure_plant_admm(max_iter=80, rho=1.2, dual_step=0.35, damp=0.4, tol=5.0)
    assert out["dual_recovery_path"] == "pure-admm"
    assert out["duals_like_monolithic"] == {}
    h = out["history"]
    assert len(h) >= 5
    r0 = h[0]["primal_residual_norm"]
    rN = h[-1]["primal_residual_norm"]
    # residual must not explode; prefer improvement or bounded
    assert rN < r0 * 1.05 or rN < 40.0
    short = out.get("shortage_residual_norm", rN)
    assert short < 25.0, short
    # CDU must stay active (not collapse)
    assert out["unit_feeds"]["cdu_charge"] > 50.0
    assert out["unit_feeds"]["fcc_feed"] > 5.0
    assert out["unit_feeds"]["coker_feed"] > 5.0


def test_pure_admm_lambda_vs_mono_econ_reasonable():
    """Not dual recovery — but L∞ should be far better than unhinged (>200) paths."""
    out = admm_price_directed_plant(recovery_path="pure-admm", max_iter=80)
    assert out["dual_recovery_path"] == "pure-admm"
    linf = float(out.get("lambda_vs_mono_Linf") or 1e9)
    # hardened target: well below pre-fix ~100–460
    assert linf < 80.0, linf
    # free λ never copies mono dual dict
    assert out["duals_like_monolithic"] == {}


def test_pure_admm_key_stream_lambda_positive():
    out = run_pure_plant_admm(max_iter=40)
    lam = out["lambda"]
    # free-disposal duals projected ≥ 0
    assert all(v >= -1e-6 for v in lam.values())
    # valuable products should carry positive intermediate prices
    assert lam.get("reformate", 0) > 10.0 or lam.get("fcc_naphtha", 0) > 10.0
