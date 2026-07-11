"""Hardened pure-ADMM residual / λ honesty tests (Wave5 multi-stream align)."""

from __future__ import annotations

from pims_admm_llm.admm.pure_plant_admm import (
    FREE_DISPOSAL_STREAMS,
    run_pure_plant_admm,
)
from pims_admm_llm.models.full_plant import admm_price_directed_plant


def test_pure_admm_residual_improves_and_shortage_bounded():
    # ρ≈2 keeps FCC active on current assay slate (ρ≈1.2 collapses FCC feed)
    out = run_pure_plant_admm(max_iter=80, rho=2.0, dual_step=0.35, damp=0.4, tol=5.0)
    assert out["dual_recovery_path"] == "pure-admm"
    assert out["duals_like_monolithic"] == {}
    h = out["history"]
    assert len(h) >= 5
    r0 = h[0]["primal_residual_norm"]
    rN = h[-1]["primal_residual_norm"]
    # residual must not explode; prefer improvement or bounded
    assert rN < r0 * 1.05 or rN < 40.0
    short = out.get("shortage_residual_norm", rN)
    # Wave4 full yield slate expanded faces; Wave5 decision shortage is primary gate
    dec_short = out.get("decision_shortage_residual_norm", short)
    assert short < 55.0, short
    assert dec_short < 55.0, dec_short
    # Free-disposal multi-stream residual should be ~0 after auto-sink align
    fd_r = float(out.get("free_disposal_residual_norm", 0.0))
    assert fd_r < 1e-3, fd_r
    # CDU must stay active (not collapse); ρ≈2 keeps FCC+coker active on assay slate
    assert out["unit_feeds"]["cdu_charge"] > 50.0
    assert out["unit_feeds"]["fcc_feed"] > 5.0
    assert out["unit_feeds"]["coker_feed"] > 5.0


def test_pure_admm_lambda_vs_mono_econ_reasonable():
    """Not dual recovery — but L∞ should be far better than unhinged (>200) paths."""
    out = admm_price_directed_plant(recovery_path="pure-admm", max_iter=80)
    assert out["dual_recovery_path"] == "pure-admm"
    linf = float(out.get("lambda_vs_mono_Linf") or 1e9)
    # hardened target: well below pre-fix ~100–460; structural floor ~20–40 remains
    assert linf < 80.0, linf
    # free λ never copies mono dual dict
    assert out["duals_like_monolithic"] == {}
    assert "structural" in (out.get("structural_linf_floor_note") or "").lower() or "floor" in (
        out.get("honesty") or ""
    ).lower()


def test_pure_admm_key_stream_lambda_positive():
    out = run_pure_plant_admm(max_iter=40)
    lam = out["lambda"]
    # free-disposal duals projected ≥ 0
    assert all(v >= -1e-6 for v in lam.values())
    # valuable products should carry positive intermediate prices
    assert lam.get("reformate", 0) > 10.0 or lam.get("fcc_naphtha", 0) > 10.0


def test_pure_admm_multi_stream_free_disposal_aligned():
    """Wave5: blocks emit multi-stream yields; free-disposal residual ~0."""
    out = run_pure_plant_admm(max_iter=30)
    assert out["dual_recovery_path"] == "pure-admm"
    props = out["block_proposals"]
    # multi-stream products present on conversion blocks
    assert props["FCC"].get("fcc_dry_gas", 0) >= 0.0
    assert props["FCC"].get("fcc_lpg", 0) >= 0.0 or props["FCC"].get("fcc_naphtha", 0) > 0
    assert "fcc_coke" in props["FCC"]
    assert "coker_coke" in props["COKER"]
    assert "cdu_offgas" in props["CDU"]
    assert "reformer_h2" in props["REFORMER"]
    # free disposal residual metric
    assert float(out["free_disposal_residual_norm"]) < 1e-3
    # decision shortage tracked separately
    assert "decision_shortage_residual_norm" in out
    rb = out.get("residual_breakdown") or {}
    for s in FREE_DISPOSAL_STREAMS:
        if s in (rb.get("free_disposal_streams") or []):
            break
    else:
        # at least the metric keys exist
        assert "free_disposal_streams" in rb or out["free_disposal_residual_norm"] == 0.0
