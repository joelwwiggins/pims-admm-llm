"""Property-driven yield vectors for CDU / FCC / Delayed Coker / Reformer.

Hard LP still uses linear yield coefficients; properties + process conditions
set those coefficients before the solve (PIMS-style assay → LP rows). Soft
nonlinear notes stay in the LLM layer.

Wave4: each unit exposes a full product slate (dry gas, LPG, liquids, coke)
so every yield stream can be routed somewhere in the superstructure.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .properties import FeedProperties
from .unit_specs import merge_process_conditions


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def cdu_yields_from_assay(props: FeedProperties, tbp_cut_vol: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Map crude assay + optional TBP cuts → CDU product yields (vol frac of charge).

    If TBP cuts provided, use them with mild API/S corrections; else synthesize from API/CCR.
    Includes ``cdu_offgas`` (fuel-gas equiv, small). Liquid cuts still sum ≈ 1.0 so
    existing mass-balance tests pass; offgas is extra fuel credit fraction on charge.
    """
    if tbp_cut_vol:
        y = {
            "cdu_naphtha": float(tbp_cut_vol.get("naphtha_ibp_350f", 0.2)),
            "cdu_distillate": float(tbp_cut_vol.get("distillate_350_650f", 0.25)),
            "cdu_gasoil": float(tbp_cut_vol.get("gasoil_650_1050f", 0.28)),
            "cdu_resid": float(tbp_cut_vol.get("resid_1050f_plus", 0.27)),
        }
    else:
        # Rough API-driven cut: lighter crude → more lights, less resid
        api = props.api
        naph = _clamp(0.12 + 0.0045 * (api - 20.0), 0.08, 0.35)
        dist = _clamp(0.22 + 0.0025 * (api - 20.0), 0.15, 0.35)
        go = _clamp(0.30 - 0.001 * (api - 25.0), 0.20, 0.35)
        resid = _clamp(1.0 - naph - dist - go, 0.10, 0.50)
        # renormalize
        s = naph + dist + go + resid
        y = {
            "cdu_naphtha": naph / s,
            "cdu_distillate": dist / s,
            "cdu_gasoil": go / s,
            "cdu_resid": resid / s,
        }

    # Sulfur / CCR nudge: heavier residual bias if high CCR
    if props.ccr_wt > 5.0:
        shift = min(0.04, 0.003 * (props.ccr_wt - 5.0))
        y["cdu_resid"] += shift
        y["cdu_naphtha"] = max(0.05, y["cdu_naphtha"] - shift * 0.4)
        y["cdu_distillate"] = max(0.05, y["cdu_distillate"] - shift * 0.3)
        y["cdu_gasoil"] = max(0.05, y["cdu_gasoil"] - shift * 0.3)
    s = sum(y.values())
    out = {k: v / s for k, v in y.items()}
    # Overhead offgas / fuel (not part of liquid sum) — planning credit
    out["cdu_offgas"] = _clamp(0.008 + 0.0004 * props.api * 0.1, 0.005, 0.02)
    return out


def gasoil_props_from_crude(props: FeedProperties) -> FeedProperties:
    """Approximate atmospheric/vacuum gasoil properties from parent crude."""
    return FeedProperties(
        name=f"{props.name}_gasoil",
        api=_clamp(props.api - 8.0, 15.0, 35.0),
        sulfur_wt=props.sulfur_wt * 1.1,
        ccr_wt=_clamp(props.ccr_wt * 0.25, 0.2, 6.0),
        nitrogen_ppm=props.nitrogen_ppm * 1.2,
        paraffins_vol=props.paraffins_vol * 0.9,
        naphthenes_vol=props.naphthenes_vol,
        aromatics_vol=_clamp(props.aromatics_vol * 1.1, 0.1, 0.7),
    )


def resid_props_from_crude(props: FeedProperties) -> FeedProperties:
    return FeedProperties(
        name=f"{props.name}_resid",
        api=_clamp(props.api - 18.0, 5.0, 25.0),
        sulfur_wt=props.sulfur_wt * 1.4,
        ccr_wt=_clamp(props.ccr_wt * 2.2, 5.0, 25.0),
        nitrogen_ppm=props.nitrogen_ppm * 1.5,
        paraffins_vol=props.paraffins_vol * 0.7,
        naphthenes_vol=props.naphthenes_vol * 0.8,
        aromatics_vol=_clamp(props.aromatics_vol * 1.3, 0.2, 0.8),
    )


def _fcc_severity_delta(conditions: Mapping[str, Any]) -> float:
    """Map ROT / C/O / activity to conversion delta around base severity."""
    rot = float(conditions.get("riser_outlet_temp_f", 980.0))
    co = float(conditions.get("catalyst_to_oil", 6.5))
    act = float(conditions.get("catalyst_activity", 68.0))
    preheat = float(conditions.get("feed_preheat_temp_f", 550.0))
    # Base design: 980 F, C/O 6.5, MAT 68
    d = 0.0
    d += 0.00055 * (rot - 980.0)  # ~0.055 per 100 F
    d += 0.012 * (co - 6.5)
    d += 0.0015 * (act - 68.0)
    d += 0.00008 * (preheat - 550.0)
    return _clamp(d, -0.12, 0.12)


def fcc_yields(
    feed: FeedProperties,
    conditions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """FCC yield on fresh feed.

    Liquid products are vol/vol charge (dry gas, LPG, naphtha, LCO, slurry).
    ``fcc_coke`` is wt frac of fresh feed (regenerator heat balance credit).

    Typical planning ranges (vol% feed unless noted):
      dry gas 3–8, LPG 12–25, gasoline 40–55, LCO 15–25, slurry 5–15, coke 4–8 wt%.
    """
    cond = merge_process_conditions("FCC", conditions)
    conversion = _clamp(
        0.58 + 0.008 * (feed.api - 22.0) - 0.025 * feed.ccr_wt + _fcc_severity_delta(cond),
        0.35,
        0.82,
    )
    recycle = float(cond.get("recycle_ratio", 0.0))
    coke_wt = _clamp(0.045 + 0.012 * feed.ccr_wt + 0.02 * (1.0 - conversion), 0.035, 0.10)
    liquid_vol = _clamp(0.96 - 0.55 * coke_wt, 0.88, 0.96)

    # Converted products scale with conversion; bottoms with (1-conversion).
    # Explicit API/CCR terms keep lighter/low-CCR feeds above heavy on gasoline.
    dry = _clamp(0.035 + 0.055 * conversion + 0.01 * recycle, 0.03, 0.09)
    lpg = _clamp(0.11 + 0.14 * conversion, 0.10, 0.26)
    naph = _clamp(
        0.38 + 0.28 * conversion + 0.004 * (feed.api - 22.0) - 0.02 * feed.ccr_wt,
        0.30,
        0.56,
    )
    unconv = 1.0 - conversion
    lco = _clamp(0.14 + 0.32 * unconv, 0.12, 0.30)
    slurry = _clamp(0.06 + 0.28 * unconv + 0.012 * feed.ccr_wt, 0.04, 0.22)

    raw = dry + lpg + naph + lco + slurry
    scale = liquid_vol / raw if raw > 1e-9 else 1.0
    return {
        "fcc_dry_gas": dry * scale,
        "fcc_lpg": lpg * scale,
        "fcc_naphtha": naph * scale,
        "fcc_lco": lco * scale,
        "fcc_slurry": slurry * scale,
        "fcc_coke": coke_wt,
    }


def coker_yields(
    feed: FeedProperties,
    conditions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Delayed coker yields from resid CCR + severity.

    Liquids: dry gas, LPG, naphtha, gasoil (vol/vol feed).
    ``coker_coke`` wt frac of feed.
    """
    cond = merge_process_conditions("COKER", conditions)
    drum_t = float(cond.get("drum_outlet_temp_f", 920.0))
    recycle = float(cond.get("recycle_ratio", 0.15))
    # Higher CCR → more coke, less liquid
    liquid = _clamp(0.72 - 0.018 * (feed.ccr_wt - 8.0) - 0.15 * recycle, 0.42, 0.78)
    # Severity nudge
    liquid = _clamp(liquid + 0.00015 * (drum_t - 920.0), 0.40, 0.80)
    coke_wt = _clamp(1.0 - liquid - 0.04, 0.12, 0.35)  # residual solids + gas makeup

    # Split liquids
    dry = _clamp(0.06 + 0.002 * (feed.ccr_wt - 8.0), 0.04, 0.12)
    lpg = _clamp(0.05 + 0.001 * (drum_t - 900.0) / 10.0, 0.03, 0.09)
    naph_frac = _clamp(0.22 + 0.002 * (20.0 - feed.api), 0.15, 0.30)
    # remaining liquid after gas/lpg
    mid = max(0.05, liquid - dry - lpg)
    naph = mid * naph_frac
    go = mid * (1.0 - naph_frac)
    # Renormalize liquids to `liquid`
    liq_sum = dry + lpg + naph + go
    if liq_sum > 1e-9:
        sc = liquid / liq_sum
        dry, lpg, naph, go = dry * sc, lpg * sc, naph * sc, go * sc
    return {
        "coker_dry_gas": dry,
        "coker_lpg": lpg,
        "coker_naphtha": naph,
        "coker_gasoil": go,
        "coker_coke": coke_wt,
    }


def reformer_yields(
    feed: FeedProperties,
    conditions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Catalytic reformer yields: C5+ reformate, net H2, C1–C4 lights."""
    cond = merge_process_conditions("REFORMER", conditions)
    n_a = feed.n_plus_a
    wait = float(cond.get("weighted_wait_avg_f", 940.0))
    p_psig = float(cond.get("pressure_psig", 150.0))
    # base reformate ~ 0.88; better N+A slightly higher liquid, high N hurts
    y_ref = _clamp(
        0.86
        + 0.08 * (n_a - 0.55)
        - 0.00002 * max(0.0, feed.nitrogen_ppm - 500)
        + 0.00005 * (wait - 940.0)
        - 0.00015 * (p_psig - 150.0),
        0.78,
        0.94,
    )
    # lights + H2 fill the rest of feed vol (planning approximation)
    remainder = max(0.04, 1.0 - y_ref)
    h2 = _clamp(0.25 * remainder + 0.01 * (n_a - 0.5), 0.015, 0.08)
    lights = max(0.02, remainder - h2)
    return {
        "reformate": y_ref,
        "reformer_h2": h2,
        "reformer_lights": lights,
    }


def naphtha_props_fcc(feed_go: FeedProperties) -> FeedProperties:
    return FeedProperties(
        name="fcc_naphtha",
        api=_clamp(feed_go.api + 25.0, 45.0, 65.0),
        sulfur_wt=feed_go.sulfur_wt * 0.15,
        ccr_wt=0.05,
        nitrogen_ppm=feed_go.nitrogen_ppm * 0.4,
        paraffins_vol=0.25,
        naphthenes_vol=0.30,
        aromatics_vol=0.45,
    )


def naphtha_props_coker(feed_resid: FeedProperties) -> FeedProperties:
    return FeedProperties(
        name="coker_naphtha",
        api=_clamp(feed_resid.api + 30.0, 45.0, 60.0),
        sulfur_wt=feed_resid.sulfur_wt * 0.2,
        ccr_wt=0.1,
        nitrogen_ppm=feed_resid.nitrogen_ppm * 0.5,
        paraffins_vol=0.35,
        naphthenes_vol=0.28,
        aromatics_vol=0.37,
    )


def hdt_naph_yields(conditions: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Soft HDT liquid recovery (vol on naphtha charge)."""
    cond = merge_process_conditions("HDT_NAPH", conditions)
    # Higher severity slightly more lights
    t = float(cond.get("reactor_inlet_temp_f", 550.0))
    lights = _clamp(0.015 + 0.00002 * (t - 550.0), 0.01, 0.04)
    return {"hdt_naphtha": 1.0 - lights, "hdt_lights": lights}
