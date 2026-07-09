"""Property-driven yield vectors for CDU / FCC / Delayed Coker / Reformer.

Hard LP still uses linear yield coefficients; properties set those coefficients
before the solve (PIMS-style assay → LP rows). Soft nonlinear notes stay in the LLM layer.
"""

from __future__ import annotations

from typing import Dict, Mapping

from .properties import FeedProperties


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def cdu_yields_from_assay(props: FeedProperties, tbp_cut_vol: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Map crude assay + optional TBP cuts → CDU product yields (vol frac of charge).

    If TBP cuts provided, use them with mild API/S corrections; else synthesize from API/CCR.
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
    return {k: v / s for k, v in y.items()}


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


def fcc_yields(feed: FeedProperties) -> Dict[str, float]:
    """FCC yield on fresh feed (vol/vol charge). Conversion driven by API & CCR."""
    conversion = _clamp(0.58 + 0.008 * (feed.api - 22.0) - 0.025 * feed.ccr_wt, 0.35, 0.78)
    # product slate of converted + unconverted
    naph = 0.48 * conversion
    lco = 0.22 * conversion + 0.55 * (1.0 - conversion)
    slurry = 0.10 * conversion + 0.45 * (1.0 - conversion)
    # coke/gas lumped as loss from liquid (not an LP stream here)
    liquid = naph + lco + slurry
    if liquid <= 0:
        liquid = 1.0
    # normalize liquid yields to sum <= 0.95 (coke/gas ~5%+)
    scale = min(1.0, 0.93 / liquid)
    return {
        "fcc_naphtha": naph * scale,
        "fcc_lco": lco * scale,
        "fcc_slurry": slurry * scale,
    }


def coker_yields(feed: FeedProperties) -> Dict[str, float]:
    """Delayed coker liquid yields from resid CCR (volume on fresh feed)."""
    # Higher CCR → more coke, less liquid
    liquid = _clamp(0.72 - 0.018 * (feed.ccr_wt - 8.0), 0.45, 0.78)
    naph_frac = _clamp(0.22 + 0.002 * (20.0 - feed.api), 0.15, 0.30)
    go_frac = 1.0 - naph_frac
    return {
        "coker_naphtha": liquid * naph_frac,
        "coker_gasoil": liquid * go_frac,
        # coke not modeled as market stream in MVP LP
    }


def reformer_yields(feed: FeedProperties) -> Dict[str, float]:
    """Catalytic reformer C5+ reformate yield from naphtha N+A and nitrogen poison."""
    n_a = feed.n_plus_a
    # base reformate ~ 0.88; better N+A slightly higher liquid, high N hurts
    y_ref = _clamp(0.86 + 0.08 * (n_a - 0.55) - 0.00002 * max(0.0, feed.nitrogen_ppm - 500), 0.78, 0.94)
    return {"reformate": y_ref}


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
