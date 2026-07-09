"""Process-pool MIP: discrete severity / mode selection for conversion units.

PIMS-style process pools replace continuous severity with a small catalog of
operating modes (FCC ROT bands, coker recycle bands). Each mode owns a fixed
yield table. PuLP binary variables (SOS1-style: exactly one mode active)
select which table multiplies feed into product streams.

This module is intentionally standalone so full_plant.py stays LP-pure; the
bench / optional helpers can attach process-pool mode selection without a deep
plant rewrite.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pulp

from .properties import FeedProperties
from .unit_specs import merge_process_conditions
from .yields import coker_yields, fcc_yields

# ---------------------------------------------------------------------------
# Discrete mode catalogs (planning-grade bands)
# ---------------------------------------------------------------------------

# FCC severity via riser outlet temperature (ROT) bands + mild C/O coupling
FCC_ROT_MODES: List[Dict[str, Any]] = [
    {
        "id": "rot_low",
        "label": "FCC low severity (ROT ~960 F)",
        "conditions": {
            "riser_outlet_temp_f": 960.0,
            "catalyst_to_oil": 5.8,
            "catalyst_activity": 65.0,
            "feed_preheat_temp_f": 530.0,
            "recycle_ratio": 0.0,
        },
    },
    {
        "id": "rot_mid",
        "label": "FCC design severity (ROT ~980 F)",
        "conditions": {
            "riser_outlet_temp_f": 980.0,
            "catalyst_to_oil": 6.5,
            "catalyst_activity": 68.0,
            "feed_preheat_temp_f": 550.0,
            "recycle_ratio": 0.0,
        },
    },
    {
        "id": "rot_high",
        "label": "FCC high severity (ROT ~1005 F)",
        "conditions": {
            "riser_outlet_temp_f": 1005.0,
            "catalyst_to_oil": 7.2,
            "catalyst_activity": 72.0,
            "feed_preheat_temp_f": 570.0,
            "recycle_ratio": 0.05,
        },
    },
]

# Delayed coker recycle-ratio bands (higher recycle → more coke, less liquid)
COKER_RECYCLE_MODES: List[Dict[str, Any]] = [
    {
        "id": "rec_low",
        "label": "Coker low recycle (~0.05)",
        "conditions": {
            "drum_outlet_temp_f": 910.0,
            "recycle_ratio": 0.05,
            "drum_pressure_psig": 20.0,
        },
    },
    {
        "id": "rec_mid",
        "label": "Coker design recycle (~0.15)",
        "conditions": {
            "drum_outlet_temp_f": 920.0,
            "recycle_ratio": 0.15,
            "drum_pressure_psig": 25.0,
        },
    },
    {
        "id": "rec_high",
        "label": "Coker high recycle (~0.30)",
        "conditions": {
            "drum_outlet_temp_f": 930.0,
            "recycle_ratio": 0.30,
            "drum_pressure_psig": 30.0,
        },
    },
]


# Default product netbacks ($/bbl vol-equiv) for standalone mode-selection MIP
DEFAULT_STREAM_VALUES: Dict[str, float] = {
    "fcc_dry_gas": 35.0,
    "fcc_lpg": 45.0,
    "fcc_naphtha": 95.0,
    "fcc_lco": 85.0,
    "fcc_slurry": 55.0,
    "fcc_coke": 12.0,  # credit on feed basis (wt → planning credit)
    "coker_dry_gas": 35.0,
    "coker_lpg": 45.0,
    "coker_naphtha": 88.0,
    "coker_gasoil": 75.0,
    "coker_coke": 18.0,
}


def list_fcc_rot_modes() -> List[Dict[str, Any]]:
    return deepcopy(FCC_ROT_MODES)


def list_coker_recycle_modes() -> List[Dict[str, Any]]:
    return deepcopy(COKER_RECYCLE_MODES)


def default_gasoil_feed() -> FeedProperties:
    return FeedProperties(
        name="vgo_pool",
        api=24.0,
        sulfur_wt=0.8,
        ccr_wt=1.2,
        nitrogen_ppm=900.0,
        paraffins_vol=0.30,
        naphthenes_vol=0.35,
        aromatics_vol=0.35,
    )


def default_resid_feed() -> FeedProperties:
    return FeedProperties(
        name="resid_pool",
        api=12.0,
        sulfur_wt=2.5,
        ccr_wt=12.0,
        nitrogen_ppm=2500.0,
        paraffins_vol=0.25,
        naphthenes_vol=0.30,
        aromatics_vol=0.45,
    )


# ---------------------------------------------------------------------------
# Yield table generation per discrete mode
# ---------------------------------------------------------------------------


def build_fcc_mode_yield_tables(
    feed: Optional[FeedProperties] = None,
    modes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Map FCC mode id → yield vector (from yields.fcc_yields + mode conditions)."""
    feed = feed or default_gasoil_feed()
    modes = list(modes) if modes is not None else FCC_ROT_MODES
    out: Dict[str, Dict[str, float]] = {}
    for m in modes:
        mid = str(m["id"])
        cond = merge_process_conditions("FCC", m.get("conditions") or {})
        out[mid] = dict(fcc_yields(feed, cond))
    return out


def build_coker_mode_yield_tables(
    feed: Optional[FeedProperties] = None,
    modes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Map coker mode id → yield vector."""
    feed = feed or default_resid_feed()
    modes = list(modes) if modes is not None else COKER_RECYCLE_MODES
    out: Dict[str, Dict[str, float]] = {}
    for m in modes:
        mid = str(m["id"])
        cond = merge_process_conditions("COKER", m.get("conditions") or {})
        out[mid] = dict(coker_yields(feed, cond))
    return out


def build_process_pool_yield_library(
    gasoil: Optional[FeedProperties] = None,
    resid: Optional[FeedProperties] = None,
    fcc_modes: Optional[Sequence[Mapping[str, Any]]] = None,
    coker_modes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full library used by the MIP: modes + precomputed yield tables."""
    fcc_modes = list(fcc_modes) if fcc_modes is not None else FCC_ROT_MODES
    coker_modes = list(coker_modes) if coker_modes is not None else COKER_RECYCLE_MODES
    return {
        "fcc_modes": deepcopy(list(fcc_modes)),
        "coker_modes": deepcopy(list(coker_modes)),
        "fcc_yields_by_mode": build_fcc_mode_yield_tables(gasoil, fcc_modes),
        "coker_yields_by_mode": build_coker_mode_yield_tables(resid, coker_modes),
        "feed": {
            "gasoil": (gasoil or default_gasoil_feed()).__dict__,
            "resid": (resid or default_resid_feed()).__dict__,
        },
        "note": (
            "Discrete process-pool tables: SOS1/binary mode selection picks exactly "
            "one FCC ROT band and one coker recycle band."
        ),
    }


# ---------------------------------------------------------------------------
# PuLP helpers: binary / SOS1-style mode selection
# ---------------------------------------------------------------------------


@dataclass
class ModeSelectionVars:
    """Binary indicators for one unit's process-pool modes (exactly one on)."""

    unit: str
    binaries: Dict[str, pulp.LpVariable]
    mode_ids: List[str]

    def selected_id(self) -> Optional[str]:
        for mid, y in self.binaries.items():
            raw: Any = pulp.value(y)
            if raw is None:
                continue
            try:
                if float(raw) > 0.5:
                    return mid
            except (TypeError, ValueError):
                continue
        return None


def add_mode_selection(
    prob: pulp.LpProblem,
    mode_ids: Sequence[str],
    *,
    prefix: str,
    unit: str = "",
) -> ModeSelectionVars:
    """Add binary y_m with sum_m y_m = 1 (SOS1-style exclusive mode selection)."""
    if not mode_ids:
        raise ValueError("mode_ids must be non-empty")
    bins: Dict[str, pulp.LpVariable] = {}
    for mid in mode_ids:
        bins[mid] = pulp.LpVariable(f"{prefix}_{mid}", cat=pulp.LpBinary)
    # Exactly one mode — classic process-pool / special-ordered-set type 1 on binaries
    prob += pulp.lpSum(bins[m] for m in mode_ids) == 1, f"{prefix}_select_one"
    return ModeSelectionVars(unit=unit, binaries=bins, mode_ids=list(mode_ids))


def yield_from_modes(
    mode_vars: ModeSelectionVars,
    yields_by_mode: Mapping[str, Mapping[str, float]],
    stream: str,
    feed_var: pulp.LpVariable | float,
) -> pulp.LpAffineExpression:
    """Process-pool product: sum_m y_m * yield[m,stream] * feed.

    With binary y_m and sum y=1 this is exact discrete selection (not a continuous
    convex combination of severities).
    """
    terms = []
    for mid, ybin in mode_vars.binaries.items():
        coeff = float((yields_by_mode.get(mid) or {}).get(stream, 0.0))
        terms.append(coeff * ybin * feed_var)
    return pulp.lpSum(terms)


# ---------------------------------------------------------------------------
# Standalone process-pool MIP (demo / scale-up / tests)
# ---------------------------------------------------------------------------


@dataclass
class ProcessPoolResult:
    status: str
    feasible: bool
    objective: float
    fcc_mode: Optional[str]
    coker_mode: Optional[str]
    fcc_mode_selection: Dict[str, float]
    coker_mode_selection: Dict[str, float]
    fcc_feed: float
    coker_feed: float
    streams: Dict[str, float]
    yields_selected: Dict[str, Dict[str, float]]
    solve_time_s: float
    n_binaries: int
    meta: Dict[str, Any] = field(default_factory=dict)
    problem: Optional[pulp.LpProblem] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "objective": self.objective,
            "fcc_mode": self.fcc_mode,
            "coker_mode": self.coker_mode,
            "fcc_mode_selection": self.fcc_mode_selection,
            "coker_mode_selection": self.coker_mode_selection,
            "fcc_feed": self.fcc_feed,
            "coker_feed": self.coker_feed,
            "streams": self.streams,
            "yields_selected": self.yields_selected,
            "solve_time_s": self.solve_time_s,
            "n_binaries": self.n_binaries,
            "meta": self.meta,
        }


def solve_process_pool_mip(
    *,
    gasoil: Optional[FeedProperties] = None,
    resid: Optional[FeedProperties] = None,
    fcc_feed_kbd: float = 40.0,
    coker_feed_kbd: float = 25.0,
    stream_values: Optional[Mapping[str, float]] = None,
    fcc_modes: Optional[Sequence[Mapping[str, Any]]] = None,
    coker_modes: Optional[Sequence[Mapping[str, Any]]] = None,
    fix_fcc_mode: Optional[str] = None,
    fix_coker_mode: Optional[str] = None,
    msg: bool = False,
    time_limit: Optional[int] = None,
) -> ProcessPoolResult:
    """Solve a small process-pool MIP: pick FCC ROT band + coker recycle band.

    Feed rates are fixed (capacity-like). Binaries select yield tables; objective
    is product netback value of mode-selected yields. No deep coupling to
    full_plant — suitable for tests, demos, and scale-up portfolio copies.
    """
    t0 = time.perf_counter()
    gasoil = gasoil or default_gasoil_feed()
    resid = resid or default_resid_feed()
    fcc_modes = list(fcc_modes) if fcc_modes is not None else FCC_ROT_MODES
    coker_modes = list(coker_modes) if coker_modes is not None else COKER_RECYCLE_MODES
    values = dict(DEFAULT_STREAM_VALUES)
    if stream_values:
        values.update(dict(stream_values))

    fcc_y = build_fcc_mode_yield_tables(gasoil, fcc_modes)
    cok_y = build_coker_mode_yield_tables(resid, coker_modes)
    fcc_ids = [str(m["id"]) for m in fcc_modes]
    cok_ids = [str(m["id"]) for m in coker_modes]

    prob = pulp.LpProblem("process_pool_mip", pulp.LpMaximize)
    fcc_sel = add_mode_selection(prob, fcc_ids, prefix="fcc_mode", unit="FCC")
    cok_sel = add_mode_selection(prob, cok_ids, prefix="coker_mode", unit="COKER")

    if fix_fcc_mode is not None:
        if fix_fcc_mode not in fcc_sel.binaries:
            raise KeyError(f"unknown fcc mode {fix_fcc_mode!r}; have {fcc_ids}")
        prob += fcc_sel.binaries[fix_fcc_mode] == 1, "fix_fcc_mode"
    if fix_coker_mode is not None:
        if fix_coker_mode not in cok_sel.binaries:
            raise KeyError(f"unknown coker mode {fix_coker_mode!r}; have {cok_ids}")
        prob += cok_sel.binaries[fix_coker_mode] == 1, "fix_coker_mode"

    # Fixed feeds (continuous capacity dummies keep MIP structure extensible)
    fcc_feed = pulp.LpVariable("fcc_feed", lowBound=0.0, upBound=float(fcc_feed_kbd))
    cok_feed = pulp.LpVariable("coker_feed", lowBound=0.0, upBound=float(coker_feed_kbd))
    # Prefer full feed utilization when value is positive
    prob += fcc_feed == float(fcc_feed_kbd), "fcc_feed_fixed"
    prob += cok_feed == float(coker_feed_kbd), "coker_feed_fixed"

    fcc_streams = sorted({s for y in fcc_y.values() for s in y})
    cok_streams = sorted({s for y in cok_y.values() for s in y})

    # Stream production variables linked to selected mode yields
    prod: Dict[str, pulp.LpVariable] = {}
    for s in fcc_streams:
        prod[s] = pulp.LpVariable(f"prod_{s}", lowBound=0.0)
        # Big-M free form via identity: prod = sum y_m * coeff_m * feed
        # With binary exclusive selection this is linear in (y * feed) products —
        # y binary, feed fixed → y*feed is linear in y. Use fixed feed rate.
        prob += (
            prod[s]
            == pulp.lpSum(
                float(fcc_y[mid].get(s, 0.0)) * fcc_sel.binaries[mid] * float(fcc_feed_kbd)
                for mid in fcc_ids
            ),
            f"bal_{s}",
        )
    for s in cok_streams:
        prod[s] = pulp.LpVariable(f"prod_{s}", lowBound=0.0)
        prob += (
            prod[s]
            == pulp.lpSum(
                float(cok_y[mid].get(s, 0.0)) * cok_sel.binaries[mid] * float(coker_feed_kbd)
                for mid in cok_ids
            ),
            f"bal_{s}",
        )

    obj = pulp.lpSum(float(values.get(s, 0.0)) * prod[s] for s in prod)
    # Tiny mode preference so ties break toward mid design if values equal
    obj += 0.01 * fcc_sel.binaries.get("rot_mid", 0)
    obj += 0.01 * cok_sel.binaries.get("rec_mid", 0)
    prob += obj

    solver_kwargs: Dict[str, Any] = {"msg": msg}
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    status_code = prob.solve(solver)
    status = pulp.LpStatus.get(status_code, str(status_code))
    feasible = status == "Optimal"

    def _v(x) -> float:
        val = pulp.value(x)
        return float(val) if val is not None else 0.0

    fcc_choice = fcc_sel.selected_id()
    cok_choice = cok_sel.selected_id()
    fcc_sel_map = {m: _v(fcc_sel.binaries[m]) for m in fcc_ids}
    cok_sel_map = {m: _v(cok_sel.binaries[m]) for m in cok_ids}

    streams = {s: _v(prod[s]) for s in prod}
    yields_sel: Dict[str, Dict[str, float]] = {}
    if fcc_choice and fcc_choice in fcc_y:
        yields_sel["fcc"] = dict(fcc_y[fcc_choice])
    if cok_choice and cok_choice in cok_y:
        yields_sel["coker"] = dict(cok_y[cok_choice])

    # PuLP stores Binary as Integer 0–1; isBinary() is the reliable check.
    n_bin = sum(1 for v in prob.variables() if v.isBinary())

    return ProcessPoolResult(
        status=status,
        feasible=feasible,
        objective=_v(prob.objective) if feasible else float("nan"),
        fcc_mode=fcc_choice,
        coker_mode=cok_choice,
        fcc_mode_selection=fcc_sel_map,
        coker_mode_selection=cok_sel_map,
        fcc_feed=_v(fcc_feed),
        coker_feed=_v(cok_feed),
        streams=streams,
        yields_selected=yields_sel,
        solve_time_s=time.perf_counter() - t0,
        n_binaries=n_bin,
        meta={
            "kind": "process_pool_mip",
            "fcc_rot_bands": fcc_ids,
            "coker_recycle_bands": cok_ids,
            "stream_values": values,
            "note": (
                "SOS1-style binary mode selection: exactly one FCC ROT band and "
                "one coker recycle band; yield tables from yields.fcc_yields / coker_yields."
            ),
        },
        problem=prob,
    )


def process_pool_once(
    *,
    scale: int = 1,
    msg: bool = False,
) -> ProcessPoolResult:
    """Convenience: one MIP solve; scale multiplies feed rates (heavier MIP load)."""
    scale = max(1, int(scale))
    return solve_process_pool_mip(
        fcc_feed_kbd=40.0 * scale,
        coker_feed_kbd=25.0 * scale,
        msg=msg,
    )


def attach_process_pool_to_plant_yields(
    base_yields: Mapping[str, Any],
    pool_result: ProcessPoolResult,
) -> Dict[str, Any]:
    """Optional helper: merge selected process-pool yields into a yield-table dict.

    Does not mutate full_plant; callers (bench / what-if) can pass the result into
    custom solves. Prefer this over deep full_plant edits.
    """
    out = dict(base_yields)
    ys = pool_result.yields_selected or {}
    if "fcc" in ys:
        out["fcc"] = dict(ys["fcc"])
    if "coker" in ys:
        out["coker"] = dict(ys["coker"])
    meta = dict(out.get("process_pool") or {})
    meta.update(
        {
            "fcc_mode": pool_result.fcc_mode,
            "coker_mode": pool_result.coker_mode,
            "objective": pool_result.objective,
            "feasible": pool_result.feasible,
        }
    )
    out["process_pool"] = meta
    return out
