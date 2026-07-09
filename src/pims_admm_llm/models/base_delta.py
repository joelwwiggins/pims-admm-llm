"""PIMS-style BASE + DELTA LP submodels for conversion units.

Each unit submodel exposes:
  - product yield streams (every product has an exit)
  - BASE yields at reference feed + process conditions
  - DELTA columns for feed attributes and process-condition deviations
  - product stream compositions (planning property vectors)
  - default exit destinations when the flowsheet has no drawn edge

Process conditions are first-class drivers of the yield vector (not just
decorative meta). For LP embedding, use ``process_modes()`` (SOS1 discrete
severity bands) or ``evaluate()`` at fixed conditions.

Scope note: CDU + FCC are complete; other units can reuse the same pattern.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .stream_composition import (
    FAMILY_DISTILLATE,
    FAMILY_GASOIL,
    FAMILY_LIGHT_ENDS,
    FAMILY_NAPHTHA,
    FAMILY_RESID,
    FAMILY_SOLID,
    StreamComposition,
    get_stream,
)
from .unit_specs import DEFAULT_PROCESS_CONDITIONS, merge_process_conditions


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ProductExit:
    """Where a product leaves the unit block if no flowsheet edge is drawn."""

    stream: str
    default_sink: str  # FUEL_GAS | LPG | GASOLINE | DIESEL | FO | REGEN_HEAT | SELL | FCC | ...
    basis: str = "vol"  # vol | wt | fuel
    required: bool = True
    alt_sinks: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class BaseDeltaModel:
    """Serializable BASE/DELTA package for one unit."""

    unit: str
    products: List[str]
    base_yields: Dict[str, float]  # product → yield on fresh feed
    # delta[product][driver] = dy / d(driver) at base point
    deltas: Dict[str, Dict[str, float]]
    reference_feed: Dict[str, float]
    reference_conditions: Dict[str, Any]
    exits: List[ProductExit]
    drivers: List[str]  # ordered driver names used in deltas
    product_compositions: Dict[str, Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "products": list(self.products),
            "base_yields": dict(self.base_yields),
            "deltas": {p: dict(d) for p, d in self.deltas.items()},
            "reference_feed": dict(self.reference_feed),
            "reference_conditions": deepcopy(self.reference_conditions),
            "exits": [
                {
                    "stream": e.stream,
                    "default_sink": e.default_sink,
                    "basis": e.basis,
                    "required": e.required,
                    "alt_sinks": list(e.alt_sinks),
                    "note": e.note,
                }
                for e in self.exits
            ],
            "drivers": list(self.drivers),
            "product_compositions": deepcopy(self.product_compositions),
            "notes": list(self.notes),
        }

    def missing_exits(self) -> List[str]:
        exit_streams = {e.stream for e in self.exits}
        return [p for p in self.products if p not in exit_streams]

    def evaluate(
        self,
        feed: Optional[Mapping[str, float]] = None,
        conditions: Optional[Mapping[str, Any]] = None,
        clamp_products: bool = True,
    ) -> Dict[str, float]:
        """Apply BASE + Σ DELTA_j * (x_j − x_j0) → yield vector."""
        feed = dict(self.reference_feed if feed is None else feed)
        cond = merge_process_conditions(self.unit, conditions)
        # Flatten process drivers into numeric map
        x = dict(feed)
        for k, v in cond.items():
            if isinstance(v, (int, float)):
                x[k] = float(v)
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        x[f"{k}.{kk}"] = float(vv)

        y: Dict[str, float] = {}
        for p in self.products:
            base = float(self.base_yields.get(p, 0.0))
            dy = 0.0
            for drv, coef in self.deltas.get(p, {}).items():
                x0 = self._driver_ref(drv)
                xv = float(x.get(drv, x0))
                dy += float(coef) * (xv - x0)
            val = base + dy
            if clamp_products:
                val = max(0.0, val)
            y[p] = val

        # Renormalize liquid products if unit policy says so
        y = self._postprocess_yields(y, cond)
        return y

    def _driver_ref(self, drv: str) -> float:
        """Numeric reference value for a BASE/DELTA driver name."""
        if drv in self.reference_feed:
            v = self.reference_feed[drv]
            if isinstance(v, (int, float)):
                return float(v)
        if drv in self.reference_conditions:
            v = self.reference_conditions[drv]
            if isinstance(v, (int, float)):
                return float(v)
        if drv.startswith("cut_points_f."):
            sub = drv.split(".", 1)[1]
            cuts = self.reference_conditions.get("cut_points_f") or {}
            if isinstance(cuts, Mapping) and sub in cuts:
                cv = cuts[sub]
                if isinstance(cv, (int, float)):
                    return float(cv)
        return 0.0

    def _postprocess_yields(
        self, y: Dict[str, float], cond: Mapping[str, Any]
    ) -> Dict[str, float]:
        return y

    def compositions_at(
        self,
        feed: Optional[Mapping[str, float]] = None,
        conditions: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, StreamComposition]:
        """Return product compositions (base library + mild severity shifts)."""
        feed = dict(self.reference_feed if feed is None else feed)
        cond = merge_process_conditions(self.unit, conditions)
        out: Dict[str, StreamComposition] = {}
        for p, raw in self.product_compositions.items():
            comp = StreamComposition.from_mapping(p, raw)
            out[p] = self._adjust_composition(p, comp, feed, cond)
        return out

    def _adjust_composition(
        self,
        product: str,
        comp: StreamComposition,
        feed: Mapping[str, float],
        cond: Mapping[str, Any],
    ) -> StreamComposition:
        return comp


# ---------------------------------------------------------------------------
# CDU atmospheric tower
# ---------------------------------------------------------------------------

CDU_PRODUCTS = [
    "cdu_offgas",
    "cdu_naphtha_light",
    "cdu_naphtha_heavy",
    "cdu_distillate",
    "cdu_gasoil",
    "cdu_resid",
]

CDU_EXITS: List[ProductExit] = [
    ProductExit("cdu_offgas", "FUEL_GAS", basis="fuel", alt_sinks=["SELL"], note="overhead fuel"),
    ProductExit(
        "cdu_naphtha_light",
        "GASOLINE",
        alt_sinks=["BLENDER", "SELL"],
        note="SR light → gasoline pool",
    ),
    ProductExit(
        "cdu_naphtha_heavy",
        "REFORMER",
        alt_sinks=["GASOLINE", "BLENDER", "SELL"],
        note="SR heavy → reformer preferred",
    ),
    ProductExit(
        "cdu_distillate",
        "DIESEL",
        alt_sinks=["BLENDER", "SELL"],
        note="kero/diesel",
    ),
    ProductExit(
        "cdu_gasoil",
        "FCC",
        alt_sinks=["DIESEL", "SELL", "POOL_FCC"],
        note="VGO → FCC preferred",
    ),
    ProductExit(
        "cdu_resid",
        "FO",
        alt_sinks=["COKER", "SELL", "POOL_COKER"],
        note="resid → FO / coker",
    ),
]


def build_cdu_base_delta(
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> BaseDeltaModel:
    """CDU BASE/DELTA at planning reference.

    Drivers:
      feed: api, sulfur_wt, ccr_wt
      process: flash_zone_temp_f, overflash_frac, cut_points_f.naphtha_ep,
               cut_points_f.distillate_ep, cut_points_f.gasoil_ep
    """
    ref_feed = {
        "api": 30.0,
        "sulfur_wt": 1.0,
        "ccr_wt": 2.0,
        "nitrogen_ppm": 1000.0,
    }
    if reference_feed:
        ref_feed.update({k: float(v) for k, v in reference_feed.items() if isinstance(v, (int, float))})

    ref_cond = merge_process_conditions("CDU", reference_conditions)
    # Base liquid slate from API-driven cuts (consistent with yields.cdu_yields_from_assay spirit)
    api = ref_feed["api"]
    naph = _clamp(0.12 + 0.0045 * (api - 20.0), 0.08, 0.35)
    dist = _clamp(0.22 + 0.0025 * (api - 20.0), 0.15, 0.35)
    go = _clamp(0.30 - 0.001 * (api - 25.0), 0.20, 0.35)
    resid = _clamp(1.0 - naph - dist - go, 0.10, 0.50)
    s = naph + dist + go + resid
    naph, dist, go, resid = naph / s, dist / s, go / s, resid / s
    light_frac = 0.40
    base = {
        "cdu_offgas": 0.01,
        "cdu_naphtha_light": naph * light_frac,
        "cdu_naphtha_heavy": naph * (1.0 - light_frac),
        "cdu_distillate": dist,
        "cdu_gasoil": go,
        "cdu_resid": resid,
    }

    # Finite-difference-ish planning deltas (per unit of driver)
    # Positive naphtha_ep → more naphtha, less distillate
    deltas: Dict[str, Dict[str, float]] = {p: {} for p in CDU_PRODUCTS}
    # API: lighter crude → more lights
    for p, coef in {
        "cdu_naphtha_light": 0.0010,
        "cdu_naphtha_heavy": 0.0012,
        "cdu_distillate": 0.0008,
        "cdu_gasoil": -0.0005,
        "cdu_resid": -0.0025,
        "cdu_offgas": 0.00005,
    }.items():
        deltas[p]["api"] = coef
    for p, coef in {
        "cdu_resid": 0.004,
        "cdu_gasoil": 0.001,
        "cdu_naphtha_light": -0.0015,
        "cdu_naphtha_heavy": -0.0015,
        "cdu_distillate": -0.001,
        "cdu_offgas": 0.0,
    }.items():
        deltas[p]["ccr_wt"] = coef
    for p, coef in {
        "cdu_resid": 0.002,
        "cdu_gasoil": 0.001,
        "cdu_distillate": 0.0005,
        "cdu_naphtha_light": -0.001,
        "cdu_naphtha_heavy": -0.001,
        "cdu_offgas": 0.0002,
    }.items():
        deltas[p]["sulfur_wt"] = coef

    # Process: flash zone / cut points (°F)
    for p, coef in {
        "cdu_gasoil": 0.00008,
        "cdu_resid": -0.00010,
        "cdu_distillate": 0.00002,
        "cdu_naphtha_light": 0.0,
        "cdu_naphtha_heavy": 0.0,
        "cdu_offgas": 0.00001,
    }.items():
        deltas[p]["flash_zone_temp_f"] = coef
    for p, coef in {
        "cdu_naphtha_light": 0.00015,
        "cdu_naphtha_heavy": 0.00010,
        "cdu_distillate": -0.00020,
        "cdu_gasoil": -0.00005,
        "cdu_resid": 0.0,
        "cdu_offgas": 0.0,
    }.items():
        deltas[p]["cut_points_f.naphtha_ep"] = coef
    for p, coef in {
        "cdu_distillate": 0.00012,
        "cdu_gasoil": -0.00010,
        "cdu_naphtha_light": 0.0,
        "cdu_naphtha_heavy": 0.0,
        "cdu_resid": -0.00002,
        "cdu_offgas": 0.0,
    }.items():
        deltas[p]["cut_points_f.distillate_ep"] = coef
    for p, coef in {
        "cdu_gasoil": 0.00010,
        "cdu_resid": -0.00010,
        "cdu_distillate": 0.0,
        "cdu_naphtha_light": 0.0,
        "cdu_naphtha_heavy": 0.0,
        "cdu_offgas": 0.0,
    }.items():
        deltas[p]["cut_points_f.gasoil_ep"] = coef
    for p, coef in {
        "cdu_gasoil": 0.15,
        "cdu_resid": -0.10,
        "cdu_distillate": -0.05,
        "cdu_naphtha_light": 0.0,
        "cdu_naphtha_heavy": 0.0,
        "cdu_offgas": 0.0,
    }.items():
        deltas[p]["overflash_frac"] = coef

    comps = {p: get_stream(p).to_dict() for p in CDU_PRODUCTS}
    drivers = [
        "api",
        "sulfur_wt",
        "ccr_wt",
        "flash_zone_temp_f",
        "overflash_frac",
        "cut_points_f.naphtha_ep",
        "cut_points_f.distillate_ep",
        "cut_points_f.gasoil_ep",
    ]

    model = _CDUModel(
        unit="CDU",
        products=list(CDU_PRODUCTS),
        base_yields=base,
        deltas=deltas,
        reference_feed=ref_feed,
        reference_conditions=ref_cond,
        exits=list(CDU_EXITS),
        drivers=drivers,
        product_compositions=comps,
        notes=[
            "Liquid cuts renormalized to sum≈1; cdu_offgas is additive fuel fraction.",
            "Cut-point and flash-zone drivers are process conditions inside the unit block.",
            "Every product has a ProductExit (default sink + alts).",
        ],
    )
    return model


class _CDUModel(BaseDeltaModel):
    def _postprocess_yields(
        self, y: Dict[str, float], cond: Mapping[str, Any]
    ) -> Dict[str, float]:
        off = y.get("cdu_offgas", 0.01)
        liquids = [p for p in self.products if p != "cdu_offgas"]
        s = sum(max(0.0, y[p]) for p in liquids)
        if s > 1e-12:
            for p in liquids:
                y[p] = max(0.0, y[p]) / s
        y["cdu_offgas"] = _clamp(off, 0.005, 0.03)
        return y

    def evaluate(
        self,
        feed: Optional[Mapping[str, float]] = None,
        conditions: Optional[Mapping[str, Any]] = None,
        clamp_products: bool = True,
    ) -> Dict[str, float]:
        # Flatten nested cut points into reference_conditions numeric lookup
        feed_m = dict(self.reference_feed if feed is None else feed)
        cond = merge_process_conditions(self.unit, conditions)
        # stash flat process numbers onto a synthetic feed map for parent logic
        flat = dict(feed_m)
        for k, v in cond.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)
        cuts = cond.get("cut_points_f") or {}
        if isinstance(cuts, Mapping):
            for ck, cv in cuts.items():
                if isinstance(cv, (int, float)):
                    flat[f"cut_points_f.{ck}"] = float(cv)
        # temporarily extend reference_conditions with flat keys for x0
        old_ref = self.reference_conditions
        ref_flat = dict(old_ref)
        for k, v in old_ref.items():
            if isinstance(v, (int, float)):
                ref_flat[k] = float(v)
        cuts0 = old_ref.get("cut_points_f") or {}
        if isinstance(cuts0, Mapping):
            for ck, cv in cuts0.items():
                if isinstance(cv, (int, float)):
                    ref_flat[f"cut_points_f.{ck}"] = float(cv)
        self.reference_conditions = ref_flat
        try:
            y: Dict[str, float] = {}
            for p in self.products:
                base = float(self.base_yields.get(p, 0.0))
                dy = 0.0
                for drv, coef in self.deltas.get(p, {}).items():
                    x0 = float(ref_flat.get(drv, self.reference_feed.get(drv, 0.0)))
                    xv = float(flat.get(drv, x0))
                    dy += float(coef) * (xv - x0)
                val = base + dy
                if clamp_products:
                    val = max(0.0, val)
                y[p] = val
            y = self._postprocess_yields(y, cond)
            return y
        finally:
            self.reference_conditions = old_ref


# ---------------------------------------------------------------------------
# FCC
# ---------------------------------------------------------------------------

FCC_PRODUCTS = [
    "fcc_dry_gas",
    "fcc_lpg",
    "fcc_naphtha",
    "fcc_lco",
    "fcc_slurry",
    "fcc_coke",
]

FCC_EXITS: List[ProductExit] = [
    ProductExit("fcc_dry_gas", "FUEL_GAS", basis="vol", alt_sinks=["SELL"], note="C1–C2 fuel"),
    ProductExit("fcc_lpg", "LPG", basis="vol", alt_sinks=["FUEL_GAS", "SELL"], note="C3/C4"),
    ProductExit(
        "fcc_naphtha",
        "GASOLINE",
        basis="vol",
        alt_sinks=["HDT_NAPH", "BLENDER", "SELL"],
        note="cat gasoline — not reformer",
    ),
    ProductExit(
        "fcc_lco",
        "DIESEL",
        basis="vol",
        alt_sinks=["FO", "BLENDER", "SELL"],
        note="LCO",
    ),
    ProductExit(
        "fcc_slurry",
        "FO",
        basis="vol",
        alt_sinks=["SELL", "COKER"],
        note="slurry oil",
    ),
    ProductExit(
        "fcc_coke",
        "REGEN_HEAT",
        basis="wt",
        alt_sinks=[],
        note="burned in regenerator",
    ),
]


def build_fcc_base_delta(
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> BaseDeltaModel:
    """FCC BASE/DELTA with ROT, C/O, activity, preheat, recycle as process drivers."""
    ref_feed = {
        "api": 22.0,
        "sulfur_wt": 0.45,
        "ccr_wt": 0.4,
        "nitrogen_ppm": 800.0,
        "metals_ni_v_ppm": 1.0,
        "uop_k": 11.8,
    }
    if reference_feed:
        ref_feed.update({k: float(v) for k, v in reference_feed.items() if isinstance(v, (int, float))})

    ref_cond = merge_process_conditions("FCC", reference_conditions)

    # Base yields at reference (vol frac feed; coke wt)
    conversion = 0.62
    base = {
        "fcc_dry_gas": 0.045,
        "fcc_lpg": 0.16,
        "fcc_naphtha": 0.46,
        "fcc_lco": 0.18,
        "fcc_slurry": 0.09,
        "fcc_coke": 0.055,
    }
    # light renormalize liquids to ~0.935
    liq = ["fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry"]
    s = sum(base[p] for p in liq)
    scale = 0.935 / s
    for p in liq:
        base[p] *= scale

    deltas: Dict[str, Dict[str, float]] = {p: {} for p in FCC_PRODUCTS}
    # Feed API
    for p, coef in {
        "fcc_naphtha": 0.004,
        "fcc_lpg": 0.0015,
        "fcc_dry_gas": 0.0005,
        "fcc_lco": -0.002,
        "fcc_slurry": -0.003,
        "fcc_coke": -0.001,
    }.items():
        deltas[p]["api"] = coef
    for p, coef in {
        "fcc_coke": 0.012,
        "fcc_slurry": 0.015,
        "fcc_naphtha": -0.020,
        "fcc_lpg": -0.005,
        "fcc_lco": 0.002,
        "fcc_dry_gas": 0.001,
    }.items():
        deltas[p]["ccr_wt"] = coef
    for p, coef in {
        "fcc_naphtha": -0.01,
        "fcc_coke": 0.002,
        "fcc_slurry": 0.005,
        "fcc_lpg": 0.0,
        "fcc_lco": 0.003,
        "fcc_dry_gas": 0.0,
    }.items():
        deltas[p]["sulfur_wt"] = coef

    # Process conditions — primary optimization handles
    for p, coef in {
        "fcc_naphtha": 0.00045,
        "fcc_lpg": 0.00025,
        "fcc_dry_gas": 0.00012,
        "fcc_lco": -0.00030,
        "fcc_slurry": -0.00035,
        "fcc_coke": -0.00005,
    }.items():
        deltas[p]["riser_outlet_temp_f"] = coef
    for p, coef in {
        "fcc_naphtha": 0.012,
        "fcc_lpg": 0.008,
        "fcc_dry_gas": 0.004,
        "fcc_lco": -0.010,
        "fcc_slurry": -0.012,
        "fcc_coke": 0.002,
    }.items():
        deltas[p]["catalyst_to_oil"] = coef
    for p, coef in {
        "fcc_naphtha": 0.0012,
        "fcc_lpg": 0.0006,
        "fcc_dry_gas": 0.0002,
        "fcc_lco": -0.0008,
        "fcc_slurry": -0.0010,
        "fcc_coke": -0.0001,
    }.items():
        deltas[p]["catalyst_activity"] = coef
    for p, coef in {
        "fcc_naphtha": 0.00006,
        "fcc_lpg": 0.00003,
        "fcc_dry_gas": 0.00001,
        "fcc_lco": -0.00004,
        "fcc_slurry": -0.00005,
        "fcc_coke": -0.00001,
    }.items():
        deltas[p]["feed_preheat_temp_f"] = coef
    for p, coef in {
        "fcc_dry_gas": 0.02,
        "fcc_lpg": 0.03,
        "fcc_naphtha": 0.02,
        "fcc_lco": -0.04,
        "fcc_slurry": -0.03,
        "fcc_coke": 0.01,
    }.items():
        deltas[p]["recycle_ratio"] = coef

    comps = {p: get_stream(p).to_dict() for p in FCC_PRODUCTS}
    drivers = [
        "api",
        "sulfur_wt",
        "ccr_wt",
        "riser_outlet_temp_f",
        "catalyst_to_oil",
        "catalyst_activity",
        "feed_preheat_temp_f",
        "recycle_ratio",
    ]

    return _FCCModel(
        unit="FCC",
        products=list(FCC_PRODUCTS),
        base_yields=base,
        deltas=deltas,
        reference_feed=ref_feed,
        reference_conditions=ref_cond,
        exits=list(FCC_EXITS),
        drivers=drivers,
        product_compositions=comps,
        notes=[
            "Liquids renormalized; fcc_coke is wt fraction (exit REGEN_HEAT).",
            "ROT / C/O / activity / preheat / recycle are optimizable process drivers.",
            "Use process_modes() for SOS1 LP embedding of severity bands.",
        ],
    )


class _FCCModel(BaseDeltaModel):
    def evaluate(
        self,
        feed: Optional[Mapping[str, float]] = None,
        conditions: Optional[Mapping[str, Any]] = None,
        clamp_products: bool = True,
    ) -> Dict[str, float]:
        feed_m = dict(self.reference_feed if feed is None else feed)
        cond = merge_process_conditions(self.unit, conditions)
        flat = dict(feed_m)
        for k, v in cond.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)
        ref_flat = dict(self.reference_feed)
        for k, v in self.reference_conditions.items():
            if isinstance(v, (int, float)):
                ref_flat[k] = float(v)

        y: Dict[str, float] = {}
        for p in self.products:
            base = float(self.base_yields.get(p, 0.0))
            dy = 0.0
            for drv, coef in self.deltas.get(p, {}).items():
                x0 = float(ref_flat.get(drv, 0.0))
                xv = float(flat.get(drv, x0))
                dy += float(coef) * (xv - x0)
            val = base + dy
            if clamp_products:
                val = max(0.0, val)
            y[p] = val
        return self._postprocess_yields(y, cond)

    def _postprocess_yields(
        self, y: Dict[str, float], cond: Mapping[str, Any]
    ) -> Dict[str, float]:
        coke = _clamp(y.get("fcc_coke", 0.05), 0.03, 0.12)
        liquids = ["fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry"]
        liquid_vol = _clamp(0.96 - 0.55 * coke, 0.88, 0.96)
        s = sum(max(0.0, y[p]) for p in liquids)
        if s > 1e-12:
            sc = liquid_vol / s
            for p in liquids:
                y[p] = max(0.0, y[p]) * sc
        y["fcc_coke"] = coke
        return y

    def _adjust_composition(
        self,
        product: str,
        comp: StreamComposition,
        feed: Mapping[str, float],
        cond: Mapping[str, Any],
    ) -> StreamComposition:
        # Mild severity effect on FCC naphtha RON / olefins
        if product == "fcc_naphtha":
            rot = float(cond.get("riser_outlet_temp_f", 980.0))
            d = (rot - 980.0) / 100.0
            comp.ron = _clamp(comp.ron + 1.5 * d, 88.0, 96.0)
            comp.olefins_vol = _clamp(comp.olefins_vol + 0.03 * d, 0.15, 0.45)
            comp.sulfur_wt = max(0.001, float(feed.get("sulfur_wt", 0.45)) * 0.12)
        if product in ("fcc_lco", "fcc_slurry"):
            comp.sulfur_wt = float(feed.get("sulfur_wt", 0.45)) * (0.8 if product == "fcc_lco" else 1.5)
            comp.ccr_wt = float(feed.get("ccr_wt", 0.4)) * (0.5 if product == "fcc_lco" else 2.0)
        return comp


def process_modes_fcc(
    model: Optional[BaseDeltaModel] = None,
    feed: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """SOS1 process modes for FCC — each mode is a full yield vector at fixed conditions.

    Modes vary riser_outlet_temp_f (and mild C/O couple) so the unit-block LP can
    optimize process severity with binary mode selection.
    """
    m = model or build_fcc_base_delta()
    feed = dict(m.reference_feed if feed is None else feed)
    modes = []
    specs = [
        ("rot_low", 940.0, 5.5),
        ("rot_mid", 980.0, 6.5),
        ("rot_high", 1020.0, 7.5),
    ]
    for mid, rot, co in specs:
        cond = {
            "riser_outlet_temp_f": rot,
            "catalyst_to_oil": co,
            "catalyst_activity": float(m.reference_conditions.get("catalyst_activity", 68.0)),
            "feed_preheat_temp_f": float(m.reference_conditions.get("feed_preheat_temp_f", 550.0)),
            "recycle_ratio": float(m.reference_conditions.get("recycle_ratio", 0.0)),
        }
        y = m.evaluate(feed=feed, conditions=cond)
        comps = {k: v.to_dict() for k, v in m.compositions_at(feed=feed, conditions=cond).items()}
        modes.append(
            {
                "id": mid,
                "unit": "FCC",
                "conditions": cond,
                "yields": y,
                "compositions": comps,
                "exits": [e.__dict__ for e in m.exits],
            }
        )
    return modes


def process_modes_cdu(
    model: Optional[BaseDeltaModel] = None,
    feed: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """CDU cut-point / flash modes for unit-block optimization."""
    m = model or build_cdu_base_delta()
    feed = dict(m.reference_feed if feed is None else feed)
    modes = []
    specs = [
        (
            "cuts_light",
            {"flash_zone_temp_f": 700.0, "cut_points_f": {"naphtha_ep": 380.0, "distillate_ep": 670.0, "gasoil_ep": 1070.0}},
        ),
        (
            "cuts_mid",
            {"flash_zone_temp_f": 680.0, "cut_points_f": {"naphtha_ep": 350.0, "distillate_ep": 650.0, "gasoil_ep": 1050.0}},
        ),
        (
            "cuts_heavy",
            {"flash_zone_temp_f": 660.0, "cut_points_f": {"naphtha_ep": 320.0, "distillate_ep": 630.0, "gasoil_ep": 1020.0}},
        ),
    ]
    for mid, cond in specs:
        y = m.evaluate(feed=feed, conditions=cond)
        comps = {k: v.to_dict() for k, v in m.compositions_at(feed=feed, conditions=cond).items()}
        modes.append(
            {
                "id": mid,
                "unit": "CDU",
                "conditions": cond,
                "yields": y,
                "compositions": comps,
                "exits": [e.__dict__ for e in m.exits],
            }
        )
    return modes


def assert_every_product_has_exit(model: BaseDeltaModel) -> None:
    missing = model.missing_exits()
    if missing:
        raise ValueError(f"{model.unit}: products without exit: {missing}")


def unit_submodels_cdu_fcc(
    crude_feed: Optional[Mapping[str, float]] = None,
    gasoil_feed: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience: build both submodels + modes + validate exits."""
    cdu = build_cdu_base_delta(reference_feed=crude_feed)
    # Gasoil feed for FCC from CDU gasoil composition defaults
    go = get_stream("cdu_gasoil")
    go_feed = {
        "api": go.api,
        "sulfur_wt": go.sulfur_wt,
        "ccr_wt": go.ccr_wt,
        "nitrogen_ppm": go.nitrogen_ppm,
        "metals_ni_v_ppm": go.metals_ni_v_ppm,
    }
    if gasoil_feed:
        go_feed.update({k: float(v) for k, v in gasoil_feed.items() if isinstance(v, (int, float))})
    fcc = build_fcc_base_delta(reference_feed=go_feed)
    assert_every_product_has_exit(cdu)
    assert_every_product_has_exit(fcc)
    return {
        "CDU": cdu.to_dict(),
        "FCC": fcc.to_dict(),
        "CDU_modes": process_modes_cdu(cdu, crude_feed),
        "FCC_modes": process_modes_fcc(fcc, go_feed),
        "models": {"CDU": cdu, "FCC": fcc},
    }
