"""Assay import → heart/swing cuts → CDU fractionation LP.

Industry planning pattern (PIMS/Aspen-style):
  1. Import assay as ordered narrow TBP cuts (volume + properties).
  2. Split the library into **heart cuts** (fixed to one product) and
     **swing cuts** (shared on a product boundary; allocation is LP vars).
  3. Linear combination of hearts + swung material matches effective cut points.
  4. Product properties = volume-weighted blend of contributing cuts.
  5. Unit mass balance: sum(product vol) = charge (within assay yield coverage).

Swing formulation on boundary between products A and B for swing volume V_s:
  v_to_A + v_to_B = V_s * charge
  0 ≤ v_to_A, v_to_B
Cut-point severity is represented by the free allocation (or fixed via
process cut-point targets).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pulp

from .assay_loader import default_assays_path, load_assays_json, load_json
from .stream_composition import StreamComposition, api_to_sg


PathLike = Union[str, Path]

# Default product stack (light → heavy)
DEFAULT_PRODUCTS = (
    "cdu_naphtha",
    "cdu_distillate",
    "cdu_gasoil",
    "cdu_resid",
)

# Boundary names between consecutive products
DEFAULT_BOUNDARIES = (
    ("cdu_naphtha", "cdu_distillate"),
    ("cdu_distillate", "cdu_gasoil"),
    ("cdu_gasoil", "cdu_resid"),
)

# Planning TBP edges (°C) matching product_map midpoints
DEFAULT_EDGES_C = {
    "cdu_naphtha": (5.0, 200.0),
    "cdu_distillate": (200.0, 370.0),
    "cdu_gasoil": (370.0, 550.0),
    "cdu_resid": (550.0, 750.0),
}


@dataclass
class AssayCut:
    """Single narrow assay cut (heart or raw)."""

    id: str
    tbp_start_c: float
    tbp_end_c: float
    yield_vol: float
    yield_wt: float = 0.0
    api: float = 30.0
    density_15c_g_cc: float = 0.876
    sulfur_wt: float = 0.5
    ccr_wt: float = 0.0
    nitrogen_ppm: float = 0.0
    ron: float = 0.0
    paraffins_vol: float = 0.33
    naphthenes_vol: float = 0.33
    aromatics_vol: float = 0.34
    nickel_ppm: float = 0.0
    vanadium_ppm: float = 0.0
    asphaltenes_wt: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)

    @property
    def tbp_mid_c(self) -> float:
        return 0.5 * (self.tbp_start_c + self.tbp_end_c)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_composition(self, name: Optional[str] = None) -> StreamComposition:
        return StreamComposition(
            name=name or self.id,
            api=self.api,
            specific_gravity=self.density_15c_g_cc or api_to_sg(self.api),
            sulfur_wt=self.sulfur_wt,
            ccr_wt=self.ccr_wt,
            nitrogen_ppm=self.nitrogen_ppm,
            ron=self.ron,
            paraffins_vol=self.paraffins_vol,
            naphthenes_vol=self.naphthenes_vol,
            aromatics_vol=self.aromatics_vol,
            metals_ni_v_ppm=self.nickel_ppm + self.vanadium_ppm,
            tbp_10_f=self.tbp_start_c * 1.8 + 32.0,
            tbp_50_f=self.tbp_mid_c * 1.8 + 32.0,
            tbp_90_f=self.tbp_end_c * 1.8 + 32.0,
        )


@dataclass
class HeartCut:
    product: str
    cut: AssayCut

    def to_dict(self) -> Dict[str, Any]:
        return {"product": self.product, "cut": self.cut.to_dict()}


@dataclass
class SwingCut:
    """Boundary swing volume shared by light_product and heavy_product."""

    id: str
    light_product: str
    heavy_product: str
    cut: AssayCut

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "light_product": self.light_product,
            "heavy_product": self.heavy_product,
            "cut": self.cut.to_dict(),
        }


@dataclass
class AssayPackage:
    name: str
    reference: str
    whole_crude: Dict[str, Any]
    cuts: List[AssayCut]
    products: Tuple[str, ...] = DEFAULT_PRODUCTS
    edges_c: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_EDGES_C)
    )
    source_path: str = ""

    def total_vol(self) -> float:
        return sum(c.yield_vol for c in self.cuts)

    def normalize_vol(self) -> "AssayPackage":
        """Return copy with yield_vol renormalized to sum 1.0."""
        pkg = deepcopy(self)
        s = pkg.total_vol() or 1.0
        for c in pkg.cuts:
            c.yield_vol = c.yield_vol / s
            if c.yield_wt:
                # keep wt consistent scale if present
                pass
        wt = sum(c.yield_wt for c in pkg.cuts)
        if wt > 1e-12:
            for c in pkg.cuts:
                c.yield_wt = c.yield_wt / wt
        return pkg


def _cut_from_mapping(row: Mapping[str, Any], idx: int = 0) -> AssayCut:
    api = float(row.get("api", 30.0))
    dens = float(row.get("density_15c_g_cc") or api_to_sg(api))
    known = {
        "id",
        "tbp_start_c",
        "tbp_end_c",
        "yield_vol",
        "yield_wt",
        "api",
        "density_15c_g_cc",
        "sulfur_wt",
        "ccr_wt",
        "nitrogen_ppm",
        "ron",
        "paraffins_vol",
        "naphthenes_vol",
        "aromatics_vol",
        "nickel_ppm",
        "vanadium_ppm",
        "asphaltenes_wt",
    }
    extras = {}
    for k, v in row.items():
        if k not in known:
            try:
                extras[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return AssayCut(
        id=str(row.get("id", f"cut_{idx}")),
        tbp_start_c=float(row.get("tbp_start_c", 0.0)),
        tbp_end_c=float(row.get("tbp_end_c", 100.0)),
        yield_vol=float(row.get("yield_vol", 0.0)),
        yield_wt=float(row.get("yield_wt", 0.0)),
        api=api,
        density_15c_g_cc=dens,
        sulfur_wt=float(row.get("sulfur_wt", 0.5)),
        ccr_wt=float(row.get("ccr_wt", 0.0)),
        nitrogen_ppm=float(row.get("nitrogen_ppm", 0.0)),
        ron=float(row.get("ron", 0.0)),
        paraffins_vol=float(row.get("paraffins_vol", 0.33)),
        naphthenes_vol=float(row.get("naphthenes_vol", 0.33)),
        aromatics_vol=float(row.get("aromatics_vol", 0.34)),
        nickel_ppm=float(row.get("nickel_ppm", 0.0)),
        vanadium_ppm=float(row.get("vanadium_ppm", 0.0)),
        asphaltenes_wt=float(row.get("asphaltenes_wt", 0.0)),
        extras=extras,
    )


def import_detailed_assay_json(path: PathLike) -> AssayPackage:
    """Import EMTEC-style detailed cut table JSON."""
    path = Path(path)
    raw = load_json(path)
    meta = raw.get("meta") or {}
    whole = raw.get("whole_crude") or {}
    cuts = [_cut_from_mapping(r, i) for i, r in enumerate(raw.get("cuts") or [])]
    edges = dict(DEFAULT_EDGES_C)
    pmap = raw.get("product_map") or {}
    for prod, rng in pmap.items():
        edges[prod] = (float(rng["tbp_lo_c"]), float(rng["tbp_hi_c"]))
    dcp = raw.get("default_cut_points_c") or {}
    if dcp:
        # rebuild edges from cut points
        nep = float(dcp.get("naphtha_ep", 200.0))
        dep = float(dcp.get("distillate_ep", 370.0))
        gep = float(dcp.get("gasoil_ep", 550.0))
        edges = {
            "cdu_naphtha": (5.0, nep),
            "cdu_distillate": (nep, dep),
            "cdu_gasoil": (dep, gep),
            "cdu_resid": (gep, 750.0),
        }
    return AssayPackage(
        name=str(meta.get("name") or whole.get("name") or path.stem),
        reference=str(meta.get("reference", "")),
        whole_crude=dict(whole),
        cuts=cuts,
        edges_c=edges,
        source_path=str(path),
    ).normalize_vol()


def import_crude_from_assays_package(
    crude_name: str,
    assays_path: PathLike | None = None,
    *,
    detailed_override_path: PathLike | None = None,
) -> AssayPackage:
    """Import named crude from data/assays/crudes.json.

    If a detailed cut table exists (e.g. cold_lake_blend_clkbl23b.json) or
    ``detailed_override_path`` is set, use that. Else synthesize 4 heart + 3
    swing cuts from ``tbp_cut_vol``.
    """
    if detailed_override_path:
        return import_detailed_assay_json(detailed_override_path)

    # Known detailed libraries
    name_key = crude_name.replace(" ", "_")
    candidates = [
        Path("data/assays/cold_lake_blend_clkbl23b.json"),
        Path(__file__).resolve().parents[3] / "data" / "assays" / "cold_lake_blend_clkbl23b.json",
    ]
    if "cold_lake" in name_key.lower() or name_key.lower() == "cold_lake_blend":
        for p in candidates:
            if p.is_file():
                return import_detailed_assay_json(p)

    pkg = load_assays_json(assays_path)
    crude = None
    for c in pkg.get("crudes") or []:
        if str(c.get("name", "")).lower() == crude_name.lower():
            crude = c
            break
    if crude is None:
        raise KeyError(f"crude {crude_name!r} not found in assays package")

    # Prefer detailed_cuts key if present
    if crude.get("cuts"):
        return AssayPackage(
            name=str(crude["name"]),
            reference=str(crude.get("reference", "")),
            whole_crude={k: v for k, v in crude.items() if k not in ("cuts", "tbp_cut_vol")},
            cuts=[_cut_from_mapping(r, i) for i, r in enumerate(crude["cuts"])],
            source_path=str(assays_path or default_assays_path()),
        ).normalize_vol()

    return synthesize_from_tbp_cut_vol(crude)


def synthesize_from_tbp_cut_vol(crude: Mapping[str, Any]) -> AssayPackage:
    """Build 4 hearts + 3 boundary swings from coarse TBP vol fractions.

    Each product band: 70% heart + 15% light-side swing + 15% heavy-side swing
    (ends: only one swing). Properties inherit whole crude with mild TBP skew.
    """
    tbp = crude.get("tbp_cut_vol") or {}
    # Map keys
    y_n = float(tbp.get("naphtha_ibp_350f", tbp.get("naphtha", 0.2)))
    y_d = float(tbp.get("distillate_350_650f", tbp.get("distillate", 0.25)))
    y_g = float(tbp.get("gasoil_650_1050f", tbp.get("gasoil", 0.28)))
    y_r = float(tbp.get("resid_1050f_plus", tbp.get("residue", 0.27)))
    s = y_n + y_d + y_g + y_r or 1.0
    y_n, y_d, y_g, y_r = y_n / s, y_d / s, y_g / s, y_r / s

    api = float(crude.get("api", 30.0))
    sul = float(crude.get("sulfur_wt", crude.get("sulfur_wt_pct", 1.0)))
    ccr = float(crude.get("ccr_wt", 2.0))

    def make_cut(cid: str, lo: float, hi: float, yv: float, api_off: float, s_fac: float, ccr_fac: float) -> AssayCut:
        return AssayCut(
            id=cid,
            tbp_start_c=lo,
            tbp_end_c=hi,
            yield_vol=yv,
            yield_wt=yv,
            api=max(0.5, api + api_off),
            density_15c_g_cc=api_to_sg(max(0.5, api + api_off)),
            sulfur_wt=max(0.001, sul * s_fac),
            ccr_wt=max(0.0, ccr * ccr_fac),
            nitrogen_ppm=float(crude.get("nitrogen_ppm", 1000.0)) * s_fac,
            paraffins_vol=float(crude.get("paraffins_vol", 0.33)),
            naphthenes_vol=float(crude.get("naphthenes_vol", 0.33)),
            aromatics_vol=float(crude.get("aromatics_vol", 0.34)),
        )

    # Split each band: heart 0.70, swing-out 0.15 each side (edges 0.85 heart + 0.15 swing)
    cuts: List[AssayCut] = []
    # naphtha
    cuts.append(make_cut("heart_naph", 5, 170, y_n * 0.70, +25, 0.05, 0.0))
    cuts.append(make_cut("swing_naph_dist", 170, 200, y_n * 0.15 + y_d * 0.15, +10, 0.15, 0.0))
    # distillate heart uses remaining after contributing to swings
    cuts.append(make_cut("heart_dist", 200, 340, y_d * 0.70, 0.0, 0.4, 0.05))
    cuts.append(make_cut("swing_dist_go", 340, 370, y_d * 0.15 + y_g * 0.15, -5, 0.7, 0.1))
    cuts.append(make_cut("heart_go", 370, 520, y_g * 0.70, -12, 1.0, 0.25))
    cuts.append(make_cut("swing_go_resid", 520, 550, y_g * 0.15 + y_r * 0.15, -18, 1.3, 0.8))
    cuts.append(make_cut("heart_resid", 550, 750, y_r * 0.70, -25, 1.6, 2.0))

    # renorm (construction may not sum exactly to 1)
    s2 = sum(c.yield_vol for c in cuts) or 1.0
    for c in cuts:
        c.yield_vol /= s2
        c.yield_wt = c.yield_vol

    return AssayPackage(
        name=str(crude.get("name", "crude")),
        reference=str(crude.get("reference", "")),
        whole_crude=dict(crude),
        cuts=cuts,
        source_path="synthesized_from_tbp_cut_vol",
    )


def product_for_midpoint(
    mid_c: float,
    edges: Mapping[str, Tuple[float, float]],
) -> Optional[str]:
    for prod, (lo, hi) in edges.items():
        if lo <= mid_c < hi or (hi >= 700 and mid_c >= lo):
            return prod
    # clamp
    if mid_c < 100:
        return "cdu_naphtha"
    return "cdu_resid"


def build_heart_swing_library(
    assay: AssayPackage,
    *,
    swing_half_width_c: float = 15.0,
) -> Tuple[List[HeartCut], List[SwingCut]]:
    """Classify assay cuts into hearts and boundary swings.

    A cut is a **swing** if its midpoint lies within ``swing_half_width_c`` of a
    product boundary. Otherwise it is a **heart** fixed to the product whose
    TBP window contains the midpoint.
    """
    edges = assay.edges_c
    # ordered boundaries (temp, light, heavy)
    bounds: List[Tuple[float, str, str]] = []
    ordered = list(DEFAULT_PRODUCTS)
    for i in range(len(ordered) - 1):
        light, heavy = ordered[i], ordered[i + 1]
        # boundary temp = light hi = heavy lo
        btemp = edges[light][1]
        bounds.append((btemp, light, heavy))

    hearts: List[HeartCut] = []
    swings: List[SwingCut] = []

    for cut in assay.cuts:
        mid = cut.tbp_mid_c
        # nearest boundary?
        nearest = None
        nearest_dist = 1e9
        for btemp, light, heavy in bounds:
            d = abs(mid - btemp)
            if d < nearest_dist:
                nearest_dist = d
                nearest = (btemp, light, heavy)
        if nearest is not None and nearest_dist <= swing_half_width_c:
            _, light, heavy = nearest
            swings.append(
                SwingCut(
                    id=f"swing_{cut.id}",
                    light_product=light,
                    heavy_product=heavy,
                    cut=cut,
                )
            )
        else:
            prod = product_for_midpoint(mid, edges) or "cdu_resid"
            hearts.append(HeartCut(product=prod, cut=cut))

    return hearts, swings


def _blend_props(parts: List[Tuple[float, AssayCut]]) -> Dict[str, float]:
    """Volume-weighted property blend of (vol_flow, cut) pairs."""
    tot = sum(max(0.0, v) for v, _ in parts)
    if tot < 1e-12:
        return {
            "api": 0.0,
            "sulfur_wt": 0.0,
            "ccr_wt": 0.0,
            "nitrogen_ppm": 0.0,
            "ron": 0.0,
            "density_15c_g_cc": 0.0,
            "metals_ni_v_ppm": 0.0,
        }
    def wavg(attr: str) -> float:
        return sum(max(0.0, v) * float(getattr(c, attr)) for v, c in parts) / tot

    return {
        "api": wavg("api"),
        "sulfur_wt": wavg("sulfur_wt"),
        "ccr_wt": wavg("ccr_wt"),
        "nitrogen_ppm": wavg("nitrogen_ppm"),
        "ron": wavg("ron"),
        "density_15c_g_cc": wavg("density_15c_g_cc"),
        "metals_ni_v_ppm": sum(
            max(0.0, v) * (c.nickel_ppm + c.vanadium_ppm) for v, c in parts
        )
        / tot,
        "paraffins_vol": wavg("paraffins_vol"),
        "naphthenes_vol": wavg("naphthenes_vol"),
        "aromatics_vol": wavg("aromatics_vol"),
    }


@dataclass
class CduSwingResult:
    status: str
    charge_kbd: float
    product_yields_vol: Dict[str, float]  # frac of charge
    product_rates_kbd: Dict[str, float]
    product_properties: Dict[str, Dict[str, float]]
    swing_allocations: Dict[str, Dict[str, Any]]  # swing_id → allocation detail
    hearts: List[Dict[str, Any]]
    swings: List[Dict[str, Any]]
    mass_balance: Dict[str, Any]
    assay_name: str
    cut_points_c: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "charge_kbd": self.charge_kbd,
            "product_yields_vol": dict(self.product_yields_vol),
            "product_rates_kbd": dict(self.product_rates_kbd),
            "product_properties": self.product_properties,
            "swing_allocations": self.swing_allocations,
            "hearts": self.hearts,
            "swings": self.swings,
            "mass_balance": self.mass_balance,
            "assay_name": self.assay_name,
            "cut_points_c": self.cut_points_c,
            "meta": self.meta,
        }


def solve_cdu_swing_cuts(
    assay: AssayPackage,
    *,
    charge_kbd: float = 100.0,
    # Prefer light allocation on swing (0=all heavy, 1=all light)
    swing_light_frac: Optional[Mapping[str, float]] = None,
    # Or free optimize economics
    optimize: bool = True,
    prices: Optional[Mapping[str, float]] = None,
    # Soft targets for product sulfur (optional)
    max_sulfur: Optional[Mapping[str, float]] = None,
    msg: bool = False,
) -> CduSwingResult:
    """Fractionate assay with heart/swing LP; close mass balance + blend props."""
    assay = assay.normalize_vol()
    hearts, swings = build_heart_swing_library(assay)
    products = list(assay.products)
    price = {
        "cdu_naphtha": 90.0,
        "cdu_distillate": 95.0,
        "cdu_gasoil": 75.0,
        "cdu_resid": 45.0,
    }
    if prices:
        price.update(prices)

    Q = float(charge_kbd)
    prob = pulp.LpProblem("cdu_heart_swing", pulp.LpMaximize if optimize else pulp.LpMinimize)

    # Heart contributions fixed
    heart_rate: Dict[str, float] = {p: 0.0 for p in products}
    heart_parts: Dict[str, List[Tuple[float, AssayCut]]] = {p: [] for p in products}
    for h in hearts:
        rate = h.cut.yield_vol * Q
        heart_rate[h.product] = heart_rate.get(h.product, 0.0) + rate
        heart_parts[h.product].append((rate, h.cut))

    # Swing vars: to_light, to_heavy
    swing_light_vars: Dict[str, pulp.LpVariable] = {}
    swing_heavy_vars: Dict[str, pulp.LpVariable] = {}
    for s in swings:
        total = s.cut.yield_vol * Q
        vl = pulp.LpVariable(f"sw_L_{s.id}", lowBound=0, upBound=total)
        vh = pulp.LpVariable(f"sw_H_{s.id}", lowBound=0, upBound=total)
        prob += vl + vh == total, f"swing_bal_{s.id}"
        swing_light_vars[s.id] = vl
        swing_heavy_vars[s.id] = vh
        if swing_light_frac and s.id in swing_light_frac:
            # fix allocation
            frac = max(0.0, min(1.0, float(swing_light_frac[s.id])))
            prob += vl == frac * total
        elif swing_light_frac is not None and not optimize:
            # default 50/50 when not optimizing and no map
            pass

    # Product totals
    prod_rate: Dict[str, pulp.LpAffineExpression] = {}
    for p in products:
        terms = [heart_rate.get(p, 0.0)]
        for s in swings:
            if s.light_product == p:
                terms.append(swing_light_vars[s.id])
            if s.heavy_product == p:
                terms.append(swing_heavy_vars[s.id])
        prod_rate[p] = pulp.lpSum(terms)

    # Mass balance: sum products = sum cuts * Q
    total_cut = assay.total_vol() * Q
    prob += pulp.lpSum(prod_rate[p] for p in products) == total_cut, "unit_mass_balance"

    # Objective: maximize product value (or min 0 dummy)
    if optimize:
        prob += pulp.lpSum(price.get(p, 50.0) * prod_rate[p] for p in products)
    else:
        # fix swings 50/50 if not specified
        for s in swings:
            if not (swing_light_frac and s.id in swing_light_frac):
                tot = s.cut.yield_vol * Q
                prob += swing_light_vars[s.id] == 0.5 * tot
        prob += 0.0

    # Optional property soft constraints are post-check only (bilinear if hard on blend)
    # Skip hard property constraints in LP (linear mass only); report properties after.

    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus.get(status_code, str(status_code))

    def _val(x: Any) -> float:
        v = pulp.value(x)
        return float(v) if v is not None else 0.0

    rates = {p: _val(prod_rate[p]) for p in products}
    yields = {p: (rates[p] / Q if Q > 1e-12 else 0.0) for p in products}

    # Reconstruct property blend parts with solved swings
    swing_alloc: Dict[str, Dict[str, float]] = {}
    prop_parts: Dict[str, List[Tuple[float, AssayCut]]] = {p: list(heart_parts[p]) for p in products}
    for s in swings:
        vl = _val(swing_light_vars[s.id])
        vh = _val(swing_heavy_vars[s.id])
        swing_alloc[s.id] = {
            "light_product": s.light_product,
            "heavy_product": s.heavy_product,
            "to_light_kbd": vl,
            "to_heavy_kbd": vh,
            "light_frac": vl / (vl + vh) if (vl + vh) > 1e-12 else 0.5,
            "cut_id": s.cut.id,
        }
        if vl > 1e-12:
            prop_parts[s.light_product].append((vl, s.cut))
        if vh > 1e-12:
            prop_parts[s.heavy_product].append((vh, s.cut))

    product_properties = {p: _blend_props(prop_parts[p]) for p in products}

    sum_rates = sum(rates.values())
    mb_gap = abs(sum_rates - total_cut)
    # Property mass check: sulfur mass in ≈ sulfur mass out
    s_in = float(assay.whole_crude.get("sulfur_wt", 0.0)) * Q  # wt% * kbd approx
    # use vol * sulfur as planning proxy
    s_out = sum(rates[p] * product_properties[p].get("sulfur_wt", 0.0) for p in products)
    # Better: cut-weighted
    s_in_cuts = sum(c.yield_vol * Q * c.sulfur_wt for c in assay.cuts)
    s_out_cuts = s_out
    s_gap = abs(s_in_cuts - s_out_cuts)

    cut_points = {
        "naphtha_ep_c": assay.edges_c["cdu_naphtha"][1],
        "distillate_ep_c": assay.edges_c["cdu_distillate"][1],
        "gasoil_ep_c": assay.edges_c["cdu_gasoil"][1],
    }

    mb = {
        "ok": mb_gap < 1e-3 * max(1.0, Q) + 1e-6 and s_gap < 0.05 * max(1.0, s_in_cuts) + 1e-3,
        "charge_kbd": Q,
        "sum_products_kbd": sum_rates,
        "assay_vol_coverage": assay.total_vol(),
        "vol_balance_gap": mb_gap,
        "sulfur_mass_in": s_in_cuts,
        "sulfur_mass_out": s_out_cuts,
        "sulfur_balance_gap": s_gap,
        "heart_count": len(hearts),
        "swing_count": len(swings),
    }

    return CduSwingResult(
        status=status,
        charge_kbd=Q,
        product_yields_vol=yields,
        product_rates_kbd=rates,
        product_properties=product_properties,
        swing_allocations=swing_alloc,
        hearts=[h.to_dict() for h in hearts],
        swings=[s.to_dict() for s in swings],
        mass_balance=mb,
        assay_name=assay.name,
        cut_points_c=cut_points,
        meta={
            "reference": assay.reference,
            "source_path": assay.source_path,
            "whole_crude": assay.whole_crude,
            "optimize": optimize,
            "model": "cdu_heart_swing",
        },
    )


def cdu_yields_and_props_from_assay(
    crude_name: str,
    *,
    charge_kbd: float = 100.0,
    optimize: bool = True,
) -> Dict[str, Any]:
    """Convenience: import crude → swing CDU → yields + properties dict."""
    assay = import_crude_from_assays_package(crude_name)
    res = solve_cdu_swing_cuts(assay, charge_kbd=charge_kbd, optimize=optimize)
    return res.to_dict()


def list_importable_assays(assays_path: PathLike | None = None) -> List[str]:
    pkg = load_assays_json(assays_path)
    names = [str(c.get("name")) for c in pkg.get("crudes") or []]
    # ensure Cold Lake detailed present
    if not any("cold_lake" in n.lower() for n in names):
        names.append("Cold_Lake_Blend")
    return names
