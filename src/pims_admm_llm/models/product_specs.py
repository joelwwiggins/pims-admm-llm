"""Product quality specs + blender purchase intermediates (W2 crude→cat→blender).

Owns finished-product gates for:
  - gasoline (multi-component blender: RON min, S max; optional RVP report)
  - sweet_gasoil / sour_gasoil (single-stream S/API/cetane gates + reclass)
  - naphtha intermediate sell
and make-buy purchase streams (naphtha / alkylate / reformate) with qualities.

Does not own tanks, H2/BTU utilities, or ADMM coordination (sibling waves).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None  # type: ignore

from .quality_blender import (
    GasolineQualityConfig,
    QualityComponent,
    QualityConstraintMeta,
    add_gasoline_quality_constraints,
    blend_quality_closed_form,
    component_deltas,
    resolve_base,
)


# ---------------------------------------------------------------------------
# Paths / load
# ---------------------------------------------------------------------------


def _repo_roots() -> List[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[3],
        here.parents[2],
        Path.cwd(),
        Path("/home/joel/projects/pims-admm-llm"),
    ]


def default_product_specs_path() -> Path:
    name = Path("data") / "product_specs.json"
    for root in _repo_roots():
        cand = root / name
        if cand.is_file():
            return cand
    return _repo_roots()[0] / name


def load_product_specs(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else default_product_specs_path()
    with p.open() as f:
        data = json.load(f)
    if "products" not in data:
        raise ValueError(f"product_specs missing 'products': {p}")
    return data


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecLimits:
    """Numeric limits for one product. Absent keys are None (not enforced)."""

    min_ron: Optional[float] = None
    max_ron: Optional[float] = None
    max_sulfur_wt: Optional[float] = None
    min_sulfur_wt: Optional[float] = None
    min_api: Optional[float] = None
    max_api: Optional[float] = None
    min_cetane: Optional[float] = None
    max_rvp_psi: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SpecLimits":
        d = dict(d or {})
        def _f(k: str) -> Optional[float]:
            if k not in d or d[k] is None:
                return None
            return float(d[k])

        return cls(
            min_ron=_f("min_ron"),
            max_ron=_f("max_ron"),
            max_sulfur_wt=_f("max_sulfur_wt"),
            min_sulfur_wt=_f("min_sulfur_wt"),
            min_api=_f("min_api"),
            max_api=_f("max_api"),
            min_cetane=_f("min_cetane"),
            max_rvp_psi=_f("max_rvp_psi"),
        )

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in (
            "min_ron",
            "max_ron",
            "max_sulfur_wt",
            "min_sulfur_wt",
            "min_api",
            "max_api",
            "min_cetane",
            "max_rvp_psi",
        ):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


@dataclass
class FinishedProduct:
    name: str
    price_usd_per_bbl: float
    specs: SpecLimits
    max_demand_kbd: Optional[float] = None
    min_demand_kbd: float = 0.0
    blend_model: Optional[str] = None
    base_stream: Optional[str] = None
    components: List[str] = field(default_factory=list)
    source_streams: List[str] = field(default_factory=list)
    fallback_product: Optional[str] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "price_usd_per_bbl": self.price_usd_per_bbl,
            "max_demand_kbd": self.max_demand_kbd,
            "min_demand_kbd": self.min_demand_kbd,
            "specs": self.specs.as_dict(),
            "blend_model": self.blend_model,
            "base_stream": self.base_stream,
            "components": list(self.components),
            "source_streams": list(self.source_streams),
            "fallback_product": self.fallback_product,
            "notes": self.notes,
        }


@dataclass
class PurchaseIntermediate:
    name: str
    price_usd_per_bbl: float
    max_kbd: float
    properties: Dict[str, float]
    role: str = "gasoline_blendstock"
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "price_usd_per_bbl": self.price_usd_per_bbl,
            "max_kbd": self.max_kbd,
            "properties": dict(self.properties),
            "role": self.role,
            "notes": self.notes,
        }

    def quality_component(self) -> QualityComponent:
        return QualityComponent(
            stream=self.name,
            ron=float(self.properties.get("ron", 0.0)),
            sulfur_wt=float(self.properties.get("sulfur_wt", 0.0)),
        )


@dataclass
class ProductSpecCatalog:
    products: Dict[str, FinishedProduct]
    purchases: Dict[str, PurchaseIntermediate]
    component_properties: Dict[str, Dict[str, float]]
    meta: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def prices(self) -> Dict[str, float]:
        """Flat price map for plant objective (product + purchase keys)."""
        out: Dict[str, float] = {}
        for p in self.products.values():
            out[p.name] = p.price_usd_per_bbl
        for b in self.purchases.values():
            out[b.name] = b.price_usd_per_bbl
        return out

    def gasoline(self) -> FinishedProduct:
        return self.products["gasoline"]

    def purchase_names(self, role: Optional[str] = "gasoline_blendstock") -> List[str]:
        if role is None:
            return list(self.purchases)
        return [k for k, v in self.purchases.items() if v.role == role]

    def gasoline_components(self) -> Dict[str, QualityComponent]:
        """All gasoline pool components (plant streams + purchases)."""
        gas = self.gasoline()
        out: Dict[str, QualityComponent] = {}
        for s in gas.components:
            if s in self.purchases:
                out[s] = self.purchases[s].quality_component()
            else:
                row = dict(self.component_properties.get(s) or {})
                out[s] = QualityComponent(
                    stream=s,
                    ron=float(row.get("ron", 0.0)),
                    sulfur_wt=float(row.get("sulfur_wt", 0.0)),
                )
        return out

    def gasoline_config(self) -> GasolineQualityConfig:
        gas = self.gasoline()
        specs = gas.specs
        return GasolineQualityConfig(
            min_ron=float(specs.min_ron if specs.min_ron is not None else 87.0),
            max_sulfur_wt=float(
                specs.max_sulfur_wt if specs.max_sulfur_wt is not None else 0.01
            ),
            model=str(gas.blend_model or "delta_base"),
            base_stream=str(gas.base_stream or "buy_alkylate"),
        )

    def stream_props(self, stream: str) -> Dict[str, float]:
        if stream in self.purchases:
            return dict(self.purchases[stream].properties)
        return dict(self.component_properties.get(stream) or {})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "source_path": self.source_path,
            "products": {k: v.as_dict() for k, v in self.products.items()},
            "purchases": {k: v.as_dict() for k, v in self.purchases.items()},
            "component_properties": {
                k: dict(v) for k, v in self.component_properties.items()
            },
        }


def build_catalog(raw: Mapping[str, Any] | None = None, path: str | Path | None = None) -> ProductSpecCatalog:
    if raw is None:
        p = Path(path) if path else default_product_specs_path()
        raw = load_product_specs(p)
        source = str(p)
    else:
        source = str(path) if path else None

    products: Dict[str, FinishedProduct] = {}
    for name, row in (raw.get("products") or {}).items():
        r = dict(row)
        products[name] = FinishedProduct(
            name=str(r.get("name", name)),
            price_usd_per_bbl=float(r.get("price_usd_per_bbl", 0.0)),
            specs=SpecLimits.from_dict(r.get("specs")),
            max_demand_kbd=(
                float(r["max_demand_kbd"]) if r.get("max_demand_kbd") is not None else None
            ),
            min_demand_kbd=float(r.get("min_demand_kbd", 0.0)),
            blend_model=r.get("blend_model"),
            base_stream=r.get("base_stream"),
            components=list(r.get("components") or []),
            source_streams=list(r.get("source_streams") or []),
            fallback_product=r.get("fallback_product"),
            notes=str(r.get("notes") or ""),
        )

    purchases: Dict[str, PurchaseIntermediate] = {}
    for name, row in (raw.get("purchase_intermediates") or {}).items():
        r = dict(row)
        purchases[name] = PurchaseIntermediate(
            name=str(r.get("name", name)),
            price_usd_per_bbl=float(r.get("price_usd_per_bbl", 0.0)),
            max_kbd=float(r.get("max_kbd", 0.0)),
            properties={k: float(v) for k, v in dict(r.get("properties") or {}).items()},
            role=str(r.get("role") or "gasoline_blendstock"),
            notes=str(r.get("notes") or ""),
        )

    # purchases also act as component property sources
    comp = {k: dict(v) for k, v in dict(raw.get("component_properties") or {}).items()}
    for name, pur in purchases.items():
        comp.setdefault(name, dict(pur.properties))

    return ProductSpecCatalog(
        products=products,
        purchases=purchases,
        component_properties=comp,
        meta=dict(raw.get("meta") or {}),
        source_path=source,
    )


# ---------------------------------------------------------------------------
# Closed-form quality gates (single stream + blend)
# ---------------------------------------------------------------------------


def check_stream_meets_spec(
    props: Mapping[str, float],
    specs: SpecLimits,
    *,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Evaluate single-stream properties against SpecLimits."""
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    def _cmp(name: str, val: Optional[float], limit: Optional[float], how: str) -> None:
        if limit is None:
            return
        if val is None:
            failures.append(f"{name}:missing")
            checks[name] = {"value": None, "limit": limit, "ok": False}
            return
        if how == "min":
            ok = float(val) + tol >= float(limit)
        else:
            ok = float(val) <= float(limit) + tol
        checks[name] = {"value": float(val), "limit": float(limit), "ok": ok}
        if not ok:
            failures.append(name)

    _cmp("ron_min", props.get("ron"), specs.min_ron, "min")
    _cmp("ron_max", props.get("ron"), specs.max_ron, "max")
    _cmp("sulfur_max", props.get("sulfur_wt"), specs.max_sulfur_wt, "max")
    _cmp("sulfur_min", props.get("sulfur_wt"), specs.min_sulfur_wt, "min")
    _cmp("api_min", props.get("api"), specs.min_api, "min")
    _cmp("api_max", props.get("api"), specs.max_api, "max")
    _cmp("cetane_min", props.get("cetane"), specs.min_cetane, "min")
    _cmp("rvp_max", props.get("rvp_psi"), specs.max_rvp_psi, "max")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "checks": checks,
    }


def classify_gasoil_stream(
    stream: str,
    props: Mapping[str, float],
    catalog: ProductSpecCatalog,
) -> Dict[str, Any]:
    """Sweet → sour → fuel_oil reclass ladder for distillate/LCO."""
    sweet = catalog.products.get("sweet_gasoil")
    sour = catalog.products.get("sour_gasoil")
    fo = catalog.products.get("fuel_oil")

    ladder: List[Tuple[str, FinishedProduct]] = []
    if sweet:
        ladder.append(("sweet_gasoil", sweet))
    if sour:
        ladder.append(("sour_gasoil", sour))
    if fo:
        ladder.append(("fuel_oil", fo))

    for grade, prod in ladder:
        res = check_stream_meets_spec(props, prod.specs)
        if res["ok"]:
            return {
                "stream": stream,
                "grade": grade,
                "price_usd_per_bbl": prod.price_usd_per_bbl,
                "meets": res,
                "props": dict(props),
            }
    # last resort FO even if soft-fail
    last = ladder[-1] if ladder else ("unsellable", None)
    grade = last[0]
    price = last[1].price_usd_per_bbl if last[1] else 0.0
    return {
        "stream": stream,
        "grade": grade,
        "price_usd_per_bbl": price,
        "meets": check_stream_meets_spec(props, last[1].specs) if last[1] else {"ok": False},
        "props": dict(props),
        "forced": True,
    }


def evaluate_gasoline_blend(
    volumes: Mapping[str, float],
    catalog: Optional[ProductSpecCatalog] = None,
) -> Dict[str, Any]:
    """Closed-form gasoline blend RON/S vs catalog specs."""
    cat = catalog or build_catalog()
    components = cat.gasoline_components()
    cfg = cat.gasoline_config()
    blend = blend_quality_closed_form(volumes, components, cfg)
    specs = cat.gasoline().specs
    meets = check_stream_meets_spec(
        {"ron": blend["ron"], "sulfur_wt": blend["sulfur_wt"]},
        specs,
    )
    # RVP volume-weighted report (not LP-enforced by default)
    vtot = float(blend.get("volume") or 0.0)
    if vtot > 1e-12:
        rvp = 0.0
        for s, x in volumes.items():
            if float(x) <= 0:
                continue
            props = cat.stream_props(s)
            rvp += float(props.get("rvp_psi", 0.0)) * float(x)
        rvp /= vtot
    else:
        rvp = 0.0
    rvp_ok = True
    if specs.max_rvp_psi is not None and vtot > 1e-12:
        rvp_ok = rvp <= float(specs.max_rvp_psi) + 1e-9

    return {
        "blend": blend,
        "specs": specs.as_dict(),
        "meets_hard": meets,
        "rvp_psi": rvp,
        "rvp_ok": rvp_ok,
        "ok": bool(meets["ok"]),
        "price_usd_per_bbl": cat.gasoline().price_usd_per_bbl,
    }


def evaluate_product_slate(
    *,
    gasoline_volumes: Mapping[str, float],
    stream_volumes: Mapping[str, float],
    stream_props_override: Optional[Mapping[str, Mapping[str, float]]] = None,
    catalog: Optional[ProductSpecCatalog] = None,
) -> Dict[str, Any]:
    """Score full W2 product slate (gas + sweet/sour GO + naphtha).

    stream_volumes keys: cdu_distillate, fcc_lco, cdu_naphtha_sell, etc.
    """
    cat = catalog or build_catalog()
    ov = {k: dict(v) for k, v in dict(stream_props_override or {}).items()}

    gas = evaluate_gasoline_blend(gasoline_volumes, cat)

    go_results = {}
    for stream in ("cdu_distillate", "fcc_lco"):
        vol = float(stream_volumes.get(stream, 0.0))
        props = ov.get(stream) or cat.stream_props(stream)
        go_results[stream] = {
            "volume": vol,
            **classify_gasoil_stream(stream, props, cat),
        }

    # naphtha intermediate sell
    naph_vol = float(stream_volumes.get("cdu_naphtha_sell", stream_volumes.get("naphtha", 0.0)))
    naph_props = ov.get("cdu_naphtha") or cat.stream_props("cdu_naphtha")
    naph_prod = cat.products.get("naphtha")
    naph_check = (
        check_stream_meets_spec(naph_props, naph_prod.specs)
        if naph_prod
        else {"ok": True, "failures": [], "checks": {}}
    )
    naph = {
        "volume": naph_vol,
        "props": naph_props,
        "meets": naph_check,
        "price_usd_per_bbl": naph_prod.price_usd_per_bbl if naph_prod else 0.0,
        "ok": bool(naph_check.get("ok", True)),
    }

    # revenue proxy (no purchases netted here)
    rev = gas["price_usd_per_bbl"] * float(gas["blend"].get("volume") or 0.0)
    for st, row in go_results.items():
        rev += float(row["price_usd_per_bbl"]) * float(row["volume"])
    if naph_check.get("ok", True):
        rev += float(naph["price_usd_per_bbl"]) * naph_vol

    return {
        "gasoline": gas,
        "gasoil": go_results,
        "naphtha": naph,
        "revenue_proxy": rev,
        "ok": bool(gas["ok"] and naph["ok"]),
        "catalog_path": cat.source_path,
    }


# ---------------------------------------------------------------------------
# PuLP helpers: purchases + gasoline quality
# ---------------------------------------------------------------------------


@dataclass
class PurchaseVars:
    """Purchase LP variables + cost expression pieces."""

    variables: Dict[str, Any]
    cost_terms: List[Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def add_purchase_variables(
    prob: Any,
    catalog: ProductSpecCatalog,
    *,
    allow: bool = True,
    names: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> PurchaseVars:
    """Create non-negative purchase vars bounded by catalog max_kbd."""
    if pulp is None:  # pragma: no cover
        raise RuntimeError("pulp required")

    want = list(names) if names is not None else catalog.purchase_names()
    variables: Dict[str, Any] = {}
    cost_terms: List[Any] = []
    meta_rows: Dict[str, Any] = {}
    for name in want:
        pur = catalog.purchases.get(name)
        if pur is None:
            continue
        ub = float(pur.max_kbd) if allow else 0.0
        var = pulp.LpVariable(f"{prefix}{name}", lowBound=0, upBound=ub)
        variables[name] = var
        cost_terms.append(float(pur.price_usd_per_bbl) * var)
        meta_rows[name] = pur.as_dict()
    return PurchaseVars(variables=variables, cost_terms=cost_terms, meta=meta_rows)


def add_catalog_gasoline_constraints(
    prob: Any,
    *,
    product_var: Any,
    volume_vars: Mapping[str, Any],
    catalog: Optional[ProductSpecCatalog] = None,
    ron_name: str = "qual_gas_min_ron",
    sulfur_name: str = "qual_gas_max_s",
) -> QualityConstraintMeta:
    """Wire catalog gasoline specs into LP via quality_blender delta-base."""
    cat = catalog or build_catalog()
    components = cat.gasoline_components()
    # only components present in volume_vars
    use = {s: c for s, c in components.items() if s in volume_vars}
    cfg = cat.gasoline_config()
    return add_gasoline_quality_constraints(
        prob,
        product_var=product_var,
        volume_vars=volume_vars,
        components=use,
        cfg=cfg,
        ron_name=ron_name,
        sulfur_name=sulfur_name,
    )


def apply_catalog_prices(
    base_prices: Optional[Mapping[str, float]] = None,
    catalog: Optional[ProductSpecCatalog] = None,
) -> Dict[str, float]:
    """Merge catalog product/purchase prices into a price dict."""
    cat = catalog or build_catalog()
    out = dict(base_prices or {})
    out.update(cat.prices())
    return out


def default_ron_sulfur_maps(
    catalog: Optional[ProductSpecCatalog] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compatibility maps matching crude_cat_blender RON/SULFUR dicts."""
    cat = catalog or build_catalog()
    ron: Dict[str, float] = {}
    sulfur: Dict[str, float] = {}
    for s, c in cat.gasoline_components().items():
        ron[s] = float(c.ron)
        sulfur[s] = float(c.sulfur_wt)
    return ron, sulfur


def summarize_w2_quality(
    *,
    gasoline_volumes: Mapping[str, float],
    purchases: Mapping[str, float],
    products: Mapping[str, float],
    stream_props_override: Optional[Mapping[str, Mapping[str, float]]] = None,
    catalog: Optional[ProductSpecCatalog] = None,
) -> Dict[str, Any]:
    """Post-solve quality report for VERDICT / CrudeCatBlenderResult.quality."""
    cat = catalog or build_catalog()
    vols = dict(gasoline_volumes)
    # ensure purchase volumes included
    for k, v in purchases.items():
        if k.startswith("buy_") or k in cat.purchases:
            vols[k] = float(v)

    slate = evaluate_product_slate(
        gasoline_volumes=vols,
        stream_volumes={
            "cdu_distillate": float(products.get("sweet_gasoil", 0.0)),
            "fcc_lco": float(products.get("sour_gasoil", 0.0)),
            "cdu_naphtha_sell": float(
                products.get("naphtha_intermediate", products.get("naphtha", 0.0))
            ),
        },
        stream_props_override=stream_props_override,
        catalog=cat,
    )
    gas = slate["gasoline"]
    return {
        "gasoline_ron": gas["blend"]["ron"],
        "gasoline_sulfur_wt": gas["blend"]["sulfur_wt"],
        "gasoline_rvp_psi": gas["rvp_psi"],
        "ron_min": cat.gasoline().specs.min_ron,
        "s_max": cat.gasoline().specs.max_sulfur_wt,
        "ron_ok": gas["ok"] and gas["meets_hard"]["ok"],
        "s_ok": gas["meets_hard"]["ok"],
        "rvp_ok": gas["rvp_ok"],
        "naphtha": slate["naphtha"],
        "gasoil": slate["gasoil"],
        "purchases_available": list(cat.purchases),
        "model": "product_specs_w2",
        "source_path": cat.source_path,
    }
