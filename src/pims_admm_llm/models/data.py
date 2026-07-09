"""Synthetic crude / product data loaders for the toy refinery LP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class CrudeAssay:
    name: str
    price_usd_per_bbl: float
    max_supply_kbd: float
    yields: Dict[str, float]
    api: float = 0.0
    sulfur_wt_pct: float = 0.0


@dataclass
class ProductSpec:
    name: str
    price_usd_per_bbl: float
    max_demand_kbd: float


@dataclass
class RefineryData:
    crudes: List[CrudeAssay]
    products: Dict[str, ProductSpec]
    cdu_capacity_kbd: float
    blend_recipes: Dict[str, Dict[str, float]]
    intermediates: List[str] = field(
        default_factory=lambda: ["naphtha", "distillate", "gasoil", "residue"]
    )


def load_crude_data(path: str | Path | None = None) -> RefineryData:
    if path is None:
        # repo root / data / synthetic_crudes.json
        here = Path(__file__).resolve()
        path = here.parents[3] / "data" / "synthetic_crudes.json"
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    crudes = [
        CrudeAssay(
            name=c["name"],
            price_usd_per_bbl=float(c["price_usd_per_bbl"]),
            max_supply_kbd=float(c["max_supply_kbd"]),
            yields={k: float(v) for k, v in c["yields"].items()},
            api=float(c.get("api", 0.0)),
            sulfur_wt_pct=float(c.get("sulfur_wt_pct", 0.0)),
        )
        for c in raw["crudes"]
    ]
    products = {
        name: ProductSpec(
            name=name,
            price_usd_per_bbl=float(p["price_usd_per_bbl"]),
            max_demand_kbd=float(p["max_demand_kbd"]),
        )
        for name, p in raw["products"].items()
    }
    return RefineryData(
        crudes=crudes,
        products=products,
        cdu_capacity_kbd=float(raw["cdu_capacity_kbd"]),
        blend_recipes={
            prod: {comp: float(f) for comp, f in recipe.items()}
            for prod, recipe in raw["blend_recipes"].items()
        },
    )
