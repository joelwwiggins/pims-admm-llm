"""Synthetic crude / product / inventory / utilities data loaders for the toy refinery LP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CrudeAssay:
    name: str
    price_usd_per_bbl: float
    max_supply_kbd: float
    yields: Dict[str, float]
    api: float = 0.0
    sulfur_wt_pct: float = 0.0
    # utility intensity per bbl charged (fuel_gas, steam, power)
    utility_use: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProductSpec:
    name: str
    price_usd_per_bbl: float
    max_demand_kbd: float
    utility_use: Dict[str, float] = field(default_factory=dict)


@dataclass
class InventorySpec:
    """Single-period tank for one intermediate stream (kbd units)."""

    stream: str
    start_kbd: float
    capacity_kbd: float
    holding_cost_usd_per_bbl: float = 0.0


@dataclass
class UtilitySpec:
    """Shared utility resource with capacity and unit cost."""

    name: str
    capacity: float
    cost_usd_per_unit: float
    unit: str = "unit"


@dataclass
class RefineryData:
    crudes: List[CrudeAssay]
    products: Dict[str, ProductSpec]
    cdu_capacity_kbd: float
    blend_recipes: Dict[str, Dict[str, float]]
    intermediates: List[str] = field(
        default_factory=lambda: ["naphtha", "distillate", "gasoil", "residue"]
    )
    inventory: Dict[str, InventorySpec] = field(default_factory=dict)
    utilities: Dict[str, UtilitySpec] = field(default_factory=dict)
    utility_names: List[str] = field(
        default_factory=lambda: ["fuel_gas", "steam", "power"]
    )

    def linking_streams(self) -> List[str]:
        """Primary ADMM linking variables: intermediate material balances."""
        return list(self.intermediates)

    def linking_utilities(self) -> List[str]:
        """Secondary linking variables: shared utility balances."""
        return list(self.utility_names)


def _repo_root_candidates(here: Path) -> List[Path]:
    """Likely repo roots relative to this module file."""
    # .../src/pims_admm_llm/models/data.py → parents[3] = repo root
    return [
        here.parents[3],
        here.parents[2],
        Path.cwd(),
        Path("/home/joel/projects/pims-admm-llm"),
    ]


def default_data_path() -> Path:
    """Resolve data/synthetic_crudes.json from installed layout, cwd, or fixed repo path."""
    here = Path(__file__).resolve()
    name = Path("data") / "synthetic_crudes.json"
    for root in _repo_root_candidates(here):
        candidate = root / name
        if candidate.is_file():
            return candidate
    # last resort: first candidate even if missing (caller will raise)
    return _repo_root_candidates(here)[0] / name


def load_crude_data(path: str | Path | None = None) -> RefineryData:
    """Load synthetic (or realistic export-shaped) crude / product / inventory data."""
    if path is None:
        path = default_data_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Crude data not found at {path}. Expected synthetic_crudes.json under repo data/."
        )
    with path.open() as f:
        raw = json.load(f)

    intermediates = list(
        raw.get("intermediates")
        or ["naphtha", "distillate", "gasoil", "residue"]
    )

    crudes: List[CrudeAssay] = []
    for c in raw["crudes"]:
        yields = {k: float(v) for k, v in c["yields"].items()}
        # normalize missing intermediate keys to 0
        for i in intermediates:
            yields.setdefault(i, 0.0)
        crudes.append(
            CrudeAssay(
                name=c["name"],
                price_usd_per_bbl=float(c["price_usd_per_bbl"]),
                max_supply_kbd=float(c["max_supply_kbd"]),
                yields=yields,
                api=float(c.get("api", 0.0)),
                sulfur_wt_pct=float(c.get("sulfur_wt_pct", 0.0)),
                utility_use={
                    k: float(v) for k, v in (c.get("utility_use") or {}).items()
                },
            )
        )

    products: Dict[str, ProductSpec] = {}
    for name, p in raw["products"].items():
        products[name] = ProductSpec(
            name=name,
            price_usd_per_bbl=float(p["price_usd_per_bbl"]),
            max_demand_kbd=float(p["max_demand_kbd"]),
            utility_use={
                k: float(v) for k, v in (p.get("utility_use") or {}).items()
            },
        )

    inventory: Dict[str, InventorySpec] = {}
    inv_raw = raw.get("inventory") or {}
    for stream, spec in inv_raw.items():
        inventory[stream] = InventorySpec(
            stream=stream,
            start_kbd=float(spec.get("start_kbd", 0.0)),
            capacity_kbd=float(spec.get("capacity_kbd", 1e9)),
            holding_cost_usd_per_bbl=float(spec.get("holding_cost_usd_per_bbl", 0.0)),
        )
    # default empty inventory specs for any intermediate missing in JSON
    for stream in intermediates:
        if stream not in inventory:
            inventory[stream] = InventorySpec(
                stream=stream,
                start_kbd=0.0,
                capacity_kbd=1e6,
                holding_cost_usd_per_bbl=0.0,
            )

    utilities: Dict[str, UtilitySpec] = {}
    util_raw = raw.get("utilities") or {}
    for uname, uspec in util_raw.items():
        utilities[uname] = UtilitySpec(
            name=uname,
            capacity=float(uspec.get("capacity", 1e9)),
            cost_usd_per_unit=float(uspec.get("cost_usd_per_unit", 0.0)),
            unit=str(uspec.get("unit", "unit")),
        )
    utility_names = list(utilities.keys()) or ["fuel_gas", "steam", "power"]
    # ensure all named utilities exist
    for uname in utility_names:
        utilities.setdefault(
            uname,
            UtilitySpec(name=uname, capacity=1e6, cost_usd_per_unit=0.0),
        )

    return RefineryData(
        crudes=crudes,
        products=products,
        cdu_capacity_kbd=float(raw["cdu_capacity_kbd"]),
        blend_recipes={
            prod: {comp: float(f) for comp, f in recipe.items()}
            for prod, recipe in raw["blend_recipes"].items()
        },
        intermediates=intermediates,
        inventory=inventory,
        utilities=utilities,
        utility_names=utility_names,
    )


def validate_refinery_data(data: RefineryData) -> List[str]:
    """Return list of validation warnings/errors (empty => OK)."""
    issues: List[str] = []
    if not data.crudes:
        issues.append("no crudes defined")
    if data.cdu_capacity_kbd <= 0:
        issues.append("cdu_capacity_kbd must be positive")
    for c in data.crudes:
        s = sum(c.yields.get(i, 0.0) for i in data.intermediates)
        if abs(s - 1.0) > 0.05:
            issues.append(f"crude {c.name} yield sum={s:.3f} (expected ~1.0)")
        if c.max_supply_kbd < 0:
            issues.append(f"crude {c.name} max_supply negative")
    for prod, recipe in data.blend_recipes.items():
        if prod not in data.products:
            issues.append(f"blend recipe for unknown product {prod}")
        s = sum(recipe.values())
        if abs(s - 1.0) > 0.05:
            issues.append(f"recipe {prod} fractions sum={s:.3f} (expected ~1.0)")
        for comp in recipe:
            if comp not in data.intermediates:
                issues.append(f"recipe {prod} uses unknown intermediate {comp}")
    for stream in data.intermediates:
        inv = data.inventory.get(stream)
        if inv is None:
            issues.append(f"missing inventory for {stream}")
        elif inv.capacity_kbd < inv.start_kbd:
            issues.append(f"inventory {stream}: capacity < start")
    return issues
