"""Unit stream catalogs, feed-pooler topology, and process-condition definitions.

PIMS/HYSYS-style planning view: every conversion unit declares
  - feed quality vectors (inputs)
  - operating variables (process conditions)
  - product yield streams that must each route somewhere (pool / unit / sell / fuel)

Topology rule: in front of each process unit is a feed pooler (tank) with an optional
direct-to-unit bypass arc so the optimizer can route via inventory or straight in.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Process conditions (planning-grade defaults; accessible in LP meta + UI)
# ---------------------------------------------------------------------------

DEFAULT_PROCESS_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "CDU": {
        "flash_zone_temp_f": 680.0,
        "overflash_frac": 0.02,
        "atm_tower_pressure_psig": 5.0,
        "cut_points_f": {
            "naphtha_ep": 350.0,
            "distillate_ep": 650.0,
            "gasoil_ep": 1050.0,
        },
    },
    "FCC": {
        "riser_outlet_temp_f": 980.0,  # ROT / severity
        "catalyst_to_oil": 6.5,
        "catalyst_activity": 68.0,  # relative MAT index
        "additive_rate_wt_pct_cat": 0.0,
        "feed_preheat_temp_f": 550.0,
        "recycle_ratio": 0.0,
        "regenerator_temp_f": 1320.0,
    },
    "COKER": {
        "drum_outlet_temp_f": 920.0,
        "drum_pressure_psig": 25.0,
        "recycle_ratio": 0.15,
        "cycle_time_hr": 16.0,
        "furnace_coil_outlet_temp_f": 920.0,
    },
    "REFORMER": {
        "severity_wait_avg_f": 940.0,
        "pressure_psig": 150.0,
        "h2_hc_ratio": 4.0,
        "space_velocity_whsv": 1.5,
        "severity_severity_count": 3,
    },
    "HDT_NAPH": {
        "reactor_inlet_temp_f": 550.0,
        "pressure_psig": 600.0,
        "h2_partial_pressure_psia": 450.0,
        "lhsv": 2.0,
        "h2_oil_scf_bbl": 800.0,
    },
    "BLENDER": {
        "blend_mode": "delta_base",
        "ron_spec": 87.0,
        "max_sulfur_wt": 0.01,
    },
}


# Feed quality vector keys tracked for each unit (planning LP / inspector)
DEFAULT_FEED_QUALITY_KEYS: Dict[str, List[str]] = {
    "CDU": ["api", "sulfur_wt", "ccr_wt", "nitrogen_ppm", "tbp_curve"],
    "FCC": [
        "api",
        "sulfur_wt",
        "ccr_wt",
        "uop_k",
        "aniline_point_f",
        "metals_ni_v_ppm",
        "basic_nitrogen_ppm",
        "tbp_curve",
    ],
    "COKER": ["api", "sulfur_wt", "ccr_wt", "asphaltenes_wt", "viscosity_cst_210f"],
    "REFORMER": ["api", "sulfur_wt", "nitrogen_ppm", "paraffins_vol", "naphthenes_vol", "aromatics_vol", "n_plus_a"],
    "HDT_NAPH": ["api", "sulfur_wt", "nitrogen_ppm", "olefins_vol", "ron"],
    "BLENDER": ["ron", "sulfur_wt", "rvp", "density_api"],
}


# Product yield streams that MUST exist and be routed for each unit
# basis: vol = vol frac of fresh feed; wt = wt frac of fresh feed; fuel = fuel-gas equiv vol
UNIT_YIELD_STREAMS: Dict[str, List[Dict[str, Any]]] = {
    "CDU": [
        {"stream": "cdu_naphtha_light", "basis": "vol", "typical_range_pct": [8, 18], "default_routes": ["BLENDER"]},
        {"stream": "cdu_naphtha_heavy", "basis": "vol", "typical_range_pct": [8, 18], "default_routes": ["POOL_REFORMER", "REFORMER", "BLENDER"]},
        {"stream": "cdu_distillate", "basis": "vol", "typical_range_pct": [15, 35], "default_routes": ["BLENDER"]},
        {"stream": "cdu_gasoil", "basis": "vol", "typical_range_pct": [20, 35], "default_routes": ["POOL_FCC", "FCC", "BLENDER", "SELL"]},
        {"stream": "cdu_resid", "basis": "vol", "typical_range_pct": [10, 45], "default_routes": ["POOL_COKER", "COKER", "BLENDER"]},
        {"stream": "cdu_offgas", "basis": "fuel", "typical_range_pct": [0.5, 2], "default_routes": ["FUEL_GAS"]},
    ],
    "FCC": [
        {"stream": "fcc_dry_gas", "basis": "vol", "typical_range_pct": [3, 8], "default_routes": ["FUEL_GAS"], "note": "H2, C1–C2"},
        {"stream": "fcc_lpg", "basis": "vol", "typical_range_pct": [12, 25], "default_routes": ["LPG", "FUEL_GAS"], "note": "C3/C4 incl. olefins"},
        {"stream": "fcc_naphtha", "basis": "vol", "typical_range_pct": [40, 55], "default_routes": ["POOL_FCC_NAPH", "BLENDER", "HDT_NAPH"]},
        {"stream": "fcc_lco", "basis": "vol", "typical_range_pct": [15, 25], "default_routes": ["BLENDER"]},
        {"stream": "fcc_slurry", "basis": "vol", "typical_range_pct": [5, 15], "default_routes": ["BLENDER"]},
        {"stream": "fcc_coke", "basis": "wt", "typical_range_pct": [4, 8], "default_routes": ["REGEN_HEAT"], "note": "burned in regenerator"},
    ],
    "COKER": [
        {"stream": "coker_dry_gas", "basis": "vol", "typical_range_pct": [4, 10], "default_routes": ["FUEL_GAS"]},
        {"stream": "coker_lpg", "basis": "vol", "typical_range_pct": [3, 8], "default_routes": ["LPG", "FUEL_GAS"]},
        {"stream": "coker_naphtha", "basis": "vol", "typical_range_pct": [12, 22], "default_routes": ["POOL_COKER_NAPH", "HDT_NAPH", "BLENDER"]},
        {"stream": "coker_gasoil", "basis": "vol", "typical_range_pct": [40, 60], "default_routes": ["BLENDER", "POOL_FCC"]},
        {"stream": "coker_coke", "basis": "wt", "typical_range_pct": [15, 30], "default_routes": ["COKE_SALES"], "note": "petcoke product/credit"},
    ],
    "REFORMER": [
        {"stream": "reformate", "basis": "vol", "typical_range_pct": [78, 94], "default_routes": ["BLENDER"]},
        {"stream": "reformer_h2", "basis": "fuel", "typical_range_pct": [2, 6], "default_routes": ["H2_GRID", "FUEL_GAS"], "note": "net H2 make"},
        {"stream": "reformer_lights", "basis": "vol", "typical_range_pct": [5, 15], "default_routes": ["FUEL_GAS", "LPG"]},
    ],
    "HDT_NAPH": [
        {"stream": "hdt_naphtha", "basis": "vol", "typical_range_pct": [96, 99], "default_routes": ["BLENDER"]},
        {"stream": "hdt_lights", "basis": "vol", "typical_range_pct": [1, 4], "default_routes": ["FUEL_GAS"]},
    ],
}


# Feed poolers in front of process units (optional inventory) + direct bypass
FEED_POOLERS: List[Dict[str, Any]] = [
    {
        "pool": "POOL_FCC",
        "alias_tank": "TANK_GASOIL",
        "unit": "FCC",
        "stream": "cdu_gasoil",
        "via_pool_arc": "go_to_fcc",
        "direct_arc": "go_direct_to_fcc",
        "sources": ["CDU"],
        "note": "VGO/gasoil pooler in front of FCC; direct CDU→FCC bypass allowed",
    },
    {
        "pool": "POOL_COKER",
        "alias_tank": "TANK_RESID",
        "unit": "COKER",
        "stream": "cdu_resid",
        "via_pool_arc": "resid_to_coker",
        "direct_arc": "resid_direct_to_coker",
        "sources": ["CDU"],
        "note": "Resid pooler in front of coker; direct resid→coker bypass",
    },
    {
        "pool": "POOL_REFORMER",
        "alias_tank": "TANK_REFORMER_FEED",
        "unit": "REFORMER",
        "stream": "cdu_naphtha_heavy",
        "via_pool_arc": "ref_pool_to_reformer",
        "direct_arc": "sr_heavy_to_reformer",
        "sources": ["CDU", "TANK_FCC_NAPH", "TANK_COKER_NAPH"],
        "note": "Naphtha feed pooler for reformer; heavy SR default direct also open",
    },
    {
        "pool": "POOL_HDT",
        "alias_tank": "TANK_HDT_FEED",
        "unit": "HDT_NAPH",
        "stream": "cracked_naphtha",
        "via_pool_arc": "hdt_pool_to_hdt",
        "direct_arc": "cracked_naph_direct_to_hdt",
        "sources": ["FCC", "COKER", "TANK_FCC_NAPH", "TANK_COKER_NAPH"],
        "note": "Cracked naphtha pooler ahead of HDT",
    },
]


def default_process_conditions(unit: str) -> Dict[str, Any]:
    return deepcopy(DEFAULT_PROCESS_CONDITIONS.get(unit.upper(), {}))


def merge_process_conditions(
    unit: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base = default_process_conditions(unit)
    if overrides:
        base.update(dict(overrides))
    return base


def unit_yield_stream_names(unit: str) -> List[str]:
    return [row["stream"] for row in UNIT_YIELD_STREAMS.get(unit.upper(), [])]


def all_required_product_streams() -> List[str]:
    names: List[str] = []
    for rows in UNIT_YIELD_STREAMS.values():
        for row in rows:
            names.append(row["stream"])
    return names


def unit_catalog() -> Dict[str, Any]:
    """Serializable catalog for routing.json / API / UI."""
    units = {}
    for utype in sorted(set(list(DEFAULT_PROCESS_CONDITIONS) + list(UNIT_YIELD_STREAMS))):
        units[utype] = {
            "process_conditions": default_process_conditions(utype),
            "feed_quality_keys": list(DEFAULT_FEED_QUALITY_KEYS.get(utype, [])),
            "yield_streams": deepcopy(UNIT_YIELD_STREAMS.get(utype, [])),
        }
    return {
        "version": "wave4-unit-streams",
        "units": units,
        "feed_poolers": deepcopy(FEED_POOLERS),
        "notes": [
            "Every yield stream must have ≥1 routing arc (pool, unit, product, fuel, sell).",
            "Feed poolers sit in front of conversion units; direct arcs are decision variables.",
            "Process conditions are accessible meta / severity drivers; hard LP uses linear yield coeffs.",
        ],
    }


def validate_yields_cover_catalog(yields_by_unit: Mapping[str, Mapping[str, float]]) -> List[str]:
    """Return list of missing stream keys (empty if complete). Coke/gas keys required for conversion units."""
    missing: List[str] = []
    for unit, rows in UNIT_YIELD_STREAMS.items():
        y = yields_by_unit.get(unit) or yields_by_unit.get(unit.lower()) or {}
        # CDU naphtha is split later — accept cdu_naphtha as covering light+heavy
        for row in rows:
            s = row["stream"]
            if s in y:
                continue
            if unit == "CDU" and s in ("cdu_naphtha_light", "cdu_naphtha_heavy") and "cdu_naphtha" in y:
                continue
            if unit == "HDT_NAPH":
                # HDT treated as soft attribute path in MVP LP
                continue
            missing.append(f"{unit}:{s}")
    return missing
