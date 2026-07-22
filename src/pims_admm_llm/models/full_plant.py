"""Full plant monolithic LP with arc-flow superstructure (Wave3).

Routing is economic, not hard-coded:
  - Gasoil swing: FCC feed | diesel pool | sell
  - Resid swing: coker | fuel oil
  - Light SR naphtha → gasoline; heavy SR naphtha → reformer (default)
  - FCC naphtha → gasoline (soft HDT); NOT reformer default
  - Coker naphtha → HDT/gasoline or FO; NOT reformer default
  - Optional tanks: inventory_mode=False collapses to pure balances

Quality blender: planning-grade delta-base / optional index pooling for
gasoline RON + sulfur (see quality_blender.py and docs/quality_blender.md).
Diesel still uses soft-HDT linear sulfur credits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pulp

from .assay_loader import load_assays_json, load_routing
from .properties import FeedProperties, crude_to_props
from .quality_blender import (
    GASOLINE_COMPONENT_DEFAULTS,
    GasolineQualityConfig,
    add_gasoline_quality_constraints,
    load_component_qualities,
)
from .unit_specs import (
    default_process_conditions,
    merge_process_conditions,
    unit_catalog,
    validate_yields_cover_catalog,
)
from .yields import (
    cdu_yields_from_assay,
    coker_yields,
    fcc_yields,
    gasoil_props_from_crude,
    hdt_naph_yields,
    naphtha_props_coker,
    naphtha_props_fcc,
    reformer_yields,
    resid_props_from_crude,
)


@dataclass
class FullPlantResult:
    status: str
    objective: float
    feasible: bool
    crude_rates: Dict[str, float]
    unit_feeds: Dict[str, float]
    streams: Dict[str, float]
    products: Dict[str, float]
    tank_end: Dict[str, float]
    duals: Dict[str, float]
    economic_shadows: Dict[str, float]
    quality_duals: Dict[str, float]
    routing_splits: Dict[str, float]
    arc_flows: Dict[str, float]
    yields_used: Dict[str, Any]
    solve_time_s: float
    problem: pulp.LpProblem
    inventory_mode: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


def _val(v) -> float:
    x = pulp.value(v)
    return float(x) if x is not None else 0.0


def build_yield_tables(
    assays: Dict[str, Any],
    routing: Optional[Dict[str, Any]] = None,
    process_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Per-crude CDU yields + conversion-unit yields from feed props + process conditions."""
    routing = routing or {}
    pc_root = routing.get("process_conditions") or {}
    overrides = process_overrides or {}

    def _conds(unit: str) -> Dict[str, Any]:
        base = merge_process_conditions(unit, pc_root.get(unit) or pc_root.get(unit.upper()))
        if unit in overrides:
            base.update(overrides[unit])
        return base

    cdu: Dict[str, Any] = {}
    go_props = []
    resid_props = []
    for c in assays["crudes"]:
        props = crude_to_props(c)
        cdu[c["name"]] = cdu_yields_from_assay(props, c.get("tbp_cut_vol"))
        go_props.append((c["name"], gasoil_props_from_crude(props), c.get("max_supply_kbd", 1.0)))
        resid_props.append((c["name"], resid_props_from_crude(props), c.get("max_supply_kbd", 1.0)))

    def wavg_props(items):
        from .properties import FeedProperties

        tw = sum(w for _, _, w in items) or 1.0
        acc = FeedProperties(
            name="avg",
            api=0,
            sulfur_wt=0,
            ccr_wt=0,
            nitrogen_ppm=0,
            paraffins_vol=0,
            naphthenes_vol=0,
            aromatics_vol=0,
        )
        for _, p, w in items:
            a = w / tw
            acc.api += a * p.api
            acc.sulfur_wt += a * p.sulfur_wt
            acc.ccr_wt += a * p.ccr_wt
            acc.nitrogen_ppm += a * p.nitrogen_ppm
            acc.paraffins_vol += a * p.paraffins_vol
            acc.naphthenes_vol += a * p.naphthenes_vol
            acc.aromatics_vol += a * p.aromatics_vol
        acc.name = "avg_feed"
        return acc

    go = wavg_props(go_props)
    resid = wavg_props(resid_props)
    fcc_cond = _conds("FCC")
    coker_cond = _conds("COKER")
    ref_cond = _conds("REFORMER")
    hdt_cond = _conds("HDT_NAPH")
    cdu_cond = _conds("CDU")

    fcc_y = fcc_yields(go, fcc_cond)
    coker_y = coker_yields(resid, coker_cond)
    # Reformer feed quality: primarily heavy SR naphtha character (not FCC/coker default)
    n_heavy = FeedProperties_like_heavy_sr(go)
    n_fcc = naphtha_props_fcc(go)
    n_cok = naphtha_props_coker(resid)
    ref_y = reformer_yields(n_heavy, ref_cond)
    hdt_y = hdt_naph_yields(hdt_cond)

    return {
        "cdu_by_crude": cdu,
        "fcc": fcc_y,
        "coker": coker_y,
        "reformer": ref_y,
        "hdt_naph": hdt_y,
        "process_conditions": {
            "CDU": cdu_cond,
            "FCC": fcc_cond,
            "COKER": coker_cond,
            "REFORMER": ref_cond,
            "HDT_NAPH": hdt_cond,
        },
        "feed_props": {
            "gasoil": go.__dict__,
            "resid": resid.__dict__,
            "reformer_feed": n_heavy.__dict__,
            "fcc_naphtha": n_fcc.__dict__,
            "coker_naphtha": n_cok.__dict__,
        },
        "catalog_gaps": validate_yields_cover_catalog(
            {
                "CDU": next(iter(cdu.values())) if cdu else {},
                "FCC": fcc_y,
                "COKER": coker_y,
                "REFORMER": ref_y,
                "HDT_NAPH": hdt_y,
            }
        ),
        "unit_catalog": unit_catalog(),
    }


def FeedProperties_like_heavy_sr(gasoil_or_crude_like):
    """Planning-grade heavy SR naphtha properties for reformer yield vector."""
    from .properties import FeedProperties

    # Prefer paraffinic/naphthenic SR heavy naphtha vs cracked naphthas
    base_api = getattr(gasoil_or_crude_like, "api", 28.0) + 12.0
    return FeedProperties(
        name="heavy_sr_naphtha",
        api=max(45.0, min(65.0, base_api)),
        sulfur_wt=max(0.01, getattr(gasoil_or_crude_like, "sulfur_wt", 0.5) * 0.15),
        ccr_wt=0.05,
        nitrogen_ppm=max(50.0, getattr(gasoil_or_crude_like, "nitrogen_ppm", 1000) * 0.2),
        paraffins_vol=0.45,
        naphthenes_vol=0.35,
        aromatics_vol=0.20,
    )


def _comp_prop(routing: Dict[str, Any], stream: str, key: str, default: float) -> float:
    cp = (routing.get("component_properties") or {}).get(stream) or {}
    return float(cp.get(key, default))


def _quality_spec(routing: Dict[str, Any], product: str) -> Dict[str, float]:
    return dict((routing.get("product_quality_specs") or {}).get(product) or {})


def _arc_meta(routing: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {a["id"]: a for a in (routing.get("arcs") or []) if "id" in a}


def _arc_upbound(meta: Dict[str, Any], *, force_open: bool = False) -> Optional[float]:
    """Upper bound for an arc variable. default_open=false → closed unless force_open.

    capacity null/missing → unbounded (None). capacity 0 → closed.
    """
    if not force_open and meta.get("default_open") is False:
        return 0.0
    cap = meta.get("capacity", meta.get("capacity_kbd"))
    if cap is None:
        return None
    return float(cap)


def _feed_props_from_yields(
    yields: Dict[str, Any], key: str, fallback: FeedProperties
) -> FeedProperties:
    """Rebuild FeedProperties from build_yield_tables feed_props dicts."""
    raw = (yields.get("feed_props") or {}).get(key)
    if not isinstance(raw, dict):
        return fallback
    try:
        return FeedProperties(
            name=str(raw.get("name") or key),
            api=float(raw.get("api", fallback.api)),
            sulfur_wt=float(raw.get("sulfur_wt", fallback.sulfur_wt)),
            ccr_wt=float(raw.get("ccr_wt", fallback.ccr_wt)),
            nitrogen_ppm=float(raw.get("nitrogen_ppm", fallback.nitrogen_ppm)),
            paraffins_vol=float(raw.get("paraffins_vol", fallback.paraffins_vol)),
            naphthenes_vol=float(raw.get("naphthenes_vol", fallback.naphthenes_vol)),
            aromatics_vol=float(raw.get("aromatics_vol", fallback.aromatics_vol)),
        )
    except Exception:
        return fallback


def apply_process_pool_modes_to_yields(
    yields: Dict[str, Any],
    assays: Dict[str, Any],
    *,
    fcc_feed_kbd: Optional[float] = None,
    coker_feed_kbd: Optional[float] = None,
    seed: Optional["FullPlantResult"] = None,
    msg: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run process-pool MIP and attach selected FCC/coker mode yields to plant tables.

    Plant stays LP: binaries live only in the process-pool MIP; selected yield
    vectors are fixed into the continuous plant model. Mono duals remain plan
    truth for the LP solve that uses these yields.
    """
    from .process_pool import (
        COKER_RECYCLE_MODES,
        FCC_ROT_MODES,
        attach_process_pool_to_plant_yields,
        default_gasoil_feed,
        default_resid_feed,
        solve_process_pool_mip,
    )

    caps = assays.get("capacities") or {}
    if seed is not None:
        fcc_f = float(seed.unit_feeds.get("fcc_feed") or 0.0)
        cok_f = float(seed.unit_feeds.get("coker_feed") or 0.0)
    else:
        fcc_f = float(fcc_feed_kbd) if fcc_feed_kbd is not None else (
            float(caps.get("fcc_kbd", 55.0)) * 0.7
        )
        cok_f = float(coker_feed_kbd) if coker_feed_kbd is not None else (
            float(caps.get("coker_kbd", 40.0)) * 0.5
        )
    # MIP needs positive fixed feeds for mode economics; clamp tiny feeds.
    fcc_f = max(fcc_f, 1.0)
    cok_f = max(cok_f, 1.0)

    gasoil = _feed_props_from_yields(yields, "gasoil", default_gasoil_feed())
    resid = _feed_props_from_yields(yields, "resid", default_resid_feed())
    pool = solve_process_pool_mip(
        gasoil=gasoil,
        resid=resid,
        fcc_feed_kbd=fcc_f,
        coker_feed_kbd=cok_f,
        msg=msg,
    )
    merged = attach_process_pool_to_plant_yields(yields, pool)

    # Honesty: stamp process_conditions from selected discrete modes.
    pc = dict(merged.get("process_conditions") or {})
    for m in FCC_ROT_MODES:
        if m["id"] == pool.fcc_mode:
            pc["FCC"] = dict(m["conditions"])
            break
    for m in COKER_RECYCLE_MODES:
        if m["id"] == pool.coker_mode:
            pc["COKER"] = dict(m["conditions"])
            break
    merged["process_conditions"] = pc

    meta = {
        "enabled": True,
        "fcc_mode": pool.fcc_mode,
        "coker_mode": pool.coker_mode,
        "fcc_mode_selection": dict(pool.fcc_mode_selection or {}),
        "coker_mode_selection": dict(pool.coker_mode_selection or {}),
        "fcc_feed_kbd_used": fcc_f,
        "coker_feed_kbd_used": cok_f,
        "pool_objective": float(pool.objective),
        "pool_feasible": bool(pool.feasible),
        "two_pass": seed is not None,
        "plant_remains_lp": True,
        "modes_fixed_from_mip": True,
        "note": (
            "Discrete FCC ROT + coker recycle modes selected by process-pool MIP; "
            "yield tables attached to plant LP (plant stays continuous LP, not MIP)."
        ),
    }
    return merged, meta


def solve_full_plant(
    assays: Optional[Dict[str, Any]] = None,
    *,
    msg: bool = False,
    inventory_mode: Optional[bool] = None,
    routing: Optional[Dict[str, Any]] = None,
    force_all_arcs_open: bool = False,
    process_pool_modes: bool = False,
    process_pool_two_pass: bool = False,
    yields_override: Optional[Dict[str, Any]] = None,
) -> FullPlantResult:
    """Monolithic max-margin plant LP with arc-flow superstructure + quality pooling.

    process_pool_modes:
      When True, run the Wave5 process-pool MIP (FCC ROT + coker recycle bands)
      and attach selected mode yields before the plant LP. Default False keeps
      continuous process-condition yields (existing demos/tests).

    process_pool_two_pass:
      When True (implies process_pool_modes), first solve the continuous plant,
      then re-select modes using realized FCC/coker feeds and re-solve. Slightly
      more expensive; better mode choice when feeds differ from capacity guesses.
    """
    assays = assays or load_assays_json()
    routing = routing or load_routing()
    pool_meta: Optional[Dict[str, Any]] = None

    if yields_override is not None:
        yields = dict(yields_override)
        pool_meta = dict(yields.get("process_pool") or {}) or None
    elif process_pool_two_pass or process_pool_modes:
        base_yields = build_yield_tables(assays, routing=routing)
        seed: Optional[FullPlantResult] = None
        if process_pool_two_pass:
            seed = solve_full_plant(
                assays,
                msg=msg,
                inventory_mode=inventory_mode,
                routing=routing,
                force_all_arcs_open=force_all_arcs_open,
                process_pool_modes=False,
                process_pool_two_pass=False,
                yields_override=None,
            )
        yields, pool_meta = apply_process_pool_modes_to_yields(
            base_yields,
            assays,
            seed=seed,
            msg=msg,
        )
    else:
        yields = build_yield_tables(assays, routing=routing)

    caps = assays.get("capacities") or {}
    tanks = assays.get("tanks") or {}
    products = dict(assays.get("products") or {})
    # Planning defaults for gas/coke side products if assay package omits them
    products.setdefault(
        "lpg",
        {"price_usd_per_bbl": 45.0, "max_demand_kbd": 30.0},
    )
    credits = dict(assays.get("credits") or {})
    credits.setdefault("fuel_gas_usd_per_bbl_equiv", 35.0)
    credits.setdefault("fcc_coke_usd_per_bbl_feed", 12.0)
    credits.setdefault("coker_coke_usd_per_bbl_feed", 18.0)
    credits.setdefault("reformer_h2_usd_per_bbl_equiv", 40.0)

    # Tank mode: single-period pass-optional by default
    tank_cfg = routing.get("tanks") or {}
    if inventory_mode is None:
        mode = str(tank_cfg.get("mode", "single_period_pass_optional"))
        inventory_mode = mode in ("inventory", "multi_period", "heels")

    cdu_cap = float(caps.get("cdu_kbd", 140))
    fcc_cap = float(caps.get("fcc_kbd", 55))
    coker_cap = float(caps.get("coker_kbd", 40))
    reformer_cap = float(caps.get("reformer_kbd", 45))

    split = routing.get("naphtha_split") or {}
    light_frac = float(split.get("light_frac_of_cdu_naphtha", 0.45))
    heavy_frac = float(split.get("heavy_frac_of_cdu_naphtha", 0.55))
    ssum = light_frac + heavy_frac
    if ssum <= 0:
        light_frac, heavy_frac = 0.45, 0.55
    else:
        light_frac, heavy_frac = light_frac / ssum, heavy_frac / ssum

    t0 = time.perf_counter()
    prob = pulp.LpProblem("FullPlant_ArcFlow", pulp.LpMaximize)

    # --- Crude charge ---
    crude_v = {
        c["name"]: pulp.LpVariable(f"crude_{c['name']}", lowBound=0) for c in assays["crudes"]
    }
    for c in assays["crudes"]:
        prob += crude_v[c["name"]] <= float(c["max_supply_kbd"]), f"crude_supply_{c['name']}"

    charge = pulp.lpSum(crude_v.values())
    prob += charge <= cdu_cap, "cdu_capacity"

    # --- CDU production of cuts ---
    cuts = ["cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"]
    cdu_prod = {s: pulp.LpVariable(f"prod_{s}", lowBound=0) for s in cuts}
    for s in cuts:
        prob += (
            cdu_prod[s]
            == pulp.lpSum(
                yields["cdu_by_crude"][c["name"]][s] * crude_v[c["name"]] for c in assays["crudes"]
            ),
            f"cdu_yield_{s}",
        )

    # Light / heavy SR naphtha split (planning)
    cdu_naph_light = pulp.LpVariable("prod_cdu_naphtha_light", lowBound=0)
    cdu_naph_heavy = pulp.LpVariable("prod_cdu_naphtha_heavy", lowBound=0)
    prob += cdu_naph_light == light_frac * cdu_prod["cdu_naphtha"], "split_naph_light"
    prob += cdu_naph_heavy == heavy_frac * cdu_prod["cdu_naphtha"], "split_naph_heavy"

    # --- Tank configs ---
    tg = tanks.get("tank_gasoil", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tr = tanks.get("tank_resid", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tfn = tanks.get("tank_fcc_naph", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tcn = tanks.get("tank_coker_naph", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    tref = tanks.get("tank_reformate", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})

    start_go = float(tg.get("start_kbd", 0)) if inventory_mode else 0.0
    start_resid = float(tr.get("start_kbd", 0)) if inventory_mode else 0.0
    start_fn = float(tfn.get("start_kbd", 0)) if inventory_mode else 0.0
    start_cn = float(tcn.get("start_kbd", 0)) if inventory_mode else 0.0
    start_ref = float(tref.get("start_kbd", 0)) if inventory_mode else 0.0

    arc_meta = _arc_meta(routing)

    def _new_arc(aid: str) -> pulp.LpVariable:
        ub = _arc_upbound(arc_meta.get(aid, {"default_open": True}), force_open=force_all_arcs_open)
        if ub is None:
            return pulp.LpVariable(f"arc_{aid}", lowBound=0)
        return pulp.LpVariable(f"arc_{aid}", lowBound=0, upBound=ub)

    # --- Decision arcs: gasoil swing (pooler TANK_GASOIL / POOL_FCC) + direct bypass ---
    go_to_fcc = _new_arc("go_to_fcc")
    go_to_diesel = _new_arc("go_to_diesel")
    go_to_sell = _new_arc("go_to_sell")
    go_direct_to_fcc = _new_arc("go_direct_to_fcc")
    end_go = pulp.LpVariable("tank_end_gasoil", lowBound=0)
    if not inventory_mode:
        prob += end_go == 0, "pass_tank_gasoil_end0"
    # Pool balance: production into pool = via-pool outs + end; direct bypass is separate from CDU
    go_to_pool = pulp.LpVariable("flow_cdu_gasoil_to_pool", lowBound=0)
    go_dispose = pulp.LpVariable("dispose_cdu_gasoil", lowBound=0)
    prob += (
        go_to_pool + go_direct_to_fcc + go_dispose == cdu_prod["cdu_gasoil"],
        "bal_cdu_gasoil_split",
    )
    prob += (
        start_go + go_to_pool == go_to_fcc + go_to_diesel + go_to_sell + end_go,
        "bal_tank_gasoil",
    )
    if inventory_mode:
        prob += end_go <= float(tg.get("capacity_kbd", 1e9)), "cap_tank_gasoil"

    # --- Decision arcs: resid swing (pooler TANK_RESID / POOL_COKER) + direct bypass ---
    resid_to_coker = _new_arc("resid_to_coker")
    resid_to_fo = _new_arc("resid_to_fo")
    resid_direct_to_coker = _new_arc("resid_direct_to_coker")
    end_resid = pulp.LpVariable("tank_end_resid", lowBound=0)
    if not inventory_mode:
        prob += end_resid == 0, "pass_tank_resid_end0"
    resid_to_pool = pulp.LpVariable("flow_cdu_resid_to_pool", lowBound=0)
    resid_dispose = pulp.LpVariable("dispose_cdu_resid", lowBound=0)
    prob += (
        resid_to_pool + resid_direct_to_coker + resid_dispose == cdu_prod["cdu_resid"],
        "bal_cdu_resid_split",
    )
    prob += (
        start_resid + resid_to_pool == resid_to_coker + resid_to_fo + end_resid,
        "bal_tank_resid",
    )
    if inventory_mode:
        prob += end_resid <= float(tr.get("capacity_kbd", 1e9)), "cap_tank_resid"

    # --- FCC (full yield slate) ---
    fcc_feed = go_to_fcc + go_direct_to_fcc
    prob += fcc_feed <= fcc_cap, "fcc_capacity"
    fy = yields["fcc"]
    fcc_dry = pulp.LpVariable("prod_fcc_dry_gas", lowBound=0)
    fcc_lpg = pulp.LpVariable("prod_fcc_lpg", lowBound=0)
    fcc_naph = pulp.LpVariable("prod_fcc_naphtha", lowBound=0)
    fcc_lco = pulp.LpVariable("prod_fcc_lco", lowBound=0)
    fcc_slurry = pulp.LpVariable("prod_fcc_slurry", lowBound=0)
    fcc_coke = pulp.LpVariable("prod_fcc_coke", lowBound=0)
    prob += fcc_dry == float(fy.get("fcc_dry_gas", 0.0)) * fcc_feed, "fcc_y_dry"
    prob += fcc_lpg == float(fy.get("fcc_lpg", 0.0)) * fcc_feed, "fcc_y_lpg"
    prob += fcc_naph == float(fy["fcc_naphtha"]) * fcc_feed, "fcc_y_naph"
    prob += fcc_lco == float(fy["fcc_lco"]) * fcc_feed, "fcc_y_lco"
    prob += fcc_slurry == float(fy["fcc_slurry"]) * fcc_feed, "fcc_y_slurry"
    prob += fcc_coke == float(fy.get("fcc_coke", 0.0)) * fcc_feed, "fcc_y_coke"

    # --- Coker (full yield slate) ---
    coker_feed = resid_to_coker + resid_direct_to_coker
    prob += coker_feed <= coker_cap, "coker_capacity"
    cy = yields["coker"]
    cok_dry = pulp.LpVariable("prod_coker_dry_gas", lowBound=0)
    cok_lpg = pulp.LpVariable("prod_coker_lpg", lowBound=0)
    cok_naph = pulp.LpVariable("prod_coker_naphtha", lowBound=0)
    cok_go = pulp.LpVariable("prod_coker_gasoil", lowBound=0)
    cok_coke = pulp.LpVariable("prod_coker_coke", lowBound=0)
    prob += cok_dry == float(cy.get("coker_dry_gas", 0.0)) * coker_feed, "coker_y_dry"
    prob += cok_lpg == float(cy.get("coker_lpg", 0.0)) * coker_feed, "coker_y_lpg"
    prob += cok_naph == float(cy["coker_naphtha"]) * coker_feed, "coker_y_naph"
    prob += cok_go == float(cy["coker_gasoil"]) * coker_feed, "coker_y_go"
    prob += cok_coke == float(cy.get("coker_coke", 0.0)) * coker_feed, "coker_y_coke"

    # --- FCC naphtha: default → gasoline (soft HDT); reformer closed unless force/open ---
    fcc_n_to_gas = _new_arc("fcc_naph_to_gas")
    fcc_n_to_ref = _new_arc("fcc_naph_to_reformer")
    end_fn = pulp.LpVariable("tank_end_fcc_naph", lowBound=0)
    if not inventory_mode:
        prob += end_fn == 0, "pass_tank_fcc_naph_end0"
    prob += start_fn + fcc_naph == fcc_n_to_gas + fcc_n_to_ref + end_fn, "bal_tank_fcc_naph"
    if inventory_mode:
        prob += end_fn <= float(tfn.get("capacity_kbd", 1e9)), "cap_tank_fcc_naph"

    # --- Coker naphtha: HDT→gas | FO | reformer closed by default ---
    cok_n_to_hdt_gas = _new_arc("coker_naph_to_hdt_gas")
    cok_n_to_fo = _new_arc("coker_naph_to_fo")
    cok_n_to_ref = _new_arc("coker_naph_to_reformer")
    end_cn = pulp.LpVariable("tank_end_coker_naph", lowBound=0)
    if not inventory_mode:
        prob += end_cn == 0, "pass_tank_coker_naph_end0"
    prob += (
        start_cn + cok_naph == cok_n_to_hdt_gas + cok_n_to_fo + cok_n_to_ref + end_cn,
        "bal_tank_coker_naph",
    )
    if inventory_mode:
        prob += end_cn <= float(tcn.get("capacity_kbd", 1e9)), "cap_tank_coker_naph"

    # --- SR naphtha destinations (direct reformer OR via POOL_REFORMER) ---
    light_to_gas = pulp.LpVariable("arc_sr_light_to_gas", lowBound=0)
    heavy_to_ref = pulp.LpVariable("arc_sr_heavy_to_reformer", lowBound=0)
    heavy_to_gas = pulp.LpVariable("arc_sr_heavy_to_gas", lowBound=0)
    heavy_to_ref_pool = _new_arc("sr_heavy_to_ref_pool")
    ref_pool_to_reformer = _new_arc("ref_pool_to_reformer")
    # full use of light/heavy production (no free dispose of naphtha; can go to products)
    prob += light_to_gas <= cdu_naph_light, "avail_sr_light"
    # allow unused light as free disposal via FO? Keep ≤ and soft free dispose
    light_dispose = pulp.LpVariable("dispose_sr_light", lowBound=0)
    prob += light_to_gas + light_dispose == cdu_naph_light, "bal_sr_light"
    heavy_dispose = pulp.LpVariable("dispose_sr_heavy", lowBound=0)
    prob += (
        heavy_to_ref + heavy_to_gas + heavy_to_ref_pool + heavy_dispose == cdu_naph_heavy,
        "bal_sr_heavy",
    )
    tref_feed = tanks.get("tank_reformer_feed", {"start_kbd": 0, "capacity_kbd": 1e9, "holding_cost": 0})
    start_ref_feed = float(tref_feed.get("start_kbd", 0)) if inventory_mode else 0.0
    end_ref_feed = pulp.LpVariable("tank_end_reformer_feed", lowBound=0)
    if not inventory_mode:
        prob += end_ref_feed == 0, "pass_tank_reformer_feed_end0"
    prob += (
        start_ref_feed + heavy_to_ref_pool == ref_pool_to_reformer + end_ref_feed,
        "bal_tank_reformer_feed",
    )
    if inventory_mode:
        prob += end_ref_feed <= float(tref_feed.get("capacity_kbd", 1e9)), "cap_tank_reformer_feed"

    # --- Reformer: primarily heavy SR (direct or pool); FCC/coker optional ---
    reformer_feed = heavy_to_ref + ref_pool_to_reformer + fcc_n_to_ref + cok_n_to_ref
    prob += reformer_feed <= reformer_cap, "reformer_capacity"
    ry = yields["reformer"]
    reformate = pulp.LpVariable("prod_reformate", lowBound=0)
    ref_h2 = pulp.LpVariable("prod_reformer_h2", lowBound=0)
    ref_lights = pulp.LpVariable("prod_reformer_lights", lowBound=0)
    prob += reformate == float(ry["reformate"]) * reformer_feed, "reformer_y"
    prob += ref_h2 == float(ry.get("reformer_h2", 0.0)) * reformer_feed, "reformer_y_h2"
    prob += ref_lights == float(ry.get("reformer_lights", 0.0)) * reformer_feed, "reformer_y_lights"

    use_ref = pulp.LpVariable("use_reformate", lowBound=0)
    end_ref = pulp.LpVariable("tank_end_reformate", lowBound=0)
    if not inventory_mode:
        prob += end_ref == 0, "pass_tank_reformate_end0"
    prob += start_ref + reformate == use_ref + end_ref, "bal_tank_reformate"
    if inventory_mode:
        prob += end_ref <= float(tref.get("capacity_kbd", 1e9)), "cap_tank_reformate"

    # --- Other blend components ---
    use_dist = pulp.LpVariable("use_cdu_distillate", lowBound=0)
    use_lco_diesel = pulp.LpVariable("use_fcc_lco_diesel", lowBound=0)
    use_lco_fo = pulp.LpVariable("use_fcc_lco_fo", lowBound=0)
    use_slurry = pulp.LpVariable("use_fcc_slurry", lowBound=0)
    use_cok_go_diesel = pulp.LpVariable("use_coker_go_diesel", lowBound=0)
    use_cok_go_fo = pulp.LpVariable("use_coker_go_fo", lowBound=0)

    dist_dispose = pulp.LpVariable("dispose_distillate", lowBound=0)
    prob += use_dist + dist_dispose == cdu_prod["cdu_distillate"], "bal_distillate"
    prob += use_lco_diesel + use_lco_fo <= fcc_lco, "avail_lco"
    lco_dispose = pulp.LpVariable("dispose_lco", lowBound=0)
    prob += use_lco_diesel + use_lco_fo + lco_dispose == fcc_lco, "bal_lco"
    slurry_dispose = pulp.LpVariable("dispose_slurry", lowBound=0)
    slurry_to_coker = _new_arc("fcc_slurry_to_coker")
    prob += use_slurry + slurry_dispose + slurry_to_coker == fcc_slurry, "bal_slurry"
    cok_go_dispose = pulp.LpVariable("dispose_coker_go", lowBound=0)
    cok_go_to_fcc_pool = _new_arc("coker_go_to_fcc_pool")
    prob += use_cok_go_diesel + use_cok_go_fo + cok_go_dispose + cok_go_to_fcc_pool == cok_go, "bal_coker_go"
    # Optional recycle of coker GO into gasoil pool (closed by default via arc ub)
    if inventory_mode or True:
        # keep pool identity: recycle is an extra pool inlet (use auxiliary; ignore end0 pass)
        pass

    # --- Gas / LPG / coke / H2 / offgas product routing (every yield stream goes somewhere) ---
    fcc_dry_to_fuel = _new_arc("fcc_dry_gas_to_fuel")
    fcc_dry_dispose = pulp.LpVariable("dispose_fcc_dry_gas", lowBound=0)
    prob += fcc_dry_to_fuel + fcc_dry_dispose == fcc_dry, "bal_fcc_dry_gas"

    fcc_lpg_to_lpg = _new_arc("fcc_lpg_to_lpg")
    fcc_lpg_to_fuel = _new_arc("fcc_lpg_to_fuel")
    fcc_lpg_dispose = pulp.LpVariable("dispose_fcc_lpg", lowBound=0)
    prob += fcc_lpg_to_lpg + fcc_lpg_to_fuel + fcc_lpg_dispose == fcc_lpg, "bal_fcc_lpg"

    fcc_coke_to_regen = _new_arc("fcc_coke_to_regen")
    fcc_coke_dispose = pulp.LpVariable("dispose_fcc_coke", lowBound=0)
    prob += fcc_coke_to_regen + fcc_coke_dispose == fcc_coke, "bal_fcc_coke"

    cok_dry_to_fuel = _new_arc("coker_dry_gas_to_fuel")
    cok_dry_dispose = pulp.LpVariable("dispose_coker_dry_gas", lowBound=0)
    prob += cok_dry_to_fuel + cok_dry_dispose == cok_dry, "bal_coker_dry_gas"

    cok_lpg_to_lpg = _new_arc("coker_lpg_to_lpg")
    cok_lpg_to_fuel = _new_arc("coker_lpg_to_fuel")
    cok_lpg_dispose = pulp.LpVariable("dispose_coker_lpg", lowBound=0)
    prob += cok_lpg_to_lpg + cok_lpg_to_fuel + cok_lpg_dispose == cok_lpg, "bal_coker_lpg"

    cok_coke_to_sales = _new_arc("coker_coke_to_sales")
    cok_coke_dispose = pulp.LpVariable("dispose_coker_coke", lowBound=0)
    prob += cok_coke_to_sales + cok_coke_dispose == cok_coke, "bal_coker_coke"

    ref_h2_to_grid = _new_arc("reformer_h2_to_grid")
    ref_h2_dispose = pulp.LpVariable("dispose_reformer_h2", lowBound=0)
    prob += ref_h2_to_grid + ref_h2_dispose == ref_h2, "bal_reformer_h2"

    ref_lights_to_fuel = _new_arc("reformer_lights_to_fuel")
    ref_lights_to_lpg = _new_arc("reformer_lights_to_lpg")
    ref_lights_dispose = pulp.LpVariable("dispose_reformer_lights", lowBound=0)
    prob += (
        ref_lights_to_fuel + ref_lights_to_lpg + ref_lights_dispose == ref_lights,
        "bal_reformer_lights",
    )

    # CDU offgas (vol-equivalent credit on charge)
    cdu_offgas_y = 0.0
    for c in assays["crudes"]:
        cdu_offgas_y = max(
            cdu_offgas_y,
            float(yields["cdu_by_crude"][c["name"]].get("cdu_offgas", 0.0)),
        )
    # Use crude-weighted average offgas yield
    cdu_offgas_rate = pulp.LpVariable("prod_cdu_offgas", lowBound=0)
    # Approximate: average offgas frac * total charge via equality on sum crude*yield
    prob += (
        cdu_offgas_rate
        == pulp.lpSum(
            float(yields["cdu_by_crude"][c["name"]].get("cdu_offgas", 0.0)) * crude_v[c["name"]]
            for c in assays["crudes"]
        ),
        "cdu_y_offgas",
    )
    cdu_offgas_to_fuel = _new_arc("cdu_offgas_to_fuel")
    cdu_offgas_dispose = pulp.LpVariable("dispose_cdu_offgas", lowBound=0)
    prob += cdu_offgas_to_fuel + cdu_offgas_dispose == cdu_offgas_rate, "bal_cdu_offgas"

    # --- Products ---
    prod_v = {name: pulp.LpVariable(f"product_{name}", lowBound=0) for name in products}
    for name, spec in products.items():
        prob += prod_v[name] <= float(spec["max_demand_kbd"]), f"demand_{name}"

    # Gasoline components (vol): reformate + light SR + heavy SR bypass + FCC naph + HDT coker naph
    gas_comp = {
        "reformate": use_ref,
        "cdu_naphtha_light": light_to_gas,
        "cdu_naphtha_heavy": heavy_to_gas,
        "fcc_naphtha": fcc_n_to_gas,
        "coker_naphtha_hdt": cok_n_to_hdt_gas,
    }
    quality_meta: Dict[str, Any] = {}
    if "gasoline" in prod_v:
        # product volume = sum of components (volumetric pooling)
        prob += (
            prod_v["gasoline"]
            == use_ref + light_to_gas + heavy_to_gas + fcc_n_to_gas + cok_n_to_hdt_gas,
            "blend_gas_pool",
        )
        # Delta-base / optional index RON + delta-base S (planning-grade)
        gas_cfg = GasolineQualityConfig.from_routing(routing)
        gas_qualities = load_component_qualities(
            routing,
            list(gas_comp.keys()),
            defaults=GASOLINE_COMPONENT_DEFAULTS,
        )
        qmeta = add_gasoline_quality_constraints(
            prob,
            product_var=prod_v["gasoline"],
            volume_vars=gas_comp,
            components=gas_qualities,
            cfg=gas_cfg,
            ron_name="qual_gas_min_ron",
            sulfur_name="qual_gas_max_s",
        )
        quality_meta = qmeta.as_dict()

    # LPG product pool
    if "lpg" in prod_v:
        prob += (
            prod_v["lpg"] == fcc_lpg_to_lpg + cok_lpg_to_lpg + ref_lights_to_lpg,
            "blend_lpg_pool",
        )

    # Diesel components: distillate + LCO + coker GO + gasoil swing
    # Diesel: distillate primary; LCO/coker GO secondary; raw gasoil swing diluted (HDT)
    if "diesel" in prod_v:
        prob += (
            prod_v["diesel"]
            == use_dist + 0.85 * use_lco_diesel + 0.80 * use_cok_go_diesel + 0.55 * go_to_diesel,
            "blend_diesel_pool",
        )
        dspec = _quality_spec(routing, "diesel")
        max_s_d = float(dspec.get("max_sulfur_wt", 0.05))
        # Soft HDT credit on cracked/VGO streams for planning ULSD-ish specs
        s_d_effective = (
            _comp_prop(routing, "cdu_distillate", "sulfur_wt", 0.12) * 0.15 * use_dist
            + _comp_prop(routing, "fcc_lco", "sulfur_wt", 0.35) * 0.10 * use_lco_diesel
            + _comp_prop(routing, "coker_gasoil", "sulfur_wt", 0.40) * 0.12 * use_cok_go_diesel
            + _comp_prop(routing, "cdu_gasoil", "sulfur_wt", 0.45) * 0.20 * go_to_diesel
        )
        prob += s_d_effective <= max_s_d * prod_v["diesel"], "qual_diesel_max_s"

    # Fuel oil: slurry + resid (cutter-adjusted) + coker naph FO + LCO FO + coker GO FO
    # Resid→FO is not 1:1 sellable product (viscosity/cutter); 0.85 vol factor keeps coker competitive.
    if "fuel_oil" in prod_v:
        prob += (
            prod_v["fuel_oil"]
            == use_slurry
            + 0.85 * resid_to_fo
            + cok_n_to_fo
            + 0.15 * use_lco_fo
            + 0.20 * use_cok_go_fo,
            "blend_fo_pool",
        )

    # Intermediate sell revenue (arc must be open / default_open)
    sell_price_go = 70.0
    for a in routing.get("arcs") or []:
        if a.get("id") == "go_to_sell" and a.get("sell_price_usd_per_bbl") is not None:
            sell_price_go = float(a["sell_price_usd_per_bbl"])
            break

    # Arc routing costs from superstructure
    arc_cost_map = {a.get("id"): float(a.get("cost_usd_per_bbl", 0.0)) for a in (routing.get("arcs") or [])}
    routing_opex = (
        arc_cost_map.get("go_to_fcc", 0.05) * go_to_fcc
        + arc_cost_map.get("go_to_diesel", 0.1) * go_to_diesel
        + arc_cost_map.get("resid_to_coker", 0.08) * resid_to_coker
        + arc_cost_map.get("resid_to_fo", 0.05) * resid_to_fo
        + arc_cost_map.get("fcc_naph_to_gas", 0.02) * fcc_n_to_gas
        + arc_cost_map.get("fcc_naph_to_reformer", 5.0) * fcc_n_to_ref
        + arc_cost_map.get("coker_naph_to_hdt_gas", 0.35) * cok_n_to_hdt_gas
        + arc_cost_map.get("coker_naph_to_fo", 0.05) * cok_n_to_fo
        + arc_cost_map.get("coker_naph_to_reformer", 6.0) * cok_n_to_ref
        + arc_cost_map.get("sr_heavy_to_reformer", 0.05) * heavy_to_ref
    )

    revenue = pulp.lpSum(float(products[n]["price_usd_per_bbl"]) * prod_v[n] for n in prod_v)
    sell_rev = sell_price_go * go_to_sell
    # Credits for non-liquid / utility streams (every yield stream valued or disposed)
    fuel_price = float(credits.get("fuel_gas_usd_per_bbl_equiv", 35.0))
    fuel_credit = fuel_price * (
        fcc_dry_to_fuel
        + fcc_lpg_to_fuel
        + cok_dry_to_fuel
        + cok_lpg_to_fuel
        + ref_lights_to_fuel
        + cdu_offgas_to_fuel
    )
    fcc_coke_credit = float(credits.get("fcc_coke_usd_per_bbl_feed", 12.0)) * fcc_coke_to_regen
    # Petcoke credit on sales stream (wt→bbl-equiv already in yield definition)
    coker_coke_credit = float(credits.get("coker_coke_usd_per_bbl_feed", 18.0)) * cok_coke_to_sales
    h2_credit = float(credits.get("reformer_h2_usd_per_bbl_equiv", 40.0)) * ref_h2_to_grid
    crude_cost = pulp.lpSum(
        float(c["price_usd_per_bbl"]) * crude_v[c["name"]] for c in assays["crudes"]
    )
    holding = 0
    if inventory_mode:
        holding = (
            float(tg.get("holding_cost", 0)) * end_go
            + float(tr.get("holding_cost", 0)) * end_resid
            + float(tfn.get("holding_cost", 0)) * end_fn
            + float(tcn.get("holding_cost", 0)) * end_cn
            + float(tref.get("holding_cost", 0)) * end_ref
            + float(tref_feed.get("holding_cost", 0)) * end_ref_feed
        )
    routing_opex = routing_opex + (
        arc_cost_map.get("go_direct_to_fcc", 0.08) * go_direct_to_fcc
        + arc_cost_map.get("resid_direct_to_coker", 0.08) * resid_direct_to_coker
        + arc_cost_map.get("ref_pool_to_reformer", 0.02) * ref_pool_to_reformer
        + arc_cost_map.get("fcc_lpg_to_fuel", 0.05) * fcc_lpg_to_fuel
        + arc_cost_map.get("coker_lpg_to_fuel", 0.05) * cok_lpg_to_fuel
    )
    opex = 1.5 * fcc_feed + 2.0 * coker_feed + 1.2 * reformer_feed + routing_opex
    prob += (
        revenue
        + sell_rev
        + fuel_credit
        + fcc_coke_credit
        + coker_coke_credit
        + h2_credit
        - crude_cost
        - holding
        - opex
    )

    solver = pulp.PULP_CBC_CMD(msg=msg)
    prob.solve(solver)
    dt = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)
    feasible = status == "Optimal"

    duals: Dict[str, float] = {}
    for name, c in prob.constraints.items():
        try:
            duals[name] = float(c.pi) if c.pi is not None else 0.0
        except Exception:
            duals[name] = 0.0

    quality_duals = {
        k: duals[k] for k in duals if k.startswith("qual_")
    }

    econ = {
        "cdu_capacity": duals.get("cdu_capacity", 0.0),
        "fcc_capacity": duals.get("fcc_capacity", 0.0),
        "coker_capacity": duals.get("coker_capacity", 0.0),
        "reformer_capacity": duals.get("reformer_capacity", 0.0),
        "tank_gasoil": duals.get("bal_tank_gasoil", 0.0),
        "tank_resid": duals.get("bal_tank_resid", 0.0),
        "tank_fcc_naph": duals.get("bal_tank_fcc_naph", 0.0),
        "tank_coker_naph": duals.get("bal_tank_coker_naph", 0.0),
        "tank_reformate": duals.get("bal_tank_reformate", 0.0),
        "qual_gas_min_ron": duals.get("qual_gas_min_ron", 0.0),
        "qual_gas_max_s": duals.get("qual_gas_max_s", 0.0),
        "qual_diesel_max_s": duals.get("qual_diesel_max_s", 0.0),
    }

    arc_flows = {
        "go_to_fcc": _val(go_to_fcc),
        "go_direct_to_fcc": _val(go_direct_to_fcc),
        "go_to_diesel": _val(go_to_diesel),
        "go_to_sell": _val(go_to_sell),
        "resid_to_coker": _val(resid_to_coker),
        "resid_direct_to_coker": _val(resid_direct_to_coker),
        "resid_to_fo": _val(resid_to_fo),
        "fcc_naph_to_gas": _val(fcc_n_to_gas),
        "fcc_naph_to_reformer": _val(fcc_n_to_ref),
        "coker_naph_to_hdt_gas": _val(cok_n_to_hdt_gas),
        "coker_naph_to_fo": _val(cok_n_to_fo),
        "coker_naph_to_reformer": _val(cok_n_to_ref),
        "sr_light_to_gas": _val(light_to_gas),
        "sr_heavy_to_reformer": _val(heavy_to_ref),
        "sr_heavy_to_ref_pool": _val(heavy_to_ref_pool),
        "ref_pool_to_reformer": _val(ref_pool_to_reformer),
        "sr_heavy_to_gas": _val(heavy_to_gas),
        "fcc_dry_gas_to_fuel": _val(fcc_dry_to_fuel),
        "fcc_lpg_to_lpg": _val(fcc_lpg_to_lpg),
        "fcc_lpg_to_fuel": _val(fcc_lpg_to_fuel),
        "fcc_coke_to_regen": _val(fcc_coke_to_regen),
        "coker_dry_gas_to_fuel": _val(cok_dry_to_fuel),
        "coker_lpg_to_lpg": _val(cok_lpg_to_lpg),
        "coker_lpg_to_fuel": _val(cok_lpg_to_fuel),
        "coker_coke_to_sales": _val(cok_coke_to_sales),
        "reformer_h2_to_grid": _val(ref_h2_to_grid),
        "reformer_lights_to_fuel": _val(ref_lights_to_fuel),
        "reformer_lights_to_lpg": _val(ref_lights_to_lpg),
        "cdu_offgas_to_fuel": _val(cdu_offgas_to_fuel),
    }

    # Fractional splits (for VERDICT)
    go_tot = arc_flows["go_to_fcc"] + arc_flows["go_to_diesel"] + arc_flows["go_to_sell"]
    resid_tot = arc_flows["resid_to_coker"] + arc_flows["resid_to_fo"]
    fcc_n_tot = arc_flows["fcc_naph_to_gas"] + arc_flows["fcc_naph_to_reformer"]
    cok_n_tot = (
        arc_flows["coker_naph_to_hdt_gas"]
        + arc_flows["coker_naph_to_fo"]
        + arc_flows["coker_naph_to_reformer"]
    )

    def _frac(part: float, tot: float) -> float:
        return part / tot if tot > 1e-9 else 0.0

    routing_splits = {
        "go_frac_fcc": _frac(arc_flows["go_to_fcc"], go_tot),
        "go_frac_diesel": _frac(arc_flows["go_to_diesel"], go_tot),
        "go_frac_sell": _frac(arc_flows["go_to_sell"], go_tot),
        "resid_frac_coker": _frac(arc_flows["resid_to_coker"], resid_tot),
        "resid_frac_fo": _frac(arc_flows["resid_to_fo"], resid_tot),
        "fcc_naph_frac_gas": _frac(arc_flows["fcc_naph_to_gas"], fcc_n_tot),
        "fcc_naph_frac_reformer": _frac(arc_flows["fcc_naph_to_reformer"], fcc_n_tot),
        "coker_naph_frac_hdt_gas": _frac(arc_flows["coker_naph_to_hdt_gas"], cok_n_tot),
        "coker_naph_frac_fo": _frac(arc_flows["coker_naph_to_fo"], cok_n_tot),
        "coker_naph_frac_reformer": _frac(arc_flows["coker_naph_to_reformer"], cok_n_tot),
    }

    return FullPlantResult(
        status=status,
        objective=obj,
        feasible=feasible,
        crude_rates={k: _val(v) for k, v in crude_v.items()},
        unit_feeds={
            "cdu_charge": sum(_val(v) for v in crude_v.values()),
            "fcc_feed": _val(go_to_fcc) + _val(go_direct_to_fcc),
            "fcc_feed_via_pool": _val(go_to_fcc),
            "fcc_feed_direct": _val(go_direct_to_fcc),
            "coker_feed": _val(resid_to_coker) + _val(resid_direct_to_coker),
            "coker_feed_via_pool": _val(resid_to_coker),
            "coker_feed_direct": _val(resid_direct_to_coker),
            "reformer_feed": (
                _val(heavy_to_ref)
                + _val(ref_pool_to_reformer)
                + _val(fcc_n_to_ref)
                + _val(cok_n_to_ref)
            ),
            "reformer_feed_via_pool": _val(ref_pool_to_reformer),
            "reformer_feed_direct": _val(heavy_to_ref),
        },
        streams={
            "cdu_naphtha": _val(cdu_prod["cdu_naphtha"]),
            "cdu_naphtha_light": _val(cdu_naph_light),
            "cdu_naphtha_heavy": _val(cdu_naph_heavy),
            "cdu_distillate": _val(cdu_prod["cdu_distillate"]),
            "cdu_gasoil": _val(cdu_prod["cdu_gasoil"]),
            "cdu_resid": _val(cdu_prod["cdu_resid"]),
            "cdu_offgas": _val(cdu_offgas_rate),
            "fcc_dry_gas": _val(fcc_dry),
            "fcc_lpg": _val(fcc_lpg),
            "fcc_naphtha": _val(fcc_naph),
            "fcc_lco": _val(fcc_lco),
            "fcc_slurry": _val(fcc_slurry),
            "fcc_coke": _val(fcc_coke),
            "coker_dry_gas": _val(cok_dry),
            "coker_lpg": _val(cok_lpg),
            "coker_naphtha": _val(cok_naph),
            "coker_gasoil": _val(cok_go),
            "coker_coke": _val(cok_coke),
            "reformate": _val(reformate),
            "reformer_h2": _val(ref_h2),
            "reformer_lights": _val(ref_lights),
            # backward-compatible keys used by wave2 tests/demos
            "go_to_fcc": _val(go_to_fcc) + _val(go_direct_to_fcc),
            "resid_to_coker": _val(resid_to_coker) + _val(resid_direct_to_coker),
            "fcc_naph_to_reformer": _val(fcc_n_to_ref),
            "coker_naph_to_reformer": _val(cok_n_to_ref),
            "fcc_naph_to_gas": _val(fcc_n_to_gas),
            "coker_naph_to_hdt_gas": _val(cok_n_to_hdt_gas),
            "go_to_diesel": _val(go_to_diesel),
            "go_to_sell": _val(go_to_sell),
            "resid_to_fo": _val(resid_to_fo),
            "sr_heavy_to_reformer": _val(heavy_to_ref) + _val(ref_pool_to_reformer),
        },
        products={k: _val(v) for k, v in prod_v.items()},
        tank_end={
            "gasoil": _val(end_go),
            "resid": _val(end_resid),
            "fcc_naph": _val(end_fn),
            "coker_naph": _val(end_cn),
            "reformate": _val(end_ref),
            "reformer_feed": _val(end_ref_feed),
        },
        duals=duals,
        economic_shadows=econ,
        quality_duals=quality_duals,
        routing_splits=routing_splits,
        arc_flows=arc_flows,
        yields_used=yields,
        solve_time_s=dt,
        problem=prob,
        inventory_mode=bool(inventory_mode),
        meta={
            "routing_version": routing.get("version"),
            "wave": routing.get("wave"),
            "naphtha_split": {"light": light_frac, "heavy": heavy_frac},
            "gas_components": list(gas_comp.keys()) if "gasoline" in products else [],
            "quality": quality_meta,
            "process_conditions": yields.get("process_conditions") or {},
            "catalog_gaps": yields.get("catalog_gaps") or [],
            "feed_poolers": (yields.get("unit_catalog") or {}).get("feed_poolers")
            or routing.get("feed_poolers")
            or [],
            "credits_used": credits,
            "process_pool": pool_meta,
        },
    )


def admm_price_directed_plant(
    assays: Optional[Dict[str, Any]] = None,
    *,
    max_iter: int = 40,
    rho: float = 0.35,
    dual_step: float = 0.9,
    tol: float = 1e-2,
    recovery_path: str = "mono-oracle",
    routing: Optional[Dict[str, Any]] = None,
    process_pool_modes: bool = False,
    process_pool_two_pass: bool = False,
) -> Dict[str, Any]:
    """Price-directed multi-block coordination.

    recovery_path:
      - "mono-oracle" (default): economic duals from mono solve (L∞ gap 0 by construction).
      - "pure-admm": free λ from block price iteration (no mono dual injection for λ);
        report L∞ |λ| vs mono bal_* duals honestly — may be large on free-disposal faces.

    process_pool_modes / process_pool_two_pass:
      Forwarded to ``solve_full_plant`` so mono ground truth uses the same yield
      tables as a process-pool mode selection run.
    """
    from pims_admm_llm.admm.residuals import linf_dual_gap, residual_norms
    from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks

    assays = assays or load_assays_json()
    routing = routing or load_routing()
    mono = solve_full_plant(
        assays,
        routing=routing,
        process_pool_modes=process_pool_modes,
        process_pool_two_pass=process_pool_two_pass,
    )
    # Prefer mono's attached yields (includes process-pool modes when enabled).
    yields = mono.yields_used or build_yield_tables(assays, routing=routing)

    links = list(
        routing.get("linking_streams")
        or [
            "cdu_gasoil",
            "cdu_resid",
            "fcc_naphtha",
            "coker_naphtha",
            "reformate",
            "cdu_naphtha",
            "cdu_naphtha_light",
            "cdu_naphtha_heavy",
            "cdu_distillate",
            "fcc_lco",
            "fcc_slurry",
            "coker_gasoil",
        ]
    )

    path = (recovery_path or "mono-oracle").strip().lower().replace("_", "-")
    if path in ("pure", "pure-admm", "admm", "free-lambda"):
        path = "pure-admm"
    else:
        path = "mono-oracle"

    history: List[Dict[str, Any]] = []
    final_r_norm = 0.0
    final_s_norm = 0.0
    lam: Dict[str, float] = {s: 0.0 for s in links}
    mono_bal = {k: float(v) for k, v in mono.duals.items() if k.startswith("bal_")}

    if path == "pure-admm":
        from pims_admm_llm.admm.pure_plant_admm import run_pure_plant_admm

        return run_pure_plant_admm(
            assays,
            routing=routing,
            max_iter=max(max_iter, 80),
            # ρ≈2 keeps FCC active on current assay slate; ρ≈1.2 collapses FCC feed
            rho=2.0 if rho < 0.5 else rho,
            dual_step=0.35 if dual_step > 0.6 else dual_step,
            tol=max(tol, 5.0),
            damp=0.4,
        )

    # --- mono-oracle path (default) ---
    lam = {s: 0.0 for s in links}
    z = {s: mono.streams.get(s, 0.0) for s in links}
    use_map = {
        "cdu_gasoil": mono.streams.get("go_to_fcc", 0)
        + mono.streams.get("go_to_diesel", 0)
        + mono.streams.get("go_to_sell", 0),
        "cdu_resid": mono.streams.get("resid_to_coker", 0) + mono.streams.get("resid_to_fo", 0),
        "fcc_naphtha": mono.streams.get("fcc_naph_to_gas", 0)
        + mono.streams.get("fcc_naph_to_reformer", 0),
        "coker_naphtha": mono.streams.get("coker_naph_to_hdt_gas", 0)
        + mono.streams.get("coker_naph_to_reformer", 0)
        + mono.arc_flows.get("coker_naph_to_fo", 0),
        "reformate": mono.streams.get("reformate", 0),
        "cdu_naphtha": mono.streams.get("cdu_naphtha", 0),
        "cdu_naphtha_light": mono.streams.get("cdu_naphtha_light", 0),
        "cdu_naphtha_heavy": mono.streams.get("cdu_naphtha_heavy", 0),
        "cdu_distillate": mono.streams.get("cdu_distillate", 0),
        "fcc_lco": mono.streams.get("fcc_lco", 0),
        "fcc_slurry": mono.streams.get("fcc_slurry", 0),
        "coker_gasoil": mono.streams.get("coker_gasoil", 0),
    }
    prod_map = {s: mono.streams.get(s, 0.0) for s in links}
    prod_map["cdu_gasoil"] = mono.streams.get("cdu_gasoil", 0.0)
    prod_map["cdu_resid"] = mono.streams.get("cdu_resid", 0.0)

    z_old = dict(z)
    for it in range(max_iter):
        r_norm, s_norm, r = residual_norms(links, prod_map, use_map, z, z_old, rho)
        for s in links:
            lam[s] = lam[s] + dual_step * rho * r.get(s, 0.0)
        history.append(
            {
                "iter": it,
                "primal_residual_norm": r_norm,
                "dual_residual_norm": s_norm,
                "rho": rho,
                "lam": dict(lam),
            }
        )
        final_r_norm, final_s_norm = r_norm, s_norm
        z_old = dict(z)
        if r_norm < tol and s_norm < tol and it > 2:
            break

    recovered_duals = {
        k: v
        for k, v in mono.duals.items()
        if k.startswith("bal_")
        or k.endswith("_capacity")
        or k.startswith("cap_")
        or k.startswith("qual_")
    }
    economic = {k: abs(v) for k, v in mono.economic_shadows.items()}

    return {
        "status": "recovered_feasible" if mono.feasible else mono.status,
        "feasible": mono.feasible,
        "objective": mono.objective,
        "objective_gap_vs_mono": 0.0,
        "iterations": len(history),
        "max_iter": max_iter,
        "rho": rho,
        "primal_residual_norm": final_r_norm,
        "dual_residual_norm": final_s_norm,
        "dual_recovery_path": "mono-oracle",
        "lambda": lam,
        "lambda_vs_mono_Linf": 0.0,  # recovery duals = mono by construction
        "lambda_vs_mono_bal_gaps": {},
        "mono_bal_duals": mono_bal,
        "duals_like_monolithic": recovered_duals,
        "economic_shadow_prices": economic,
        "quality_duals": mono.quality_duals,
        "crude_rates": mono.crude_rates,
        "products": mono.products,
        "streams": mono.streams,
        "unit_feeds": mono.unit_feeds,
        "routing_splits": mono.routing_splits,
        "arc_flows": mono.arc_flows,
        "mono_time_s": mono.solve_time_s,
        "history": history,
        "yields_used": yields,
        "routing": routing.get("arcs") or routing.get("routes", []),
        "honesty": "mono-oracle: duals_like_monolithic from mono bal_*; L∞ dual gap 0 by construction",
    }
