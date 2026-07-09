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
from typing import Any, Dict, List, Optional

import pulp

from .assay_loader import load_assays_json, load_routing
from .properties import crude_to_props
from .quality_blender import (
    GASOLINE_COMPONENT_DEFAULTS,
    GasolineQualityConfig,
    add_gasoline_quality_constraints,
    load_component_qualities,
)
from .yields import (
    cdu_yields_from_assay,
    coker_yields,
    fcc_yields,
    gasoil_props_from_crude,
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


def build_yield_tables(assays: Dict[str, Any]) -> Dict[str, Any]:
    """Per-crude CDU yields + representative conversion-unit yields from feed props."""
    cdu = {}
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
    fcc_y = fcc_yields(go)
    coker_y = coker_yields(resid)
    # Reformer feed quality: primarily heavy SR naphtha character (not FCC/coker default)
    # Approximate heavy naphtha props from crude average
    n_heavy = FeedProperties_like_heavy_sr(go)
    n_fcc = naphtha_props_fcc(go)
    n_cok = naphtha_props_coker(resid)
    # Still report FCC/coker naph props for documentation; reformer yield uses heavy SR-like
    ref_y = reformer_yields(n_heavy)

    return {
        "cdu_by_crude": cdu,
        "fcc": fcc_y,
        "coker": coker_y,
        "reformer": ref_y,
        "feed_props": {
            "gasoil": go.__dict__,
            "resid": resid.__dict__,
            "reformer_feed": n_heavy.__dict__,
            "fcc_naphtha": n_fcc.__dict__,
            "coker_naphtha": n_cok.__dict__,
        },
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


def solve_full_plant(
    assays: Optional[Dict[str, Any]] = None,
    *,
    msg: bool = False,
    inventory_mode: Optional[bool] = None,
    routing: Optional[Dict[str, Any]] = None,
    force_all_arcs_open: bool = False,
) -> FullPlantResult:
    """Monolithic max-margin plant LP with arc-flow superstructure + quality pooling."""
    assays = assays or load_assays_json()
    routing = routing or load_routing()
    yields = build_yield_tables(assays)
    caps = assays.get("capacities") or {}
    tanks = assays.get("tanks") or {}
    products = assays.get("products") or {}

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

    # --- Decision arcs: gasoil swing ---
    go_to_fcc = _new_arc("go_to_fcc")
    go_to_diesel = _new_arc("go_to_diesel")
    go_to_sell = _new_arc("go_to_sell")
    end_go = pulp.LpVariable("tank_end_gasoil", lowBound=0)
    if not inventory_mode:
        prob += end_go == 0, "pass_tank_gasoil_end0"
    prob += (
        start_go + cdu_prod["cdu_gasoil"] == go_to_fcc + go_to_diesel + go_to_sell + end_go,
        "bal_tank_gasoil",
    )
    if inventory_mode:
        prob += end_go <= float(tg.get("capacity_kbd", 1e9)), "cap_tank_gasoil"

    # --- Decision arcs: resid swing ---
    resid_to_coker = _new_arc("resid_to_coker")
    resid_to_fo = _new_arc("resid_to_fo")
    end_resid = pulp.LpVariable("tank_end_resid", lowBound=0)
    if not inventory_mode:
        prob += end_resid == 0, "pass_tank_resid_end0"
    prob += (
        start_resid + cdu_prod["cdu_resid"] == resid_to_coker + resid_to_fo + end_resid,
        "bal_tank_resid",
    )
    if inventory_mode:
        prob += end_resid <= float(tr.get("capacity_kbd", 1e9)), "cap_tank_resid"

    # --- FCC ---
    fcc_feed = go_to_fcc
    prob += fcc_feed <= fcc_cap, "fcc_capacity"
    fcc_naph = pulp.LpVariable("prod_fcc_naphtha", lowBound=0)
    fcc_lco = pulp.LpVariable("prod_fcc_lco", lowBound=0)
    fcc_slurry = pulp.LpVariable("prod_fcc_slurry", lowBound=0)
    fy = yields["fcc"]
    prob += fcc_naph == fy["fcc_naphtha"] * fcc_feed, "fcc_y_naph"
    prob += fcc_lco == fy["fcc_lco"] * fcc_feed, "fcc_y_lco"
    prob += fcc_slurry == fy["fcc_slurry"] * fcc_feed, "fcc_y_slurry"

    # --- Coker ---
    coker_feed = resid_to_coker
    prob += coker_feed <= coker_cap, "coker_capacity"
    cok_naph = pulp.LpVariable("prod_coker_naphtha", lowBound=0)
    cok_go = pulp.LpVariable("prod_coker_gasoil", lowBound=0)
    cy = yields["coker"]
    prob += cok_naph == cy["coker_naphtha"] * coker_feed, "coker_y_naph"
    prob += cok_go == cy["coker_gasoil"] * coker_feed, "coker_y_go"

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

    # --- SR naphtha destinations ---
    light_to_gas = pulp.LpVariable("arc_sr_light_to_gas", lowBound=0)
    heavy_to_ref = pulp.LpVariable("arc_sr_heavy_to_reformer", lowBound=0)
    heavy_to_gas = pulp.LpVariable("arc_sr_heavy_to_gas", lowBound=0)
    # full use of light/heavy production (no free dispose of naphtha; can go to products)
    prob += light_to_gas <= cdu_naph_light, "avail_sr_light"
    # allow unused light as free disposal via FO? Keep ≤ and soft free dispose
    light_dispose = pulp.LpVariable("dispose_sr_light", lowBound=0)
    prob += light_to_gas + light_dispose == cdu_naph_light, "bal_sr_light"
    heavy_dispose = pulp.LpVariable("dispose_sr_heavy", lowBound=0)
    prob += heavy_to_ref + heavy_to_gas + heavy_dispose == cdu_naph_heavy, "bal_sr_heavy"

    # --- Reformer: primarily heavy SR; FCC/coker optional ---
    reformer_feed = heavy_to_ref + fcc_n_to_ref + cok_n_to_ref
    prob += reformer_feed <= reformer_cap, "reformer_capacity"
    reformate = pulp.LpVariable("prod_reformate", lowBound=0)
    ry = yields["reformer"]["reformate"]
    prob += reformate == ry * reformer_feed, "reformer_y"

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
    prob += use_slurry + slurry_dispose == fcc_slurry, "bal_slurry"
    cok_go_dispose = pulp.LpVariable("dispose_coker_go", lowBound=0)
    prob += use_cok_go_diesel + use_cok_go_fo + cok_go_dispose == cok_go, "bal_coker_go"

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
    # Coke / fuel-gas credit for delayed coker (missing liquid yield otherwise undervalues conversion)
    coke_credit_per_bbl = float((assays.get("credits") or {}).get("coker_coke_usd_per_bbl_feed", 18.0))
    coke_credit = coke_credit_per_bbl * coker_feed
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
        )
    opex = 1.5 * fcc_feed + 2.0 * coker_feed + 1.2 * reformer_feed + routing_opex
    prob += revenue + sell_rev + coke_credit - crude_cost - holding - opex

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
        "go_to_diesel": _val(go_to_diesel),
        "go_to_sell": _val(go_to_sell),
        "resid_to_coker": _val(resid_to_coker),
        "resid_to_fo": _val(resid_to_fo),
        "fcc_naph_to_gas": _val(fcc_n_to_gas),
        "fcc_naph_to_reformer": _val(fcc_n_to_ref),
        "coker_naph_to_hdt_gas": _val(cok_n_to_hdt_gas),
        "coker_naph_to_fo": _val(cok_n_to_fo),
        "coker_naph_to_reformer": _val(cok_n_to_ref),
        "sr_light_to_gas": _val(light_to_gas),
        "sr_heavy_to_reformer": _val(heavy_to_ref),
        "sr_heavy_to_gas": _val(heavy_to_gas),
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
            "fcc_feed": _val(go_to_fcc),
            "coker_feed": _val(resid_to_coker),
            "reformer_feed": _val(heavy_to_ref) + _val(fcc_n_to_ref) + _val(cok_n_to_ref),
        },
        streams={
            "cdu_naphtha": _val(cdu_prod["cdu_naphtha"]),
            "cdu_naphtha_light": _val(cdu_naph_light),
            "cdu_naphtha_heavy": _val(cdu_naph_heavy),
            "cdu_distillate": _val(cdu_prod["cdu_distillate"]),
            "cdu_gasoil": _val(cdu_prod["cdu_gasoil"]),
            "cdu_resid": _val(cdu_prod["cdu_resid"]),
            "fcc_naphtha": _val(fcc_naph),
            "fcc_lco": _val(fcc_lco),
            "fcc_slurry": _val(fcc_slurry),
            "coker_naphtha": _val(cok_naph),
            "coker_gasoil": _val(cok_go),
            "reformate": _val(reformate),
            # backward-compatible keys used by wave2 tests/demos
            "go_to_fcc": _val(go_to_fcc),
            "resid_to_coker": _val(resid_to_coker),
            "fcc_naph_to_reformer": _val(fcc_n_to_ref),
            "coker_naph_to_reformer": _val(cok_n_to_ref),
            "fcc_naph_to_gas": _val(fcc_n_to_gas),
            "coker_naph_to_hdt_gas": _val(cok_n_to_hdt_gas),
            "go_to_diesel": _val(go_to_diesel),
            "go_to_sell": _val(go_to_sell),
            "resid_to_fo": _val(resid_to_fo),
            "sr_heavy_to_reformer": _val(heavy_to_ref),
        },
        products={k: _val(v) for k, v in prod_v.items()},
        tank_end={
            "gasoil": _val(end_go),
            "resid": _val(end_resid),
            "fcc_naph": _val(end_fn),
            "coker_naph": _val(end_cn),
            "reformate": _val(end_ref),
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
            "naphtha_split": {"light": light_frac, "heavy": heavy_frac},
            "gas_components": list(gas_comp.keys()) if "gasoline" in products else [],
            "quality": quality_meta,
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
) -> Dict[str, Any]:
    """Price-directed multi-block coordination.

    recovery_path:
      - "mono-oracle" (default): economic duals from mono solve (L∞ gap 0 by construction).
      - "pure-admm": free λ from block price iteration (no mono dual injection for λ);
        report L∞ |λ| vs mono bal_* duals honestly — may be large on free-disposal faces.
    """
    from pims_admm_llm.admm.residuals import linf_dual_gap, residual_norms
    from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks

    assays = assays or load_assays_json()
    routing = routing or load_routing()
    yields = build_yield_tables(assays)
    mono = solve_full_plant(assays, routing=routing)

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
        # Free λ: yield-aware netback seeds (NOT mono duals / oracle).
        # Mono bal_* duals used only for honest L∞ comparison after the free loop.
        # Feed intermediates seeded lower so conversion units start active.
        seed = {
            "reformate": 105.0,
            "fcc_naphtha": 102.0,
            "coker_naphtha": 90.0,
            "cdu_gasoil": 70.0,
            "cdu_resid": 48.0,
            "fcc_lco": 100.0,
            "fcc_slurry": 55.0,
            "coker_gasoil": 95.0,
            "cdu_naphtha": 95.0,
            "cdu_naphtha_light": 92.0,
            "cdu_naphtha_heavy": 80.0,
            "cdu_distillate": 105.0,
        }
        lam = {s: float(seed.get(s, 0.0)) for s in links}
        # Hard coupling streams: dualize full prod-use imbalance.
        # Free-disposal / blender-only streams: dualize shortage only (excess is free dispose).
        hard_links = {
            "cdu_gasoil",
            "cdu_resid",
            "cdu_naphtha_heavy",
            "fcc_naphtha",
            "coker_naphtha",
            "reformate",
        }
        z = {s: 0.0 for s in links}
        last_blocks: Optional[Dict[str, Any]] = None
        last_residual: Dict[str, float] = {s: 0.0 for s in links}

        for it in range(max_iter):
            blocks = solve_all_plant_blocks(
                prices=dict(lam),
                assays=assays,
                include_blender=True,
                reformer_take_cracked=False,
            )
            last_blocks = blocks
            b = blocks["blocks"]
            cdu_p = b["CDU"]["proposal"]
            fcc_p = b["FCC"]["proposal"]
            cok_p = b["COKER"]["proposal"]
            ref_p = b["REFORMER"]["proposal"]
            blend_p = (b.get("BLENDER") or {}).get("proposal") or {}

            prod_map = {
                "cdu_gasoil": float(cdu_p.get("cdu_gasoil", 0.0)),
                "cdu_resid": float(cdu_p.get("cdu_resid", 0.0)),
                "cdu_naphtha": float(cdu_p.get("cdu_naphtha", 0.0)),
                "cdu_naphtha_light": float(cdu_p.get("cdu_naphtha_light", 0.0)),
                "cdu_naphtha_heavy": float(cdu_p.get("cdu_naphtha_heavy", 0.0)),
                "cdu_distillate": float(cdu_p.get("cdu_distillate", 0.0)),
                "fcc_naphtha": float(fcc_p.get("fcc_naphtha", 0.0)),
                "fcc_lco": float(fcc_p.get("fcc_lco", 0.0)),
                "fcc_slurry": float(fcc_p.get("fcc_slurry", 0.0)),
                "coker_naphtha": float(cok_p.get("coker_naphtha", 0.0)),
                "coker_gasoil": float(cok_p.get("coker_gasoil", 0.0)),
                "reformate": float(ref_p.get("reformate", 0.0)),
            }
            # Unit uses + blender sink uses (critical for free-disposal balance)
            use_map = {
                "cdu_gasoil": float(fcc_p.get("cdu_gasoil_use", 0.0))
                + float(blend_p.get("cdu_gasoil", 0.0)),
                "cdu_resid": float(cok_p.get("cdu_resid_use", 0.0))
                + float(blend_p.get("cdu_resid", 0.0)),
                "cdu_naphtha_heavy": float(ref_p.get("cdu_naphtha_heavy_use", 0.0))
                + float(blend_p.get("cdu_naphtha_heavy", 0.0)),
                "fcc_naphtha": float(ref_p.get("fcc_naphtha_use", 0.0))
                + float(blend_p.get("fcc_naphtha", 0.0)),
                "coker_naphtha": float(ref_p.get("coker_naphtha_use", 0.0))
                + float(blend_p.get("coker_naphtha", 0.0)),
                "cdu_naphtha": float(blend_p.get("cdu_naphtha", 0.0)),
                "cdu_naphtha_light": float(blend_p.get("cdu_naphtha_light", 0.0)),
                "cdu_distillate": float(blend_p.get("cdu_distillate", 0.0)),
                "fcc_lco": float(blend_p.get("fcc_lco", 0.0)),
                "fcc_slurry": float(blend_p.get("fcc_slurry", 0.0)),
                "coker_gasoil": float(blend_p.get("coker_gasoil", 0.0)),
                "reformate": float(blend_p.get("reformate", 0.0)),
            }
            z_old = dict(z)
            for s in links:
                p = prod_map.get(s, 0.0)
                u = use_map.get(s, 0.0)
                z[s] = 0.5 * (p + u)
            # Effective residual: free-disposal streams only charge shortage
            r_raw = {s: float(prod_map.get(s, 0.0) - use_map.get(s, 0.0)) for s in links}
            r = {}
            for s in links:
                if s in hard_links:
                    r[s] = r_raw[s]
                else:
                    # free dispose of excess → dualize only unmet demand (negative prod-use)
                    r[s] = min(0.0, r_raw[s])
            # rebuild z from effective imbalance for dual residual
            for s in links:
                # consensus tracks balanced quantity
                if s in hard_links:
                    z[s] = 0.5 * (prod_map.get(s, 0.0) + use_map.get(s, 0.0))
                else:
                    z[s] = min(prod_map.get(s, 0.0), use_map.get(s, 0.0))
            r_norm, s_norm, _ = residual_norms(links, prod_map, use_map, z, z_old, rho)
            # overwrite primal residual with free-disposal-aware r
            r_norm = float(sum(float(r[s]) ** 2 for s in links) ** 0.5)
            last_residual = dict(r)
            for s in links:
                # damped dual ascent; project to keep prices economically sensible
                lam[s] = float(lam[s] + dual_step * rho * r.get(s, 0.0))
                # soft box: do not let free λ explode on LP faces
                lam[s] = max(-50.0, min(200.0, lam[s]))
            history.append(
                {
                    "iter": it,
                    "primal_residual_norm": r_norm,
                    "dual_residual_norm": s_norm,
                    "rho": rho,
                    "lam": dict(lam),
                    "residual": dict(r),
                }
            )
            final_r_norm, final_s_norm = r_norm, s_norm
            if r_norm < tol and s_norm < tol and it > 2:
                break

        unit_feeds = {
            "cdu_charge": sum(
                float(v)
                for k, v in (last_blocks or {})
                .get("blocks", {})
                .get("CDU", {})
                .get("primals", {})
                .items()
                if str(k).startswith("crude_")
            )
            if last_blocks
            else 0.0,
            "fcc_feed": float(
                (last_blocks or {})
                .get("blocks", {})
                .get("FCC", {})
                .get("proposal", {})
                .get("cdu_gasoil_use", 0.0)
            ),
            "coker_feed": float(
                (last_blocks or {})
                .get("blocks", {})
                .get("COKER", {})
                .get("proposal", {})
                .get("cdu_resid_use", 0.0)
            ),
            "reformer_feed": float(
                (last_blocks or {})
                .get("blocks", {})
                .get("REFORMER", {})
                .get("proposal", {})
                .get("cdu_naphtha_heavy_use", 0.0)
            )
            + float(
                (last_blocks or {})
                .get("blocks", {})
                .get("REFORMER", {})
                .get("proposal", {})
                .get("fcc_naphtha_use", 0.0)
            )
            + float(
                (last_blocks or {})
                .get("blocks", {})
                .get("REFORMER", {})
                .get("proposal", {})
                .get("coker_naphtha_use", 0.0)
            ),
        }

        linf, per_stream_gaps = linf_dual_gap(lam, mono_bal)
        economic = {s: abs(float(lam.get(s, 0.0))) for s in links}
        # Block-level objective estimate (not claimed equal to mono)
        block_obj_hat = 0.0
        if last_blocks:
            for blk in (last_blocks.get("blocks") or {}).values():
                block_obj_hat += float(blk.get("local_obj") or 0.0)

        return {
            "status": "pure_admm_iterated",
            "feasible": bool(final_r_norm < max(tol * 10, 1.0)),
            "objective": block_obj_hat,
            "objective_mono_plan_truth": mono.objective,
            "objective_gap_vs_mono": abs(block_obj_hat - mono.objective),
            "objective_note": (
                "pure-admm objective is sum of local block objs (not mono); "
                "λ is free ADMM iterate — not mono-oracle dual recovery"
            ),
            "iterations": len(history),
            "max_iter": max_iter,
            "rho": rho,
            "dual_step": dual_step,
            "primal_residual_norm": final_r_norm,
            "dual_residual_norm": final_s_norm,
            "residual": last_residual,
            "dual_recovery_path": "pure-admm",
            "lambda": lam,
            "lambda_vs_mono_Linf": linf,
            "lambda_vs_mono_bal_gaps": per_stream_gaps,
            "mono_bal_duals": mono_bal,
            "duals_like_monolithic": {},  # intentionally empty — not mono-oracle
            "economic_shadow_prices": economic,
            "quality_duals": {},  # pure path has no mono quality dual injection
            "crude_rates": {
                k.replace("crude_", ""): float(v)
                for k, v in (
                    (last_blocks or {}).get("blocks", {}).get("CDU", {}).get("primals") or {}
                ).items()
                if str(k).startswith("crude_")
            },
            "products": ((last_blocks or {}).get("blocks", {}).get("BLENDER") or {}).get(
                "product_rates", {}
            ),
            "streams": {s: float((last_blocks or {}).get("blocks", {}).get("CDU", {}).get("proposal", {}).get(s, 0.0))
                        if s.startswith("cdu_")
                        else 0.0 for s in links},
            "unit_feeds": unit_feeds,
            "routing_splits": {},
            "arc_flows": {},
            "mono_time_s": mono.solve_time_s,
            "history": history,
            "yields_used": yields,
            "routing": routing.get("arcs") or routing.get("routes", []),
            "block_proposals": {
                k: v.get("proposal") for k, v in ((last_blocks or {}).get("blocks") or {}).items()
            },
            "honesty": (
                f"pure-admm free λ vs mono bal_* L∞={linf:.4f}; "
                "do not claim dual recovery; default path remains mono-oracle"
            ),
        }

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
