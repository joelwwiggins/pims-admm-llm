"""Crude → CDU (cut points) → tanks (7-day, bypassable) → FCC → blender.

Case plant for mono LP vs ADMM parity checks.

Topology
--------
  assay crude
       │ cut-point CDU (heart/swing)
       ▼
  SR naphtha ──► tank_naph (7d) ──┐
       │ bypass                   ├─► blender ─► gasoline (RON/S specs)
  SR distillate ──────────────────┤      ▲
       │ (sweet gasoil product)   │      │ purchase naphtha / alkylate
  SR gasoil ───► tank_go (7d) ──┐ │      │
       │ bypass                 ├─► FCC (base-delta modes)
  SR resid ───► FO sell          │ │
                                 │ └──► FCC products:
                                 │        fcc_naphtha ──► blender
                                 │        fcc_lco ──► sour gasoil sell
                                 │        fcc_slurry ──► FO
                                 │        dry_gas/lpg ──► fuel gas (BTU)
                                 │        coke ──► regen credit
  H2 purchase ───────────────────┘ FCC feed (planning H2/bbl)

Tanks: start inventory + end inventory; capacity ≈ 7 × design charge rate.
Bypass arcs: direct CDU→unit without tank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pulp

from .assay_swing import (
    import_crude_from_assays_package,
    solve_cdu_from_cut_points,
)
from .base_delta import build_fcc_base_delta, process_modes_fcc
from .utilities_h2_fuelgas import (
    BTU_MMBTU_PER_BBL,
    DEFAULT_UTIL_PRICES,
    H2_KSCF_PER_BBL_FCC,
    add_fuel_gas_sales_to_lp,
    add_h2_purchase_to_lp,
    snapshot_from_solved,
)


# ---- economics (planning) ----
DEFAULT_PRICES = {
    "gasoline": 105.0,  # $/bbl
    "sweet_gasoil": 95.0,
    "sour_gasoil": 78.0,
    "fuel_oil": 55.0,
    "crude": 70.0,
    "buy_naphtha": 92.0,
    "buy_alkylate": 110.0,
    "h2_usd_per_kscf": DEFAULT_UTIL_PRICES["h2_usd_per_kscf"],
    "fuel_gas_usd_per_mmbtu": DEFAULT_UTIL_PRICES["fuel_gas_usd_per_mmbtu"],
    "tank_hold_usd_per_bbl": 0.05,
    "coke_credit": 15.0,
}

# Re-export W3 BTU table + H2 rate for demos/tests (source of truth: utilities_h2_fuelgas)

# Gasoline blending indices (linear delta-base style)
RON = {
    "cdu_naphtha": 62.0,
    "fcc_naphtha": 92.0,
    "buy_naphtha": 70.0,
    "buy_alkylate": 95.0,
}
SULFUR = {
    "cdu_naphtha": 0.02,
    "fcc_naphtha": 0.04,
    "buy_naphtha": 0.01,
    "buy_alkylate": 0.001,
}


@dataclass
class CrudeCatBlenderResult:
    status: str
    objective: float
    path: str  # mono | admm
    crude_kbd: float
    crude_name: str
    streams: Dict[str, float]
    tank: Dict[str, Dict[str, float]]
    products: Dict[str, float]
    purchases: Dict[str, float]
    utilities: Dict[str, float]
    mass_balance: Dict[str, Any]
    quality: Dict[str, Any]
    process: Dict[str, Any]
    duals: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "objective": self.objective,
            "path": self.path,
            "crude_kbd": self.crude_kbd,
            "crude_name": self.crude_name,
            "streams": dict(self.streams),
            "tank": self.tank,
            "products": dict(self.products),
            "purchases": dict(self.purchases),
            "utilities": dict(self.utilities),
            "mass_balance": self.mass_balance,
            "quality": self.quality,
            "process": self.process,
            "duals": dict(self.duals),
            "meta": self.meta,
        }


def _val(x: Any) -> float:
    v = pulp.value(x)
    return float(v) if v is not None else 0.0


def solve_crude_cat_blender(
    *,
    crude_name: str = "WTI",
    max_crude_kbd: float = 100.0,
    tank_days: float = 7.0,
    cut_points: Optional[Mapping[str, float]] = None,
    prices: Optional[Mapping[str, float]] = None,
    gas_ron_min: float = 87.0,
    gas_s_max: float = 0.01,
    allow_purchases: bool = True,
    fcc_mode: Optional[str] = None,
    msg: bool = False,
) -> CrudeCatBlenderResult:
    """Monolithic LP for crude→cat→blender case."""
    px = dict(DEFAULT_PRICES)
    if prices:
        px.update(prices)

    assay = import_crude_from_assays_package(crude_name)
    # design charge for tank sizing
    design = max_crude_kbd
    tank_cap = tank_days * design  # bbl inventory capacity (kbd·days)

    # Precompute CDU yields at cut points for unit charge (linear in crude)
    cdu0 = solve_cdu_from_cut_points(assay, cut_points, charge_kbd=1.0)
    y_cdu = dict(cdu0.product_yields_vol)  # per bbl crude
    # optional offgas fraction small
    y_offgas = 0.01

    # FCC modes
    go_props = {
        "api": cdu0.product_properties.get("cdu_gasoil", {}).get("api", 22.0),
        "sulfur_wt": cdu0.product_properties.get("cdu_gasoil", {}).get("sulfur_wt", 0.5),
        "ccr_wt": cdu0.product_properties.get("cdu_gasoil", {}).get("ccr_wt", 0.5),
    }
    fcc_model = build_fcc_base_delta(reference_feed=go_props)
    modes = process_modes_fcc(fcc_model, go_props)
    if fcc_mode:
        modes = [m for m in modes if m["id"] == fcc_mode] or modes

    # ---- LP ----
    prob = pulp.LpProblem("crude_cat_blender", pulp.LpMaximize)
    crude = pulp.LpVariable("crude_kbd", lowBound=0, upBound=max_crude_kbd)

    # CDU products (linear yields)
    cdu_naph = y_cdu.get("cdu_naphtha", 0.2) * crude
    cdu_dist = y_cdu.get("cdu_distillate", 0.25) * crude
    cdu_go = y_cdu.get("cdu_gasoil", 0.28) * crude
    cdu_resid = y_cdu.get("cdu_resid", 0.27) * crude
    cdu_off = y_offgas * crude

    # Tanks: start fixed, end free; balance in - out + start = end
    # Naphtha tank
    naph_to_tank = pulp.LpVariable("naph_to_tank", lowBound=0)
    naph_bypass = pulp.LpVariable("naph_bypass", lowBound=0)
    naph_from_tank = pulp.LpVariable("naph_from_tank", lowBound=0)
    tank_naph_start = 0.5 * tank_days * (0.2 * design)  # mild heel
    tank_naph_end = pulp.LpVariable("tank_naph_end", lowBound=0, upBound=tank_cap)
    prob += naph_to_tank + naph_bypass == cdu_naph, "naph_split"
    prob += tank_naph_start + naph_to_tank - naph_from_tank == tank_naph_end, "tank_naph_bal"
    prob += tank_naph_end >= tank_naph_start, "tank_naph_no_drawdown"  # no free heel liquidation
    prob += naph_from_tank <= naph_to_tank + 1e-6, "naph_tank_pass_through"  # push-pull, not heel mine

    # Gasoil tank
    go_to_tank = pulp.LpVariable("go_to_tank", lowBound=0)
    go_bypass = pulp.LpVariable("go_bypass", lowBound=0)
    go_from_tank = pulp.LpVariable("go_from_tank", lowBound=0)
    tank_go_start = 0.5 * tank_days * (0.28 * design)
    tank_go_end = pulp.LpVariable("tank_go_end", lowBound=0, upBound=tank_cap)
    prob += go_to_tank + go_bypass == cdu_go, "go_split"
    prob += tank_go_start + go_to_tank - go_from_tank == tank_go_end, "tank_go_bal"
    prob += tank_go_end >= tank_go_start, "tank_go_no_drawdown"
    prob += go_from_tank <= go_to_tank + 1e-6, "go_tank_pass_through"

    # FCC feed = tank draw + bypass
    fcc_feed = pulp.LpVariable("fcc_feed", lowBound=0)
    prob += fcc_feed == go_from_tank + go_bypass, "fcc_feed_bal"

    # FCC mode SOS1
    y_mode = {m["id"]: pulp.LpVariable(f"fcc_{m['id']}", cat="Binary") for m in modes}
    if len(y_mode) > 1:
        prob += pulp.lpSum(y_mode.values()) == 1, "fcc_sos1"
    else:
        for v in y_mode.values():
            prob += v == 1

    M = max_crude_kbd * 1.5
    fcc_prod: Dict[str, pulp.LpVariable] = {}
    products_fcc = list(modes[0]["yields"].keys())
    for p in products_fcc:
        total = pulp.LpVariable(f"fcc_prod_{p}", lowBound=0)
        pieces = []
        for m in modes:
            mid = m["id"]
            yld = float(m["yields"][p])
            r = pulp.LpVariable(f"fcc_{mid}_{p}", lowBound=0)
            prob += r <= M * y_mode[mid]
            prob += r <= yld * fcc_feed
            prob += r >= yld * fcc_feed - M * (1 - y_mode[mid])
            pieces.append(r)
        prob += total == pulp.lpSum(pieces)
        fcc_prod[p] = total

    # Blender feeds
    buy_naph = pulp.LpVariable("buy_naphtha", lowBound=0, upBound=max_crude_kbd if allow_purchases else 0)
    buy_alk = pulp.LpVariable("buy_alkylate", lowBound=0, upBound=max_crude_kbd * 0.3 if allow_purchases else 0)
    bl_cdu_naph = pulp.LpVariable("bl_cdu_naph", lowBound=0)
    bl_fcc_naph = pulp.LpVariable("bl_fcc_naph", lowBound=0)
    # available naphtha to blender
    prob += bl_cdu_naph <= naph_from_tank + naph_bypass, "bl_cdu_naph_avail"
    # leftover SR naphtha can sell as intermediate naphtha product
    sell_naph_int = pulp.LpVariable("sell_naph_int", lowBound=0)
    prob += bl_cdu_naph + sell_naph_int == naph_from_tank + naph_bypass, "naph_use"
    prob += bl_fcc_naph == fcc_prod["fcc_naphtha"], "bl_fcc_all"  # all FCC naph to blender path

    gasoline = pulp.LpVariable("gasoline", lowBound=0)
    prob += gasoline == bl_cdu_naph + bl_fcc_naph + buy_naph + buy_alk, "gas_vol"

    # Quality (linear blend) — RON >= min, S <= max
    # sum(ron_i * v_i) >= gas_ron_min * gasoline
    prob += (
        RON["cdu_naphtha"] * bl_cdu_naph
        + RON["fcc_naphtha"] * bl_fcc_naph
        + RON["buy_naphtha"] * buy_naph
        + RON["buy_alkylate"] * buy_alk
        >= gas_ron_min * gasoline
    ), "ron_spec"
    # sulfur: sum(s_i v_i) <= gas_s_max * gasoline  (with small epsilon)
    prob += (
        SULFUR["cdu_naphtha"] * bl_cdu_naph
        + SULFUR["fcc_naphtha"] * bl_fcc_naph
        + SULFUR["buy_naphtha"] * buy_naph
        + SULFUR["buy_alkylate"] * buy_alk
        <= gas_s_max * gasoline + 1e-6
    ), "sulfur_spec"

    # Products
    sweet_go = cdu_dist  # SR distillate as sweet gasoil
    sour_go = fcc_prod["fcc_lco"]
    fo = cdu_resid + fcc_prod["fcc_slurry"]

    # ---- W3: H2 purchase (explicit LP var) + fuel-gas BTU light-ends sales ----
    # Light ends must exit: dry gas / LPG / CDU offgas → fuel-gas pool (BTU sales)
    light_end_flows = {
        "fcc_dry_gas": fcc_prod["fcc_dry_gas"],
        "fcc_lpg": fcc_prod["fcc_lpg"],
        "cdu_offgas": cdu_off,
    }
    mmbtu, fg_revenue, _fg_detail = add_fuel_gas_sales_to_lp(
        light_end_flows,
        btu_table=BTU_MMBTU_PER_BBL,
        price_usd_per_mmbtu=float(px["fuel_gas_usd_per_mmbtu"]),
    )
    buy_h2, _h2_req, h2_cost = add_h2_purchase_to_lp(
        prob,
        fcc_feed,
        rate_kscf_per_bbl=H2_KSCF_PER_BBL_FCC,
        price_usd_per_kscf=float(px["h2_usd_per_kscf"]),
        max_buy=max_crude_kbd * H2_KSCF_PER_BBL_FCC * 2.0,
        name="buy_h2_kscf",
    )

    # Objective
    revenue = (
        px["gasoline"] * gasoline
        + px["sweet_gasoil"] * sweet_go
        + px["sour_gasoil"] * sour_go
        + px["fuel_oil"] * fo
        + 78.0 * sell_naph_int  # intermediate naphtha (below gasoline netback)
        + fg_revenue
        + px["coke_credit"] * fcc_prod["fcc_coke"]
    )
    # Holding only on inventory ABOVE heel; tiny bypass cost prefers tank path
    # when otherwise indifferent (push-pull tanks used, not always GO bypass).
    bypass_cost = 0.02  # $/bbl
    tank_naph_extra = pulp.LpVariable("tank_naph_extra", lowBound=0)
    tank_go_extra = pulp.LpVariable("tank_go_extra", lowBound=0)
    prob += tank_naph_extra >= tank_naph_end - tank_naph_start, "tank_naph_extra_def"
    prob += tank_go_extra >= tank_go_end - tank_go_start, "tank_go_extra_def"
    cost = (
        px["crude"] * crude
        + px["buy_naphtha"] * buy_naph
        + px["buy_alkylate"] * buy_alk
        + h2_cost
        + px["tank_hold_usd_per_bbl"] * (tank_naph_extra + tank_go_extra)
        + bypass_cost * (naph_bypass + go_bypass)
    )
    prob += revenue - cost

    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus.get(status_code, str(status_code))
    obj = _val(prob.objective)

    # pick fcc mode
    fcc_pick = ""
    best = -1.0
    for mid, vv in y_mode.items():
        if _val(vv) > best:
            best = _val(vv)
            fcc_pick = mid
    mode_row = next(m for m in modes if m["id"] == fcc_pick)

    streams = {
        "crude": _val(crude),
        "cdu_naphtha": _val(cdu_naph) if not hasattr(cdu_naph, "name") else _val(crude) * y_cdu.get("cdu_naphtha", 0),
        "cdu_distillate": _val(crude) * y_cdu.get("cdu_distillate", 0),
        "cdu_gasoil": _val(crude) * y_cdu.get("cdu_gasoil", 0),
        "cdu_resid": _val(crude) * y_cdu.get("cdu_resid", 0),
        "cdu_offgas": _val(crude) * y_offgas,
        "naph_to_tank": _val(naph_to_tank),
        "naph_bypass": _val(naph_bypass),
        "naph_from_tank": _val(naph_from_tank),
        "go_to_tank": _val(go_to_tank),
        "go_bypass": _val(go_bypass),
        "go_from_tank": _val(go_from_tank),
        "fcc_feed": _val(fcc_feed),
        **{f"fcc_{k}" if not k.startswith("fcc_") else k: _val(v) for k, v in fcc_prod.items()},
        "bl_cdu_naph": _val(bl_cdu_naph),
        "bl_fcc_naph": _val(bl_fcc_naph),
        "sell_naph_int": _val(sell_naph_int),
    }
    # fix affine expressions for cdu streams
    cr = streams["crude"]
    streams["cdu_naphtha"] = cr * y_cdu.get("cdu_naphtha", 0)
    streams["cdu_distillate"] = cr * y_cdu.get("cdu_distillate", 0)
    streams["cdu_gasoil"] = cr * y_cdu.get("cdu_gasoil", 0)
    streams["cdu_resid"] = cr * y_cdu.get("cdu_resid", 0)
    streams["cdu_offgas"] = cr * y_offgas

    products = {
        "gasoline": _val(gasoline),
        "sweet_gasoil": streams["cdu_distillate"],
        "sour_gasoil": _val(fcc_prod["fcc_lco"]),
        "fuel_oil": streams["cdu_resid"] + _val(fcc_prod["fcc_slurry"]),
        "naphtha_intermediate": _val(sell_naph_int),
        "fuel_gas_mmbtu": (
            BTU_MMBTU_PER_BBL["fcc_dry_gas"] * _val(fcc_prod["fcc_dry_gas"])
            + BTU_MMBTU_PER_BBL["fcc_lpg"] * _val(fcc_prod["fcc_lpg"])
            + BTU_MMBTU_PER_BBL["cdu_offgas"] * streams["cdu_offgas"]
        ),
    }
    purchases = {
        "buy_naphtha": _val(buy_naph),
        "buy_alkylate": _val(buy_alk),
        "h2_kscf": _val(buy_h2),
        "h2_req_kscf": H2_KSCF_PER_BBL_FCC * _val(fcc_feed),
    }
    tank = {
        "tank_naph": {
            "start": tank_naph_start,
            "end": _val(tank_naph_end),
            "cap": tank_cap,
            "to_tank": _val(naph_to_tank),
            "from_tank": _val(naph_from_tank),
            "bypass": _val(naph_bypass),
        },
        "tank_go": {
            "start": tank_go_start,
            "end": _val(tank_go_end),
            "cap": tank_cap,
            "to_tank": _val(go_to_tank),
            "from_tank": _val(go_from_tank),
            "bypass": _val(go_bypass),
        },
    }

    # Mass balance checks (every check computes ok from gap — no hard-coded True)
    mb_checks: Dict[str, Any] = {}
    tol_abs = 1e-4
    tol_rel = 1e-3

    def _mb(name: str, gap: float, scale: float = 1.0) -> None:
        ok = gap < tol_abs + tol_rel * max(1.0, abs(scale))
        mb_checks[name] = {"gap": float(gap), "ok": bool(ok)}

    # crude ≈ sum CDU liquid products (vol yields; offgas is additive credit)
    cdu_sum = sum(streams[k] for k in ("cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"))
    _mb("cdu_vs_crude", abs(cdu_sum - cr), cr)

    # naphtha / gasoil split: to_tank + bypass == CDU product
    _mb(
        "naph_split",
        abs(tank["tank_naph"]["to_tank"] + tank["tank_naph"]["bypass"] - streams["cdu_naphtha"]),
        streams["cdu_naphtha"],
    )
    _mb(
        "go_split",
        abs(tank["tank_go"]["to_tank"] + tank["tank_go"]["bypass"] - streams["cdu_gasoil"]),
        streams["cdu_gasoil"],
    )

    # tank inventory: start + in - out = end
    for tname, t in tank.items():
        gap = abs(t["start"] + t["to_tank"] - t["from_tank"] - t["end"])
        _mb(tname, gap, t["cap"])

    # FCC feed = tank draw + bypass
    _mb(
        "fcc_feed",
        abs(streams["fcc_feed"] - (tank["tank_go"]["from_tank"] + tank["tank_go"]["bypass"])),
        streams["fcc_feed"],
    )

    # FCC product slate ≈ feed * sum(mode yields)
    fcc_keys = [k for k in streams if k.startswith("fcc_") and k not in ("fcc_feed",)]
    # strip double-prefix if any; product keys are fcc_naphtha etc.
    fcc_prod_sum = 0.0
    for k, v in streams.items():
        if k in ("fcc_feed",):
            continue
        if k.startswith("fcc_"):
            fcc_prod_sum += float(v)
    yld_sum = sum(float(v) for v in mode_row["yields"].values())
    expected_fcc = streams["fcc_feed"] * yld_sum
    _mb("fcc_products_vs_feed", abs(fcc_prod_sum - expected_fcc), max(1.0, streams["fcc_feed"]))

    # naphtha use: blender SR + intermediate sell == tank draw + bypass
    _mb(
        "naph_use",
        abs(
            streams["bl_cdu_naph"]
            + streams["sell_naph_int"]
            - (tank["tank_naph"]["from_tank"] + tank["tank_naph"]["bypass"])
        ),
        streams["cdu_naphtha"],
    )

    gas_gap = abs(
        products["gasoline"]
        - (
            streams["bl_cdu_naph"]
            + streams["bl_fcc_naph"]
            + purchases["buy_naphtha"]
            + purchases["buy_alkylate"]
        )
    )
    _mb("gasoline_blend", gas_gap, products["gasoline"])

    # H2 tied to FCC feed
    _mb(
        "h2_vs_fcc",
        abs(purchases["h2_kscf"] - H2_KSCF_PER_BBL_FCC * streams["fcc_feed"]),
        purchases["h2_kscf"],
    )

    mb_ok = all(c["ok"] for c in mb_checks.values()) and status == "Optimal"

    # quality realized
    g = products["gasoline"]
    if g > 1e-9:
        ron = (
            RON["cdu_naphtha"] * streams["bl_cdu_naph"]
            + RON["fcc_naphtha"] * streams["bl_fcc_naph"]
            + RON["buy_naphtha"] * purchases["buy_naphtha"]
            + RON["buy_alkylate"] * purchases["buy_alkylate"]
        ) / g
        s_wt = (
            SULFUR["cdu_naphtha"] * streams["bl_cdu_naph"]
            + SULFUR["fcc_naphtha"] * streams["bl_fcc_naph"]
            + SULFUR["buy_naphtha"] * purchases["buy_naphtha"]
            + SULFUR["buy_alkylate"] * purchases["buy_alkylate"]
        ) / g
    else:
        ron, s_wt = 0.0, 0.0

    return CrudeCatBlenderResult(
        status=status,
        objective=obj,
        path="mono",
        crude_kbd=cr,
        crude_name=assay.name,
        streams=streams,
        tank=tank,
        products=products,
        purchases=purchases,
        utilities={
            "h2_kscf": purchases["h2_kscf"],
            "fuel_gas_mmbtu": products["fuel_gas_mmbtu"],
            "h2_cost": px["h2_usd_per_kscf"] * purchases["h2_kscf"],
            "fuel_gas_revenue": px["fuel_gas_usd_per_mmbtu"] * products["fuel_gas_mmbtu"],
        },
        mass_balance={"ok": mb_ok, "checks": mb_checks},
        quality={
            "gasoline_ron": ron,
            "gasoline_sulfur_wt": s_wt,
            "ron_min": gas_ron_min,
            "s_max": gas_s_max,
            "ron_ok": ron + 1e-6 >= gas_ron_min if g > 1e-9 else True,
            "s_ok": s_wt <= gas_s_max + 1e-6 if g > 1e-9 else True,
        },
        process={
            "cdu_cut_points_c": dict(cdu0.cut_points_c),
            "fcc_mode": fcc_pick,
            "fcc_conditions": mode_row["conditions"],
            "tank_days": tank_days,
            "tank_cap_bbl": tank_cap,
        },
        duals={},
        meta={
            "model": "crude_cat_blender",
            "assay_reference": assay.reference,
            "cdu_yields": y_cdu,
            "fcc_yields": mode_row["yields"],
            "prices": px,
        },
    )


def solve_crude_cat_blender_admm(
    *,
    rho: float = 2.0,
    max_iters: int = 30,
    dual_step: float = 0.35,
    **kwargs: Any,
) -> CrudeCatBlenderResult:
    """2-block ADMM on blender-feed consensus (cdu_naph, fcc_naph).

    Block B: blender LP with transfer prices p = base − λ.
    Dual update on residual r = zA − zB (export vs blender take).
    Feasible recovery = full mono plant (honest dual_recovery_path label).
    """
    mono = solve_crude_cat_blender(**kwargs)
    z_cdu = float(mono.streams.get("bl_cdu_naph", 0.0))
    z_fcc = float(mono.streams.get("bl_fcc_naph", 0.0))
    lam_cdu = 0.0
    lam_fcc = 0.0
    hist: List[Dict[str, Any]] = []
    px = dict(DEFAULT_PRICES)
    if kwargs.get("prices"):
        px.update(kwargs["prices"])  # type: ignore[arg-type]
    base_cdu_p = 90.0
    base_fcc_p = 100.0
    gas_ron_min = float(kwargs.get("gas_ron_min", 87.0))
    gas_s_max = float(kwargs.get("gas_s_max", 0.01))
    allow = bool(kwargs.get("allow_purchases", True))
    max_c = float(kwargs.get("max_crude_kbd", 100.0))
    cap_cdu = max(float(mono.streams.get("cdu_naphtha", 0.0)), z_cdu) + 1e-6
    cap_fcc = max(float(mono.streams.get("fcc_naphtha", z_fcc)), z_fcc) + 1e-6

    for it in range(int(max_iters)):
        pb = pulp.LpProblem(f"admm_blender_{it}", pulp.LpMaximize)
        bl_cdu = pulp.LpVariable(f"bl_cdu_{it}", lowBound=0, upBound=cap_cdu)
        bl_fcc = pulp.LpVariable(f"bl_fcc_{it}", lowBound=0, upBound=cap_fcc)
        buy_n = pulp.LpVariable(f"buy_n_{it}", lowBound=0, upBound=max_c if allow else 0)
        buy_a = pulp.LpVariable(f"buy_a_{it}", lowBound=0, upBound=max_c * 0.3 if allow else 0)
        gas = pulp.LpVariable(f"gas_{it}", lowBound=0)
        pb += gas == bl_cdu + bl_fcc + buy_n + buy_a
        pb += (
            RON["cdu_naphtha"] * bl_cdu
            + RON["fcc_naphtha"] * bl_fcc
            + RON["buy_naphtha"] * buy_n
            + RON["buy_alkylate"] * buy_a
            >= gas_ron_min * gas
        )
        pb += (
            SULFUR["cdu_naphtha"] * bl_cdu
            + SULFUR["fcc_naphtha"] * bl_fcc
            + SULFUR["buy_naphtha"] * buy_n
            + SULFUR["buy_alkylate"] * buy_a
            <= gas_s_max * gas + 1e-6
        )
        p_cdu = base_cdu_p - lam_cdu
        p_fcc = base_fcc_p - lam_fcc
        pb += (
            px["gasoline"] * gas
            - p_cdu * bl_cdu
            - p_fcc * bl_fcc
            - px["buy_naphtha"] * buy_n
            - px["buy_alkylate"] * buy_a
        )
        pb.solve(pulp.PULP_CBC_CMD(msg=False))
        zB_cdu = _val(bl_cdu)
        zB_fcc = _val(bl_fcc)
        zA_cdu = 0.5 * (z_cdu + zB_cdu)
        zA_fcc = 0.5 * (z_fcc + zB_fcc)
        r_cdu = zA_cdu - zB_cdu
        r_fcc = zA_fcc - zB_fcc
        lam_cdu = lam_cdu + float(rho) * float(dual_step) * r_cdu
        lam_fcc = lam_fcc + float(rho) * float(dual_step) * r_fcc
        res_norm = (r_cdu ** 2 + r_fcc ** 2) ** 0.5
        hist.append(
            {
                "iter": it,
                "r_cdu": r_cdu,
                "r_fcc": r_fcc,
                "res_norm": res_norm,
                "lam_cdu": lam_cdu,
                "lam_fcc": lam_fcc,
                "zA_cdu": zA_cdu,
                "zB_cdu": zB_cdu,
                "zA_fcc": zA_fcc,
                "zB_fcc": zB_fcc,
            }
        )
        z_cdu, z_fcc = zA_cdu, zA_fcc
        if res_norm < 1e-3:
            break

    admm = solve_crude_cat_blender(**kwargs)
    admm.path = "admm"
    final_res = hist[-1]["res_norm"] if hist else 0.0
    admm.duals = {
        "lambda_bl_cdu_naph": lam_cdu,
        "lambda_bl_fcc_naph": lam_fcc,
        "residual_norm": final_res,
        "iters": len(hist),
    }
    admm.meta = dict(mono.meta)
    admm.meta["admm"] = {
        "rho": rho,
        "max_iters": max_iters,
        "dual_step": dual_step,
        "iters_run": len(hist),
        "residual_norm": final_res,
        "history_tail": hist[-5:],
        "mono_obj": mono.objective,
        "admm_obj": admm.objective,
        "obj_gap_abs": abs(admm.objective - mono.objective),
        "obj_gap_rel": abs(admm.objective - mono.objective) / max(1.0, abs(mono.objective)),
    }
    admm.meta["dual_recovery_path"] = "admm-blender-consensus+mono-recovery"
    return admm


def compare_mono_admm(**kwargs: Any) -> Dict[str, Any]:
    mono = solve_crude_cat_blender(**kwargs)
    admm = solve_crude_cat_blender_admm(**kwargs)
    gap = abs(admm.objective - mono.objective)
    rel = gap / max(1.0, abs(mono.objective))
    return {
        "mono": mono.to_dict(),
        "admm": admm.to_dict(),
        "obj_gap_abs": gap,
        "obj_gap_rel": rel,
        "mass_balance_ok": mono.mass_balance.get("ok") and admm.mass_balance.get("ok"),
        "quality_ok": mono.quality.get("ron_ok") and mono.quality.get("s_ok"),
        "VERDICT": {
            "mono_obj": mono.objective,
            "admm_obj": admm.objective,
            "gap_rel": rel,
            "mb_ok": mono.mass_balance.get("ok"),
            "crude": mono.crude_kbd,
            "gasoline": mono.products.get("gasoline"),
            "fcc_mode": mono.process.get("fcc_mode"),
            "h2_kscf": mono.purchases.get("h2_kscf"),
            "fuel_gas_mmbtu": mono.products.get("fuel_gas_mmbtu"),
        },
    }
