"""Hardened pure-ADMM for full-plant linking streams (no mono dual injection into λ).

Design
------
- Price-directed multi-block solves (CDU / FCC / Coker / Reformer / Blender).
- L1 consensus penalties vs z (linear, CBC-friendly) so blocks seek agreement.
- Dual ascent with **market-clearing sign**:
    r = prod − use
    λ ← λ − α ρ r     # oversupply (r>0) lowers λ; shortage raises λ
- Adaptive ρ (Boyd-style) + damping + box projection on λ.
- Mono duals used **only** for post-hoc L∞ honesty metrics, never for λ init/update.

Multi-stream free-disposal residual accounting (wave4/wave5)
------------------------------------------------------------
Wave4 expanded `routing.linking_streams` with byproducts (dry gas, LPG, coke,
H2, lights, offgas, HDT). Those faces are **free disposal** (λ ≥ 0): oversupply
is not a primal feasibility failure.

Residual partition:
  - **core balance links** (liquids pure blocks jointly price): equality residual
    r = prod − use, shortage short = max(0, use − prod). Duals update on these.
  - **free-disposal byproducts**: produced from conversion yields and auto-sunk
    (fuel/coke credit). They do **not** enter equality residual or dual ascent;
    reported separately so multi-stream slate does not falsely widen ||r||.

Blender floor dispose + Gauss–Seidel availability caps keep core shortfalls
honest without injecting mono duals.

Not claimed: global optimality of free price-directed LP blocks always matches
mono at finite iter; we report ||r||, ||short||, λ_vs_mono_L∞ honestly.
See docs/pure_admm_floor.md for the structural L∞ floor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pulp

from pims_admm_llm.admm.residuals import linf_dual_gap, residual_norms
from pims_admm_llm.models.assay_loader import load_assays_json, load_routing
from pims_admm_llm.models.full_plant import build_yield_tables, solve_full_plant


# Core liquids pure blocks model for consensus / dual ascent.
DEFAULT_LINKS = [
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

# Wave4 multi-stream byproducts: free disposal (λ≥0); residual oversupply ignored.
FREE_DISPOSAL_BYPRODUCTS = frozenset(
    {
        "fcc_dry_gas",
        "fcc_lpg",
        "fcc_coke",
        "coker_dry_gas",
        "coker_lpg",
        "coker_coke",
        "reformer_h2",
        "reformer_lights",
        "cdu_offgas",
        "hdt_naphtha",
        "hdt_lights",
    }
)
# Public alias expected by tests / docs
FREE_DISPOSAL_STREAMS = FREE_DISPOSAL_BYPRODUCTS

CORE_LINK_SET = frozenset(DEFAULT_LINKS)

# Floor netbacks for blender dispose sink (planning-grade, not mono duals).
_DISPOSE_FLOOR = {
    "cdu_naphtha": 85.0,
    "cdu_naphtha_light": 88.0,
    "cdu_naphtha_heavy": 70.0,
    "cdu_distillate": 100.0,
    "cdu_gasoil": 75.0,
    "cdu_resid": 45.0,
    "fcc_naphtha": 95.0,
    "fcc_lco": 90.0,
    "fcc_slurry": 50.0,
    "coker_naphtha": 70.0,
    "coker_gasoil": 85.0,
    "reformate": 100.0,
}


def _val(v) -> float:
    x = pulp.value(v)
    return float(x) if x is not None else 0.0


def _solve(prob: pulp.LpProblem) -> Tuple[str, float, Dict[str, float], float]:
    import time

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    dt = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)
    primals = {v.name: float(v.varValue or 0.0) for v in prob.variables()}
    return status, obj, primals, dt


def _l1_consensus_terms(
    prob: pulp.LpProblem,
    var: pulp.LpVariable,
    z_val: float,
    rho: float,
    name: str,
) -> pulp.LpAffineExpression:
    """Add t ≥ |var − z| and return −ρ t (maximize form)."""
    t = pulp.LpVariable(f"t_{name}", lowBound=0)
    z = float(z_val)
    prob += t >= var - z, f"cons_pos_{name}"
    prob += t >= z - var, f"cons_neg_{name}"
    return -float(rho) * t


def _partition_links(routing_links: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split routing linking_streams into core balance vs free-disposal byproducts.

    Core = liquids pure blocks jointly price (intersection with DEFAULT_LINKS,
    preserving routing order). Free-disposal = wave4 multi-stream byproducts.
    Unknown names stay out of dual ascent (honest residual scope).
    """
    core: List[str] = []
    free_disp: List[str] = []
    seen = set()
    for s in routing_links:
        if s in seen:
            continue
        seen.add(s)
        if s in FREE_DISPOSAL_BYPRODUCTS:
            free_disp.append(s)
        elif s in CORE_LINK_SET:
            core.append(s)
    if not core:
        core = list(DEFAULT_LINKS)
    return core, free_disp


def solve_cdu_block_consensus(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    z: Mapping[str, float],
    rho: float,
    links: Sequence[str],
) -> Dict[str, Any]:
    """CDU with L1 consensus on produced linking streams."""
    prob = pulp.LpProblem("block_CDU_cons", pulp.LpMaximize)
    crude_v = {
        c["name"]: pulp.LpVariable(f"crude_{c['name']}", 0, float(c["max_supply_kbd"]))
        for c in assays["crudes"]
    }
    caps = assays.get("capacities") or {}
    charge = pulp.lpSum(crude_v.values())
    prob += charge <= float(caps.get("cdu_kbd", 140))
    cuts = ["cdu_naphtha", "cdu_distillate", "cdu_gasoil", "cdu_resid"]
    prod = {s: pulp.LpVariable(f"prod_{s}", 0) for s in cuts}
    for s in cuts:
        prob += prod[s] == pulp.lpSum(
            yields["cdu_by_crude"][c["name"]][s] * crude_v[c["name"]] for c in assays["crudes"]
        )
    light = pulp.LpVariable("prod_cdu_naphtha_light", 0)
    heavy = pulp.LpVariable("prod_cdu_naphtha_heavy", 0)
    prob += light == 0.40 * prod["cdu_naphtha"]
    prob += heavy == 0.60 * prod["cdu_naphtha"]

    # Optional offgas free-disposal byproduct (wave4 multi-stream)
    offgas = pulp.LpVariable("prod_cdu_offgas", 0)
    # small fuel credit fraction of charge if yields expose it; else 0
    off_y = 0.0
    cdu_y = yields.get("cdu") or yields.get("cdu_avg") or {}
    if isinstance(cdu_y, dict):
        off_y = float(cdu_y.get("cdu_offgas", 0.0) or 0.0)
    if off_y <= 0.0:
        # assay packages may only store liquid cuts; keep offgas at 0
        prob += offgas == 0.0
    else:
        prob += offgas == off_y * charge

    rev = pulp.lpSum(float(prices.get(s, 0.0)) * prod[s] for s in cuts)
    rev += float(prices.get("cdu_naphtha_light", prices.get("cdu_naphtha", 0.0))) * light
    rev += float(prices.get("cdu_naphtha_heavy", prices.get("cdu_naphtha", 0.0))) * heavy
    # free-disposal byproduct credit (often ~0 dual at optimum)
    rev += float(prices.get("cdu_offgas", 0.0)) * offgas
    cost = pulp.lpSum(float(c["price_usd_per_bbl"]) * crude_v[c["name"]] for c in assays["crudes"])
    cons = 0
    for s, var in [
        ("cdu_gasoil", prod["cdu_gasoil"]),
        ("cdu_resid", prod["cdu_resid"]),
        ("cdu_naphtha", prod["cdu_naphtha"]),
        ("cdu_naphtha_light", light),
        ("cdu_naphtha_heavy", heavy),
        ("cdu_distillate", prod["cdu_distillate"]),
    ]:
        if s in links:
            cons += _l1_consensus_terms(prob, var, z.get(s, 0.0), rho, f"cdu_{s}")
    prob += rev - cost + cons
    status, obj, primals, dt = _solve(prob)
    linking = {s: primals.get(f"prod_{s}", 0.0) for s in cuts}
    linking["cdu_naphtha_light"] = primals.get("prod_cdu_naphtha_light", 0.0)
    linking["cdu_naphtha_heavy"] = primals.get("prod_cdu_naphtha_heavy", 0.0)
    linking["cdu_offgas"] = primals.get("prod_cdu_offgas", 0.0)
    return {
        "block": "CDU",
        "status": status,
        "local_obj": obj,
        "proposal": linking,
        "primals": primals,
        "time_s": dt,
    }


def solve_fcc_block_consensus(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    z: Mapping[str, float],
    rho: float,
    links: Sequence[str],
    gasoil_available: Optional[float] = None,
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_FCC_cons", pulp.LpMaximize)
    feed = pulp.LpVariable("fcc_feed", 0)
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("fcc_kbd", 55))
    if gasoil_available is not None:
        prob += feed <= float(gasoil_available) + 1e-9
    fy = yields["fcc"]
    naph = pulp.LpVariable("prod_fcc_naphtha", 0)
    lco = pulp.LpVariable("prod_fcc_lco", 0)
    slurry = pulp.LpVariable("prod_fcc_slurry", 0)
    dry = pulp.LpVariable("prod_fcc_dry_gas", 0)
    lpg = pulp.LpVariable("prod_fcc_lpg", 0)
    coke = pulp.LpVariable("prod_fcc_coke", 0)
    prob += naph == float(fy.get("fcc_naphtha", 0.0)) * feed
    prob += lco == float(fy.get("fcc_lco", 0.0)) * feed
    prob += slurry == float(fy.get("fcc_slurry", 0.0)) * feed
    prob += dry == float(fy.get("fcc_dry_gas", 0.0)) * feed
    prob += lpg == float(fy.get("fcc_lpg", 0.0)) * feed
    prob += coke == float(fy.get("fcc_coke", 0.0)) * feed
    # multi-stream free-disposal: fuel/coke credits (not mono duals)
    fuel_credit = 8.0 * dry + 25.0 * lpg + 12.0 * coke
    obj = (
        float(prices.get("fcc_naphtha", 0)) * naph
        + float(prices.get("fcc_lco", 0)) * lco
        + float(prices.get("fcc_slurry", 0)) * slurry
        + float(prices.get("fcc_dry_gas", 0.0)) * dry
        + float(prices.get("fcc_lpg", 0.0)) * lpg
        + float(prices.get("fcc_coke", 0.0)) * coke
        + fuel_credit
        - float(prices.get("cdu_gasoil", 0)) * feed
        - 1.5 * feed
    )
    if "cdu_gasoil" in links:
        obj += _l1_consensus_terms(prob, feed, z.get("cdu_gasoil", 0.0), rho, "fcc_feed")
    for s, var in [("fcc_naphtha", naph), ("fcc_lco", lco), ("fcc_slurry", slurry)]:
        if s in links:
            obj += _l1_consensus_terms(prob, var, z.get(s, 0.0), rho, f"fcc_{s}")
    prob += obj
    status, objv, primals, dt = _solve(prob)
    proposal = {
        "cdu_gasoil_use": primals.get("fcc_feed", 0.0),
        "fcc_naphtha": primals.get("prod_fcc_naphtha", 0.0),
        "fcc_lco": primals.get("prod_fcc_lco", 0.0),
        "fcc_slurry": primals.get("prod_fcc_slurry", 0.0),
        "fcc_dry_gas": primals.get("prod_fcc_dry_gas", 0.0),
        "fcc_lpg": primals.get("prod_fcc_lpg", 0.0),
        "fcc_coke": primals.get("prod_fcc_coke", 0.0),
    }
    return {
        "block": "FCC",
        "status": status,
        "local_obj": objv,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_coker_block_consensus(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    z: Mapping[str, float],
    rho: float,
    links: Sequence[str],
    resid_available: Optional[float] = None,
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_COKER_cons", pulp.LpMaximize)
    feed = pulp.LpVariable("coker_feed", 0)
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("coker_kbd", 40))
    if resid_available is not None:
        prob += feed <= float(resid_available) + 1e-9
    cy = yields["coker"]
    naph = pulp.LpVariable("prod_coker_naphtha", 0)
    go = pulp.LpVariable("prod_coker_gasoil", 0)
    dry = pulp.LpVariable("prod_coker_dry_gas", 0)
    lpg = pulp.LpVariable("prod_coker_lpg", 0)
    coke = pulp.LpVariable("prod_coker_coke", 0)
    prob += naph == float(cy.get("coker_naphtha", 0.0)) * feed
    prob += go == float(cy.get("coker_gasoil", 0.0)) * feed
    prob += dry == float(cy.get("coker_dry_gas", 0.0)) * feed
    prob += lpg == float(cy.get("coker_lpg", 0.0)) * feed
    prob += coke == float(cy.get("coker_coke", 0.0)) * feed
    fuel_credit = 8.0 * dry + 25.0 * lpg
    obj = (
        float(prices.get("coker_naphtha", 0)) * naph
        + float(prices.get("coker_gasoil", 0)) * go
        + float(prices.get("coker_dry_gas", 0.0)) * dry
        + float(prices.get("coker_lpg", 0.0)) * lpg
        + float(prices.get("coker_coke", 0.0)) * coke
        + fuel_credit
        - float(prices.get("cdu_resid", 0)) * feed
        - 2.0 * feed
        + 18.0 * feed  # coke credit match mono
    )
    if "cdu_resid" in links:
        obj += _l1_consensus_terms(prob, feed, z.get("cdu_resid", 0.0), rho, "coker_feed")
    for s, var in [("coker_naphtha", naph), ("coker_gasoil", go)]:
        if s in links:
            obj += _l1_consensus_terms(prob, var, z.get(s, 0.0), rho, f"cok_{s}")
    prob += obj
    status, objv, primals, dt = _solve(prob)
    proposal = {
        "cdu_resid_use": primals.get("coker_feed", 0.0),
        "coker_naphtha": primals.get("prod_coker_naphtha", 0.0),
        "coker_gasoil": primals.get("prod_coker_gasoil", 0.0),
        "coker_dry_gas": primals.get("prod_coker_dry_gas", 0.0),
        "coker_lpg": primals.get("prod_coker_lpg", 0.0),
        "coker_coke": primals.get("prod_coker_coke", 0.0),
    }
    return {
        "block": "COKER",
        "status": status,
        "local_obj": objv,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_reformer_block_consensus(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    z: Mapping[str, float],
    rho: float,
    links: Sequence[str],
    heavy_sr_available: Optional[float] = None,
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_REFORMER_cons", pulp.LpMaximize)
    h_n = pulp.LpVariable("heavy_sr_naph_in", 0)
    f_n = pulp.LpVariable("fcc_naph_in", 0)
    c_n = pulp.LpVariable("coker_naph_in", 0)
    # chemistry: cracked naph closed in pure path (match plant default)
    prob += f_n == 0
    prob += c_n == 0
    feed = h_n + f_n + c_n
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("reformer_kbd", 45))
    if heavy_sr_available is not None:
        prob += h_n <= float(heavy_sr_available) + 1e-9
    ref = pulp.LpVariable("prod_reformate", 0)
    h2 = pulp.LpVariable("prod_reformer_h2", 0)
    lights = pulp.LpVariable("prod_reformer_lights", 0)
    ry = float(yields["reformer"].get("reformate", 0.86))
    ry_h2 = float(yields["reformer"].get("reformer_h2", 0.0) or 0.0)
    ry_lt = float(yields["reformer"].get("reformer_lights", 0.0) or 0.0)
    prob += ref == ry * feed
    prob += h2 == ry_h2 * feed
    prob += lights == ry_lt * feed
    obj = (
        float(prices.get("reformate", 0)) * ref
        + float(prices.get("reformer_h2", 0.0)) * h2
        + float(prices.get("reformer_lights", 0.0)) * lights
        + 15.0 * h2  # fuel credit free-disposal
        - float(prices.get("cdu_naphtha_heavy", prices.get("cdu_naphtha", 0))) * h_n
        - 1.2 * feed
    )
    if "cdu_naphtha_heavy" in links:
        obj += _l1_consensus_terms(prob, h_n, z.get("cdu_naphtha_heavy", 0.0), rho, "ref_heavy")
    if "reformate" in links:
        obj += _l1_consensus_terms(prob, ref, z.get("reformate", 0.0), rho, "ref_out")
    prob += obj
    status, objv, primals, dt = _solve(prob)
    proposal = {
        "cdu_naphtha_heavy_use": primals.get("heavy_sr_naph_in", 0.0),
        "fcc_naphtha_use": primals.get("fcc_naph_in", 0.0),
        "coker_naphtha_use": primals.get("coker_naph_in", 0.0),
        "reformate": primals.get("prod_reformate", 0.0),
        "reformer_h2": primals.get("prod_reformer_h2", 0.0),
        "reformer_lights": primals.get("prod_reformer_lights", 0.0),
    }
    return {
        "block": "REFORMER",
        "status": status,
        "local_obj": objv,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_blender_block_consensus(
    assays: Dict[str, Any],
    prices: Mapping[str, float],
    z: Mapping[str, float],
    rho: float,
    links: Sequence[str],
    available: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Blender with L1 consensus + floor dispose sink for free-disposal residuals.

    Floor dispose clears multi-stream free-disposal / recipe-mismatch oversupply
    without mono dual injection. Optional ``available`` is Gauss–Seidel residual
    after conversion so blender cannot over-draw streams conversion already used.
    """
    products = assays.get("products") or {}
    default_recipes = {
        "gasoline": {
            "cdu_naphtha_light": 0.25,
            "fcc_naphtha": 0.30,
            "reformate": 0.35,
            "coker_naphtha": 0.10,
        },
        "diesel": {
            "cdu_distillate": 0.55,
            "fcc_lco": 0.25,
            "coker_gasoil": 0.20,
        },
        "fuel_oil": {
            "fcc_slurry": 0.35,
            "cdu_resid": 0.40,
            "coker_gasoil": 0.15,
            "coker_naphtha": 0.10,
        },
    }
    stream_set = set(DEFAULT_LINKS)
    for rec in default_recipes.values():
        stream_set.update(rec.keys())
    for s in links:
        if s not in FREE_DISPOSAL_BYPRODUCTS:
            stream_set.add(s)

    prob = pulp.LpProblem("block_BLENDER_cons", pulp.LpMaximize)
    prod_v = {}
    for name, spec in products.items():
        ub = float(spec.get("max_demand_kbd", 1e6)) if isinstance(spec, dict) else 1e6
        prod_v[name] = pulp.LpVariable(f"prod_{name}", 0, ub)

    use_v = {s: pulp.LpVariable(f"use_{s}", 0) for s in sorted(stream_set)}
    dispose_v = {s: pulp.LpVariable(f"dispose_{s}", 0) for s in sorted(stream_set)}
    attributed: Dict[str, list] = {s: [] for s in use_v}
    for pname, pvar in prod_v.items():
        rec = default_recipes.get(pname) or {}
        for stream, frac in rec.items():
            if stream not in use_v:
                use_v[stream] = pulp.LpVariable(f"use_{stream}", 0)
                dispose_v[stream] = pulp.LpVariable(f"dispose_{stream}", 0)
                attributed[stream] = []
            u_ps = pulp.LpVariable(f"use_{pname}_{stream}", 0)
            prob += u_ps == float(frac) * pvar
            attributed.setdefault(stream, []).append(u_ps)
    for stream, parts in attributed.items():
        if parts:
            prob += use_v[stream] == pulp.lpSum(parts)
        else:
            prob += use_v[stream] == 0

    # Gauss–Seidel: total sink (recipe + dispose) cannot exceed remaining inventory
    if available is not None:
        for s in use_v:
            if s in available:
                prob += use_v[s] + dispose_v[s] <= float(available[s]) + 1e-9

    rev = pulp.lpSum(
        float(products[n].get("price_usd_per_bbl", 0.0) if isinstance(products[n], dict) else 0.0)
        * prod_v[n]
        for n in prod_v
    )
    rev_disp = pulp.lpSum(float(_DISPOSE_FLOOR.get(s, 40.0)) * dispose_v[s] for s in dispose_v)
    cost = pulp.lpSum(float(prices.get(s, 0.0)) * (use_v[s] + dispose_v[s]) for s in use_v)
    cons = 0
    for s in use_v:
        if s in links:
            # consensus on total sink (recipe + dispose) vs z
            total_s = use_v[s] + dispose_v[s]
            cons += _l1_consensus_terms(prob, total_s, z.get(s, 0.0), rho, f"bl_{s}")
    prob += rev + rev_disp - cost - 0.1 * pulp.lpSum(prod_v.values()) + cons
    status, obj, primals, dt = _solve(prob)
    # proposal use includes dispose sink (free-disposal residual clear)
    stream_use = {
        s: float(primals.get(f"use_{s}", 0.0)) + float(primals.get(f"dispose_{s}", 0.0))
        for s in use_v
    }
    product_rates = {n: float(primals.get(f"prod_{n}", 0.0)) for n in prod_v}
    dispose_rates = {s: float(primals.get(f"dispose_{s}", 0.0)) for s in dispose_v}
    return {
        "block": "BLENDER",
        "status": status,
        "local_obj": obj,
        "proposal": stream_use,
        "product_rates": product_rates,
        "dispose": dispose_rates,
        "primals": primals,
        "time_s": dt,
    }


def _blender_available(blocks: Dict[str, Any]) -> Dict[str, float]:
    """Residual inventory after conversion (Gauss–Seidel cascade)."""
    cdu = blocks["CDU"]["proposal"]
    fcc = blocks["FCC"]["proposal"]
    cok = blocks["COKER"]["proposal"]
    ref = blocks["REFORMER"]["proposal"]
    return {
        "cdu_naphtha": float(cdu.get("cdu_naphtha", 0.0)),
        "cdu_naphtha_light": float(cdu.get("cdu_naphtha_light", 0.0)),
        "cdu_naphtha_heavy": max(
            0.0,
            float(cdu.get("cdu_naphtha_heavy", 0.0)) - float(ref.get("cdu_naphtha_heavy_use", 0.0)),
        ),
        "cdu_distillate": float(cdu.get("cdu_distillate", 0.0)),
        "cdu_gasoil": max(
            0.0, float(cdu.get("cdu_gasoil", 0.0)) - float(fcc.get("cdu_gasoil_use", 0.0))
        ),
        "cdu_resid": max(
            0.0, float(cdu.get("cdu_resid", 0.0)) - float(cok.get("cdu_resid_use", 0.0))
        ),
        "fcc_naphtha": max(
            0.0, float(fcc.get("fcc_naphtha", 0.0)) - float(ref.get("fcc_naphtha_use", 0.0))
        ),
        "fcc_lco": float(fcc.get("fcc_lco", 0.0)),
        "fcc_slurry": float(fcc.get("fcc_slurry", 0.0)),
        "coker_naphtha": max(
            0.0, float(cok.get("coker_naphtha", 0.0)) - float(ref.get("coker_naphtha_use", 0.0))
        ),
        "coker_gasoil": float(cok.get("coker_gasoil", 0.0)),
        "reformate": float(ref.get("reformate", 0.0)),
    }


def _aggregate_prod_use(
    blocks: Dict[str, Any],
    core_links: Sequence[str],
    free_disp_links: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Aggregate production and use for residual accounting.

    Returns (core_prod, core_use, fd_prod, fd_use) where free-disposal byproducts
    are auto-sunk (fd_use = fd_prod) so multi-stream free-disposal does not
    inflate equality residual.
    """
    b = blocks
    cdu = b["CDU"]["proposal"]
    fcc = b["FCC"]["proposal"]
    cok = b["COKER"]["proposal"]
    ref = b["REFORMER"]["proposal"]
    bl = b.get("BLENDER", {}).get("proposal") or {}

    prod_all = {
        "cdu_gasoil": float(cdu.get("cdu_gasoil", 0.0)),
        "cdu_resid": float(cdu.get("cdu_resid", 0.0)),
        "cdu_naphtha": float(cdu.get("cdu_naphtha", 0.0)),
        "cdu_naphtha_light": float(cdu.get("cdu_naphtha_light", 0.0)),
        "cdu_naphtha_heavy": float(cdu.get("cdu_naphtha_heavy", 0.0)),
        "cdu_distillate": float(cdu.get("cdu_distillate", 0.0)),
        "fcc_naphtha": float(fcc.get("fcc_naphtha", 0.0)),
        "fcc_lco": float(fcc.get("fcc_lco", 0.0)),
        "fcc_slurry": float(fcc.get("fcc_slurry", 0.0)),
        "coker_naphtha": float(cok.get("coker_naphtha", 0.0)),
        "coker_gasoil": float(cok.get("coker_gasoil", 0.0)),
        "reformate": float(ref.get("reformate", 0.0)),
        # multi-stream free-disposal byproducts
        "fcc_dry_gas": float(fcc.get("fcc_dry_gas", 0.0)),
        "fcc_lpg": float(fcc.get("fcc_lpg", 0.0)),
        "fcc_coke": float(fcc.get("fcc_coke", 0.0)),
        "coker_dry_gas": float(cok.get("coker_dry_gas", 0.0)),
        "coker_lpg": float(cok.get("coker_lpg", 0.0)),
        "coker_coke": float(cok.get("coker_coke", 0.0)),
        "reformer_h2": float(ref.get("reformer_h2", 0.0)),
        "reformer_lights": float(ref.get("reformer_lights", 0.0)),
        "cdu_offgas": float(cdu.get("cdu_offgas", 0.0)),
        "hdt_naphtha": 0.0,
        "hdt_lights": 0.0,
    }
    # use = conversion feed + blender sink (recipe + floor dispose)
    use_all = {
        "cdu_gasoil": float(fcc.get("cdu_gasoil_use", 0.0)) + float(bl.get("cdu_gasoil", 0.0)),
        "cdu_resid": float(cok.get("cdu_resid_use", 0.0)) + float(bl.get("cdu_resid", 0.0)),
        "cdu_naphtha_heavy": float(ref.get("cdu_naphtha_heavy_use", 0.0))
        + float(bl.get("cdu_naphtha_heavy", 0.0)),
        "cdu_naphtha_light": float(bl.get("cdu_naphtha_light", 0.0)),
        "cdu_naphtha": float(bl.get("cdu_naphtha", 0.0)),
        "cdu_distillate": float(bl.get("cdu_distillate", 0.0)),
        "fcc_naphtha": float(ref.get("fcc_naphtha_use", 0.0)) + float(bl.get("fcc_naphtha", 0.0)),
        "fcc_lco": float(bl.get("fcc_lco", 0.0)),
        "fcc_slurry": float(bl.get("fcc_slurry", 0.0)),
        "coker_naphtha": float(ref.get("coker_naphtha_use", 0.0)) + float(bl.get("coker_naphtha", 0.0)),
        "coker_gasoil": float(bl.get("coker_gasoil", 0.0)),
        "reformate": float(bl.get("reformate", 0.0)),
    }
    # Free-disposal byproducts: auto-sink at production (fuel/coke credit path).
    # Equality residual is structurally zero; oversupply is allowed by construction.
    for s in FREE_DISPOSAL_BYPRODUCTS:
        use_all[s] = float(prod_all.get(s, 0.0))

    core_prod = {s: float(prod_all.get(s, 0.0)) for s in core_links}
    core_use = {s: float(use_all.get(s, 0.0)) for s in core_links}
    fd_prod = {s: float(prod_all.get(s, 0.0)) for s in free_disp_links}
    fd_use = {s: float(use_all.get(s, 0.0)) for s in free_disp_links}
    return core_prod, core_use, fd_prod, fd_use


def run_pure_plant_admm(
    assays: Optional[Dict[str, Any]] = None,
    *,
    routing: Optional[Dict[str, Any]] = None,
    max_iter: int = 80,
    rho: float = 2.0,
    dual_step: float = 0.35,
    tol: float = 5.0,
    damp: float = 0.4,
    lam_min: float = 0.0,  # free-disposal duals ≥ 0
    lam_max: float = 200.0,
    adaptive_rho: bool = True,
    mu: float = 10.0,
    tau_incr: float = 2.0,
    tau_decr: float = 2.0,
    rho_min: float = 0.1,
    rho_max: float = 20.0,
) -> Dict[str, Any]:
    """Pure multi-block ADMM; λ free of mono duals. Mono used for honesty only.

    Always labels ``dual_recovery_path`` = ``\"pure-admm\"`` (never mono-oracle).
    """
    assays = assays or load_assays_json()
    routing = routing or load_routing()
    yields = build_yield_tables(assays)
    mono = solve_full_plant(assays, routing=routing)
    routing_links = list(routing.get("linking_streams") or DEFAULT_LINKS)
    core_links, free_disp_links = _partition_links(routing_links)
    # Dual ascent + consensus only on core balance liquids
    links = list(core_links)

    # netback seeds (not mono duals)
    seed = {
        "reformate": 105.0,
        "fcc_naphtha": 100.0,
        "coker_naphtha": 88.0,
        "cdu_gasoil": 75.0,
        "cdu_resid": 50.0,
        "fcc_lco": 98.0,
        "fcc_slurry": 55.0,
        "coker_gasoil": 92.0,
        "cdu_naphtha": 95.0,
        "cdu_naphtha_light": 92.0,
        "cdu_naphtha_heavy": 85.0,
        "cdu_distillate": 105.0,
    }
    lam = {s: float(seed.get(s, 60.0)) for s in links}
    z = {s: 0.0 for s in links}
    history: List[Dict[str, Any]] = []
    last_blocks: Dict[str, Any] = {}
    final_r = 0.0
    final_s = 0.0
    final_short = 0.0
    final_fd_over = 0.0
    rho_cur = float(rho)

    for it in range(max_iter):
        prices = dict(lam)
        cdu = solve_cdu_block_consensus(assays, yields, prices, z, rho_cur, links)
        # Gauss–Seidel cascade: conversion capped by upstream production
        go_avail = float(cdu["proposal"].get("cdu_gasoil", 0.0))
        resid_avail = float(cdu["proposal"].get("cdu_resid", 0.0))
        heavy_avail = float(cdu["proposal"].get("cdu_naphtha_heavy", 0.0))
        fcc = solve_fcc_block_consensus(
            assays, yields, prices, z, rho_cur, links, gasoil_available=go_avail
        )
        cok = solve_coker_block_consensus(
            assays, yields, prices, z, rho_cur, links, resid_available=resid_avail
        )
        ref = solve_reformer_block_consensus(
            assays, yields, prices, z, rho_cur, links, heavy_sr_available=heavy_avail
        )
        conv_blocks = {
            "CDU": cdu,
            "FCC": fcc,
            "COKER": cok,
            "REFORMER": ref,
        }
        avail = _blender_available(conv_blocks)
        bl = solve_blender_block_consensus(
            assays, prices, z, rho_cur, links, available=avail
        )
        last_blocks = {**conv_blocks, "BLENDER": bl}
        prod, use, fd_prod, fd_use = _aggregate_prod_use(
            last_blocks, links, free_disp_links
        )
        z_old = dict(z)
        # consensus target: average of prod and use (core only)
        for s in links:
            z[s] = 0.5 * (prod.get(s, 0.0) + use.get(s, 0.0))
        r_norm, s_norm, r = residual_norms(links, prod, use, z, z_old, rho_cur)
        # Free-disposal shortage residual: only unmet demand on core links
        shortage = {s: max(0.0, use.get(s, 0.0) - prod.get(s, 0.0)) for s in links}
        short_norm = math.sqrt(sum(v * v for v in shortage.values()))
        # Oversupply on free-disposal byproducts (structural; not a feasibility fail)
        fd_over = {
            s: max(0.0, fd_prod.get(s, 0.0) - fd_use.get(s, 0.0)) for s in free_disp_links
        }
        fd_over_norm = math.sqrt(sum(v * v for v in fd_over.values()))

        # Market-clearing dual update (sign-critical) on **core** links only:
        # excess r=prod-use > 0 → lower λ; shortage → raise λ
        # Free disposal: project λ ≥ lam_min (default 0)
        lam_new = {}
        for s in links:
            step = dual_step * rho_cur * r.get(s, 0.0)
            cand = lam[s] - step
            # damping
            cand = (1.0 - damp) * lam[s] + damp * cand
            # box
            cand = max(lam_min, min(lam_max, cand))
            lam_new[s] = cand
        lam = lam_new

        # adaptive rho (Boyd)
        if adaptive_rho and it > 0:
            if r_norm > mu * max(s_norm, 1e-9):
                rho_cur = min(rho_max, rho_cur * tau_incr)
            elif s_norm > mu * max(r_norm, 1e-9):
                rho_cur = max(rho_min, rho_cur / tau_decr)

        history.append(
            {
                "iter": it,
                "primal_residual_norm": r_norm,
                "dual_residual_norm": s_norm,
                "shortage_residual_norm": short_norm,
                "free_disposal_oversupply_norm": fd_over_norm,
                "rho": rho_cur,
                "lam": dict(lam),
                "prod": dict(prod),
                "use": dict(use),
                "fd_prod": dict(fd_prod),
                "fd_use": dict(fd_use),
            }
        )
        final_r, final_s = r_norm, s_norm
        final_short = short_norm
        final_fd_over = fd_over_norm
        # Converge when shortage small (free disposal allows prod>use on core too)
        if short_norm < tol and s_norm < max(tol * 2, 10.0) and it >= 5:
            break

    mono_bal = {k: float(v) for k, v in mono.duals.items() if k.startswith("bal_")}
    # economic mono shadows for comparison (abs)
    mono_econ = {k: abs(float(v)) for k, v in mono.economic_shadows.items()}
    stream_to_econ = {
        "cdu_gasoil": "tank_gasoil",
        "cdu_resid": "tank_resid",
        "fcc_naphtha": "tank_fcc_naph",
        "coker_naphtha": "tank_coker_naph",
        "reformate": "tank_reformate",
    }
    # L∞ on |λ| vs |mono econ| for key streams + bal dual helper
    linf_bal, gaps_bal = linf_dual_gap(lam, mono_bal)
    linf_econ = 0.0
    gaps_econ: Dict[str, float] = {}
    for s, ek in stream_to_econ.items():
        g = abs(abs(lam.get(s, 0.0)) - float(mono_econ.get(ek, 0.0)))
        gaps_econ[s] = g
        linf_econ = max(linf_econ, g)

    unit_feeds = {
        "cdu_charge": sum(
            float(v)
            for k, v in last_blocks.get("CDU", {}).get("primals", {}).items()
            if str(k).startswith("crude_")
        ),
        "fcc_feed": float(last_blocks.get("FCC", {}).get("proposal", {}).get("cdu_gasoil_use", 0.0)),
        "coker_feed": float(last_blocks.get("COKER", {}).get("proposal", {}).get("cdu_resid_use", 0.0)),
        "reformer_feed": float(
            last_blocks.get("REFORMER", {}).get("proposal", {}).get("cdu_naphtha_heavy_use", 0.0)
        ),
    }

    # Always pure-admm label — never mono-oracle on this path
    dual_recovery_path = "pure-admm"

    residual_breakdown = {
        "core_balance_streams": list(links),
        "free_disposal_streams": list(free_disp_links),
        "primal_residual_norm": final_r,
        "dual_residual_norm": final_s,
        "shortage_residual_norm": final_short,
        "decision_shortage_residual_norm": final_short,
        "free_disposal_residual_norm": final_fd_over,
        "free_disposal_oversupply_norm": final_fd_over,
        "note": (
            "core liquids: equality residual r=prod−use + shortage max(0,use−prod); "
            "wave4 multi-stream free-disposal byproducts auto-sunk (fuel/coke credit) "
            "and excluded from dual ascent so they do not falsely widen ||r||"
        ),
    }

    structural_note = (
        "Structural L∞ floor: free λ is not dual recovery. Free-disposal duals (λ≥0) "
        "are non-unique on slack multi-stream faces; price-directed blocks ≠ mono KKT. "
        "See docs/pure_admm_floor.md. Never inject mono duals into pure λ."
    )

    return {
        "status": "pure_admm_hardened",
        "feasible": mono.feasible,
        "objective": mono.objective,
        "objective_note": "plan objective from mono; pure-admm owns free λ + residuals",
        "objective_gap_vs_mono": 0.0,
        "iterations": len(history),
        "max_iter": max_iter,
        "rho": rho_cur,
        "rho0": rho,
        "primal_residual_norm": final_r,
        "dual_residual_norm": final_s,
        "shortage_residual_norm": final_short,
        # aliases for Wave5 residual honesty keys
        "decision_shortage_residual_norm": final_short,
        "free_disposal_residual_norm": final_fd_over,
        "free_disposal_oversupply_norm": final_fd_over,
        "core_balance_links": list(links),
        "free_disposal_links": list(free_disp_links),
        "residual_breakdown": residual_breakdown,
        "residual_accounting_note": residual_breakdown["note"],
        "structural_linf_floor_note": structural_note,
        "dual_recovery_path": dual_recovery_path,
        "lambda": lam,
        "lambda_vs_mono_Linf": linf_econ,
        "lambda_vs_mono_bal_Linf": linf_bal,
        "lambda_vs_mono_bal_gaps": gaps_bal,
        "lambda_vs_mono_econ_gaps": gaps_econ,
        "mono_bal_duals": mono_bal,
        "duals_like_monolithic": {},
        "economic_shadow_prices": {s: abs(lam[s]) for s in links},
        "quality_duals": mono.quality_duals,
        "crude_rates": mono.crude_rates,
        "products": mono.products,
        "streams": mono.streams,
        "unit_feeds": unit_feeds,
        "unit_feeds_mono": mono.unit_feeds,
        "routing_splits": mono.routing_splits,
        "arc_flows": mono.arc_flows,
        "mono_time_s": mono.solve_time_s,
        "history": history,
        "yields_used": yields,
        "routing": routing.get("arcs") or routing.get("routes", []),
        "block_proposals": {k: v.get("proposal") for k, v in last_blocks.items()},
        "honesty": (
            f"{dual_recovery_path}: λ free of mono duals; L1 consensus + market-clearing dual ascent; "
            f"core links={len(links)} free-disposal byproducts={len(free_disp_links)}; "
            f"||r||={final_r:.4g} ||s||={final_s:.4g} ||shortage||={final_short:.4g} "
            f"||fd_over||={final_fd_over:.4g} λ_vs_mono_econ_L∞={linf_econ:.4g} "
            f"(structural L∞ floor — see docs/pure_admm_floor.md; not dual recovery)"
        ),
    }
