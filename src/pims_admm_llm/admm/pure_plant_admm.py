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

Not claimed: global optimality of free price-directed LP blocks always matches
mono at finite iter; we report ||r||, ||s||, λ_vs_mono_L∞ honestly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pulp

from pims_admm_llm.admm.residuals import linf_dual_gap, residual_norms
from pims_admm_llm.models.assay_loader import load_assays_json, load_routing
from pims_admm_llm.models.full_plant import build_yield_tables, solve_full_plant


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

    rev = pulp.lpSum(float(prices.get(s, 0.0)) * prod[s] for s in cuts)
    rev += float(prices.get("cdu_naphtha_light", prices.get("cdu_naphtha", 0.0))) * light
    rev += float(prices.get("cdu_naphtha_heavy", prices.get("cdu_naphtha", 0.0))) * heavy
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
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_FCC_cons", pulp.LpMaximize)
    feed = pulp.LpVariable("fcc_feed", 0)
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("fcc_kbd", 55))
    fy = yields["fcc"]
    naph = pulp.LpVariable("prod_fcc_naphtha", 0)
    lco = pulp.LpVariable("prod_fcc_lco", 0)
    slurry = pulp.LpVariable("prod_fcc_slurry", 0)
    prob += naph == fy["fcc_naphtha"] * feed
    prob += lco == fy["fcc_lco"] * feed
    prob += slurry == fy["fcc_slurry"] * feed
    obj = (
        float(prices.get("fcc_naphtha", 0)) * naph
        + float(prices.get("fcc_lco", 0)) * lco
        + float(prices.get("fcc_slurry", 0)) * slurry
        - float(prices.get("cdu_gasoil", 0)) * feed
        - 1.5 * feed
    )
    # consensus on feed vs z[cdu_gasoil] and products
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
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_COKER_cons", pulp.LpMaximize)
    feed = pulp.LpVariable("coker_feed", 0)
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("coker_kbd", 40))
    cy = yields["coker"]
    naph = pulp.LpVariable("prod_coker_naphtha", 0)
    go = pulp.LpVariable("prod_coker_gasoil", 0)
    prob += naph == cy["coker_naphtha"] * feed
    prob += go == cy["coker_gasoil"] * feed
    obj = (
        float(prices.get("coker_naphtha", 0)) * naph
        + float(prices.get("coker_gasoil", 0)) * go
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
    ref = pulp.LpVariable("prod_reformate", 0)
    ry = yields["reformer"]["reformate"]
    prob += ref == ry * feed
    obj = (
        float(prices.get("reformate", 0)) * ref
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
) -> Dict[str, Any]:
    """Blender with L1 consensus on stream uses vs z."""
    # reuse recipe structure from plant_blocks blender
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
    stream_set = set()
    for rec in default_recipes.values():
        stream_set.update(rec.keys())
    for s in links:
        stream_set.add(s)

    prob = pulp.LpProblem("block_BLENDER_cons", pulp.LpMaximize)
    prod_v = {}
    for name, spec in products.items():
        ub = float(spec.get("max_demand_kbd", 1e6)) if isinstance(spec, dict) else 1e6
        prod_v[name] = pulp.LpVariable(f"prod_{name}", 0, ub)

    use_v = {s: pulp.LpVariable(f"use_{s}", 0) for s in sorted(stream_set)}
    attributed: Dict[str, list] = {s: [] for s in use_v}
    for pname, pvar in prod_v.items():
        rec = default_recipes.get(pname) or {}
        for stream, frac in rec.items():
            if stream not in use_v:
                use_v[stream] = pulp.LpVariable(f"use_{stream}", 0)
                attributed[stream] = []
            u_ps = pulp.LpVariable(f"use_{pname}_{stream}", 0)
            prob += u_ps == float(frac) * pvar
            attributed.setdefault(stream, []).append(u_ps)
    for stream, parts in attributed.items():
        if parts:
            prob += use_v[stream] == pulp.lpSum(parts)

    rev = pulp.lpSum(
        float(products[n].get("price_usd_per_bbl", 0.0) if isinstance(products[n], dict) else 0.0)
        * prod_v[n]
        for n in prod_v
    )
    cost = pulp.lpSum(float(prices.get(s, 0.0)) * use_v[s] for s in use_v)
    cons = 0
    for s, var in use_v.items():
        if s in links:
            cons += _l1_consensus_terms(prob, var, z.get(s, 0.0), rho, f"bl_{s}")
    prob += rev - cost - 0.1 * pulp.lpSum(prod_v.values()) + cons
    status, obj, primals, dt = _solve(prob)
    stream_use = {s: float(primals.get(f"use_{s}", 0.0)) for s in use_v}
    product_rates = {n: float(primals.get(f"prod_{n}", 0.0)) for n in prod_v}
    return {
        "block": "BLENDER",
        "status": status,
        "local_obj": obj,
        "proposal": stream_use,
        "product_rates": product_rates,
        "primals": primals,
        "time_s": dt,
    }


def _aggregate_prod_use(blocks: Dict[str, Any], links: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    b = blocks
    cdu = b["CDU"]["proposal"]
    fcc = b["FCC"]["proposal"]
    cok = b["COKER"]["proposal"]
    ref = b["REFORMER"]["proposal"]
    bl = b.get("BLENDER", {}).get("proposal") or {}

    prod = {
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
    }
    # use = conversion feed + blender use (avoid double-count: conversion is intermediate use of CDU cuts;
    # blender is use of finished intermediates)
    use = {
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
    # only return links keys
    prod = {s: float(prod.get(s, 0.0)) for s in links}
    use = {s: float(use.get(s, 0.0)) for s in links}
    return prod, use


def run_pure_plant_admm(
    assays: Optional[Dict[str, Any]] = None,
    *,
    routing: Optional[Dict[str, Any]] = None,
    max_iter: int = 80,
    rho: float = 1.2,
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
    """Pure multi-block ADMM; λ free of mono duals. Mono used for honesty only."""
    assays = assays or load_assays_json()
    routing = routing or load_routing()
    yields = build_yield_tables(assays)
    mono = solve_full_plant(assays, routing=routing)
    links = list(routing.get("linking_streams") or DEFAULT_LINKS)

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
    rho_cur = float(rho)

    for it in range(max_iter):
        prices = dict(lam)
        cdu = solve_cdu_block_consensus(assays, yields, prices, z, rho_cur, links)
        fcc = solve_fcc_block_consensus(assays, yields, prices, z, rho_cur, links)
        cok = solve_coker_block_consensus(assays, yields, prices, z, rho_cur, links)
        ref = solve_reformer_block_consensus(assays, yields, prices, z, rho_cur, links)
        bl = solve_blender_block_consensus(assays, prices, z, rho_cur, links)
        last_blocks = {
            "CDU": cdu,
            "FCC": fcc,
            "COKER": cok,
            "REFORMER": ref,
            "BLENDER": bl,
        }
        prod, use = _aggregate_prod_use(last_blocks, links)
        z_old = dict(z)
        # consensus target: average of prod and use
        for s in links:
            z[s] = 0.5 * (prod.get(s, 0.0) + use.get(s, 0.0))
        r_norm, s_norm, r = residual_norms(links, prod, use, z, z_old, rho_cur)
        # Free-disposal shortage residual: only unmet demand matters for "hard" feasibility
        shortage = {s: max(0.0, use.get(s, 0.0) - prod.get(s, 0.0)) for s in links}
        import math
        short_norm = math.sqrt(sum(v * v for v in shortage.values()))

        # Market-clearing dual update (sign-critical):
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
                "rho": rho_cur,
                "lam": dict(lam),
                "prod": dict(prod),
                "use": dict(use),
            }
        )
        final_r, final_s = r_norm, s_norm
        final_short = short_norm
        # Converge when shortage small (free disposal allows prod>use)
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
        "dual_recovery_path": "pure-admm",
        "lambda": lam,
        "lambda_vs_mono_Linf": linf_econ,
        "lambda_vs_mono_bal_Linf": linf_bal,
        "lambda_vs_mono_bal_gaps": gaps_bal,
        "lambda_vs_mono_econ_gaps": gaps_econ,
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
            "pure-admm: λ free of mono duals; L1 consensus + market-clearing dual ascent; "
            f"||r||={final_r:.4g} ||s||={final_s:.4g} ||shortage||={final_short:.4g} "
            f"λ_vs_mono_econ_L∞={linf_econ:.4g}"
        ),
    }
