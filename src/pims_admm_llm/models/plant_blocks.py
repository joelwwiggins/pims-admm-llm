"""Per-unit PuLP subproblems for ADMM / parallel solves on the full plant (Wave3).

Each block maximizes local margin given dual prices λ on linking streams.
Reformer prefers heavy SR naphtha; FCC/coker naphtha are optional non-default feeds.
Blender (+ floor dispose) sinks free-disposal streams for pure-ADMM residual balance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Tuple

import pulp

from .assay_loader import load_assays_json
from .full_plant import build_yield_tables


def _solve(prob: pulp.LpProblem) -> Tuple[str, float, Dict[str, float], float]:
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    dt = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    obj = float(pulp.value(prob.objective) or 0.0)
    primals = {v.name: float(v.varValue or 0.0) for v in prob.variables()}
    return status, obj, primals, dt


def solve_cdu_block(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
) -> Dict[str, Any]:
    """CDU: choose crudes; produce cuts; revenue = λ_prod · cuts - crude cost."""
    prob = pulp.LpProblem("block_CDU", pulp.LpMaximize)
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
    rev = (
        pulp.lpSum(float(prices.get(s, 0.0)) * prod[s] for s in cuts)
        + float(prices.get("cdu_naphtha_light", prices.get("cdu_naphtha", 0.0))) * light
        + float(prices.get("cdu_naphtha_heavy", prices.get("cdu_naphtha", 0.0))) * heavy
    )
    cost = pulp.lpSum(float(c["price_usd_per_bbl"]) * crude_v[c["name"]] for c in assays["crudes"])
    prob += rev - cost
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


def solve_fcc_block(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    gasoil_available: Optional[float] = None,
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_FCC", pulp.LpMaximize)
    feed = pulp.LpVariable("fcc_feed", 0)
    caps = assays.get("capacities") or {}
    if gasoil_available is not None:
        prob += feed <= float(gasoil_available)
    prob += feed <= float(caps.get("fcc_kbd", 55))
    fy = yields["fcc"]
    naph = pulp.LpVariable("prod_fcc_naphtha", 0)
    lco = pulp.LpVariable("prod_fcc_lco", 0)
    slurry = pulp.LpVariable("prod_fcc_slurry", 0)
    prob += naph == fy["fcc_naphtha"] * feed
    prob += lco == fy["fcc_lco"] * feed
    prob += slurry == fy["fcc_slurry"] * feed
    prob += (
        float(prices.get("fcc_naphtha", 0)) * naph
        + float(prices.get("fcc_lco", 0)) * lco
        + float(prices.get("fcc_slurry", 0)) * slurry
        - float(prices.get("cdu_gasoil", 0)) * feed
        - 1.5 * feed
    )
    status, obj, primals, dt = _solve(prob)
    proposal = {
        "cdu_gasoil_use": primals.get("fcc_feed", 0.0),
        "fcc_naphtha": primals.get("prod_fcc_naphtha", 0.0),
        "fcc_lco": primals.get("prod_fcc_lco", 0.0),
        "fcc_slurry": primals.get("prod_fcc_slurry", 0.0),
    }
    return {
        "block": "FCC",
        "status": status,
        "local_obj": obj,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_coker_block(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    resid_available: Optional[float] = None,
) -> Dict[str, Any]:
    prob = pulp.LpProblem("block_COKER", pulp.LpMaximize)
    feed = pulp.LpVariable("coker_feed", 0)
    caps = assays.get("capacities") or {}
    if resid_available is not None:
        prob += feed <= float(resid_available)
    prob += feed <= float(caps.get("coker_kbd", 40))
    cy = yields["coker"]
    naph = pulp.LpVariable("prod_coker_naphtha", 0)
    go = pulp.LpVariable("prod_coker_gasoil", 0)
    prob += naph == cy["coker_naphtha"] * feed
    prob += go == cy["coker_gasoil"] * feed
    prob += (
        float(prices.get("coker_naphtha", 0)) * naph
        + float(prices.get("coker_gasoil", 0)) * go
        - float(prices.get("cdu_resid", 0)) * feed
        - 2.0 * feed
    )
    status, obj, primals, dt = _solve(prob)
    proposal = {
        "cdu_resid_use": primals.get("coker_feed", 0.0),
        "coker_naphtha": primals.get("prod_coker_naphtha", 0.0),
        "coker_gasoil": primals.get("prod_coker_gasoil", 0.0),
    }
    return {
        "block": "COKER",
        "status": status,
        "local_obj": obj,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_reformer_block(
    assays: Dict[str, Any],
    yields: Dict[str, Any],
    prices: Mapping[str, float],
    heavy_sr_avail: Optional[float] = None,
    fcc_naph_avail: Optional[float] = None,
    coker_naph_avail: Optional[float] = None,
) -> Dict[str, Any]:
    """Reformer: primarily heavy SR naphtha; FCC/coker naph optional non-default."""
    prob = pulp.LpProblem("block_REFORMER", pulp.LpMaximize)
    h_n = pulp.LpVariable("heavy_sr_naph_in", 0)
    f_n = pulp.LpVariable("fcc_naph_in", 0)
    c_n = pulp.LpVariable("coker_naph_in", 0)
    if heavy_sr_avail is not None:
        prob += h_n <= float(heavy_sr_avail)
    if fcc_naph_avail is not None:
        prob += f_n <= float(fcc_naph_avail)
    if coker_naph_avail is not None:
        prob += c_n <= float(coker_naph_avail)
    feed = h_n + f_n + c_n
    caps = assays.get("capacities") or {}
    prob += feed <= float(caps.get("reformer_kbd", 45))
    ref = pulp.LpVariable("prod_reformate", 0)
    ry = yields["reformer"]["reformate"]
    prob += ref == ry * feed
    prob += (
        float(prices.get("reformate", 0)) * ref
        - float(prices.get("cdu_naphtha_heavy", prices.get("cdu_naphtha", 0))) * h_n
        - float(prices.get("fcc_naphtha", 0)) * f_n
        - float(prices.get("coker_naphtha", 0)) * c_n
        - 1.2 * feed
        - 3.0 * f_n
        - 4.0 * c_n
    )
    status, obj, primals, dt = _solve(prob)
    proposal = {
        "cdu_naphtha_heavy_use": primals.get("heavy_sr_naph_in", 0.0),
        "fcc_naphtha_use": primals.get("fcc_naph_in", 0.0),
        "coker_naphtha_use": primals.get("coker_naph_in", 0.0),
        "reformate": primals.get("prod_reformate", 0.0),
    }
    return {
        "block": "REFORMER",
        "status": status,
        "local_obj": obj,
        "proposal": proposal,
        "primals": primals,
        "time_s": dt,
    }


def solve_blender_block(
    assays: Dict[str, Any],
    prices: Mapping[str, float],
    available: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Sink / product block: buy linking streams at λ, sell products + floor dispose.

    Floor dispose clears free-disposal residuals for pure ADMM even when multi-component
    recipes cannot absorb every stream. Recipe is planning-grade linear (not quality pooling).
    """
    products = assays.get("products") or {}
    # Wave3 defaults only — avoid merging assay fractions (can sum > 1).
    use_recipes: Dict[str, Dict[str, float]] = {
        "gasoline": {
            "cdu_naphtha_light": 0.20,
            "cdu_naphtha": 0.15,
            "fcc_naphtha": 0.30,
            "reformate": 0.25,
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
    stream_set = {
        "cdu_naphtha",
        "cdu_naphtha_light",
        "cdu_naphtha_heavy",
        "cdu_distillate",
        "cdu_gasoil",
        "cdu_resid",
        "fcc_naphtha",
        "fcc_lco",
        "fcc_slurry",
        "coker_naphtha",
        "coker_gasoil",
        "reformate",
    }
    for rec in use_recipes.values():
        stream_set.update(rec.keys())

    floor = {
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

    prob = pulp.LpProblem("block_BLENDER", pulp.LpMaximize)
    prod_v = {
        name: pulp.LpVariable(
            f"prod_{name}",
            0,
            float(spec.get("max_demand_kbd", 1e6)) if isinstance(spec, dict) else 1e6,
        )
        for name, spec in products.items()
    }
    use_v = {s: pulp.LpVariable(f"use_{s}", 0) for s in sorted(stream_set)}
    dispose_v = {s: pulp.LpVariable(f"dispose_{s}", 0) for s in sorted(stream_set)}

    # Recipe attribution
    attributed: Dict[str, list] = {s: [] for s in use_v}
    for pname, pvar in prod_v.items():
        rec = use_recipes.get(pname) or {}
        for stream, frac in rec.items():
            if stream not in use_v:
                continue
            u_ps = pulp.LpVariable(f"use_{pname}_{stream}", 0)
            prob += u_ps == float(frac) * pvar
            attributed[stream].append(u_ps)
    for stream, parts in attributed.items():
        if parts:
            prob += use_v[stream] == pulp.lpSum(parts)
        else:
            # no recipe demand → recipe use fixed at 0; dispose handles sink
            prob += use_v[stream] == 0

    if available is not None:
        for s in use_v:
            if s in available:
                prob += use_v[s] + dispose_v[s] <= float(available[s]) + 1e-9

    rev = pulp.lpSum(
        float(products[n].get("price_usd_per_bbl", 0.0) if isinstance(products[n], dict) else 0.0)
        * prod_v[n]
        for n in prod_v
    )
    rev_disp = pulp.lpSum(float(floor.get(s, 40.0)) * dispose_v[s] for s in dispose_v)
    cost = pulp.lpSum(float(prices.get(s, 0.0)) * (use_v[s] + dispose_v[s]) for s in use_v)
    prob += rev + rev_disp - cost - 0.1 * pulp.lpSum(list(prod_v.values()))

    status, obj, primals, dt = _solve(prob)
    stream_use = {
        s: float(primals.get(f"use_{s}", 0.0)) + float(primals.get(f"dispose_{s}", 0.0))
        for s in use_v
    }
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


def solve_all_plant_blocks(
    prices: Optional[Mapping[str, float]] = None,
    assays: Optional[Dict[str, Any]] = None,
    *,
    include_blender: bool = True,
    reformer_take_cracked: bool = False,
) -> Dict[str, Any]:
    assays = assays or load_assays_json()
    yields = build_yield_tables(assays)
    prices = dict(prices or {})
    # default economic seeds from product netbacks (only fill missing keys)
    prices.setdefault("reformate", 100.0)
    prices.setdefault("fcc_naphtha", 98.0)
    prices.setdefault("coker_naphtha", 88.0)
    prices.setdefault("cdu_gasoil", 85.0)
    prices.setdefault("cdu_resid", 55.0)
    prices.setdefault("fcc_lco", 95.0)
    prices.setdefault("fcc_slurry", 50.0)
    prices.setdefault("coker_gasoil", 90.0)
    prices.setdefault("cdu_naphtha", 92.0)
    prices.setdefault("cdu_naphtha_light", 90.0)
    prices.setdefault("cdu_naphtha_heavy", 88.0)
    prices.setdefault("cdu_distillate", 100.0)

    cdu = solve_cdu_block(assays, yields, prices)
    # Cascade: upstream production caps downstream use (Gauss–Seidel ADMM-friendly)
    fcc = solve_fcc_block(
        assays, yields, prices, gasoil_available=cdu["proposal"].get("cdu_gasoil")
    )
    coker = solve_coker_block(
        assays, yields, prices, resid_available=cdu["proposal"].get("cdu_resid")
    )
    reformer = solve_reformer_block(
        assays,
        yields,
        prices,
        heavy_sr_avail=cdu["proposal"].get("cdu_naphtha_heavy"),
        fcc_naph_avail=(fcc["proposal"].get("fcc_naphtha") if reformer_take_cracked else 0.0),
        coker_naph_avail=(coker["proposal"].get("coker_naphtha") if reformer_take_cracked else 0.0),
    )
    blocks: Dict[str, Any] = {
        "CDU": cdu,
        "FCC": fcc,
        "COKER": coker,
        "REFORMER": reformer,
    }
    if include_blender:
        avail = {
            "cdu_naphtha": float(cdu["proposal"].get("cdu_naphtha", 0.0)),
            "cdu_naphtha_light": float(cdu["proposal"].get("cdu_naphtha_light", 0.0)),
            "cdu_naphtha_heavy": max(
                0.0,
                float(cdu["proposal"].get("cdu_naphtha_heavy", 0.0))
                - float(reformer["proposal"].get("cdu_naphtha_heavy_use", 0.0)),
            ),
            "cdu_distillate": float(cdu["proposal"].get("cdu_distillate", 0.0)),
            "cdu_gasoil": max(
                0.0,
                float(cdu["proposal"].get("cdu_gasoil", 0.0))
                - float(fcc["proposal"].get("cdu_gasoil_use", 0.0)),
            ),
            "cdu_resid": max(
                0.0,
                float(cdu["proposal"].get("cdu_resid", 0.0))
                - float(coker["proposal"].get("cdu_resid_use", 0.0)),
            ),
            "fcc_naphtha": max(
                0.0,
                float(fcc["proposal"].get("fcc_naphtha", 0.0))
                - float(reformer["proposal"].get("fcc_naphtha_use", 0.0)),
            ),
            "fcc_lco": float(fcc["proposal"].get("fcc_lco", 0.0)),
            "fcc_slurry": float(fcc["proposal"].get("fcc_slurry", 0.0)),
            "coker_naphtha": max(
                0.0,
                float(coker["proposal"].get("coker_naphtha", 0.0))
                - float(reformer["proposal"].get("coker_naphtha_use", 0.0)),
            ),
            "coker_gasoil": float(coker["proposal"].get("coker_gasoil", 0.0)),
            "reformate": float(reformer["proposal"].get("reformate", 0.0)),
        }
        blocks["BLENDER"] = solve_blender_block(assays, prices, available=avail)
    return {
        "blocks": blocks,
        "prices": prices,
        "yields": yields,
    }
