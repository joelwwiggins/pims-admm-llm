"""Excel PIMS-shaped model → mono + ADMM solve → results workbook.

MVP path for planners who live in Excel:
  load template/export → RefineryData → simple mono + package ADMM
  → results .xlsx + JSON-serializable report.

Sheets expected on input (see ``load_assays_excel`` / ``write_template_excel``):
  Crudes, Products, Capacities, optional Intermediates.

ADMM imports are lazy to avoid models ↔ admm circular import at package load.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from .assay_loader import (
    assays_to_refinery_data,
    load_assays_excel,
    write_template_excel,
)
from .data import RefineryData

if TYPE_CHECKING:
    from pims_admm_llm.admm.coordinator import ADMMConfig

PathLike = Union[str, Path]


def _default_admm_config():
    """Lazy import to avoid models ↔ admm circular import at package load."""
    from pims_admm_llm.admm.coordinator import ADMMConfig

    # Tuned multi-crude assay Excel path (template):
    #   ρ=8, dual_step=0.5, max_iter=120 → gap ~0.02%, online-λ L∞ ~3 vs mono.
    # Recovered blender duals can sit on a different LP face; report online λ for shadows.
    return ADMMConfig(
        backend="qp_l2",
        rho=8.0,
        max_iter=120,
        dual_step=0.5,
        stable_crude_iters=25,
        recover_primal=True,
        abs_tol=1e-3,
        rel_tol=1e-3,
        verbose=False,
    )


def load_pims_excel(path: PathLike) -> Dict[str, Any]:
    """Load PIMS-shaped workbook → assay package dict."""
    return load_assays_excel(path)


def excel_to_refinery_data(path: PathLike) -> RefineryData:
    """Load Excel and convert to classic RefineryData (naphtha/distillate/gasoil/residue)."""
    return assays_to_refinery_data(load_pims_excel(path))


def _mono_shadow_prices(duals: Dict[str, float], intermediates: list[str]) -> Dict[str, float]:
    """Economic $/bbl value of intermediates from maximize-form balance duals."""
    out: Dict[str, float] = {}
    for n in intermediates:
        bal = duals.get(f"balance_{n}")
        yld = duals.get(f"yield_{n}")
        if yld is not None:
            out[n] = float(yld)
        elif bal is not None:
            # maximize: balance dual is typically negative of economic value
            out[n] = float(-bal) if bal <= 0 else float(bal)
        else:
            out[n] = 0.0
    return out


def _online_economic_shadows(online_duals: Dict[str, float]) -> Dict[str, float]:
    """Economic $/bbl from free ADMM λ (maximize-form λ ≈ mono balance dual ≤ 0)."""
    out: Dict[str, float] = {}
    for n, lam in online_duals.items():
        v = float(lam)
        out[n] = float(-v) if v <= 0 else float(v)
    return out


def _linf_shadow_gap(
    mono_sh: Dict[str, float], admm_sh: Dict[str, float], keys: list[str]
) -> float:
    if not keys:
        return 0.0
    return max(abs(float(mono_sh.get(k, 0.0)) - float(admm_sh.get(k, 0.0))) for k in keys)


def _verdict(
    mono: Dict[str, Any],
    admm: Dict[str, Any],
    gap_rel: float,
    dual_linf: float,
    *,
    gap_tol: float = 0.005,
    dual_tol: float = 15.0,
) -> str:
    if not mono.get("feasible"):
        return "FAIL — mono infeasible"
    if not admm.get("feasible"):
        return "FAIL — ADMM infeasible / no activity"
    if gap_rel > gap_tol:
        return f"FAIL — objective gap {gap_rel:.4%} > {gap_tol:.2%}"
    if dual_linf > dual_tol:
        return (
            f"PASS_SOFT — obj gap ok ({gap_rel:.4%}) but dual L∞={dual_linf:.2f} > {dual_tol:.0f}"
        )
    return (
        f"PASS — both feasible; gap≤{gap_tol:.2%}; dual L∞≤{dual_tol:.0f} "
        f"(gap={gap_rel:.4%}, dual_L∞={dual_linf:.2f})"
    )


def model_calculations_from_data(data: RefineryData) -> Dict[str, Any]:
    """Export LP coefficients/equations used *before* ADMM runs (Excel math tables).

    Mirrors ``admm/simple_mono.py``: yields, blend recipes, obj terms, bounds,
    block map. Solver stays CBC/ADMM in Python — not Excel formulas.
    """
    inter = list(data.intermediates)
    yields_rows: List[Dict[str, Any]] = []
    for c in data.crudes:
        row: Dict[str, Any] = {
            "crude": c.name,
            "price_usd_per_bbl": float(c.price_usd_per_bbl),
            "max_supply_kbd": float(c.max_supply_kbd),
            "api": float(c.api),
            "sulfur_wt_pct": float(c.sulfur_wt_pct),
        }
        ysum = 0.0
        for i in inter:
            y = float(c.yields.get(i, 0.0))
            row[f"y_{i}"] = y
            ysum += y
        row["yield_sum"] = ysum
        yields_rows.append(row)

    blend_rows: List[Dict[str, Any]] = []
    for prod, recipe in (data.blend_recipes or {}).items():
        row: Dict[str, Any] = {"product": prod}
        rsum = 0.0
        for i in inter:
            v = float(recipe.get(i, 0.0))
            row[f"use_{i}"] = v
            rsum += v
        row["recipe_sum"] = rsum
        spec = data.products.get(prod)
        if spec:
            row["price_usd_per_bbl"] = float(spec.price_usd_per_bbl)
            row["max_demand_kbd"] = float(spec.max_demand_kbd)
        blend_rows.append(row)

    obj_terms: List[Dict[str, Any]] = []
    for c in data.crudes:
        obj_terms.append(
            {
                "term": f"crude_cost[{c.name}]",
                "variable": f"crude_{c.name}",
                "coeff": -float(c.price_usd_per_bbl),
                "unit": "USD/bbl",
                "formula": f"obj += ({-c.price_usd_per_bbl}) * crude_{c.name}",
                "block": "CDU",
            }
        )
    for name, spec in data.products.items():
        obj_terms.append(
            {
                "term": f"product_rev[{name}]",
                "variable": f"product_{name}",
                "coeff": float(spec.price_usd_per_bbl),
                "unit": "USD/bbl",
                "formula": f"obj += ({spec.price_usd_per_bbl}) * product_{name}",
                "block": "BLENDER",
            }
        )

    equations: List[Dict[str, Any]] = [
        {
            "id": "OBJ",
            "name": "margin",
            "type": "maximize",
            "equation": "max  Σ_p price_p * product_p  −  Σ_c price_c * crude_c",
            "notes": "No utility/inventory terms on classic Excel path",
        },
        {
            "id": "CAP_CDU",
            "name": "cdu_capacity",
            "type": "<=",
            "equation": f"Σ_c crude_c  ≤  {float(data.cdu_capacity_kbd)}",
            "notes": "Total CDU charge limit (kbd)",
        },
    ]
    for i in inter:
        terms = " + ".join(
            f"{float(c.yields.get(i, 0.0)):.6g}*crude_{c.name}" for c in data.crudes
        )
        equations.append(
            {
                "id": f"YLD_{i}",
                "name": f"yield_{i}",
                "type": "=",
                "equation": f"prod_{i} = {terms}",
                "notes": "CDU yield definition (linear)",
            }
        )
        equations.append(
            {
                "id": f"BAL_{i}",
                "name": f"balance_{i}",
                "type": "=",
                "equation": f"prod_{i} − use_{i} = 0",
                "notes": "Linking constraint CDU↔Blender (ADMM consensus target)",
            }
        )
        rhs_parts = []
        for p, recipe in (data.blend_recipes or {}).items():
            coef = float(recipe.get(i, 0.0))
            if abs(coef) > 1e-15:
                rhs_parts.append(f"{coef:.6g}*product_{p}")
        rhs = " + ".join(rhs_parts) if rhs_parts else "0"
        equations.append(
            {
                "id": f"BLD_{i}",
                "name": f"blend_use_{i}",
                "type": ">=",
                "equation": f"use_{i} ≥ {rhs}",
                "notes": "Blender intermediate requirement from product recipes",
            }
        )

    bounds: List[Dict[str, Any]] = []
    for c in data.crudes:
        bounds.append(
            {
                "variable": f"crude_{c.name}",
                "lo": 0.0,
                "up": float(c.max_supply_kbd),
                "block": "CDU",
            }
        )
    for name, spec in data.products.items():
        bounds.append(
            {
                "variable": f"product_{name}",
                "lo": 0.0,
                "up": float(spec.max_demand_kbd),
                "block": "BLENDER",
            }
        )
    for i in inter:
        bounds.append({"variable": f"prod_{i}", "lo": 0.0, "up": None, "block": "CDU"})
        bounds.append(
            {"variable": f"use_{i}", "lo": 0.0, "up": None, "block": "BLENDER"}
        )

    blocks = [
        {
            "block": "CDU",
            "role": "production",
            "local_vars": "crude_*, prod_*",
            "local_constraints": "cdu_capacity, yield_*",
            "sees_prices": "λ on intermediates (ADMM duals)",
            "exports": "prod_i proposals → consensus z_i",
        },
        {
            "block": "BLENDER",
            "role": "consumption / products",
            "local_vars": "use_*, product_*",
            "local_constraints": "blend_use_*",
            "sees_prices": "λ on intermediates",
            "exports": "use_i proposals → consensus z_i",
        },
        {
            "block": "MASTER_ADMM",
            "role": "coordination only (outside Excel)",
            "local_vars": "λ_i, z_i",
            "local_constraints": "dual ascent on residual r_i = prod_i − use_i",
            "sees_prices": "updates λ with dual_step / ρ",
            "exports": "converged λ, z, recovered primal",
        },
    ]
    linking = [
        {
            "stream": i,
            "cdu_var": f"prod_{i}",
            "blender_var": f"use_{i}",
            "balance": f"prod_{i} = use_{i}",
            "admm_role": "linking / consensus stream",
        }
        for i in inter
    ]
    return {
        "form": "classic_2block_excel_path",
        "source": "admm/simple_mono.py + admm/subproblems CDU/blender",
        "solver_note": (
            "Coefficients below are the model. Mono uses CBC once; ADMM iterates "
            "the same blocks. Excel does not run the solver."
        ),
        "intermediates": inter,
        "cdu_capacity_kbd": float(data.cdu_capacity_kbd),
        "yields": yields_rows,
        "blend_recipes": blend_rows,
        "objective_terms": obj_terms,
        "equations": equations,
        "bounds": bounds,
        "blocks": blocks,
        "linking": linking,
    }


def model_calc_check(
    model: Dict[str, Any], mono: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Plug mono rates into yield/blend identities (numeric audit)."""
    rows: List[Dict[str, Any]] = []
    crude_rates = mono.get("crude_rates") or {}
    prod = mono.get("intermediate_prod") or {}
    use = mono.get("intermediate_use") or {}
    products = mono.get("product_rates") or {}
    yields = {r["crude"]: r for r in model.get("yields") or []}
    blends = {r["product"]: r for r in model.get("blend_recipes") or []}
    inter = model.get("intermediates") or []

    for i in inter:
        pred = 0.0
        for cname, rate in crude_rates.items():
            yrow = yields.get(cname) or {}
            pred += float(yrow.get(f"y_{i}", 0.0)) * float(rate)
        actual = float(prod.get(i, 0.0))
        rows.append(
            {
                "check": f"yield_{i}",
                "predicted": pred,
                "actual": actual,
                "abs_err": abs(pred - actual),
                "ok": abs(pred - actual) <= 1e-4,
            }
        )
        u = float(use.get(i, 0.0))
        rows.append(
            {
                "check": f"balance_{i}",
                "predicted": actual,
                "actual": u,
                "abs_err": abs(actual - u),
                "ok": abs(actual - u) <= 1e-4,
            }
        )

    for i in inter:
        need = 0.0
        for pname, prate in products.items():
            brow = blends.get(pname) or {}
            need += float(brow.get(f"use_{i}", 0.0)) * float(prate)
        actual_u = float(use.get(i, 0.0))
        rows.append(
            {
                "check": f"blend_use_{i}_rhs",
                "predicted": need,
                "actual": actual_u,
                "abs_err": max(0.0, need - actual_u),
                "ok": actual_u + 1e-4 >= need,
            }
        )
    return rows



def block_angular_matrix(model: Dict[str, Any]) -> Dict[str, Any]:
    """Classic block-angular view of the Excel-path LP (for teaching + audit).

    Layout (constraint rows × variable columns)::

              crude_*   prod_*   use_*   product_*
      CDU:      A1        A1
      BLENDER:                     A2       A2
      LINK:               B        B

    A1 = Calc_Yields + cdu capacity; A2 = Calc_Blend; B = balance linking.
    MASTER_ADMM is not a matrix block — it updates duals λ / consensus z outside Excel.
    """
    inter = list(model.get("intermediates") or [])
    crudes = [r["crude"] for r in model.get("yields") or []]
    products = [r["product"] for r in model.get("blend_recipes") or []]
    col_vars: List[str] = (
        [f"crude_{c}" for c in crudes]
        + [f"prod_{i}" for i in inter]
        + [f"use_{i}" for i in inter]
        + [f"product_{p}" for p in products]
    )

    # Map var → column index for sparse coeff rows
    def empty_row() -> Dict[str, Any]:
        return {v: None for v in col_vars}

    matrix_rows: List[Dict[str, Any]] = []
    # --- CDU capacity ---
    r = empty_row()
    r.update(
        {
            "row_id": "CAP_CDU",
            "block": "CDU (A1)",
            "type": "<=",
            "rhs": model.get("cdu_capacity_kbd"),
            "source_sheet": "input Capacities / Calc_ModelNote",
            "editable": "edit max CDU on input Capacities sheet",
            "equation": f"Σ crude ≤ {model.get('cdu_capacity_kbd')}",
        }
    )
    for c in crudes:
        r[f"crude_{c}"] = 1.0
    matrix_rows.append(r)

    yields_by = {row["crude"]: row for row in model.get("yields") or []}
    for i in inter:
        r = empty_row()
        r.update(
            {
                "row_id": f"YLD_{i}",
                "block": "CDU (A1)",
                "type": "=",
                "rhs": 0.0,
                "source_sheet": "Calc_Yields",
                "editable": "edit y_* on Calc_Yields / input Crudes y_*",
                "equation": f"prod_{i} − Σ y[c,{i}]*crude_c = 0",
            }
        )
        r[f"prod_{i}"] = 1.0
        for c in crudes:
            y = float((yields_by.get(c) or {}).get(f"y_{i}", 0.0))
            if abs(y) > 1e-15:
                r[f"crude_{c}"] = -y
        matrix_rows.append(r)

    blends_by = {row["product"]: row for row in model.get("blend_recipes") or []}
    for i in inter:
        r = empty_row()
        r.update(
            {
                "row_id": f"BLD_{i}",
                "block": "BLENDER (A2)",
                "type": ">=",
                "rhs": 0.0,
                "source_sheet": "Calc_Blend",
                "editable": "edit use_* recipe on Calc_Blend",
                "equation": f"use_{i} − Σ recipe[p,{i}]*product_p ≥ 0",
            }
        )
        r[f"use_{i}"] = 1.0
        for p in products:
            coef = float((blends_by.get(p) or {}).get(f"use_{i}", 0.0))
            if abs(coef) > 1e-15:
                r[f"product_{p}"] = -coef
        matrix_rows.append(r)

    for i in inter:
        r = empty_row()
        r.update(
            {
                "row_id": f"BAL_{i}",
                "block": "LINKING (B)",
                "type": "=",
                "rhs": 0.0,
                "source_sheet": "Calc_Linking",
                "editable": "structure fixed; values solved; ADMM prices this row (λ)",
                "equation": f"prod_{i} − use_{i} = 0",
            }
        )
        r[f"prod_{i}"] = 1.0
        r[f"use_{i}"] = -1.0
        matrix_rows.append(r)

    # Compact block map for a second small table
    map_rows = [
        {
            "region": "A1 local",
            "block": "CDU",
            "variables": "crude_*, prod_*",
            "constraints": "CAP_CDU, YLD_*",
            "data_from": "Calc_Yields + Capacities",
            "modify_how": "Change crude yields/prices/supply on input Crudes; CDU cap on Capacities",
        },
        {
            "region": "A2 local",
            "block": "BLENDER",
            "variables": "use_*, product_*",
            "constraints": "BLD_*",
            "data_from": "Calc_Blend + Products",
            "modify_how": "Change recipes on Calc_Blend / product prices & demand on Products",
        },
        {
            "region": "B linking",
            "block": "balances",
            "variables": "prod_* and use_* (same streams)",
            "constraints": "BAL_*",
            "data_from": "Calc_Linking",
            "modify_how": "Do not edit structure; ADMM duals λ live on these rows (see Shadows)",
        },
        {
            "region": "Master",
            "block": "MASTER_ADMM",
            "variables": "λ_i, z_i, ρ",
            "constraints": "dual ascent / consensus (not in A matrix)",
            "data_from": "Summary admm_rho / dual_recovery_path",
            "modify_how": "Tune ρ, dual_step, max_iter in code/config — not Excel cells",
        },
    ]

    legend = [
        {
            "symbol": "A1",
            "meaning": "CDU block diagonal — local technology + capacity",
            "sheet": "Calc_Yields",
        },
        {
            "symbol": "A2",
            "meaning": "Blender block diagonal — recipes + product bounds",
            "sheet": "Calc_Blend",
        },
        {
            "symbol": "B",
            "meaning": "Linking columns/rows coupling blocks (material balance)",
            "sheet": "Calc_Linking / BAL_* rows",
        },
        {
            "symbol": "λ / u",
            "meaning": "ADMM dual / shadow price on linking residual",
            "sheet": "Shadows.admm_online_econ",
        },
        {
            "symbol": "z",
            "meaning": "Consensus target for intermediates (ADMM master)",
            "sheet": "not exported as activity; residual on Summary",
        },
    ]

    process = [
        {
            "step": 1,
            "name": "Edit input",
            "where": "crudes_template.xlsx: Crudes, Products, Capacities, Intermediates",
            "what": "Planner data — prices, supplies, yields, demands, CDU cap",
        },
        {
            "step": 2,
            "name": "Materialize model",
            "where": "Calc_Yields, Calc_Blend, Calc_Objective, Calc_Equations, Calc_Bounds",
            "what": "Pipeline expands input into LP coefficients (this is the A-matrix content)",
        },
        {
            "step": 3,
            "name": "See block structure",
            "where": "Calc_BlockAngular + Calc_Blocks + Calc_Linking",
            "what": "How A splits into A1, A2, B — what is local vs linking",
        },
        {
            "step": 4,
            "name": "Solve mono (truth)",
            "where": "CBC on full matrix",
            "what": "One-shot full LP → Crudes_mono, Products_mono, mono_shadow",
        },
        {
            "step": 5,
            "name": "Solve ADMM (decomposition)",
            "where": "Python coordinator; not Excel",
            "what": "Each block sub-LP with λ-penalty; master updates λ,z until ||r|| small",
        },
        {
            "step": 6,
            "name": "Read results",
            "where": "Summary, rate sheets, Shadows, Calc_Check",
            "what": "VERDICT, gap, dual L∞, economic shadows; Calc_Check proves yields/recipes hold",
        },
    ]

    return {
        "columns": col_vars,
        "matrix_rows": matrix_rows,
        "map_rows": map_rows,
        "legend": legend,
        "process": process,
    }



def submodel_matrix_tables(model: Dict[str, Any]) -> Dict[str, Any]:
    """Dense local submodel matrices A1 (CDU), A2 (Blender), and linking B.

    These are the *data* of each block (yields, recipes), not only structural labels.
    Column sets are local to each block so the submodel is readable without the full
    sparse mega-matrix.
    """
    inter = list(model.get("intermediates") or [])
    crudes = [r["crude"] for r in model.get("yields") or []]
    products = [r["product"] for r in model.get("blend_recipes") or []]
    yields_by = {row["crude"]: row for row in model.get("yields") or []}
    blends_by = {row["product"]: row for row in model.get("blend_recipes") or []}

    # --- A1 technology table (submodel data = yield matrix + economics) ---
    cdu_tech: List[Dict[str, Any]] = []
    for c in crudes:
        y = yields_by.get(c) or {}
        row: Dict[str, Any] = {
            "submodel": "CDU (A1)",
            "crude": c,
            "price_usd_per_bbl": y.get("price_usd_per_bbl"),
            "max_supply_kbd": y.get("max_supply_kbd"),
            "api": y.get("api"),
            "sulfur_wt_pct": y.get("sulfur_wt_pct"),
        }
        for i in inter:
            row[f"y_{i}"] = float(y.get(f"y_{i}", 0.0))
        row["yield_sum"] = y.get("yield_sum")
        row["local_vars"] = f"crude_{c}, prod_* (shared)"
        row["data_sheet"] = "Calc_Yields / input Crudes"
        cdu_tech.append(row)

    # Dense A1 constraint matrix: only CDU local columns
    cdu_cols = [f"crude_{c}" for c in crudes] + [f"prod_{i}" for i in inter]
    cdu_A: List[Dict[str, Any]] = []
    r: Dict[str, Any] = {
        "constraint": "CAP_CDU",
        "type": "<=",
        "rhs": model.get("cdu_capacity_kbd"),
        "meaning": "total crude charge",
        "submodel_data_from": "Capacities.cdu_kbd",
    }
    for c in crudes:
        r[f"crude_{c}"] = 1.0
    for i in inter:
        r[f"prod_{i}"] = None
    cdu_A.append(r)
    for i in inter:
        r = {
            "constraint": f"YLD_{i}",
            "type": "=",
            "rhs": 0.0,
            "meaning": f"prod_{i} = Σ y[c,{i}]*crude_c  (submodel yield tech)",
            "submodel_data_from": f"Calc_Yields.y_{i}",
        }
        for c in crudes:
            yv = float((yields_by.get(c) or {}).get(f"y_{i}", 0.0))
            r[f"crude_{c}"] = -yv
        for j in inter:
            r[f"prod_{j}"] = 1.0 if j == i else None
        cdu_A.append(r)

    # --- A2 technology table (blend recipes) ---
    blend_tech: List[Dict[str, Any]] = []
    for p in products:
        b = blends_by.get(p) or {}
        row = {
            "submodel": "BLENDER (A2)",
            "product": p,
            "price_usd_per_bbl": b.get("price_usd_per_bbl"),
            "max_demand_kbd": b.get("max_demand_kbd"),
        }
        for i in inter:
            row[f"recipe_{i}"] = float(b.get(f"use_{i}", 0.0))
        row["recipe_sum"] = b.get("recipe_sum")
        row["local_vars"] = f"product_{p}, use_* (shared)"
        row["data_sheet"] = "Calc_Blend / input Products"
        blend_tech.append(row)

    blend_cols = [f"use_{i}" for i in inter] + [f"product_{p}" for p in products]
    blend_A: List[Dict[str, Any]] = []
    for i in inter:
        r = {
            "constraint": f"BLD_{i}",
            "type": ">=",
            "rhs": 0.0,
            "meaning": f"use_{i} ≥ Σ recipe[p,{i}]*product_p  (submodel recipe tech)",
            "submodel_data_from": f"Calc_Blend.use_{i}",
        }
        for j in inter:
            r[f"use_{j}"] = 1.0 if j == i else None
        for p in products:
            coef = float((blends_by.get(p) or {}).get(f"use_{i}", 0.0))
            r[f"product_{p}"] = -coef if abs(coef) > 1e-15 else None
        blend_A.append(r)

    # --- Linking B (no local tech table; structure only + dual home) ---
    link_A: List[Dict[str, Any]] = []
    for i in inter:
        link_A.append(
            {
                "constraint": f"BAL_{i}",
                "type": "=",
                "rhs": 0.0,
                "prod_coeff": 1.0,
                "use_coeff": -1.0,
                "meaning": f"prod_{i} − use_{i} = 0",
                "submodel_data_from": "structure fixed; ADMM λ on this row → Shadows",
                "cdu_var": f"prod_{i}",
                "blender_var": f"use_{i}",
            }
        )

    # Compact "where is the submodel data" index (classic 2-block + base-delta units)
    index = [
        {
            "block": "CDU (A1)",
            "tech_table": "Submodel_CDU_Tech",
            "constraint_matrix": "Submodel_CDU_A",
            "includes": "classic path: y_i, crude prices/supply, CAP_CDU, YLD_* (Excel mono/ADMM)",
            "edit_via": "input Crudes (y_*, price, max_supply) + Capacities",
        },
        {
            "block": "BLENDER (A2)",
            "tech_table": "Submodel_Blender_Tech",
            "constraint_matrix": "Submodel_Blender_A",
            "includes": "blend recipes, product prices/demand, BLD_* rows with numeric coeffs",
            "edit_via": "input Products + recipes (Calc_Blend / pipeline defaults)",
        },
        {
            "block": "FCC (base-delta)",
            "tech_table": "Submodel_FCC (PIMS BASE/DELTA matrix)",
            "constraint_matrix": "Submodel_FCC + Submodel_FCC_FixedYield",
            "includes": "FEED_FFD, BASE, D_* deltas, E_BASE_REF, E_quality_REF (−999), FREE — Aspen How-To 07 style",
            "edit_via": "base_delta.build_fcc_base_delta (export); cascade solve_cdu_fcc",
        },
        {
            "block": "COKER (base-delta)",
            "tech_table": "Submodel_Coker (PIMS BASE/DELTA matrix)",
            "constraint_matrix": "Submodel_Coker + Submodel_Coker_FixedYield",
            "includes": "FEED_CFD, BASE, D_* deltas, E-rows, FREE — Aspen How-To 07 style",
            "edit_via": "base_delta.build_coker_base_delta (export); cascade solve_cdu_fcc_coker",
        },
        {
            "block": "LINKING (B)",
            "tech_table": "(none — balances only)",
            "constraint_matrix": "Submodel_Linking_B",
            "includes": "classic path: prod_i − use_i = 0; duals λ are ADMM/mono shadows",
            "edit_via": "do not edit; see Shadows for prices",
        },
        {
            "block": "MASTER_ADMM",
            "tech_table": "(outside Excel)",
            "constraint_matrix": "(not an A block)",
            "includes": "ρ, dual ascent, consensus z",
            "edit_via": "code ADMMConfig (rho, dual_step, max_iter)",
        },
    ]
    # Unit base-delta LP submodels (FCC + COKER) — always attached so Excel has full unit suite
    unit_bd = base_delta_unit_submodel_tables()
    return {
        "cdu_tech": cdu_tech,
        "cdu_A": cdu_A,
        "cdu_cols": cdu_cols,
        "blend_tech": blend_tech,
        "blend_A": blend_A,
        "blend_cols": blend_cols,
        "link_A": link_A,
        "index": index,
        **unit_bd,
    }


def base_delta_unit_submodel_tables() -> Dict[str, Any]:
    """Dense FCC + COKER base-delta LP submodel tables for Excel export.

    Classic Excel mono/ADMM still solves CDU+Blender only. These sheets expose the
    PIMS-style BASE/DELTA unit submodels (yields, deltas, SOS1 modes, exits, local A)
    from ``models.base_delta`` so planners can access FCC/Coker submodel data.
    """
    from .base_delta import (
        build_coker_base_delta,
        build_fcc_base_delta,
        process_modes_coker,
        process_modes_fcc,
    )

    def _flat_conditions(cond: Mapping[str, Any] | None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (cond or {}).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out[f"{k}.{kk}"] = vv
            else:
                out[str(k)] = v
        return out

    def _unit_pack(unit_name: str, model, modes: List[Dict[str, Any]]) -> Dict[str, Any]:
        products = list(model.products)
        exits_by = {e.stream: e for e in model.exits}
        # --- Tech (BASE yields + exit routing) ---
        tech: List[Dict[str, Any]] = []
        for p in products:
            ex = exits_by.get(p)
            tech.append(
                {
                    "submodel": unit_name,
                    "product": p,
                    "base_yield": float(model.base_yields.get(p, 0.0)),
                    "default_sink": getattr(ex, "default_sink", None) if ex else None,
                    "basis": getattr(ex, "basis", None) if ex else None,
                    "alt_sinks": (
                        ",".join(getattr(ex, "alt_sinks", []) or []) if ex else ""
                    ),
                    "exit_note": getattr(ex, "note", None) if ex else None,
                    "form": "BASE + DELTA · SOS1 process modes",
                    "code": f"build_{unit_name.lower()}_base_delta",
                }
            )
        # yield_sum of base (liquids may exclude coke intentionally; still report sum)
        tech_sum = sum(float(model.base_yields.get(p, 0.0)) for p in products)
        for row in tech:
            row["base_yield_sum_all_products"] = tech_sum

        # --- Reference feed + process conditions (one row each) ---
        ref_feed = [
            {"submodel": unit_name, "kind": "reference_feed", "key": k, "value": v}
            for k, v in (model.reference_feed or {}).items()
        ]
        ref_cond_flat = _flat_conditions(model.reference_conditions)
        ref_cond = [
            {"submodel": unit_name, "kind": "reference_conditions", "key": k, "value": v}
            for k, v in ref_cond_flat.items()
        ]
        drivers = [
            {"submodel": unit_name, "driver": d, "role": "BASE/DELTA process or feed quality"}
            for d in (model.drivers or [])
        ]

        # --- Deltas: product × driver ---
        deltas: List[Dict[str, Any]] = []
        for p in products:
            dmap = model.deltas.get(p) or {}
            for driver, coef in dmap.items():
                deltas.append(
                    {
                        "submodel": unit_name,
                        "product": p,
                        "driver": driver,
                        "delta_yield_per_unit": float(coef),
                        "meaning": f"Δy[{p}] / Δ{driver} around reference",
                    }
                )

        # --- Modes (SOS1 severity bands) ---
        mode_rows: List[Dict[str, Any]] = []
        for m in modes:
            row: Dict[str, Any] = {
                "submodel": unit_name,
                "mode_id": m.get("id"),
                "unit": m.get("unit", unit_name),
            }
            for ck, cv in _flat_conditions(m.get("conditions")).items():
                row[f"cond_{ck}"] = cv
            ylds = m.get("yields") or {}
            for p in products:
                row[f"y_{p}"] = float(ylds.get(p, 0.0))
            row["yield_sum"] = sum(float(ylds.get(p, 0.0)) for p in products)
            mode_rows.append(row)

        # --- Exits table ---
        exit_rows: List[Dict[str, Any]] = []
        for e in model.exits:
            exit_rows.append(
                {
                    "submodel": unit_name,
                    "stream": e.stream,
                    "default_sink": e.default_sink,
                    "basis": e.basis,
                    "required": bool(e.required),
                    "alt_sinks": ",".join(e.alt_sinks or []),
                    "note": e.note,
                }
            )

        # --- Dense local A: SOS1 modes + mode-linked yield identities ---
        # vars: feed, y_mode_*, prod_p
        # MODE_SUM: Σ y_mode = 1
        # YLD_p: prod_p − Σ_m y[m,p] * (feed · mode_m)  ≈  linearized as
        #   for teaching: coeffs of feed at each mode shown; full big-M is in cdu_fcc.py
        mode_ids = [str(m.get("id")) for m in modes]
        A: List[Dict[str, Any]] = []
        r_sum: Dict[str, Any] = {
            "constraint": "MODE_SOS1",
            "type": "=",
            "rhs": 1.0,
            "meaning": "exactly one process mode active (SOS1)",
            "submodel_data_from": f"Submodel_{unit_name}_Modes",
            "feed": None,
        }
        for mid in mode_ids:
            r_sum[f"mode_{mid}"] = 1.0
        for p in products:
            r_sum[f"prod_{p}"] = None
        A.append(r_sum)

        # Prefer mid severity as "base" capacity note
        A.append(
            {
                "constraint": f"CAP_{unit_name}",
                "type": "<=",
                "rhs": None,
                "meaning": f"feed ≤ unit capacity (set in full-plant / base_delta solve; not classic Excel mono)",
                "submodel_data_from": "Capacities / unit_specs (full plant path)",
                "feed": 1.0,
                **{f"mode_{mid}": None for mid in mode_ids},
                **{f"prod_{p}": None for p in products},
            }
        )

        # Yield rows: for each product, show mode yield coeffs on feed (teaching form)
        # prod_p = Σ_m y[m,p] * feed when mode m active
        for p in products:
            r: Dict[str, Any] = {
                "constraint": f"YLD_{p}",
                "type": "=",
                "rhs": 0.0,
                "meaning": (
                    f"prod_{p} = y[mode,{p}] * feed  (mode-linked; big-M in cdu_fcc._mode_linked_product)"
                ),
                "submodel_data_from": f"Submodel_{unit_name}_Modes.y_{p}",
                "feed": None,  # mode-dependent; see mode_* columns as y[m,p] when that mode on
            }
            for m in modes:
                mid = str(m.get("id"))
                yv = float((m.get("yields") or {}).get(p, 0.0))
                # teaching: coefficient of feed under mode m is y[m,p]; stored under mode col
                r[f"mode_{mid}"] = yv
            for j in products:
                r[f"prod_{j}"] = 1.0 if j == p else None
            A.append(r)

        # Feed balance note row (source stream)
        feed_src = "cdu_gasoil" if unit_name == "FCC" else "cdu_resid"
        A.append(
            {
                "constraint": f"FEED_FROM_{feed_src}",
                "type": "=",
                "rhs": 0.0,
                "meaning": f"feed = upstream {feed_src} (auto-wire when unit active)",
                "submodel_data_from": "base_delta.auto_wire_edges_for_units / cdu_fcc cascade",
                "feed": 1.0,
                **{f"mode_{mid}": None for mid in mode_ids},
                **{f"prod_{p}": None for p in products},
            }
        )

        notes = [
            {"submodel": unit_name, "note": n} for n in (model.notes or [])
        ]
        notes.append(
            {
                "submodel": unit_name,
                "note": (
                    "PIMS How-To 07 style: Submodel_{u} is the BASE/DELTA matrix "
                    "(FEED + BASE + DELTA_* columns, E-rows, FREE). "
                    "Classic Excel mono/ADMM still solves CDU+Blender only; "
                    "cascade solve = solve_cdu_fcc / solve_cdu_fcc_coker."
                ).format(u=unit_name if unit_name != "Coker" else "Coker"),
            }
        )
        notes.append(
            {
                "submodel": unit_name,
                "note": (
                    "Aspen video map: fixed/direct yield → Submodel_*_FixedYield "
                    "(mode columns). Base-Delta → Submodel_* (this matrix). "
                    "−999 under FEED = quality pickup placeholder (PIMS convention)."
                ),
            }
        )

        # --- PIMS How-To 07 BASE/DELTA matrix (primary planner view) ---
        # Columns: FEED + BASE + D_<driver> ; Rows: product MB, E_BASE_REF,
        # E_<quality>_REF, FREE. Matches AspenTech "Submodels continued".
        feed_tag = "FFD" if unit_name == "FCC" else "CFD"
        feed_src = "cdu_gasoil" if unit_name == "FCC" else "cdu_resid"
        feed_col = f"FEED_{feed_tag}"
        drivers_list = list(model.drivers or [])
        # Prefer feed-quality drivers first, then process conditions
        ref_all: Dict[str, Any] = {}
        ref_all.update(dict(model.reference_feed or {}))
        ref_all.update(_flat_conditions(model.reference_conditions))

        pims_matrix: List[Dict[str, Any]] = []
        # Product material-balance rows (BASE + DELTA yield coeffs)
        for p in products:
            r: Dict[str, Any] = {
                "row": f"MB_{p}",
                "row_type": "material_balance",
                "rhs": 0.0,
                feed_col: None,
                "BASE": float(model.base_yields.get(p, 0.0)),
            }
            dmap = model.deltas.get(p) or {}
            for drv in drivers_list:
                r[f"D_{drv}"] = float(dmap.get(drv, 0.0)) if drv in dmap else None
            r["meaning"] = (
                f"{p} yield = BASE·y0 + Σ D_k·(Δy/Δk); "
                f"exit→{getattr(exits_by.get(p), 'default_sink', '?')}"
            )
            r["equation"] = (
                f"activity_BASE * {r['BASE']:.6g}"
                + "".join(
                    f" + activity_D_{drv} * {float((dmap or {}).get(drv, 0.0)):.6g}"
                    for drv in drivers_list
                    if abs(float((dmap or {}).get(drv, 0.0))) > 1e-18
                )
                + f"  →  {p}"
            )
            r["pims_note"] = "yield MB row (video: material balance rows under BASE/DELTA)"
            pims_matrix.append(r)

        # E_BASE_REF: BASE = FEED (drives yields at actual feed rate)
        e_base: Dict[str, Any] = {
            "row": "E_BASE_REF",
            "row_type": "E_row",
            "rhs": 0.0,
            feed_col: 1.0,
            "BASE": -1.0,
            "meaning": f"BASE activity = FEED_{feed_tag} (feed rate drives base yields)",
            "equation": f"FEED_{feed_tag} − BASE = 0",
            "pims_note": "video: E row sets base vector equal to feed vector",
        }
        for drv in drivers_list:
            e_base[f"D_{drv}"] = None
        pims_matrix.append(e_base)

        # E_<driver>_REF: quality/condition barrel balance
        for drv in drivers_list:
            q_base = ref_all.get(drv)
            e_q: Dict[str, Any] = {
                "row": f"E_{drv}_REF",
                "row_type": "E_row_quality",
                "rhs": 0.0,
                feed_col: -999.0 if drv in (model.reference_feed or {}) else None,
                "BASE": float(q_base) if q_base is not None else None,
                "meaning": (
                    f"quality/condition balance for {drv}: "
                    f"BASE holds q_base={q_base}; D_{drv} = deviation; "
                    f"{'-999 on FEED = PIMS quality pickup from ' + feed_src if drv in (model.reference_feed or {}) else 'process condition (no stream quality pickup)'}"
                ),
                "equation": (
                    f"BASE*{q_base} + D_{drv}*1"
                    + (f" + FEED*(-999→q_stream)" if drv in (model.reference_feed or {}) else "")
                    + " = 0  (rearranged: D ≈ FEED*(q − q_base) style)"
                    if q_base is not None
                    else f"D_{drv} free deviation around reference"
                ),
                "pims_note": (
                    "video: E quality REF — base quality under BASE, deviation under DELTA, "
                    "−999 under FEED for stream quality tag"
                ),
            }
            for d2 in drivers_list:
                e_q[f"D_{d2}"] = 1.0 if d2 == drv else None
            pims_matrix.append(e_q)

        # FREE row: all DELTA columns free (+/−)
        free_row: Dict[str, Any] = {
            "row": "FREE",
            "row_type": "FREE",
            "rhs": None,
            feed_col: None,
            "BASE": None,
            "meaning": "DELTA vectors free (quality may be above or below base)",
            "equation": "D_* unrestricted sign",
            "pims_note": "video: put 1 under each DELTA in FREE row",
        }
        for drv in drivers_list:
            free_row[f"D_{drv}"] = 1.0
        pims_matrix.append(free_row)

        # CAP row (capacity on feed)
        cap_row: Dict[str, Any] = {
            "row": f"CAP_{unit_name}",
            "row_type": "capacity",
            "rhs": None,
            feed_col: 1.0,
            "BASE": None,
            "meaning": f"FEED_{feed_tag} ≤ unit capacity (full-plant / unit_specs)",
            "equation": f"FEED_{feed_tag} ≤ CAP",
            "pims_note": "capacity on feed activity",
        }
        for drv in drivers_list:
            cap_row[f"D_{drv}"] = None
        pims_matrix.append(cap_row)

        # Legend / meta rows (append as notes in matrix for in-sheet reading)
        pims_matrix.append(
            {
                "row": "META_feed_source",
                "row_type": "meta",
                "rhs": None,
                feed_col: None,
                "BASE": None,
                **{f"D_{drv}": None for drv in drivers_list},
                "meaning": f"FEED_{feed_tag} stream = {feed_src} (auto-wire when unit active)",
                "equation": f"FEED ← {feed_src}",
                "pims_note": "VolSamp analogue: SCCU cat cracker / SDHT style unit table",
            }
        )
        pims_matrix.append(
            {
                "row": "META_base_point",
                "row_type": "meta",
                "rhs": None,
                feed_col: None,
                "BASE": None,
                **{f"D_{drv}": None for drv in drivers_list},
                "meaning": "BASE yields at reference_feed + reference_conditions (see Submodel_*_Ref)",
                "equation": "y = y0(q_base, u_base)",
                "pims_note": "video: collect base quality and corresponding yield fractions first",
            }
        )

        # --- Fixed / direct yield view (video first half) via SOS1 modes ---
        fixed_yield: List[Dict[str, Any]] = []
        mode_ids = [str(m.get("id")) for m in modes]
        for p in products:
            r = {
                "row": f"MB_{p}",
                "row_type": "fixed_yield_MB",
                "rhs": 0.0,
                feed_col: None,
                "meaning": f"fixed yield fractions per discrete mode (not quality-dependent in this view)",
                "pims_note": "video: fixed/direct yield submodel — coeffs fixed on feed; modes ≈ discrete yields",
            }
            for m in modes:
                mid = str(m.get("id"))
                r[f"MODE_{mid}"] = float((m.get("yields") or {}).get(p, 0.0))
            fixed_yield.append(r)
        r_sos = {
            "row": "MODE_SOS1",
            "row_type": "SOS1",
            "rhs": 1.0,
            feed_col: None,
            "meaning": "exactly one severity mode active",
            "pims_note": "discrete mode selection (MIP/SOS1); alternative to continuous DELTA",
        }
        for mid in mode_ids:
            r_sos[f"MODE_{mid}"] = 1.0
        fixed_yield.append(r_sos)
        r_feed = {
            "row": "FEED_BALANCE",
            "row_type": "link",
            "rhs": 0.0,
            feed_col: 1.0,
            "meaning": f"feed from {feed_src}",
            "pims_note": "feed column only (fixed-yield style)",
        }
        for mid in mode_ids:
            r_feed[f"MODE_{mid}"] = None
        fixed_yield.append(r_feed)

        return {
            "tech": tech,
            "ref_feed": ref_feed,
            "ref_cond": ref_cond,
            "drivers": drivers,
            "deltas": deltas,
            "modes": mode_rows,
            "exits": exit_rows,
            "A": A,
            "notes": notes,
            "products": products,
            "mode_ids": mode_ids,
            "pims_matrix": pims_matrix,
            "fixed_yield": fixed_yield,
            "feed_col": feed_col,
            "feed_tag": feed_tag,
            "feed_src": feed_src,
            "drivers_list": drivers_list,
        }

    fcc_model = build_fcc_base_delta()
    coker_model = build_coker_base_delta()
    fcc = _unit_pack("FCC", fcc_model, list(process_modes_fcc(fcc_model)))
    coker = _unit_pack("Coker", coker_model, list(process_modes_coker(coker_model)))

    return {
        "fcc_tech": fcc["tech"],
        "fcc_ref_feed": fcc["ref_feed"],
        "fcc_ref_cond": fcc["ref_cond"],
        "fcc_drivers": fcc["drivers"],
        "fcc_deltas": fcc["deltas"],
        "fcc_modes": fcc["modes"],
        "fcc_exits": fcc["exits"],
        "fcc_A": fcc["A"],
        "fcc_notes": fcc["notes"],
        "fcc_pims_matrix": fcc["pims_matrix"],
        "fcc_fixed_yield": fcc["fixed_yield"],
        "fcc_feed_col": fcc["feed_col"],
        "coker_tech": coker["tech"],
        "coker_ref_feed": coker["ref_feed"],
        "coker_ref_cond": coker["ref_cond"],
        "coker_drivers": coker["drivers"],
        "coker_deltas": coker["deltas"],
        "coker_modes": coker["modes"],
        "coker_exits": coker["exits"],
        "coker_A": coker["A"],
        "coker_notes": coker["notes"],
        "coker_pims_matrix": coker["pims_matrix"],
        "coker_fixed_yield": coker["fixed_yield"],
        "coker_feed_col": coker["feed_col"],
    }


def run_excel_pipeline(
    input_path: PathLike,
    *,
    admm_config: Optional["ADMMConfig"] = None,
    results_xlsx: Optional[PathLike] = None,
    results_json: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Load Excel → simple mono + ADMM → optional Excel/JSON export.

    Uses the classic 2-block (CDU / Blender) path that matches package ADMM tests:
    ``solve_simple_monolithic`` + ``run_admm`` with primal recovery.
    """
    from pims_admm_llm.admm.coordinator import run_admm
    from pims_admm_llm.admm.simple_mono import solve_simple_monolithic

    t0 = time.perf_counter()
    input_path = Path(input_path)
    assays = load_pims_excel(input_path)
    data = assays_to_refinery_data(assays)
    cfg = admm_config or _default_admm_config()

    mono = solve_simple_monolithic(data, msg=False)
    mono_feasible = mono.status == "Optimal" and float(mono.objective) > 0
    mono_part: Dict[str, Any] = {
        "status": mono.status,
        "objective": float(mono.objective),
        "feasible": mono_feasible,
        "wall_time_s": float(mono.solve_time_s),
        "crude_rates": {k: float(v) for k, v in mono.crude_rates.items()},
        "product_rates": {k: float(v) for k, v in mono.product_rates.items()},
        "intermediate_prod": {k: float(v) for k, v in mono.intermediate_prod.items()},
        "intermediate_use": {k: float(v) for k, v in mono.intermediate_use.items()},
        "shadow_prices": _mono_shadow_prices(mono.duals, list(data.intermediates)),
        "duals": {k: float(v) for k, v in mono.duals.items()},
    }

    admm = run_admm(data, cfg)
    admm_activity = sum(float(v) for v in admm.crude_rates.values())
    admm_feasible = (
        (bool(admm.recovered) or bool(admm.converged))
        and admm_activity > 1.0
        and float(admm.objective) > 0
    )
    online = {k: float(v) for k, v in admm.online_duals.items()}
    # Primary shadows: free online λ (tracks mono balance duals on multi-crude slate).
    # Recovered blender duals are secondary — can sit on a different LP face.
    online_econ = _online_economic_shadows(online)
    recovered_econ = {
        k: float(v) for k, v in (admm.economic_shadow_prices or {}).items()
    }
    mono_sh = mono_part["shadow_prices"]
    dual_linf_online = _linf_shadow_gap(mono_sh, online_econ, list(data.intermediates))
    dual_linf_recovered = _linf_shadow_gap(
        mono_sh, recovered_econ, list(data.intermediates)
    )

    admm_part: Dict[str, Any] = {
        "status": admm.status,
        "converged": bool(admm.converged),
        "feasible": admm_feasible,
        "objective": float(admm.objective),
        "iteration_count": int(admm.iterations),
        "wall_time_s": float(admm.solve_time_s),
        "crude_rates": {k: float(v) for k, v in admm.crude_rates.items()},
        "product_rates": {k: float(v) for k, v in admm.product_rates.items()},
        "intermediate_prod": {k: float(v) for k, v in admm.intermediate_prod.items()},
        "intermediate_use": {k: float(v) for k, v in admm.intermediate_use.items()},
        "shadow_prices": online_econ,
        "shadow_prices_recovered": recovered_econ,
        "online_duals": online,
        "rho": float(admm.rho),
        "primal_residual": float(admm.r_norm),
        "dual_residual": float(admm.s_norm),
        "dual_recovery_path": (
            f"package-admm/{admm.backend}+recover_primal+online_lambda_shadows"
            if admm.recovered
            else f"package-admm/{admm.backend}+online_lambda_shadows"
        ),
        "recovered": bool(admm.recovered),
        "backend": admm.backend,
        "dual_linf_vs_mono_online": dual_linf_online,
        "dual_linf_vs_mono_recovered": dual_linf_recovered,
    }

    gap_abs = abs(float(admm_part["objective"]) - mono_part["objective"])
    gap_rel = gap_abs / max(abs(mono_part["objective"]), 1e-9)

    report: Dict[str, Any] = {
        "meta": {
            "input": str(input_path),
            "format": (assays.get("meta") or {}).get("format", "excel_pims_shaped"),
            "n_crudes": len(data.crudes),
            "cdu_capacity_kbd": float(data.cdu_capacity_kbd),
            "intermediates": list(data.intermediates),
            "pipeline_wall_s": time.perf_counter() - t0,
            "admm_config": {
                "backend": cfg.backend,
                "rho": cfg.rho,
                "max_iter": cfg.max_iter,
                "dual_step": cfg.dual_step,
                "recover_primal": cfg.recover_primal,
            },
            "shadow_note": (
                "Primary ADMM shadows = economic value from free online λ "
                "(-λ when maximize-form). Recovered blender duals reported "
                "separately; they need not match mono when crude slate differs."
            ),
        },
        "model": model_calculations_from_data(data),
        "mono": mono_part,
        "admm": admm_part,
        "comparison": {
            "objective_gap_abs": gap_abs,
            "objective_gap_rel": gap_rel,
            "dual_linf_online": dual_linf_online,
            "dual_linf_recovered": dual_linf_recovered,
            "both_feasible": bool(mono_part["feasible"] and admm_part["feasible"]),
            # Dual honesty roles (presentation only — VERDICT still uses online L∞).
            "dual_gate": "online_lambda",
            "verdict_dual_gate": "online_only",
            "dual_linf_online_role": "PRIMARY",
            "dual_linf_recovered_role": "SECONDARY",
            "recovered_secondary": True,
        },
        "verdict": _verdict(mono_part, admm_part, gap_rel, dual_linf_online),
    }
    # Planner honesty glance package (presentation only; no solve/VERDICT change).
    report["meta"]["planner_honesty"] = format_planner_honesty_package(report)["meta"]

    if results_xlsx:
        write_results_excel(results_xlsx, report)
        report["meta"]["results_xlsx"] = str(results_xlsx)
    if results_json:
        jp = Path(results_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, default=str))
        report["meta"]["results_json"] = str(jp)
    return report


def format_dual_honesty_summary(report: Dict[str, Any]) -> Dict[str, str]:
    """PRIMARY online-λ vs SECONDARY recovered dual honesty (presentation only).

    Pure formatter for How_to / Shadows footer / demo. Does not change VERDICT math
    (still gates on dual_linf_online only).
    """
    admm = report.get("admm") or {}
    cmp_ = report.get("comparison") or {}
    path_ = str(admm.get("dual_recovery_path") or "package-admm")
    online = cmp_.get("dual_linf_online")
    recovered = cmp_.get("dual_linf_recovered")
    try:
        online_s = f"{float(online):.4g}" if online is not None else "n/a"
    except (TypeError, ValueError):
        online_s = str(online)
    try:
        recovered_s = f"{float(recovered):.4g}" if recovered is not None else "n/a"
    except (TypeError, ValueError):
        recovered_s = str(recovered)
    return {
        "primary_role": "PRIMARY",
        "secondary_role": "SECONDARY",
        "primary_metric": "dual_linf_online",
        "secondary_metric": "dual_linf_recovered",
        "dual_linf_online": online_s,
        "dual_linf_recovered": recovered_s,
        "verdict_dual_gate": "online_only",
        "dual_gate": "online_lambda",
        "dual_recovery_path": path_,
        "recovered_secondary": "true",
        "planner_one_liner": (
            f"PRIMARY free online λ dual L∞≈{online_s} (gates VERDICT, tol≤15); "
            f"SECONDARY recovered blender dual L∞≈{recovered_s} (face-dependent; not gate); "
            f"path={path_}; not pure-ADMM dual ownership; not TF dual recovery."
        ),
        "shadows_role_banner": (
            "PRIMARY admm_online_econ = free online λ economic value (gates dual L∞ / VERDICT dual check). "
            "SECONDARY admm_recovered_econ = blender recovery LP face (may diverge; not VERDICT gate)."
        ),
    }


def format_tf_offline_units_howto() -> Dict[str, str]:
    """Static offline TF multi-unit How_to strings (isolation-safe; no TF import).

    Mirrors the honesty contract of tf_linear_blocks without importing that module
    or tensorflow (E6/E14 isolation lock). Planner-facing only — not a solve claim.
    """
    one_liner = (
        "Offline exact-linear TF/numpy kernels available for FCC + COKER + CDU "
        "(tf_linear_fcc / tf_linear_coker / tf_linear_cdu; base_delta y0/D affine). "
        "Not on this Case 1 solve (classic_2block_excel_path; mono+ADMM still CDU+Blender). "
        "Offline surface: solver=False, dual_recovery_path=None, on_excel_case1_path=False. "
        "Case 1 duals remain PRIMARY free online λ / SECONDARY recovered blender "
        "(see duals_primary_secondary); TF never owns duals."
    )
    return {
        "topic": "tf_offline_units",
        "units": "FCC+COKER+CDU",
        "on_case1_solve": "false",
        "form": "classic_2block_excel_path",
        "solver": "false",
        "dual_recovery_path": "None",
        "on_excel_case1_path": "false",
        "planner_one_liner": one_liner,
    }


def format_tf_offline_priced_howto() -> Dict[str, str]:
    """Static offline priced residual How_to strings (isolation-safe; no TF import).

    Planner-facing note that multi-unit priced residual harness exists for
    FCC+COKER+CDU exact-linear economics readiness. Does **not** import
    tf_linear_blocks / tensorflow. Not a Case 1 solve claim; duals still PRIMARY
    online-λ / SECONDARY recovered.
    """
    one_liner = (
        "Offline priced residual harness exists for FCC+COKER+CDU exact-linear "
        "economics readiness (synthetic product prices vs affine+postprocess evaluate; "
        "optional local box direction on drivers). "
        "Still not on this Case 1 solve (classic_2block_excel_path). "
        "TF/offline surface dual_recovery_path=None; prices are not ADMM λ or Case 1 shadows. "
        "Case 1 duals remain PRIMARY free online λ / SECONDARY recovered blender."
    )
    return {
        "topic": "tf_offline_priced",
        "units": "FCC+COKER+CDU",
        "on_case1_solve": "false",
        "form": "classic_2block_excel_path",
        "solver": "false",
        "dual_recovery_path": "None",
        "on_excel_case1_path": "false",
        "price_source": "synthetic_offline_demo",
        "planner_one_liner": one_liner,
    }


def format_tf_offline_timing_howto() -> Dict[str, str]:
    """Static offline block-solve timing How_to strings (isolation-safe; no TF import).

    Planner-facing note that cached multi-unit block-solve timing / readiness
    harness exists for FCC+COKER+CDU. Does **not** import tf_linear_blocks /
    tensorflow. Timings are readiness only — not Case 1 wall time; duals still
    PRIMARY online-λ / SECONDARY recovered.
    """
    one_liner = (
        "Offline cached multi-unit block-solve timing harness exists for "
        "FCC+COKER+CDU exact-linear shells (cached AffineCoeffs + numpy affine "
        "forward / optional local box step; readiness compose with parity+priced). "
        "Still not on this Case 1 solve (classic_2block_excel_path). "
        "TF/offline surface dual_recovery_path=None; timings are readiness not "
        "shadows / not online λ / not Case 1 wall time. "
        "Case 1 duals remain PRIMARY free online λ / SECONDARY recovered blender."
    )
    return {
        "topic": "tf_offline_timing",
        "units": "FCC+COKER+CDU",
        "on_case1_solve": "false",
        "form": "classic_2block_excel_path",
        "solver": "false",
        "dual_recovery_path": "None",
        "on_excel_case1_path": "false",
        "planner_one_liner": one_liner,
    }


# Static offline TF unit list for Index / Summary / meta (isolation-safe; no TF import).
_OFFLINE_TF_UNITS = "FCC,COKER,CDU"
# Index OFFLINE_TF one-liner: kernels + priced residual readiness + block-solve timing
# readiness. Hard negatives: not Case 1; TF dual_recovery_path=None; prices ≠ duals;
# timings ≠ Case 1 wall / ≠ online λ. Static only — never call live readiness reports.
_OFFLINE_TF_INDEX_WHAT = (
    "FCC+COKER+CDU exact-linear kernels offline + priced residual readiness + "
    "block-solve timing readiness — NOT on classic Case 1 solve; "
    "dual_recovery_path=None on TF surface; prices not duals; timings not Case 1 wall / not online λ"
)
_OFFLINE_TF_PRICED_NOTE = (
    "offline priced residual readiness (FCC+COKER+CDU) — synthetic prices not ADMM λ / not Case 1 shadows"
)
_OFFLINE_TF_TIMING_NOTE = (
    "offline block-solve timing readiness (FCC+COKER+CDU) — not Case 1 wall time / not duals / not online λ"
)
_OFFLINE_TF_READINESS_NOTE = (
    "offline TF readiness package: units + priced residual + block-solve timing — "
    "not on classic Case 1; dual_recovery_path=None on TF surface; not wire shipped"
)


def format_planner_honesty_package(report: Dict[str, Any]) -> Dict[str, Any]:
    """Pure composer for Index / Summary / Calc_Check / meta honesty glance.

    Isolation-safe: reuses format_dual_honesty_summary + format_tf_offline_*_howto
    helpers and report fields only — never imports tensorflow / tf_linear_blocks,
    and never calls live multi_unit_* / offline_block_solve_readiness_report.
    Presentation packaging only; does not change VERDICT math.
    Dual PRIMARY online-λ / SECONDARY recovered packaging is read-only preserve
    (#12/#14); this wave only extends offline TF readiness glance (priced + timing).
    """
    dual = format_dual_honesty_summary(report)
    tf_off = format_tf_offline_units_howto()
    tf_priced = format_tf_offline_priced_howto()
    tf_timing = format_tf_offline_timing_howto()
    model = report.get("model") or {}
    cmp_ = report.get("comparison") or {}
    form = str(model.get("form") or tf_off["form"])
    path_ = dual["dual_recovery_path"]
    online = cmp_.get("dual_linf_online")
    recovered = cmp_.get("dual_linf_recovered")
    try:
        online_f = float(online) if online is not None else None
    except (TypeError, ValueError):
        online_f = None
    try:
        recovered_f = float(recovered) if recovered is not None else None
    except (TypeError, ValueError):
        recovered_f = None

    meta = {
        "form": form,
        "model_form": form,
        "dual_gate": dual["dual_gate"],
        "verdict_dual_gate": dual["verdict_dual_gate"],
        "dual_linf_online": online_f if online_f is not None else dual["dual_linf_online"],
        "dual_linf_recovered": (
            recovered_f if recovered_f is not None else dual["dual_linf_recovered"]
        ),
        "dual_linf_online_role": dual["primary_role"],
        "dual_linf_recovered_role": dual["secondary_role"],
        "offline_tf_units": _OFFLINE_TF_UNITS,
        "offline_tf_priced_ready": True,  # static harness-existence flag; not live report
        "offline_tf_timing_ready": True,  # static harness-existence flag; not live report
        "offline_tf_priced": _OFFLINE_TF_PRICED_NOTE,
        "offline_tf_timing": _OFFLINE_TF_TIMING_NOTE,
        "offline_tf_readiness_note": _OFFLINE_TF_READINESS_NOTE,
        "on_excel_case1_path": False,
        "tf_on_excel_case1_path": False,
        "dual_recovery_path": path_,
        "tf_dual_recovery_path": None,
        "planner_one_liner": (
            f"form={form}; dual_gate={dual['dual_gate']} ({dual['verdict_dual_gate']}); "
            f"PRIMARY online L∞={dual['dual_linf_online']}; "
            f"SECONDARY recovered L∞={dual['dual_linf_recovered']}; "
            f"offline_tf_units={_OFFLINE_TF_UNITS} + priced residual readiness + "
            f"block-solve timing readiness not on Case 1; "
            f"tf_on_excel_case1_path=False; path={path_}."
        ),
    }
    index_row = {
        "block": "OFFLINE_TF",
        "sheet": "How_to_read",
        "what": _OFFLINE_TF_INDEX_WHAT,
    }
    summary_pairs = [
        ("model_form", form),
        ("dual_gate", dual["dual_gate"]),
        ("verdict_dual_gate", dual["verdict_dual_gate"]),
        ("dual_linf_online", online),
        (
            "dual_linf_online_role",
            "PRIMARY — free online λ; gates VERDICT dual L∞",
        ),
        ("dual_linf_recovered", recovered),
        (
            "dual_linf_recovered_role",
            "SECONDARY — blender recovery face; not VERDICT gate",
        ),
        ("offline_tf_units", _OFFLINE_TF_UNITS),
        ("tf_on_excel_case1_path", False),
        ("offline_tf_note", _OFFLINE_TF_READINESS_NOTE),
        ("offline_tf_priced", _OFFLINE_TF_PRICED_NOTE),
        ("offline_tf_timing", _OFFLINE_TF_TIMING_NOTE),
        ("offline_tf_readiness_note", _OFFLINE_TF_READINESS_NOTE),
    ]
    return {
        "index_row": index_row,
        "summary_pairs": summary_pairs,
        "meta": meta,
        "dual": dual,
        "tf_offline": tf_off,
        "tf_offline_priced": tf_priced,
        "tf_offline_timing": tf_timing,
    }


def planner_honesty_check_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Always-on form / dual_gate / offline_tf identity audits for Calc_Check.

    Compatible with model_calc_check columns (check, predicted, actual, abs_err, ok).
    Non-numeric honesty rows use string notes in predicted/actual; ok is boolean.
    Static only: never runs priced residual or timing harness (isolation + smoke latency).
    """
    model = report.get("model") or {}
    cmp_ = report.get("comparison") or {}
    admm = report.get("admm") or {}
    form = str(model.get("form") or "")
    form_ok = form == "classic_2block_excel_path"
    dual_gate = str(cmp_.get("dual_gate") or "")
    verdict_gate = str(cmp_.get("verdict_dual_gate") or "")
    online_role = str(cmp_.get("dual_linf_online_role") or "")
    path_ = str(admm.get("dual_recovery_path") or "")
    dual_ok = (
        dual_gate == "online_lambda"
        and verdict_gate == "online_only"
        and "PRIMARY" in online_role
        and "online_lambda" in path_
    )
    # Static honesty: offline TF is never on classic Case 1 path.
    offline_ok = True
    return [
        {
            "check": "form_classic_2block",
            "predicted": "classic_2block_excel_path",
            "actual": form or "(missing)",
            "abs_err": 0.0 if form_ok else 1.0,
            "ok": form_ok,
        },
        {
            "check": "dual_gate_online_only",
            "predicted": "online_lambda / online_only / PRIMARY",
            "actual": (
                f"dual_gate={dual_gate}; verdict_dual_gate={verdict_gate}; "
                f"role={online_role}; path has online_lambda={'online_lambda' in path_}"
            ),
            "abs_err": 0.0 if dual_ok else 1.0,
            "ok": dual_ok,
        },
        {
            "check": "offline_tf_not_on_case1",
            "predicted": f"offline_tf_units={_OFFLINE_TF_UNITS}; on_excel_case1_path=False",
            "actual": "not on classic Case 1 solve (static honesty)",
            "abs_err": 0.0,
            "ok": offline_ok,
        },
        {
            "check": "offline_tf_priced_not_duals",
            "predicted": (
                "offline priced residual readiness exists; synthetic prices not ADMM λ / "
                "not Case 1 shadows / not duals"
            ),
            "actual": "static honesty — prices not duals; dual_recovery_path=None on TF surface",
            "abs_err": 0.0,
            "ok": True,
        },
        {
            "check": "offline_tf_timing_not_case1",
            "predicted": (
                "offline block-solve timing readiness exists; timings not Case 1 wall time / "
                "not duals / not online λ"
            ),
            "actual": "static honesty — timings readiness only; not on classic Case 1 solve",
            "abs_err": 0.0,
            "ok": True,
        },
    ]


def _how_to_read_rows(report: Dict[str, Any]) -> list[tuple[str, str]]:
    """Guide for lean PIMS-style results workbook (≤15 sheets)."""
    mono = report.get("mono") or {}
    admm = report.get("admm") or {}
    cmp_ = report.get("comparison") or {}
    meta = report.get("meta") or {}
    gap_pct = 100.0 * float(cmp_.get("objective_gap_rel") or 0.0)
    dual = format_dual_honesty_summary(report)
    dual_linf = dual["dual_linf_online"]
    dual_rec = dual["dual_linf_recovered"]
    path_ = dual["dual_recovery_path"]
    tf_off = format_tf_offline_units_howto()
    tf_priced = format_tf_offline_priced_howto()
    tf_timing = format_tf_offline_timing_howto()
    return [
        (
            "goal",
            "≤15 sheets; one tab per unit submodel; FCC/Coker = Aspen How-To 07 BASE/DELTA matrices.",
        ),
        (
            "tabs",
            "Index → Calc_Yields / Calc_Blend → Submodel_CDU / Blender / FCC / Coker / Linking → Rates → Shadows → Summary → Check",
        ),
        (
            "Submodel_CDU / Submodel_Blender",
            "Each sheet has TECH + A sections (merged). Classic mono/ADMM solve uses these yields/recipes.",
        ),
        (
            "Submodel_FCC / Submodel_Coker",
            "PIMS matrix: FEED · BASE · D_* · MB_* · E_BASE_REF · E_quality_REF (−999) · FREE. No satellite tabs.",
        ),
        (
            "fcc_three_path",
            "Submodel_FCC BASE/D_* = base_delta export (planner-visible coeffs). "
            "Optional offline TF (tf_linear_blocks) = exact linear copy of the same y0/D (not this solve). "
            "Case 1 mono/ADMM remains classic_2block_excel_path (CDU+Blender); duals are not TF-owned.",
        ),
        (
            "coker_three_path",
            "Submodel_Coker BASE/D_* = base_delta export (pre-postprocess MB_* coeffs). "
            "Optional offline TF/numpy affine (tf_linear_coker) = exact linear copy of same y0/D — not this solve. "
            "Case 1 mono/ADMM remains classic_2block_excel_path (CDU+Blender duals, not TF-owned). "
            "Coker postprocess renorm is outside affine export: raw BASE/D_* ≠ full evaluate() even at reference.",
        ),
        (
            "cdu_three_path",
            "Submodel_CDU = classic mono/ADMM TECH+A yield/recipe export (Case 1 solve path) — "
            "not Aspen How-To 07 BASE/DELTA / MB_* matrix (unlike Submodel_FCC / Submodel_Coker). "
            "Optional offline TF/numpy affine (tf_linear_cdu) = exact linear copy of build_cdu_base_delta "
            "y0/D/x0 (nested cut_points_f.* drivers) — not this solve. "
            "Duals remain package-ADMM free online λ on classic path — not TF-owned. "
            "Liquid renorm + offgas clamp sit outside raw affine (full evaluate = affine + postprocess).",
        ),
        (
            "tf_offline_units",
            tf_off["planner_one_liner"],
        ),
        (
            "tf_offline_priced",
            tf_priced["planner_one_liner"],
        ),
        (
            "tf_offline_timing",
            tf_timing["planner_one_liner"],
        ),
        (
            "solve_boundary",
            f"Mono+ADMM still CDU+Blender only. Cascade FCC/Coker = solve_cdu_fcc. "
            f"This run: mono={mono.get('objective')}, admm={admm.get('objective')}, "
            f"gap={gap_pct:.4f}%, dual_L∞_PRIMARY_online={dual_linf}, "
            f"dual_L∞_SECONDARY_recovered={dual_rec}, path={path_}.",
        ),
        (
            "duals_online_lambda",
            "PRIMARY ADMM shadows on Shadows tab = free online λ (path labels online_lambda; "
            f"this-run dual L∞ online≈{dual_linf}). "
            f"SECONDARY recovered blender duals (this-run dual L∞ recovered≈{dual_rec}; face-dependent). "
            "Not pure-ADMM dual ownership; not TF dual recovery.",
        ),
        (
            "duals_primary_secondary",
            dual["planner_one_liner"],
        ),
        (
            "input",
            f"Template Crudes/Products/Capacities. Input: {meta.get('input')}. Re-solve via Python/API.",
        ),
    ]


def _sheet_header_map(ws) -> Dict[str, str]:
    """Header name -> column letter for row 1."""
    from openpyxl.utils import get_column_letter

    out: Dict[str, str] = {}
    for cell in ws[1]:
        if cell.value is None:
            continue
        out[str(cell.value)] = get_column_letter(cell.column)
    return out


def _apply_excel_formula_links(wb, model: Dict[str, Any], mono: Dict[str, Any]) -> None:
    """Minimal formulas: yield_sum / recipe_sum on Calc_Yields / Calc_Blend only.

    Lean workbook drops Live_* sprawl; re-solve stays in Python/API.
    """
    if "Calc_Yields" not in wb.sheetnames or "Calc_Blend" not in wb.sheetnames:
        return
    inter = list(model.get("intermediates") or [])
    yields = list(model.get("yields") or [])
    blends = list(model.get("blend_recipes") or [])
    n_c, n_p = len(yields), len(blends)
    if n_c == 0 or n_p == 0 or not inter:
        return
    yws, bws = wb["Calc_Yields"], wb["Calc_Blend"]
    ymap, bmap = _sheet_header_map(yws), _sheet_header_map(bws)
    y_cols = [ymap[f"y_{i}"] for i in inter if f"y_{i}" in ymap]
    if "yield_sum" in ymap and y_cols:
        for r in range(2, 2 + n_c):
            yws[f"{ymap['yield_sum']}{r}"] = "=" + "+".join(f"{c}{r}" for c in y_cols)
    u_cols = [bmap[f"use_{i}"] for i in inter if f"use_{i}" in bmap]
    if "recipe_sum" in bmap and u_cols:
        for r in range(2, 2 + n_p):
            bws[f"{bmap['recipe_sum']}{r}"] = "=" + "+".join(f"{c}{r}" for c in u_cols)


def write_results_excel(path: PathLike, report: Dict[str, Any]) -> Path:
    """Lean PIMS-style results workbook (target ≤15 sheets).

    One tab per unit submodel: CDU, Blender, FCC, Coker, Linking.
    FCC/Coker are Aspen How-To 07 BASE/DELTA matrices.
    """
    import openpyxl
    from openpyxl.styles import Alignment, Font

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    bold = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")

    mono = report.get("mono") or {}
    admm = report.get("admm") or {}
    cmp_ = report.get("comparison") or {}
    meta = report.get("meta") or {}
    model = report.get("model") or {}

    # --- How_to_read ---
    guide = wb.active
    if guide is None:
        guide = wb.create_sheet("How_to_read", 0)
    guide.title = "How_to_read"
    guide.append(["topic", "explanation"])
    guide["A1"].font = bold
    guide["B1"].font = bold
    for topic, text_ in _how_to_read_rows(report):
        guide.append([topic, text_])
    guide.column_dimensions["A"].width = 22
    guide.column_dimensions["B"].width = 100
    for row in guide.iter_rows(min_row=2, max_col=2):
        row[1].alignment = wrap
        guide.row_dimensions[row[0].row].height = 36

    def _header(ws, headers):
        ws.append(headers)
        for c in ws[1]:
            c.font = bold

    def _keys_for_rows(rows, preferred=None):
        keys: List[str] = []
        if preferred:
            keys.extend([k for k in preferred if any(k in r for r in rows)])
        dyn = []
        for r in rows:
            for k in r.keys():
                if k not in keys and (str(k).startswith("D_") or str(k).startswith("MODE_")):
                    if k not in dyn:
                        dyn.append(k)
        insert_at = None
        for marker in ("BASE", "FEED_FFD", "FEED_CFD", "rhs"):
            if marker in keys:
                insert_at = keys.index(marker) + 1
                break
        if insert_at is None:
            keys.extend(dyn)
        else:
            keys[insert_at:insert_at] = dyn
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        return keys

    def _dict_rows_sheet(name, rows, preferred=None):
        s = wb.create_sheet(name)
        if not rows:
            _header(s, ["(empty)"])
            return s
        keys = _keys_for_rows(rows, preferred)
        _header(s, keys)
        for r in rows:
            s.append([r.get(k) for k in keys])
        return s

    def _sectioned_sheet(name, sections):
        """sections: list of (section_title, rows, preferred)."""
        s = wb.create_sheet(name)
        first = True
        for title, rows, preferred in sections:
            if not first:
                s.append([])
            first = False
            s.append([f"=== {title} ==="])
            s.cell(s.max_row, 1).font = bold
            if not rows:
                s.append(["(empty)"])
                continue
            keys = _keys_for_rows(rows, preferred)
            s.append(keys)
            for c in s[s.max_row]:
                c.font = bold
            for r in rows:
                s.append([r.get(k) for k in keys])
        return s

    honesty = format_planner_honesty_package(report)
    dual = honesty["dual"]

    if model:
        sm = submodel_matrix_tables(model)
        lean_index = [
            {"block": "CDU", "sheet": "Submodel_CDU", "what": "TECH yields + A matrix (CAP/YLD) — classic solve"},
            {"block": "BLENDER", "sheet": "Submodel_Blender", "what": "TECH recipes + A matrix (BLD) — classic solve"},
            {
                "block": "FCC",
                "sheet": "Submodel_FCC",
                "what": (
                    "PIMS BASE/DELTA export/teaching matrices — not live ADMM blocks on this path"
                ),
            },
            {
                "block": "COKER",
                "sheet": "Submodel_Coker",
                "what": (
                    "PIMS BASE/DELTA export/teaching matrices — not live ADMM blocks on this path"
                ),
            },
            {"block": "LINKING", "sheet": "Submodel_Linking", "what": "prod−use balances; duals → Shadows"},
            {"block": "MASTER_ADMM", "sheet": "(outside Excel)", "what": "ρ / dual ascent / consensus — Python only"},
            honesty["index_row"],
        ]
        _dict_rows_sheet("Submodel_Index", lean_index, preferred=["block", "sheet", "what"])
        _dict_rows_sheet(
            "Calc_Yields",
            list(model.get("yields") or []),
            preferred=["crude", "price_usd_per_bbl", "max_supply_kbd", "api", "sulfur_wt_pct"],
        )
        _dict_rows_sheet(
            "Calc_Blend",
            list(model.get("blend_recipes") or []),
            preferred=["product", "price_usd_per_bbl", "max_demand_kbd"],
        )
        _sectioned_sheet(
            "Submodel_CDU",
            [
                (
                    "TECH — yields / economics",
                    list(sm.get("cdu_tech") or []),
                    ["submodel", "crude", "price_usd_per_bbl", "max_supply_kbd", "api", "sulfur_wt_pct"],
                ),
                (
                    "A — CAP + YLD constraints",
                    list(sm.get("cdu_A") or []),
                    ["constraint", "type", "rhs", "meaning", "submodel_data_from"],
                ),
            ],
        )
        _sectioned_sheet(
            "Submodel_Blender",
            [
                (
                    "TECH — recipes / economics",
                    list(sm.get("blend_tech") or []),
                    ["submodel", "product", "price_usd_per_bbl", "max_demand_kbd"],
                ),
                (
                    "A — BLD constraints",
                    list(sm.get("blend_A") or []),
                    ["constraint", "type", "rhs", "meaning", "submodel_data_from"],
                ),
            ],
        )
        _dict_rows_sheet(
            "Submodel_FCC",
            list(sm.get("fcc_pims_matrix") or []),
            preferred=["row", "row_type", "rhs", "FEED_FFD", "BASE", "meaning", "equation", "pims_note"],
        )
        _dict_rows_sheet(
            "Submodel_Coker",
            list(sm.get("coker_pims_matrix") or []),
            preferred=["row", "row_type", "rhs", "FEED_CFD", "BASE", "meaning", "equation", "pims_note"],
        )
        _dict_rows_sheet(
            "Submodel_Linking",
            list(sm.get("link_A") or []),
            preferred=[
                "constraint",
                "type",
                "rhs",
                "prod_coeff",
                "use_coeff",
                "cdu_var",
                "blender_var",
                "meaning",
            ],
        )
        check_rows = list(model_calc_check(model, mono)) + planner_honesty_check_rows(report)
        _dict_rows_sheet(
            "Calc_Check",
            check_rows,
            preferred=["check", "predicted", "actual", "abs_err", "ok"],
        )

    ws = wb.create_sheet("Summary")
    _header(ws, ["key", "value"])
    honesty_summary_keys = {k for k, _ in honesty["summary_pairs"]}
    for k, v in [
        ("verdict", report.get("verdict")),
        ("input", meta.get("input")),
        ("n_crudes", meta.get("n_crudes")),
        ("cdu_capacity_kbd", meta.get("cdu_capacity_kbd")),
        ("mono_status", mono.get("status")),
        ("mono_objective", mono.get("objective")),
        ("mono_wall_s", mono.get("wall_time_s")),
        ("admm_status", admm.get("status")),
        ("admm_objective", admm.get("objective")),
        ("admm_iters", admm.get("iteration_count")),
        ("admm_wall_s", admm.get("wall_time_s")),
        ("admm_rho", admm.get("rho")),
        ("primal_residual", admm.get("primal_residual")),
        ("dual_residual", admm.get("dual_residual")),
        ("dual_recovery_path", admm.get("dual_recovery_path")),
        ("gap_abs", cmp_.get("objective_gap_abs")),
        ("gap_rel", cmp_.get("objective_gap_rel")),
        ("both_feasible", cmp_.get("both_feasible")),
        ("pipeline_wall_s", meta.get("pipeline_wall_s")),
        ("recovered_secondary", True),
        # Honesty strip (form + dual PRIMARY gate + offline TF not-on-path).
        *honesty["summary_pairs"],
        (
            "sheet_guide",
            "How_to_read → Index → Calc_Yields/Blend → Submodel_CDU/Blender/FCC/Coker/Linking → Rates → Shadows → Summary",
        ),
        ("n_sheets_target", "≤15 lean PIMS-style"),
        (
            "index_offline_tf_note",
            "See Submodel_Index OFFLINE_TF row (FCC+COKER+CDU not on classic Case 1)",
        ),
    ]:
        # Guard against accidental double-append of honesty keys.
        if k in honesty_summary_keys and (k, v) not in honesty["summary_pairs"]:
            continue
        ws.append([k, v])

    rates = wb.create_sheet("Rates")
    _header(rates, ["kind", "name", "mono_kbd", "admm_kbd", "delta_kbd"])
    for kind, mdict, adict in [
        ("crude", mono.get("crude_rates") or {}, admm.get("crude_rates") or {}),
        ("product", mono.get("product_rates") or {}, admm.get("product_rates") or {}),
        ("inter_prod", mono.get("intermediate_prod") or {}, admm.get("intermediate_prod") or {}),
        ("inter_use", mono.get("intermediate_use") or {}, admm.get("intermediate_use") or {}),
    ]:
        for name in sorted(set(mdict) | set(adict)):
            mv = float(mdict.get(name, 0.0))
            av = float(adict.get(name, 0.0))
            rates.append([kind, name, mv, av, av - mv])

    # Shadows: PRIMARY online λ vs SECONDARY recovered (planner dual honesty surface).
    sh = wb.create_sheet("Shadows")
    sh.append(["role_banner", dual["shadows_role_banner"]])
    sh.append(
        [
            "role_online_col",
            "admm_online_econ = PRIMARY — free online λ economic value; gates dual L∞ / VERDICT dual check",
        ]
    )
    sh.append(
        [
            "role_recovered_col",
            "admm_recovered_econ = SECONDARY — blender recovery LP face; may diverge; not VERDICT gate",
        ]
    )
    sh.append(
        [
            "stream",
            "mono_shadow",
            "admm_online_econ (PRIMARY)",
            "admm_recovered_econ (SECONDARY)",
            "abs_diff_online",
        ]
    )
    for c in sh[4]:
        c.font = bold
    mono_sh = mono.get("shadow_prices") or {}
    admm_sh = admm.get("shadow_prices") or {}
    rec_sh = admm.get("shadow_prices_recovered") or {}
    for k in sorted(set(mono_sh) | set(admm_sh) | set(rec_sh)):
        m, a, r = mono_sh.get(k), admm_sh.get(k), rec_sh.get(k)
        diff = abs(float(m) - float(a)) if m is not None and a is not None else None
        sh.append([k, m, a, r, diff])
    # Footer metrics (this-run dual honesty; not stream rows).
    sh.append([])
    sh.append(["metric", "value"])
    sh.append(["dual_L∞_online_vs_mono", cmp_.get("dual_linf_online")])
    sh.append(["dual_L∞_recovered_vs_mono", cmp_.get("dual_linf_recovered")])
    sh.append(["verdict_dual_gate", "online_only"])
    sh.append(["dual_recovery_path", admm.get("dual_recovery_path")])
    sh.append(["dual_gate", "online_lambda"])
    sh.append(["recovered_secondary", True])

    if model:
        _apply_excel_formula_links(wb, model, mono)

    # Goal gate: ≤15 sheets
    if len(wb.sheetnames) > 15:
        raise RuntimeError(
            f"Excel lean goal failed: {len(wb.sheetnames)} sheets > 15: {wb.sheetnames}"
        )

    wb.save(path)
    return path



def ensure_template(path: PathLike | None = None) -> Path:
    """Regenerate PIMS-shaped template from current JSON assays."""
    if path is None:
        from .assay_loader import default_assays_path

        path = default_assays_path().parent / "crudes_template.xlsx"
    return write_template_excel(path)
