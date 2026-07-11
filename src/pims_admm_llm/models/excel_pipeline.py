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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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
        },
        "verdict": _verdict(mono_part, admm_part, gap_rel, dual_linf_online),
    }

    if results_xlsx:
        write_results_excel(results_xlsx, report)
        report["meta"]["results_xlsx"] = str(results_xlsx)
    if results_json:
        jp = Path(results_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, default=str))
        report["meta"]["results_json"] = str(jp)
    return report


def _how_to_read_rows(report: Dict[str, Any]) -> list[tuple[str, str]]:
    """Guide text for results workbook readers (PIMS users + ADMM math)."""
    mono = report.get("mono") or {}
    admm = report.get("admm") or {}
    cmp_ = report.get("comparison") or {}
    meta = report.get("meta") or {}
    gap_pct = 100.0 * float(cmp_.get("objective_gap_rel") or 0.0)
    dual_linf = cmp_.get("dual_linf_online")
    path_ = admm.get("dual_recovery_path") or "package-admm"
    return [
        (
            "purpose",
            "Audit trail: Excel PIMS-shaped input → model calculations (Calc_*) → mono CBC + "
            "block-angular ADMM. Solver is external; Excel holds coefficients, equations, rates, duals.",
        ),
        (
            "how_this_differs_from_PIMS",
            "PIMS: one Excel/CPLEX matrix. Here: 2-block LP (CDU + Blender) with explicit yields/recipes "
            "in Calc_* sheets, then ADMM coordinates linking balances with prices λ.",
        ),
        (
            "story_in_one_line",
            "Buy crudes → CDU yields intermediates → blender recipes make products; "
            "ADMM equates prod_i ≈ use_i and matches mono objective.",
        ),
        ("--- READING ORDER ---", ""),
        ("1_How_to_read", "This guide."),
        (
            "2_Calc_Yields",
            "CDU technology: y_i per crude. prod_i = Σ_c y[c,i]*crude_c.",
        ),
        (
            "3_Calc_Blend",
            "Blender recipes: use_i ≥ Σ_p recipe[p,i]*product_p.",
        ),
        (
            "4_Calc_Objective",
            "Linear margin terms: +price*product − price*crude (no utilities on classic path).",
        ),
        (
            "5_Calc_Equations",
            "Full symbolic mono LP (OBJ, CAP_CDU, YLD_*, BAL_*, BLD_*). Source: admm/simple_mono.py.",
        ),
        ("6_Calc_Bounds", "Variable bounds crude/product/prod/use."),
        (
            "7_Calc_Blocks",
            "Block-angular map up to ADMM handoff. MASTER_ADMM is coordinator only (not an Excel LP).",
        ),
        ("8_Calc_Linking", "Intermediates linking CDU prod_* to Blender use_*."),
        (
            "9_Summary",
            f"VERDICT + metrics. This run: mono={mono.get('objective')}, admm={admm.get('objective')}, "
            f"gap={gap_pct:.4f}%, dual_L∞={dual_linf}, path={path_}.",
        ),
        ("10_rate_sheets", "Optimal kbd after solve (mono vs ADMM)."),
        (
            "11_Shadows",
            "mono_shadow vs admm_online_econ (PRIMARY) vs recovered (diagnostic).",
        ),
        (
            "12_Calc_Check",
            "Plug mono rates into Calc_Yields/Calc_Blend; abs_err ~ 0 proves tables match solve.",
        ),
        ("--- MATH / SOLVER ---", ""),
        (
            "mono_LP",
            "CBC maximizes margin subject to Calc_Equations. Duals → mono_shadow.",
        ),
        (
            "ADMM_handoff",
            "Same coefficients split across CDU/BLENDER blocks; master updates λ,z,ρ outside Excel "
            "(admm/coordinator.py). Excel stops at defining the model + reporting results.",
        ),
        (
            "pass_criteria",
            "both feasible; gap_rel ≤ 0.50%; dual L∞(online) ≤ 15. See Summary.verdict.",
        ),
        (
            "input_workbook",
            "Crudes / Products / Capacities / Intermediates template. "
            f"This solve input: {meta.get('input')}.",
        ),
        (
            "code_map",
            "excel_pipeline.model_calculations_from_data → simple_mono + run_admm. "
            "CLI: python -m demos.run_excel_pipeline_demo",
        ),
    ]


def write_results_excel(path: PathLike, report: Dict[str, Any]) -> Path:
    """Write How_to_read + Calc_* model math + Summary / rates / shadows."""
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

    guide = wb.active
    if guide is None:
        guide = wb.create_sheet("How_to_read", 0)
    guide.title = "How_to_read"
    guide.append(["topic", "explanation"])
    guide["A1"].font = bold
    guide["B1"].font = bold
    for topic, text in _how_to_read_rows(report):
        guide.append([topic, text])
    guide.column_dimensions["A"].width = 28
    guide.column_dimensions["B"].width = 100
    for row in guide.iter_rows(min_row=2, max_col=2):
        row[1].alignment = wrap
        t = row[0].value
        guide.row_dimensions[row[0].row].height = (
            18 if t and str(t).startswith("---") else 40
        )

    def _header(ws, headers):
        ws.append(headers)
        for c in ws[1]:
            c.font = bold

    def _dict_rows_sheet(name, rows, preferred=None):
        s = wb.create_sheet(name)
        if not rows:
            _header(s, ["(empty)"])
            return
        keys = []
        if preferred:
            keys.extend([k for k in preferred if any(k in r for r in rows)])
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        _header(s, keys)
        for r in rows:
            s.append([r.get(k) for k in keys])

    if model:
        note = wb.create_sheet("Calc_ModelNote")
        _header(note, ["key", "value"])
        for k in ("form", "source", "solver_note", "cdu_capacity_kbd"):
            note.append([k, model.get(k)])
        note.append(["intermediates", ", ".join(model.get("intermediates") or [])])

        _dict_rows_sheet(
            "Calc_Yields",
            list(model.get("yields") or []),
            preferred=[
                "crude",
                "price_usd_per_bbl",
                "max_supply_kbd",
                "api",
                "sulfur_wt_pct",
            ],
        )
        _dict_rows_sheet(
            "Calc_Blend",
            list(model.get("blend_recipes") or []),
            preferred=["product", "price_usd_per_bbl", "max_demand_kbd"],
        )
        _dict_rows_sheet(
            "Calc_Objective",
            list(model.get("objective_terms") or []),
            preferred=["term", "variable", "coeff", "unit", "block", "formula"],
        )
        _dict_rows_sheet(
            "Calc_Equations",
            list(model.get("equations") or []),
            preferred=["id", "name", "type", "equation", "notes"],
        )
        if "Calc_Equations" in wb.sheetnames:
            wb["Calc_Equations"].column_dimensions["D"].width = 90
            wb["Calc_Equations"].column_dimensions["E"].width = 42
        _dict_rows_sheet(
            "Calc_Bounds",
            list(model.get("bounds") or []),
            preferred=["variable", "lo", "up", "block"],
        )
        _dict_rows_sheet(
            "Calc_Blocks",
            list(model.get("blocks") or []),
            preferred=[
                "block",
                "role",
                "local_vars",
                "local_constraints",
                "sees_prices",
                "exports",
            ],
        )
        _dict_rows_sheet(
            "Calc_Linking",
            list(model.get("linking") or []),
            preferred=["stream", "cdu_var", "blender_var", "balance", "admm_role"],
        )
        _dict_rows_sheet(
            "Calc_Check",
            model_calc_check(model, mono),
            preferred=["check", "predicted", "actual", "abs_err", "ok"],
        )

    ws = wb.create_sheet("Summary")
    _header(ws, ["key", "value"])
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
        ("dual_linf_online", cmp_.get("dual_linf_online")),
        ("dual_linf_recovered", cmp_.get("dual_linf_recovered")),
        ("model_form", model.get("form")),
    ]:
        ws.append([k, v])

    def rate_sheet(name, rates):
        s = wb.create_sheet(name)
        _header(s, ["name", "kbd"])
        for k, v in sorted((rates or {}).items()):
            s.append([k, float(v)])

    rate_sheet("Crudes_mono", mono.get("crude_rates") or {})
    rate_sheet("Products_mono", mono.get("product_rates") or {})
    rate_sheet("Crudes_admm", admm.get("crude_rates") or {})
    rate_sheet("Products_admm", admm.get("product_rates") or {})
    rate_sheet("Inter_prod_mono", mono.get("intermediate_prod") or {})
    rate_sheet("Inter_use_mono", mono.get("intermediate_use") or {})

    sh = wb.create_sheet("Shadows")
    _header(
        sh,
        [
            "stream",
            "mono_shadow",
            "admm_online_econ",
            "admm_recovered_econ",
            "abs_diff_online",
        ],
    )
    mono_sh = mono.get("shadow_prices") or {}
    admm_sh = admm.get("shadow_prices") or {}
    rec_sh = admm.get("shadow_prices_recovered") or {}
    for k in sorted(set(mono_sh) | set(admm_sh) | set(rec_sh)):
        m, a, r = mono_sh.get(k), admm_sh.get(k), rec_sh.get(k)
        diff = abs(float(m) - float(a)) if m is not None and a is not None else None
        sh.append([k, m, a, r, diff])

    wb.save(path)
    return path


def ensure_template(path: PathLike | None = None) -> Path:
    """Regenerate PIMS-shaped template from current JSON assays."""
    if path is None:
        from .assay_loader import default_assays_path

        path = default_assays_path().parent / "crudes_template.xlsx"
    return write_template_excel(path)
