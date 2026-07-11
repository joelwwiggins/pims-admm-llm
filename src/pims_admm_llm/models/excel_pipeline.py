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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

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


def write_results_excel(path: PathLike, report: Dict[str, Any]) -> Path:
    """Write solve results workbook (Summary / rates / shadows)."""
    import openpyxl
    from openpyxl.styles import Font

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    bold = Font(bold=True)

    mono = report.get("mono") or {}
    admm = report.get("admm") or {}
    cmp_ = report.get("comparison") or {}
    meta = report.get("meta") or {}

    ws = wb.active
    if ws is None:
        ws = wb.create_sheet("Summary", 0)
    ws.title = "Summary"
    ws.append(["key", "value"])
    ws["A1"].font = bold
    ws["B1"].font = bold
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
    ]:
        ws.append([k, v])

    def rate_sheet(name: str, rates: Dict[str, Any]) -> None:
        s = wb.create_sheet(name)
        s.append(["name", "kbd"])
        s["A1"].font = bold
        s["B1"].font = bold
        for k, v in sorted((rates or {}).items()):
            s.append([k, float(v)])

    rate_sheet("Crudes_mono", mono.get("crude_rates") or {})
    rate_sheet("Products_mono", mono.get("product_rates") or {})
    rate_sheet("Crudes_admm", admm.get("crude_rates") or {})
    rate_sheet("Products_admm", admm.get("product_rates") or {})
    rate_sheet("Inter_prod_mono", mono.get("intermediate_prod") or {})
    rate_sheet("Inter_use_mono", mono.get("intermediate_use") or {})

    sh = wb.create_sheet("Shadows")
    sh.append(
        [
            "stream",
            "mono_shadow",
            "admm_online_econ",
            "admm_recovered_econ",
            "abs_diff_online",
        ]
    )
    for c in sh[1]:
        c.font = bold
    mono_sh = mono.get("shadow_prices") or {}
    admm_sh = admm.get("shadow_prices") or {}
    rec_sh = admm.get("shadow_prices_recovered") or {}
    for k in sorted(set(mono_sh) | set(admm_sh) | set(rec_sh)):
        m = mono_sh.get(k)
        a = admm_sh.get(k)
        r = rec_sh.get(k)
        diff = None
        if m is not None and a is not None:
            diff = abs(float(m) - float(a))
        sh.append([k, m, a, r, diff])

    ws.append(["dual_linf_online", (cmp_ or {}).get("dual_linf_online")])
    ws.append(["dual_linf_recovered", (cmp_ or {}).get("dual_linf_recovered")])

    wb.save(path)
    return path


def ensure_template(path: PathLike | None = None) -> Path:
    """Regenerate PIMS-shaped template from current JSON assays."""
    if path is None:
        from .assay_loader import default_assays_path

        path = default_assays_path().parent / "crudes_template.xlsx"
    return write_template_excel(path)
