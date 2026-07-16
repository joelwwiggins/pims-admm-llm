#!/usr/bin/env python3
"""Excel PIMS-shaped → mono + ADMM → results Excel MVP demo.

Usage (repo root):
  source .venv/bin/activate
  export PYTHONPATH=src
  python -m demos.run_excel_pipeline_demo
  python -m demos.run_excel_pipeline_demo --input data/assays/crudes_template.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Excel PIMS → ADMM pipeline MVP")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="PIMS-shaped .xlsx (default: regenerate template from JSON assays)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "demos" / "output",
        help="Directory for results xlsx/json",
    )
    args = parser.parse_args(argv)

    from pims_admm_llm.models.excel_pipeline import ensure_template, run_excel_pipeline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.input is None:
        template = _REPO / "data" / "assays" / "crudes_template.xlsx"
        ensure_template(template)
        input_path = template
        print(f"Regenerated template: {template}")
    else:
        input_path = args.input
        if not input_path.is_file():
            print(f"ERROR: input not found: {input_path}", file=sys.stderr)
            return 2

    xlsx_out = args.out_dir / "excel_pipeline_results.xlsx"
    json_out = args.out_dir / "excel_pipeline_results.json"

    print("=" * 72)
    print("EXCEL PIMS → MONO + ADMM PIPELINE")
    print("=" * 72)
    print(f"Input:  {input_path}")
    print(f"Output: {xlsx_out}")
    print()

    report = run_excel_pipeline(
        input_path,
        results_xlsx=xlsx_out,
        results_json=json_out,
    )
    mono = report["mono"]
    admm = report["admm"]
    cmp_ = report["comparison"]
    meta = report["meta"]

    print(f"Crudes: {meta['n_crudes']}  CDU cap: {meta['cdu_capacity_kbd']} kbd")
    print(f"Mono:  status={mono['status']}  obj={mono['objective']:.6f}  "
          f"t={mono['wall_time_s']:.4f}s  feasible={mono['feasible']}")
    print(f"ADMM:  status={admm['status']}  obj={admm['objective']:.6f}  "
          f"iters={admm['iteration_count']}  t={admm['wall_time_s']:.4f}s  "
          f"rho={admm['rho']}  ||r||={admm['primal_residual']:.4g}")
    print(f"       dual_recovery_path={admm['dual_recovery_path']}")
    print(f"Gap:   abs={cmp_['objective_gap_abs']:.6f}  rel={cmp_['objective_gap_rel']:.6%}")
    print(
        f"Dual PRIMARY (online λ, gates VERDICT): L∞={cmp_.get('dual_linf_online')}  "
        f"[verdict_dual_gate=online_only]"
    )
    print(
        f"Dual SECONDARY (recovered blender face, not gate): L∞={cmp_.get('dual_linf_recovered')}  "
        f"[recovered_secondary=true]"
    )
    print(f"Mono crudes:   { {k: round(v, 3) for k, v in mono['crude_rates'].items() if v > 1e-6} }")
    print(f"Mono products: { {k: round(v, 3) for k, v in mono['product_rates'].items() if v > 1e-6} }")
    print(f"Shadows mono:  { {k: round(v, 2) for k, v in mono['shadow_prices'].items()} }")
    print(f"Shadows ADMM:  { {k: round(v, 2) for k, v in admm['shadow_prices'].items()} }")
    rec = admm.get('shadow_prices_recovered') or {}
    if rec:
        print(f"Shadows recov: { {k: round(v, 2) for k, v in rec.items()} }")
    print()
    print(f"Results Excel: {xlsx_out}")
    print(f"Results JSON:  {json_out}")
    print("=" * 72)
    print(f"VERDICT: {report['verdict']}")
    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
