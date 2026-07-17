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
    ph = (meta.get("planner_honesty") or {})
    offline_units = ph.get("offline_tf_units") or "FCC,COKER,CDU"
    # Static readiness flags from meta only — never import tf_linear_blocks /
    # live residual, block subproblem, multi-round coordination, plant-linking,
    # plant-named, wire-preflight, Case-1-shaped skeleton, dual-space/form
    # contract, or dual-space L∞ probe reports.
    readiness_bits = []
    if ph.get("offline_tf_priced_ready"):
        readiness_bits.append("priced")
    if ph.get("offline_tf_timing_ready"):
        readiness_bits.append("timing")
    if ph.get("offline_tf_admm_residual_ready"):
        readiness_bits.append("admm_residual")
    if ph.get("offline_tf_admm_block_subproblem_ready"):
        readiness_bits.append("admm_block_subproblem")
    if ph.get("offline_tf_admm_coordination_ready"):
        readiness_bits.append("admm_coordination")
    if ph.get("offline_tf_admm_plant_linking_ready"):
        readiness_bits.append("admm_plant_linking")
    if ph.get("offline_tf_admm_plant_named_linking_ready"):
        readiness_bits.append("admm_plant_named_linking")
    if ph.get("offline_tf_wire_preflight_ready"):
        readiness_bits.append("wire_preflight")
    if ph.get("offline_tf_case1_shaped_linking_ready"):
        readiness_bits.append("case1_shaped_linking")
    if ph.get("offline_tf_case1_dual_space_form_contract_ready"):
        readiness_bits.append("case1_dual_space_form_contract")
    if ph.get("offline_tf_case1_dual_space_linf_probe_ready"):
        readiness_bits.append("case1_dual_space_linf_probe")
    readiness_pkg = "+".join(readiness_bits) if readiness_bits else "units_only"
    wire_note = (
        "wire_shipped=False; blockers documented; structural ready ≠ wire tomorrow"
        if ph.get("offline_tf_wire_preflight_ready")
        else "not wire shipped"
    )
    case1_shaped_note = (
        "Case-1-shaped skeleton packaged (linear_quality_pooling; "
        "naphtha/distillate/gasoil/residue; skeleton λ ≠ duals; skeleton ≠ wire)"
        if ph.get("offline_tf_case1_shaped_linking_ready")
        else "no case1_shaped packaging flag"
    )
    dual_space_note = (
        "dual-space/form contract packaged (planned≠classic form registered; "
        "streams aligned; dual_linf_under_wire=unproven; dual-ban; not wire)"
        if ph.get("offline_tf_case1_dual_space_form_contract_ready")
        else "no dual_space_form_contract packaging flag"
    )
    linf_probe_note = (
        "dual-space L∞ probe packaged (unproven; not VERDICT; not wire proof; "
        "dual-ban; skeleton λ ≠ Case 1 duals; wire_shipped=False)"
        if ph.get("offline_tf_case1_dual_space_linf_probe_ready")
        else "no dual_space_linf_probe packaging flag"
    )
    print(
        f"Offline TF: units={offline_units}  readiness={readiness_pkg}  "
        f"on_excel_case1_path={ph.get('on_excel_case1_path', False)}  "
        f"(NOT on classic Case 1; dual_recovery_path=None on TF surface; "
        f"synthetic residual/subproblem/coordination/plant-linking/plant-named λ ≠ duals; "
        f"per-unit coordination ≠ plant linking; synthetic topology ≠ full plant MB; "
        f"plant-named offline demo ≠ full plant MB / ≠ live cascade; "
        f"preflight λ ≠ duals; {wire_note}; {case1_shaped_note}; {dual_space_note}; "
        f"{linf_probe_note})"
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

    # Post-solve diagnostic only (never gates VERDICT; never writes Excel).
    # Demo is allowed to import tf_linear_blocks; excel write path is not.
    try:
        from pims_admm_llm.models import tf_linear_blocks as _tlb

        bridge = _tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
            case1_package=report,
            skeleton_n_rounds=1,
            include_secondary_recovered=True,
        )
        print(
            f"Offline TF live-λ bridge (diagnostic only): "
            f"source={bridge.get('live_lambda_source')}  "
            f"L∞={bridge.get('linf')}  "
            f"bridge_ok={bridge.get('bridge_ok')}  "
            f"dual_linf_under_wire={bridge.get('dual_linf_under_wire_status')}  "
            f"online_linf_gate={bridge.get('online_linf_gate_under_tf_path')}  "
            f"dual_recovery_path={bridge.get('dual_recovery_path')}  "
            f"wire_shipped={bridge.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT dual L∞ under wire proof]"
        )
    except Exception as exc:  # pragma: no cover - demo soft-skip
        print(f"Offline TF live-λ bridge: skipped ({exc})")

    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
