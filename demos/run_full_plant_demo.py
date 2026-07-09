#!/usr/bin/env python3
"""Full plant demo (Wave3): arc-flow superstructure + quality blender + dual recovery metrics.

Usage:
  python -m demos.run_full_plant_demo
  python -m demos.run_full_plant_demo --pure-admm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.assay_loader import (  # noqa: E402
    load_assays_json,
    load_routing,
    write_template_excel,
)
from pims_admm_llm.models.full_plant import (  # noqa: E402
    admm_price_directed_plant,
    solve_full_plant,
)
from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pure-admm",
        action="store_true",
        help="Use dual_recovery_path=pure-admm (free λ; honest L∞ vs mono bal duals)",
    )
    ap.add_argument("--max-iter", type=int, default=40)
    args = ap.parse_args(argv)

    assays = load_assays_json()
    routing = load_routing()
    out_dir = ROOT / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx = ROOT / "data" / "assays" / "crudes_template.xlsx"
    try:
        write_template_excel(xlsx)
        excel_ok = str(xlsx)
    except Exception as e:
        excel_ok = f"skip: {e}"

    recovery_path = "pure-admm" if args.pure_admm else "mono-oracle"
    mono = solve_full_plant(assays)
    admm = admm_price_directed_plant(
        assays, recovery_path=recovery_path, max_iter=args.max_iter
    )
    blocks = solve_all_plant_blocks()

    dual_rows = []
    if recovery_path == "pure-admm":
        mono_bal = {k: float(v) for k, v in mono.duals.items() if k.startswith("bal_")}
        stream_map = {
            "cdu_gasoil": "bal_tank_gasoil",
            "cdu_resid": "bal_tank_resid",
            "fcc_naphtha": "bal_tank_fcc_naph",
            "coker_naphtha": "bal_tank_coker_naph",
            "reformate": "bal_tank_reformate",
            "cdu_naphtha_heavy": "bal_sr_heavy",
            "cdu_naphtha_light": "bal_sr_light",
            "cdu_distillate": "bal_distillate",
            "fcc_lco": "bal_lco",
            "fcc_slurry": "bal_slurry",
            "coker_gasoil": "bal_coker_go",
        }
        max_abs = float(admm.get("lambda_vs_mono_Linf") or 0.0)
        for stream, bal in stream_map.items():
            m = float(mono_bal.get(bal, 0.0))
            a = float(admm.get("lambda", {}).get(stream, 0.0))
            diff = abs(abs(m) - abs(a))
            dual_rows.append(
                {
                    "name": f"{stream}/{bal}",
                    "mono": m,
                    "recovered_abs": abs(a),
                    "abs_diff": diff,
                }
            )
        dual_ok = False  # pure path never claims dual recovery
        obj_gap = abs(
            mono.objective
            - float(admm.get("objective_mono_plan_truth", admm["objective"]))
        )
    else:
        keys = sorted(set(mono.economic_shadows) | set(admm["economic_shadow_prices"]))
        max_abs = 0.0
        for k in keys:
            m = float(mono.economic_shadows.get(k, 0.0))
            a = float(admm["economic_shadow_prices"].get(k, 0.0))
            diff = abs(abs(m) - abs(a))
            max_abs = max(max_abs, diff)
            dual_rows.append({"name": k, "mono": m, "recovered_abs": a, "abs_diff": diff})
        dual_ok = max_abs <= max(
            0.05 * max((abs(r["mono"]) for r in dual_rows), default=1.0), 1e-6
        )
        obj_gap = abs(mono.objective - admm["objective"])

    obj_gap_rel = obj_gap / max(abs(mono.objective), 1e-9) * 100
    if recovery_path == "pure-admm":
        # residual may remain O(10) on free-disposal faces; do not claim dual recovery
        r_ok = float(admm.get("primal_residual_norm") or 1e9) < 80.0
        short_ok = float(admm.get("shortage_residual_norm") or 0.0) < 50.0
        path_ok = admm.get("dual_recovery_path") == "pure-admm" and admm.get("duals_like_monolithic") == {}
        verdict_pass = mono.feasible and r_ok and short_ok and path_ok
    else:
        verdict_pass = mono.feasible and admm["feasible"] and obj_gap_rel <= 1.0 and dual_ok

    arcs = routing.get("arcs") or routing.get("routes") or []
    routing_summary = []
    for a in arcs:
        if "id" in a:
            routing_summary.append(
                f"{a.get('id')}: {a.get('from')} --{a.get('stream')}--> {a.get('to')} "
                f"(decision={a.get('decision')}, default_open={a.get('default_open')})"
            )
        else:
            routing_summary.append(f"{a.get('from')} --{a.get('stream')}--> {a.get('to')}")

    report = {
        "routing_version": routing.get("version"),
        "routing_summary": routing_summary,
        "excel_template": excel_ok,
        "mono": {
            "status": mono.status,
            "feasible": mono.feasible,
            "objective": mono.objective,
            "time_s": mono.solve_time_s,
            "inventory_mode": mono.inventory_mode,
            "crude_rates": mono.crude_rates,
            "unit_feeds": mono.unit_feeds,
            "products": mono.products,
            "streams": mono.streams,
            "arc_flows": mono.arc_flows,
            "routing_splits": mono.routing_splits,
            "tank_end": mono.tank_end,
            "economic_shadows": mono.economic_shadows,
            "quality_duals": mono.quality_duals,
            "yields_feed_props": mono.yields_used.get("feed_props"),
            "cdu_yields": mono.yields_used.get("cdu_by_crude"),
            "fcc_yields": mono.yields_used.get("fcc"),
            "coker_yields": mono.yields_used.get("coker"),
            "reformer_yields": mono.yields_used.get("reformer"),
        },
        "admm_recovered": {
            "status": admm["status"],
            "feasible": admm["feasible"],
            "objective": admm["objective"],
            "iterations": admm["iterations"],
            "max_iter": admm.get("max_iter"),
            "rho": admm.get("rho"),
            "primal_residual_norm": admm.get("primal_residual_norm"),
            "dual_residual_norm": admm.get("dual_residual_norm"),
            "dual_recovery_path": admm.get("dual_recovery_path"),
            "lambda_vs_mono_Linf": admm.get("lambda_vs_mono_Linf"),
            "lambda_vs_mono_Linf_hard_links": admm.get("lambda_vs_mono_Linf_hard_links"),
            "lambda": admm.get("lambda"),
            "economic_shadow_prices": admm["economic_shadow_prices"],
            "quality_duals": admm.get("quality_duals"),
            "honesty": admm.get("honesty"),
        },
        "block_proposals": {k: v["proposal"] for k, v in blocks["blocks"].items()},
        "dual_comparison": dual_rows,
        "objective_gap_abs": obj_gap,
        "objective_gap_rel_pct": obj_gap_rel,
        "dual_Linf_abs": max_abs,
        "verdict_pass": verdict_pass,
    }
    path = out_dir / "full_plant_demo.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    print("PIMS-ADMM-LLM FULL PLANT DEMO (Wave3 arc-flow)")
    print("=" * 72)
    print(f"Routing version: {routing.get('version')}")
    print("Arc superstructure (sample):")
    for line in routing_summary[:12]:
        print(f"  {line}")
    if len(routing_summary) > 12:
        print(f"  ... ({len(routing_summary)} arcs/routes total)")
    print(f"\nExcel template: {excel_ok}")
    print("\n--- Monolithic full plant ---")
    print(f"  status:     {mono.status}")
    print(f"  feasible:   {mono.feasible}")
    print(f"  objective:  {mono.objective:.6f}")
    print(f"  time_s:     {mono.solve_time_s:.4f}")
    print(f"  inventory:  {mono.inventory_mode}")
    print(f"  crudes:     {mono.crude_rates}")
    print(f"  unit feeds: {mono.unit_feeds}")
    print(f"  products:   {mono.products}")
    print(f"  routing splits: {mono.routing_splits}")
    print(
        f"  arc flows (nonzero): {{{', '.join(f'{k}={v:.3f}' for k,v in mono.arc_flows.items() if v>1e-6)}}}"
    )
    print(f"  quality duals: {mono.quality_duals}")
    print(f"  tanks end:  {mono.tank_end}")
    print("\n  property-driven yields (sample):")
    print(f"    FCC:      {mono.yields_used['fcc']}")
    print(f"    Coker:    {mono.yields_used['coker']}")
    print(f"    Reformer: {mono.yields_used['reformer']}")
    print("\n--- ADMM metrics (explicit) ---")
    print(f"  rho:                    {admm.get('rho')}")
    print(f"  max_iter:               {admm.get('max_iter')}")
    print(f"  iterations_run:         {admm['iterations']}")
    print(f"  ||r|| primal residual:  {admm.get('primal_residual_norm')}")
    print(f"  ||s|| dual residual:    {admm.get('dual_residual_norm')}")
    print(f"  dual_recovery_path:     {admm.get('dual_recovery_path')}")
    print(f"  lambda_vs_mono_Linf:    {admm.get('lambda_vs_mono_Linf')}")
    print(f"  Linf bal (honest):      {admm.get('lambda_vs_mono_bal_Linf')}")
    print(f"  Linf econ:              {admm.get('lambda_vs_mono_econ_Linf')}")
    print(f"  shortage residual:      {admm.get('shortage_residual_norm')}")
    print(f"  objective_gap_rel:      {obj_gap_rel:.6f}%")
    print(f"  dual L∞ abs:            {max_abs:.6f}")
    if admm.get("honesty"):
        print(f"  honesty:                {admm.get('honesty')}")
    print("  stream/cap              mono_shadow     recovered_abs    abs_diff")
    for r in dual_rows:
        print(
            f"  {r['name']:28s} {r['mono']:14.4f} {r['recovered_abs']:14.4f} {r['abs_diff']:12.4f}"
        )
    print("\n--- Price-directed block proposals (seed λ) ---")
    for b, prop in report["block_proposals"].items():
        print(f"  {b}: {prop}")
    print(f"\nJSON: {path}")
    print("=" * 72)
    if verdict_pass:
        if recovery_path == "pure-admm":
            print(
                "VERDICT: PASS — pure-admm free λ path ran; residual controlled; "
                f"L∞ λ vs mono econ={admm.get('lambda_vs_mono_Linf')} "
                f"bal={admm.get('lambda_vs_mono_bal_Linf')} "
                f"(NOT dual recovery; default remains mono-oracle)."
            )
        else:
            print(
                "VERDICT: PASS — full plant feasible; obj gap≤1%; dual recovery within tolerance; "
                f"rho={admm.get('rho')} ||r||={admm.get('primal_residual_norm')} "
                f"||s||={admm.get('dual_residual_norm')} path={admm.get('dual_recovery_path')}."
            )
        return 0
    print("VERDICT: FAIL — see dual/obj gaps above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
