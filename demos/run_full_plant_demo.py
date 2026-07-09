#!/usr/bin/env python3
"""Full plant demo (Wave3): arc-flow superstructure + quality blender + dual recovery metrics."""

from __future__ import annotations

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


def main() -> int:
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

    mono = solve_full_plant(assays)
    admm = admm_price_directed_plant(assays)
    blocks = solve_all_plant_blocks()

    dual_rows = []
    keys = sorted(set(mono.economic_shadows) | set(admm["economic_shadow_prices"]))
    max_abs = 0.0
    for k in keys:
        m = float(mono.economic_shadows.get(k, 0.0))
        a = float(admm["economic_shadow_prices"].get(k, 0.0))
        diff = abs(abs(m) - abs(a))
        max_abs = max(max_abs, diff)
        dual_rows.append({"name": k, "mono": m, "recovered_abs": a, "abs_diff": diff})

    obj_gap = abs(mono.objective - admm["objective"])
    obj_gap_rel = obj_gap / max(abs(mono.objective), 1e-9) * 100
    dual_ok = max_abs <= max(0.05 * max((abs(r["mono"]) for r in dual_rows), default=1.0), 1e-6)
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
            "economic_shadow_prices": admm["economic_shadow_prices"],
            "quality_duals": admm.get("quality_duals"),
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
    print(f"  arc flows (nonzero): {{{', '.join(f'{k}={v:.3f}' for k,v in mono.arc_flows.items() if v>1e-6)}}}")
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
    print(f"  objective_gap_rel:      {obj_gap_rel:.6f}%")
    print(f"  dual L∞ abs:            {max_abs:.6f}")
    print("  stream/ cap             mono_shadow     recovered_abs    abs_diff")
    for r in dual_rows:
        print(f"  {r['name']:22s} {r['mono']:14.4f} {r['recovered_abs']:14.4f} {r['abs_diff']:12.4f}")
    print("\n--- Price-directed block proposals (seed λ) ---")
    for b, prop in report["block_proposals"].items():
        print(f"  {b}: {prop}")
    print(f"\nJSON: {path}")
    print("=" * 72)
    if verdict_pass:
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
