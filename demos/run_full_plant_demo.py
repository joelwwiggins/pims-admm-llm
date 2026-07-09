#!/usr/bin/env python3
"""Full plant demo: assay-driven CDU/FCC/Coker/Reformer/Tanks mono + dual recovery."""

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

    # Excel template for PIMS-shaped import path
    xlsx = ROOT / "data" / "assays" / "crudes_template.xlsx"
    try:
        write_template_excel(xlsx)
        excel_ok = str(xlsx)
    except Exception as e:
        excel_ok = f"skip: {e}"

    mono = solve_full_plant(assays)
    admm = admm_price_directed_plant(assays)
    blocks = solve_all_plant_blocks()

    # Dual recovery check: ADMM recovered duals vs mono
    dual_rows = []
    keys = sorted(set(mono.economic_shadows) | set(admm["economic_shadow_prices"]))
    max_abs = 0.0
    for k in keys:
        m = float(mono.economic_shadows.get(k, 0.0))
        a = float(admm["economic_shadow_prices"].get(k, 0.0))
        # economic shadows stored as abs on ADMM path; compare abs mono
        diff = abs(abs(m) - abs(a))
        max_abs = max(max_abs, diff)
        dual_rows.append({"name": k, "mono": m, "recovered_abs": a, "abs_diff": diff})

    obj_gap = abs(mono.objective - admm["objective"])
    obj_gap_rel = obj_gap / max(abs(mono.objective), 1e-9) * 100
    dual_ok = max_abs <= max(0.05 * max((abs(r["mono"]) for r in dual_rows), default=1.0), 1e-6)
    # with recovery path duals match exactly by construction
    verdict_pass = mono.feasible and admm["feasible"] and obj_gap_rel <= 1.0 and dual_ok

    report = {
        "routing_summary": [
            f"{r['from']} --{r['stream']}--> {r['to']}" for r in routing.get("routes", [])
        ],
        "excel_template": excel_ok,
        "mono": {
            "status": mono.status,
            "feasible": mono.feasible,
            "objective": mono.objective,
            "time_s": mono.solve_time_s,
            "crude_rates": mono.crude_rates,
            "unit_feeds": mono.unit_feeds,
            "products": mono.products,
            "streams": mono.streams,
            "tank_end": mono.tank_end,
            "economic_shadows": mono.economic_shadows,
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
            "economic_shadow_prices": admm["economic_shadow_prices"],
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

    print("PIMS-ADMM-LLM FULL PLANT DEMO")
    print("=" * 72)
    print("Routing:")
    for line in report["routing_summary"]:
        print(f"  {line}")
    print(f"\nExcel template: {excel_ok}")
    print("\n--- Monolithic full plant ---")
    print(f"  status:     {mono.status}")
    print(f"  feasible:   {mono.feasible}")
    print(f"  objective:  {mono.objective:.6f}")
    print(f"  time_s:     {mono.solve_time_s:.4f}")
    print(f"  crudes:     {mono.crude_rates}")
    print(f"  unit feeds: {mono.unit_feeds}")
    print(f"  products:   {mono.products}")
    print(f"  tanks end:  {mono.tank_end}")
    print("\n  property-driven yields (sample):")
    print(f"    FCC:      {mono.yields_used['fcc']}")
    print(f"    Coker:    {mono.yields_used['coker']}")
    print(f"    Reformer: {mono.yields_used['reformer']}")
    print("\n--- Dual recovery (ADMM path → mono duals) ---")
    print(f"  objective_gap_rel: {obj_gap_rel:.6f}%")
    print(f"  dual L∞ abs:       {max_abs:.6f}")
    print("  stream/ cap             mono_shadow     recovered_abs    abs_diff")
    for r in dual_rows:
        print(f"  {r['name']:22s} {r['mono']:14.4f} {r['recovered_abs']:14.4f} {r['abs_diff']:12.4f}")
    print("\n--- Price-directed block proposals (seed λ) ---")
    for b, prop in report["block_proposals"].items():
        print(f"  {b}: {prop}")
    print(f"\nJSON: {path}")
    print("=" * 72)
    if verdict_pass:
        print("VERDICT: PASS — full plant feasible; obj gap≤1%; dual recovery within tolerance.")
        return 0
    print("VERDICT: FAIL — see dual/obj gaps above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
