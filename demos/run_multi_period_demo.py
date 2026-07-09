#!/usr/bin/env python3
"""Multi-period inventory tanks smoke (Wave3b W4).

Coupled mono LP: start inventory + end-of-period carries across n periods.
Default smoke cuts crude supply after period 0 so carries have economic value.

  PYTHONPATH=src python -m demos.run_multi_period_demo
  PYTHONPATH=src python -m demos.run_multi_period_demo --periods 3
  PYTHONPATH=src python -m demos.run_full_plant_demo --multi-period
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.assay_loader import load_assays_json, load_routing  # noqa: E402
from pims_admm_llm.models.multi_period import solve_multi_period  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-period inventory tank smoke demo")
    p.add_argument("--periods", type=int, default=2, help="Number of planning periods (default 2)")
    p.add_argument(
        "--inventory-mode",
        default="multi_period",
        choices=["multi_period", "inventory", "heels", "pass", "off"],
        help="Tank inventory mode (default multi_period)",
    )
    p.add_argument(
        "--crude-scale",
        default=None,
        help="Comma-separated crude max_supply scales per period (default: 1.0,0.35,...)",
    )
    p.add_argument("--msg", action="store_true", help="CBC solver log")
    args = p.parse_args(argv)

    assays = load_assays_json()
    routing = load_routing()
    out_dir = ROOT / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    crude_scale = None
    if args.crude_scale:
        crude_scale = [float(x) for x in args.crude_scale.split(",")]

    inv = args.inventory_mode
    if inv in ("pass", "off"):
        inv_flag: bool | str = False
    else:
        inv_flag = inv

    res = solve_multi_period(
        assays,
        n_periods=args.periods,
        inventory_mode=inv_flag,
        msg=args.msg,
        routing=routing,
        crude_scale=crude_scale,
    )

    # Carry chain check
    carry_ok = True
    for t in range(res.n_periods - 1):
        for k, v in res.tank_end[t].items():
            if abs(v - res.tank_start[t + 1][k]) > 1e-5:
                carry_ok = False

    open_inv = sum(res.tank_start[0].values()) if res.tank_start else 0.0
    terminal = sum(res.tank_end[-1].values()) if res.tank_end else 0.0
    mid_carry = sum(sum(c.values()) for c in res.carries[:-1]) if len(res.carries) > 1 else 0.0

    verdict_pass = (
        res.feasible
        and res.status == "Optimal"
        and carry_ok
        and (open_inv > 0 if res.inventory_mode else open_inv == 0)
    )

    report = {
        "n_periods": res.n_periods,
        "inventory_mode": res.inventory_mode,
        "status": res.status,
        "feasible": res.feasible,
        "objective": res.objective,
        "solve_time_s": res.solve_time_s,
        "meta": res.meta,
        "tank_start": res.tank_start,
        "tank_end": res.tank_end,
        "carries": res.carries,
        "period_unit_feeds": res.period_unit_feeds,
        "period_products": res.period_products,
        "period_crude_rates": res.period_crude_rates,
        "period_arc_flows": res.period_arc_flows,
        "period_objectives_approx": res.period_objectives,
        "carry_chain_ok": carry_ok,
        "open_inventory_total": open_inv,
        "terminal_inventory_total": terminal,
        "intermediate_carry_total": mid_carry,
        "verdict_pass": verdict_pass,
    }
    path = out_dir / "multi_period_demo.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    print("PIMS-ADMM-LLM MULTI-PERIOD INVENTORY TANKS SMOKE (Wave3b W4)")
    print("=" * 72)
    print(f"  n_periods:       {res.n_periods}")
    print(f"  inventory_mode:  {res.inventory_mode} ({res.meta.get('mode')})")
    print(f"  crude_scale:     {res.meta.get('crude_scale')}")
    print(f"  status:          {res.status}")
    print(f"  feasible:        {res.feasible}")
    print(f"  objective:       {res.objective:.6f}")
    print(f"  time_s:          {res.solve_time_s:.4f}")
    print(f"  open inventory:  {res.tank_start[0] if res.tank_start else {}}")
    for t in range(res.n_periods):
        print(f"\n  --- period {t} ---")
        print(f"    start tanks:  {res.tank_start[t]}")
        print(f"    end/carry:    {res.tank_end[t]}")
        print(f"    unit feeds:   {res.period_unit_feeds[t]}")
        print(f"    products:     {res.period_products[t]}")
        print(f"    crude rates:  {res.period_crude_rates[t]}")
        print(f"    period ~obj:  {res.period_objectives[t]:.4f}")
    print(f"\n  carry_chain_ok:  {carry_ok}")
    print(f"JSON: {path}")
    print("=" * 72)
    if verdict_pass:
        print(
            f"VERDICT: PASS — multi_period inventory smoke: start inv + end carries, "
            f"T={res.n_periods}, obj={res.objective:.4f}, open_inv={open_inv:.3f}."
        )
        return 0
    print("VERDICT: FAIL — see multi_period report.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
