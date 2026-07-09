#!/usr/bin/env python3
"""Demo: base-delta cascade crude→CDU→FCC[+COKER] with mass-balance report.

Run:
  PYTHONPATH=src python -m demos.run_cdu_fcc_demo
  PYTHONPATH=src python -m demos.run_cdu_fcc_demo --coker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.base_delta import auto_wire_edges_for_units, unit_submodels_cdu_fcc
from pims_admm_llm.models.cdu_fcc import solve_cdu_fcc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coker", action="store_true", help="enable delayed coker unit")
    args = ap.parse_args()

    units = ["CDU", "FCC"] + (["COKER"] if args.coker else [])
    pack = unit_submodels_cdu_fcc(include_coker=args.coker)
    print("enabled:", pack["enabled"])
    edges = auto_wire_edges_for_units(units)
    print("auto_wire edges:")
    for e in edges:
        print(f"  {e['stream']:20s} {e['from']} → {e['to']}  ({e.get('reason','')[:60]})")

    r = solve_cdu_fcc(max_crude_kbd=100.0, enable_coker=args.coker, active_units=units)
    print("\n=== SOLVE ===")
    print(f"status={r.status} obj={r.objective:.3f} crude={r.crude_kbd:.2f}")
    print(f"modes: CDU={r.cdu_mode} FCC={r.fcc_mode} COKER={r.coker_mode or 'off'}")
    print("mass_balance ok:", r.mass_balance.get("ok"))
    for name, chk in r.mass_balance.get("checks", {}).items():
        print(f"  {name:22s} gap={chk['abs_gap']:.6f} ok={chk['ok']}")
    print("streams:")
    for k, v in sorted(r.streams.items()):
        if abs(v) > 1e-6:
            print(f"  {k:22s} {v:8.3f}")

    out_dir = ROOT / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("cdu_fcc_coker_demo.json" if args.coker else "cdu_fcc_demo.json")
    path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
    print(
        f"\nVERDICT cascade status={r.status} mb_ok={r.mass_balance.get('ok')} "
        f"obj={r.objective:.4f} crude={r.crude_kbd:.3f} "
        f"units={','.join(r.enabled_units)} path={path}"
    )
    return 0 if r.status == "Optimal" and r.mass_balance.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
