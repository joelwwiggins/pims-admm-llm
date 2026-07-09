#!/usr/bin/env python3
"""Demo: process-pool MIP mode selection (FCC ROT + coker recycle bands).

Usage:
  PYTHONPATH=src python demos/run_process_pool_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.process_pool import (  # noqa: E402
    build_process_pool_yield_library,
    solve_process_pool_mip,
)


def main() -> int:
    lib = build_process_pool_yield_library()
    print("=== Process-pool yield library (mode → key streams) ===")
    for mid, y in lib["fcc_yields_by_mode"].items():
        print(f"  FCC {mid}: naphtha={y['fcc_naphtha']:.4f} lco={y['fcc_lco']:.4f} coke={y['fcc_coke']:.4f}")
    for mid, y in lib["coker_yields_by_mode"].items():
        print(f"  COKER {mid}: naphtha={y['coker_naphtha']:.4f} go={y['coker_gasoil']:.4f} coke={y['coker_coke']:.4f}")

    r = solve_process_pool_mip()
    print("\n=== solve_process_pool_mip() ===")
    print(f"status={r.status} feasible={r.feasible} obj={r.objective:.4f} t={r.solve_time_s:.4f}s")
    print(f"selected: FCC={r.fcc_mode}  COKER={r.coker_mode}  n_binaries={r.n_binaries}")
    print(f"fcc_selection={r.fcc_mode_selection}")
    print(f"coker_selection={r.coker_mode_selection}")

    out = ROOT / "demos" / "output" / "process_pool_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r.to_dict(), indent=2))
    print(f"JSON: {out}")
    print(
        f"VERDICT: {'PASS' if r.feasible and r.fcc_mode and r.coker_mode else 'FAIL'} — "
        f"process-pool MIP selects one mode per unit (SOS1 binaries)."
    )
    return 0 if r.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
