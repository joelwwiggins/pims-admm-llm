#!/usr/bin/env python3
"""Demo: crude tower → FCC with BASE/DELTA submodels + process modes + auto exits.

Run:
  cd ~/projects/pims-admm-llm && source .venv/bin/activate
  PYTHONPATH=src python -m demos.run_cdu_fcc_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.base_delta import build_cdu_base_delta, build_fcc_base_delta
from pims_admm_llm.models.cdu_fcc import solve_cdu_fcc


def main() -> int:
    cdu = build_cdu_base_delta()
    fcc = build_fcc_base_delta()
    print("=== CDU BASE yields ===")
    for k, v in cdu.base_yields.items():
        sink = next(e.default_sink for e in cdu.exits if e.stream == k)
        print(f"  {k:22s} {v:7.4f}  → {sink}")
    print("=== FCC BASE yields ===")
    for k, v in fcc.base_yields.items():
        sink = next(e.default_sink for e in fcc.exits if e.stream == k)
        print(f"  {k:22s} {v:7.4f}  → {sink}")

    r = solve_cdu_fcc(max_crude_kbd=100.0)
    print("\n=== SOLVE CDU→FCC ===")
    print(f"status={r.status} obj={r.objective:.3f} crude={r.crude_kbd:.2f} kbd")
    print(f"cdu_mode={r.cdu_mode} fcc_mode={r.fcc_mode}")
    print("process_conditions:", json.dumps(r.process_conditions, indent=2))
    print("streams (kbd):")
    for k, v in sorted(r.streams.items()):
        if abs(v) > 1e-6:
            print(f"  {k:22s} {v:8.3f}")
    print("FCC naphtha composition:", json.dumps(r.compositions.get("fcc_naphtha", {}), indent=2)[:500])
    print("auto_routes:", r.auto_routes)

    out_dir = ROOT / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cdu_fcc_demo.json"
    path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
    print(f"\nVERDICT cdu_fcc status={r.status} obj={r.objective:.4f} crude={r.crude_kbd:.3f} "
          f"fcc_naphtha={r.streams.get('fcc_naphtha', 0):.3f} mode={r.fcc_mode} "
          f"path={path}")
    return 0 if r.status == "Optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
