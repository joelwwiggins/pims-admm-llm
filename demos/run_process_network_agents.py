#!/usr/bin/env python3
"""Process-network agents on a real plant plan (areas + pushback + octane couple).

Each area (CDU, FCC, Coker, Reformer, Blender) gets actual feeds/products/duals
from the solve and can push back on ridiculous asks or flag no-wiggle-room.
Cross-unit: FCC naphtha + reformer reformate jointly own gasoline RON.

Usage:
  PYTHONPATH=src python -m demos.run_process_network_agents
  PYTHONPATH=src python -m demos.run_process_network_agents --process-pool
  PYTHONPATH=src python -m demos.run_process_network_agents --process-pool-two-pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.agents.process_network import (  # noqa: E402
    format_process_network_round,
    run_process_network_round,
)
from pims_admm_llm.models.assay_loader import load_assays_json  # noqa: E402
from pims_admm_llm.models.full_plant import solve_full_plant  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--process-pool",
        action="store_true",
        help="Attach process-pool FCC/coker modes before plant solve",
    )
    ap.add_argument(
        "--process-pool-two-pass",
        action="store_true",
        help="Two-pass process-pool (realized feeds)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Also write demos/output/process_network_round.json",
    )
    args = ap.parse_args(argv)

    assays = load_assays_json()
    use_pool = bool(args.process_pool or args.process_pool_two_pass)
    plant = solve_full_plant(
        assays,
        process_pool_modes=use_pool,
        process_pool_two_pass=bool(args.process_pool_two_pass),
    )
    if not plant.feasible:
        print("VERDICT: FAIL — plant infeasible; agents not run")
        return 1

    round_ = run_process_network_round(plant, assays=assays)
    print(format_process_network_round(round_))

    out_dir = ROOT / "demos" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "process_network_round.json"
    path.write_text(json.dumps(round_.to_dict(), indent=2, default=str))
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
