#!/usr/bin/env python3
"""Worker 4 demo: parallel block solves vs monolithic + warm-start timing.

Usage:
  cd ~/projects/pims-admm-llm
  source .venv/bin/activate
  PYTHONPATH=src python -m demos.bench_parallel
  PYTHONPATH=src python -m demos.bench_parallel --blocks 8 --iters 12 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel ADMM block solve benchmark")
    parser.add_argument("--blocks", type=int, default=4, help="Number of block solves (replicates CDU/Blender)")
    parser.add_argument("--iters", type=int, default=8, help="ADMM-like iterations for warm-start test")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--rho", type=float, default=1.0, help="L1 ADMM penalty weight")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write full BenchmarkReport JSON",
    )
    args = parser.parse_args()

    from pims_admm_llm.models.data import load_crude_data
    from pims_admm_llm.solvers.benchmark import run_parallel_benchmark

    data_path = ROOT / "data" / "synthetic_crudes.json"
    data = load_crude_data(data_path)

    report = run_parallel_benchmark(
        data=data,
        data_path=str(data_path),
        iterations=args.iters,
        max_workers=args.workers,
        n_blocks=args.blocks,
        rho=args.rho,
    )
    print(report.summary_text())

    out = args.json_out or str(ROOT / "demos" / "bench_parallel_last.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nJSON report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
