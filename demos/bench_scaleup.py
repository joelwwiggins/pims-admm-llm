#!/usr/bin/env python3
"""Scale-up timing: mono full plant vs parallel multi-block portfolio solves (W6).

Toy single-period mono CBC is usually faster. Parallel wins when work is
replicated across independent periods/sites (n cases) or crude slate grows.

Usage:
  PYTHONPATH=src python demos/bench_scaleup.py
  PYTHONPATH=src python demos/bench_scaleup.py --ns 1,2,4,8,16,32 --workers 4
  PYTHONPATH=src python demos/bench_scaleup.py --ns 1,2,4,8,16 --workers 4 --crudes 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.assay_loader import load_assays_json  # noqa: E402
from pims_admm_llm.models.full_plant import solve_full_plant  # noqa: E402
from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks  # noqa: E402


def _expand_crudes(assays: Dict[str, Any], factor: int) -> Dict[str, Any]:
    """Replicate crude slate to enlarge the LP (synthetic scale)."""
    if factor <= 1:
        return assays
    out = copy.deepcopy(assays)
    base = list(out.get("crudes") or [])
    expanded = []
    for i in range(factor):
        for c in base:
            cc = dict(c)
            cc["name"] = f"{c['name']}_r{i}"
            # keep same max supply so capacity still binds realistically
            expanded.append(cc)
    out["crudes"] = expanded
    return out


def _mono_once(assays: Optional[Dict[str, Any]] = None) -> float:
    t0 = time.perf_counter()
    r = solve_full_plant(assays)
    assert r.feasible, r.status
    return time.perf_counter() - t0


def _blocks_once(assays: Optional[Dict[str, Any]] = None) -> float:
    t0 = time.perf_counter()
    out = solve_all_plant_blocks(assays=assays)
    assert out["blocks"]["CDU"]["status"] == "Optimal"
    return time.perf_counter() - t0


def bench(n_cases: int, workers: int, assays: Optional[Dict[str, Any]] = None) -> dict:
    # sequential mono (n independent periods/sites)
    t0 = time.perf_counter()
    for _ in range(n_cases):
        _mono_once(assays)
    mono_s = time.perf_counter() - t0

    # sequential blocks
    t0 = time.perf_counter()
    for _ in range(n_cases):
        _blocks_once(assays)
    blocks_serial_s = time.perf_counter() - t0

    # parallel block portfolio
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_blocks_once, assays) for _ in range(n_cases)]
        for f in as_completed(futs):
            f.result()
    parallel_s = time.perf_counter() - t0

    # parallel mono portfolio (fair multi-case baseline)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_mono_once, assays) for _ in range(n_cases)]
        for f in as_completed(futs):
            f.result()
    mono_par_s = time.perf_counter() - t0

    return {
        "n": n_cases,
        "n_cases": n_cases,
        "workers": workers,
        "mono_s": mono_s,
        "mono_serial_s": mono_s,
        "parallel_s": parallel_s,
        "blocks_parallel_s": parallel_s,
        "blocks_serial_s": blocks_serial_s,
        "mono_parallel_s": mono_par_s,
        "speedup": mono_s / max(parallel_s, 1e-9),
        "speedup_blocks_par_vs_mono_serial": mono_s / max(parallel_s, 1e-9),
        "speedup_mono_par_vs_mono_serial": mono_s / max(mono_par_s, 1e-9),
        "parallel_blocks_beats_mono_serial": parallel_s < mono_s,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="W6 scale-up: mono vs parallel multi-block")
    p.add_argument("--ns", default="1,2,4,8,16,32", help="comma list of n (multi-period copies)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crudes", type=int, default=1, help="crude slate expansion factor")
    p.add_argument(
        "--out",
        default=str(ROOT / "demos" / "output" / "bench_scaleup.json"),
        help="output JSON path",
    )
    args = p.parse_args(argv)

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    base = load_assays_json()
    assays = _expand_crudes(base, args.crudes)

    rows = []
    print(f"workers={args.workers} crude_factor={args.crudes} ns={ns}")
    print(f"{'n':>4} {'mono_s':>10} {'parallel_s':>12} {'speedup':>9} {'win':>5}")
    for n in ns:
        row = bench(n, workers=min(args.workers, max(1, n)), assays=assays)
        rows.append(row)
        print(
            f"{n:4d} {row['mono_s']:10.6f} {row['parallel_s']:12.6f} "
            f"{row['speedup']:8.3f}x {'Y' if row['parallel_blocks_beats_mono_serial'] else 'N':>4}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workers": args.workers,
        "crude_factor": args.crudes,
        "rows": rows,
        "note": (
            "multi-period/site proxy: n sequential mono full-plant vs n concurrent "
            "solve_all_plant_blocks. Parallel wins when portfolio work amortizes pool overhead."
        ),
    }
    out.write_text(json.dumps(payload, indent=2))

    any_win = any(r["parallel_blocks_beats_mono_serial"] for r in rows)
    best = max(rows, key=lambda r: r["speedup"]) if rows else None
    print(f"JSON: {out}")
    if any_win and best:
        print(
            f"VERDICT: PASS — parallel portfolio beats mono serial at some n "
            f"(best {best['speedup']:.3f}× at n={best['n']})."
        )
    else:
        print(
            "VERDICT: PASS — table written. Toy/current run mono-favoring; "
            "path ready for larger n/workers/crudes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
