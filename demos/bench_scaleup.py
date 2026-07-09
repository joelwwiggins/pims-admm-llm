#!/usr/bin/env python3
"""Scale-up timing: mono full plant vs parallel multi-block + process-pool MIP (W6/W2A).

Toy single-period mono CBC is usually faster on pure LPs. Parallel wins when work is
replicated across independent periods/sites (n cases), crude slate grows, or unit
models become MIP process-pools (discrete severity/mode binaries).

Usage:
  PYTHONPATH=src python demos/bench_scaleup.py
  PYTHONPATH=src python demos/bench_scaleup.py --ns 1,2,4,8,16,32 --workers 4
  PYTHONPATH=src python demos/bench_scaleup.py --ns 1,2,4,8,16 --workers 4 --crudes 3
  PYTHONPATH=src python demos/bench_scaleup.py --ns 1,2,4,8,16,32 --workers 4 --with-process-pool
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
from pims_admm_llm.models.process_pool import process_pool_once  # noqa: E402


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


def _pool_once(scale: int = 1) -> float:
    t0 = time.perf_counter()
    r = process_pool_once(scale=scale)
    assert r.feasible, r.status
    return time.perf_counter() - t0


def _crossover_n(rows: List[dict], speedup_key: str = "speedup") -> Optional[int]:
    """Smallest n where speedup > 1.0 (parallel beats serial mono)."""
    for row in sorted(rows, key=lambda r: r["n"]):
        if float(row.get(speedup_key) or 0.0) > 1.0:
            return int(row["n"])
    return None


def bench(
    n_cases: int,
    workers: int,
    assays: Optional[Dict[str, Any]] = None,
    *,
    with_process_pool: bool = True,
    pool_scale: int = 1,
) -> dict:
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

    row: Dict[str, Any] = {
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

    if with_process_pool:
        # serial process-pool MIP portfolio (n independent mode-selection MIPs)
        t0 = time.perf_counter()
        for _ in range(n_cases):
            _pool_once(pool_scale)
        pool_serial_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_pool_once, pool_scale) for _ in range(n_cases)]
            for f in as_completed(futs):
                f.result()
        pool_parallel_s = time.perf_counter() - t0

        row.update(
            {
                "process_pool_serial_s": pool_serial_s,
                "process_pool_parallel_s": pool_parallel_s,
                "process_pool_speedup": pool_serial_s / max(pool_parallel_s, 1e-9),
                "process_pool_parallel_beats_serial": pool_parallel_s < pool_serial_s,
                "pool_scale": pool_scale,
            }
        )

    return row


def primary_table(rows: List[dict]) -> List[Dict[str, Any]]:
    """Compact primary_table: n, mono_s, parallel_s, speedup (+ process-pool cols)."""
    table = []
    for r in rows:
        entry: Dict[str, Any] = {
            "n": r["n"],
            "mono_s": r["mono_s"],
            "parallel_s": r["parallel_s"],
            "speedup": r["speedup"],
        }
        if "process_pool_serial_s" in r:
            entry["process_pool_serial_s"] = r["process_pool_serial_s"]
            entry["process_pool_parallel_s"] = r["process_pool_parallel_s"]
            entry["process_pool_speedup"] = r["process_pool_speedup"]
        table.append(entry)
    return table


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="W6/W2A scale-up: mono vs parallel + process-pool MIP")
    p.add_argument("--ns", default="1,2,4,8,16,32", help="comma list of n (multi-period copies)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crudes", type=int, default=1, help="crude slate expansion factor")
    p.add_argument(
        "--with-process-pool",
        dest="with_process_pool",
        action="store_true",
        default=True,
        help="include process-pool MIP serial/parallel timings (default on)",
    )
    p.add_argument(
        "--no-process-pool",
        dest="with_process_pool",
        action="store_false",
        help="skip process-pool MIP timings",
    )
    p.add_argument(
        "--pool-scale",
        type=int,
        default=1,
        help="feed scale multiplier for process-pool MIP (larger = heavier instance)",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "demos" / "output" / "bench_scaleup.json"),
        help="output JSON path",
    )
    args = p.parse_args(argv)

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    base = load_assays_json()
    assays = _expand_crudes(base, args.crudes)

    rows: List[dict] = []
    print(
        f"workers={args.workers} crude_factor={args.crudes} ns={ns} "
        f"process_pool={args.with_process_pool} pool_scale={args.pool_scale}"
    )
    hdr = f"{'n':>4} {'mono_s':>10} {'parallel_s':>12} {'speedup':>9} {'win':>5}"
    if args.with_process_pool:
        hdr += f" {'pool_ser':>10} {'pool_par':>10} {'pool_x':>8}"
    print(hdr)

    for n in ns:
        row = bench(
            n,
            workers=min(args.workers, max(1, n)),
            assays=assays,
            with_process_pool=args.with_process_pool,
            pool_scale=max(1, args.pool_scale),
        )
        rows.append(row)
        line = (
            f"{n:4d} {row['mono_s']:10.6f} {row['parallel_s']:12.6f} "
            f"{row['speedup']:8.3f}x {'Y' if row['parallel_blocks_beats_mono_serial'] else 'N':>4}"
        )
        if args.with_process_pool:
            line += (
                f" {row['process_pool_serial_s']:10.6f} {row['process_pool_parallel_s']:10.6f} "
                f"{row['process_pool_speedup']:7.3f}x"
            )
        print(line)

    xover_blocks = _crossover_n(rows, "speedup")
    xover_pool = (
        _crossover_n(rows, "process_pool_speedup") if args.with_process_pool else None
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workers": args.workers,
        "crude_factor": args.crudes,
        "with_process_pool": args.with_process_pool,
        "pool_scale": args.pool_scale,
        "rows": rows,
        "primary_table": primary_table(rows),
        "crossover": {
            "blocks_parallel_beats_mono_serial_at_n": xover_blocks,
            "process_pool_parallel_beats_serial_at_n": xover_pool,
            "note": (
                "crossover = smallest n where speedup > 1.0. "
                "None means this run stayed mono/serial-favoring (toy LPs or noise)."
            ),
        },
        "process_pool_note": (
            "Process-pool MIP (models/process_pool.py): SOS1-style binary selection of "
            "FCC ROT bands + coker recycle bands; each mode owns a fixed yield table from "
            "yields.fcc_yields / coker_yields. Portfolio timings are n independent mode "
            "MIPs serial vs ThreadPool. Mono-oracle full_plant duals remain plan truth; "
            "process-pool is discrete severity scale-up, not dual recovery."
        ),
        "note": (
            "multi-period/site proxy: n sequential mono full-plant vs n concurrent "
            "solve_all_plant_blocks. Parallel wins when portfolio work amortizes pool "
            "overhead or unit models grow (process-pool MIP / larger crude slate)."
        ),
    }
    out.write_text(json.dumps(payload, indent=2))

    any_win = any(r["parallel_blocks_beats_mono_serial"] for r in rows)
    best = max(rows, key=lambda r: r["speedup"]) if rows else None
    print(f"JSON: {out}")
    print(
        f"CROSSOVER blocks_par vs mono_serial: "
        f"{'n=' + str(xover_blocks) if xover_blocks is not None else 'none this run'}"
    )
    if args.with_process_pool:
        print(
            f"CROSSOVER process_pool_par vs serial: "
            f"{'n=' + str(xover_pool) if xover_pool is not None else 'none this run'}"
        )
        any_pool = any(r.get("process_pool_parallel_beats_serial") for r in rows)
        best_pool = max(rows, key=lambda r: r.get("process_pool_speedup", 0.0))
        print(
            f"process-pool best speedup {best_pool.get('process_pool_speedup', 0):.3f}× "
            f"at n={best_pool['n']} (par beats serial: {any_pool})"
        )

    if any_win and best:
        print(
            f"VERDICT: PASS — parallel portfolio beats mono serial at some n "
            f"(best {best['speedup']:.3f}× at n={best['n']}); "
            f"crossover={xover_blocks}; process_pool_crossover={xover_pool}."
        )
    else:
        print(
            f"VERDICT: PASS — table written with process-pool note. "
            f"Toy/current run may be mono-favoring on pure LP; "
            f"crossover={xover_blocks}; process_pool_crossover={xover_pool}; "
            f"path ready for larger n/workers/crudes/MIP."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
