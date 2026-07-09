#!/usr/bin/env python3
"""Scale-up timing: mono full plant vs parallel multi-block solves.

At toy single-plant scale mono CBC is usually faster. Parallel wins when we
replicate many independent planning cases (multi-period / multi-site proxy).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.full_plant import solve_full_plant  # noqa: E402
from pims_admm_llm.models.plant_blocks import solve_all_plant_blocks  # noqa: E402


def _mono_once() -> float:
    t0 = time.perf_counter()
    r = solve_full_plant()
    assert r.feasible
    return time.perf_counter() - t0


def _blocks_once() -> float:
    t0 = time.perf_counter()
    out = solve_all_plant_blocks()
    assert out["blocks"]["CDU"]["status"] == "Optimal"
    return time.perf_counter() - t0


def bench(n_cases: int, workers: int) -> dict:
    # sequential mono
    t0 = time.perf_counter()
    for _ in range(n_cases):
        _mono_once()
    mono_s = time.perf_counter() - t0

    # sequential blocks
    t0 = time.perf_counter()
    for _ in range(n_cases):
        _blocks_once()
    blocks_serial_s = time.perf_counter() - t0

    # parallel blocks (multi-case)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_blocks_once) for _ in range(n_cases)]
        for f in as_completed(futs):
            f.result()
    blocks_par_s = time.perf_counter() - t0

    # parallel mono cases
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_mono_once) for _ in range(n_cases)]
        for f in as_completed(futs):
            f.result()
    mono_par_s = time.perf_counter() - t0

    return {
        "n_cases": n_cases,
        "workers": workers,
        "mono_serial_s": mono_s,
        "mono_parallel_s": mono_par_s,
        "blocks_serial_s": blocks_serial_s,
        "blocks_parallel_s": blocks_par_s,
        "speedup_blocks_par_vs_mono_serial": mono_s / max(blocks_par_s, 1e-9),
        "speedup_mono_par_vs_mono_serial": mono_s / max(mono_par_s, 1e-9),
        "parallel_blocks_beats_mono_serial": blocks_par_s < mono_s,
    }


def main() -> int:
    rows = []
    for n in (1, 4, 8, 16):
        row = bench(n, workers=min(4, n))
        rows.append(row)
        print(
            f"n={n:2d} mono_ser={row['mono_serial_s']:.3f}s "
            f"blocks_par={row['blocks_parallel_s']:.3f}s "
            f"mono_par={row['mono_parallel_s']:.3f}s "
            f"blocks_vs_mono_ser={row['speedup_blocks_par_vs_mono_serial']:.2f}x "
            f"win={row['parallel_blocks_beats_mono_serial']}"
        )
    out = ROOT / "demos" / "output" / "bench_scaleup.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "note": "multi-case parallel proxy for multi-period/site"}
    out.write_text(json.dumps(payload, indent=2))
    any_win = any(r["parallel_blocks_beats_mono_serial"] or r["speedup_mono_par_vs_mono_serial"] > 1.05 for r in rows)
    print(f"JSON: {out}")
    print(
        "VERDICT: PASS — scale table written."
        + (" Parallel wins on some n." if any_win else " Toy scale still mono-favoring; parallel multi-case path ready.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
