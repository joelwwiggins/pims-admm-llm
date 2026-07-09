"""Timing benchmarks: monolithic vs sequential blocks vs parallel blocks.

Also measures warm-start benefit across repeated ADMM-like iterations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pims_admm_llm.models.blocks import solve_monolithic
from pims_admm_llm.models.data import RefineryData, load_crude_data

from .block_solvers import BlockSolveRequest, LINKING_STREAMS
from .parallel import (
    ParallelBackend,
    ParallelBlockExecutor,
    collect_warm_starts,
)


@dataclass
class BenchmarkReport:
    monolithic_time_s: float
    monolithic_status: str
    monolithic_objective: float
    sequential_wall_s: float
    thread_wall_s: float
    process_wall_s: float
    sequential_sum_block_s: float
    thread_sum_block_s: float
    process_sum_block_s: float
    thread_speedup_vs_sequential: float
    process_speedup_vs_sequential: float
    warm_start_cold_avg_s: float
    warm_start_hot_avg_s: float
    warm_start_speedup: float
    iterations: int
    n_blocks: int
    max_workers: int
    notes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            "=== Parallel subproblem benchmark (Worker 4) ===",
            f"blocks={self.n_blocks}  workers={self.max_workers}  admm_iters={self.iterations}",
            "",
            f"Monolithic PuLP solve:  {self.monolithic_time_s*1000:.2f} ms  "
            f"status={self.monolithic_status}  obj={self.monolithic_objective:.4f}",
            "",
            "One-shot multi-block solve wall times:",
            f"  sequential: {self.sequential_wall_s*1000:.2f} ms  (sum blocks {self.sequential_sum_block_s*1000:.2f} ms)",
            f"  thread:     {self.thread_wall_s*1000:.2f} ms  (sum blocks {self.thread_sum_block_s*1000:.2f} ms)  "
            f"speedup vs seq={self.thread_speedup_vs_sequential:.2f}x",
            f"  process:    {self.process_wall_s*1000:.2f} ms  (sum blocks {self.process_sum_block_s*1000:.2f} ms)  "
            f"speedup vs seq={self.process_speedup_vs_sequential:.2f}x",
            "",
            "Warm-start across ADMM-like iterations (thread backend):",
            f"  cold first iter avg block: {self.warm_start_cold_avg_s*1000:.2f} ms",
            f"  warm later iters avg block: {self.warm_start_hot_avg_s*1000:.2f} ms",
            f"  warm-start speedup: {self.warm_start_speedup:.2f}x",
            "",
        ]
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


def _dummy_prices(scale: float = 1.0) -> Dict[str, float]:
    # Mid-range intermediate values ($/bbl) as initial dual guesses
    base = {
        "naphtha": 85.0,
        "distillate": 90.0,
        "gasoil": 75.0,
        "residue": 45.0,
    }
    return {k: v * scale for k, v in base.items()}


def _expand_block_names(n_blocks: int) -> List[str]:
    """Replicate base blocks so parallel speedup is measurable on tiny LPs.

    Names stay unique for warm-start maps; solve_block maps *CDU* / *Blender* by
    substring: we pass exact 'CDU' / 'Blender' and pad with artificial clones that
    re-use the same builders via alias handling below.
    """
    base = ["CDU", "Blender"]
    if n_blocks <= 2:
        return base[:n_blocks] if n_blocks > 0 else base
    names: List[str] = []
    i = 0
    while len(names) < n_blocks:
        names.append(base[i % 2])
        i += 1
    # Disambiguate duplicates for result ordering: CDU, Blender, CDU, Blender...
    # solve_block only cares about name content (CDU vs Blender)
    return names


def run_parallel_benchmark(
    data: Optional[RefineryData] = None,
    data_path: Optional[str] = None,
    iterations: int = 8,
    max_workers: Optional[int] = None,
    n_blocks: int = 2,
    rho: float = 1.0,
) -> BenchmarkReport:
    """Full timing suite for Worker 4 acceptance criteria."""
    notes: List[str] = []
    if data is None:
        data = load_crude_data(data_path)

    # --- Monolithic baseline ---
    mono = solve_monolithic(data, msg=False)

    block_names = _expand_block_names(n_blocks)
    # Unique request names for ordering, but keep builder type
    unique_names = []
    for i, n in enumerate(block_names):
        unique_names.append(f"{n}" if block_names.count(n) == 1 else f"{n}#{i}")

    def _builder_name(uname: str) -> str:
        return "CDU" if uname.upper().startswith("CDU") else "Blender"

    prices = _dummy_prices()
    consensus = {s: 30.0 for s in LINKING_STREAMS}  # rough mid guess kbd

    def _make_requests(warm: Dict[str, Dict[str, float]] | None = None):
        warm = warm or {}
        reqs = []
        for uname in unique_names:
            bname = _builder_name(uname)
            # warm keys stored under unique name; solvers use builder name for vars
            ws = dict(warm.get(uname, warm.get(bname, {})))
            reqs.append(
                BlockSolveRequest(
                    block_name=bname,
                    prices=prices,
                    consensus=consensus,
                    rho=rho,
                    warm_start=ws,
                    time_limit_s=30.0,
                    data_path=data_path,
                )
            )
        return reqs

    # Patch: ParallelBlockExecutor returns results with builder names; for
    # duplicate CDU/Blender we need unique tracking. Use sequential path that
    # tags results after solve.
    from .block_solvers import solve_block

    def _run_backend(backend: ParallelBackend) -> tuple[float, float, list]:
        """Returns wall, sum_block, list of BlockSolveResult (tagged)."""
        reqs = _make_requests()
        # For true parallel with duplicate block types, wrap solve with tags
        if backend == ParallelBackend.SEQUENTIAL:
            t0 = time.perf_counter()
            results = []
            for uname, req in zip(unique_names, reqs):
                r = solve_block(data, req)
                r.block_name = uname  # tag
                results.append(r)
            wall = time.perf_counter() - t0
            return wall, sum(r.solve_time_s for r in results), results

        if backend == ParallelBackend.THREAD:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            t0 = time.perf_counter()
            workers = min(max_workers or 4, len(reqs))
            out: Dict[int, Any] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(solve_block, data, req): (i, uname)
                    for i, (uname, req) in enumerate(zip(unique_names, reqs))
                }
                for fut in as_completed(futs):
                    i, uname = futs[fut]
                    r = fut.result()
                    r.block_name = uname
                    out[i] = r
            wall = time.perf_counter() - t0
            results = [out[i] for i in range(len(reqs))]
            return wall, sum(r.solve_time_s for r in results), results

        # process
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from .parallel import _worker_solve

        t0 = time.perf_counter()
        workers = min(max_workers or 4, len(reqs))
        out = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for i, (uname, req) in enumerate(zip(unique_names, reqs)):
                payload = {
                    "block_name": req.block_name,
                    "prices": dict(req.prices),
                    "consensus": dict(req.consensus),
                    "rho": req.rho,
                    "warm_start": dict(req.warm_start),
                    "time_limit_s": req.time_limit_s,
                    "msg": False,
                    "data_path": data_path,
                }
                futs[ex.submit(_worker_solve, data_path, payload)] = (i, uname)
            for fut in as_completed(futs):
                i, uname = futs[fut]
                d = fut.result()
                from .parallel import _result_from_dict

                r = _result_from_dict(d)
                r.block_name = uname
                out[i] = r
        wall = time.perf_counter() - t0
        results = [out[i] for i in range(len(reqs))]
        return wall, sum(r.solve_time_s for r in results), results

    seq_wall, seq_sum, _ = _run_backend(ParallelBackend.SEQUENTIAL)
    thr_wall, thr_sum, _ = _run_backend(ParallelBackend.THREAD)
    try:
        proc_wall, proc_sum, _ = _run_backend(ParallelBackend.PROCESS)
    except Exception as e:
        notes.append(f"process backend failed: {e!r}; set process times = -1")
        proc_wall, proc_sum = -1.0, -1.0

    thr_speedup = (seq_wall / thr_wall) if thr_wall > 0 else 0.0
    proc_speedup = (seq_wall / proc_wall) if proc_wall > 0 else 0.0

    # --- Warm-start multi-iteration (thread) ---
    cold_times: List[float] = []
    hot_times: List[float] = []
    warm: Dict[str, Dict[str, float]] = {}
    prices_iter = dict(prices)

    for it in range(max(1, iterations)):
        reqs = []
        for uname in unique_names:
            bname = _builder_name(uname)
            reqs.append(
                BlockSolveRequest(
                    block_name=bname,
                    prices=prices_iter,
                    consensus=consensus,
                    rho=rho,
                    warm_start=dict(warm.get(uname, {})),
                    time_limit_s=30.0,
                )
            )
        from concurrent.futures import ThreadPoolExecutor, as_completed

        t0 = time.perf_counter()
        tagged = {}
        with ThreadPoolExecutor(max_workers=min(max_workers or 4, len(reqs))) as ex:
            futs = {
                ex.submit(solve_block, data, req): uname
                for uname, req in zip(unique_names, reqs)
            }
            for fut in as_completed(futs):
                uname = futs[fut]
                r = fut.result()
                r.block_name = uname
                tagged[uname] = r
        wall = time.perf_counter() - t0
        avg_block = sum(r.solve_time_s for r in tagged.values()) / max(1, len(tagged))
        if it == 0:
            cold_times.append(avg_block)
        else:
            hot_times.append(avg_block)

        # Update warm starts + crude dual heuristic for next iter
        warm = {u: dict(tagged[u].primal) for u in tagged}
        # Mild price update (simulates ADMM dual step)
        for s in LINKING_STREAMS:
            cdu_flow = 0.0
            bl_flow = 0.0
            for u, r in tagged.items():
                if u.upper().startswith("CDU"):
                    cdu_flow = r.linking_flows.get(s, cdu_flow)
                else:
                    bl_flow = r.linking_flows.get(s, bl_flow)
            # residual prod - use
            prices_iter[s] = prices_iter.get(s, 0.0) + 0.05 * (cdu_flow - bl_flow)
            consensus[s] = 0.5 * (cdu_flow + bl_flow)

    cold_avg = sum(cold_times) / max(1, len(cold_times))
    hot_avg = sum(hot_times) / max(1, len(hot_times)) if hot_times else cold_avg
    warm_speedup = (cold_avg / hot_avg) if hot_avg > 0 else 0.0

    notes.append(
        "Toy LPs are tiny; wall-clock parallel speedup may be <1x due to pool overhead. "
        "Use n_blocks>2 and iterations to stress. Sum-of-block-times / wall shows concurrency."
    )
    notes.append(
        "Warm-start uses PuLP setInitialValue + CBC warmStart=True; pure LPs often "
        "re-solve quickly either way — speedup is more visible on larger / MIP blocks."
    )

    workers_used = max_workers or min(4, len(unique_names))

    return BenchmarkReport(
        monolithic_time_s=mono.solve_time_s,
        monolithic_status=mono.status,
        monolithic_objective=mono.objective,
        sequential_wall_s=seq_wall,
        thread_wall_s=thr_wall,
        process_wall_s=proc_wall,
        sequential_sum_block_s=seq_sum,
        thread_sum_block_s=thr_sum,
        process_sum_block_s=proc_sum,
        thread_speedup_vs_sequential=thr_speedup,
        process_speedup_vs_sequential=proc_speedup,
        warm_start_cold_avg_s=cold_avg,
        warm_start_hot_avg_s=hot_avg,
        warm_start_speedup=warm_speedup,
        iterations=iterations,
        n_blocks=len(unique_names),
        max_workers=workers_used,
        notes=notes,
        raw={
            "mono_duals_sample": dict(list(mono.duals.items())[:8]),
            "final_prices": prices_iter,
            "final_consensus": consensus,
        },
    )
