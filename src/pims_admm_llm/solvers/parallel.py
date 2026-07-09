"""Parallel execution of block LP subproblems.

Backends:
  - sequential: for baseline timing
  - thread: concurrent.futures.ThreadPoolExecutor (good when CBC releases GIL)
  - process: concurrent.futures.ProcessPoolExecutor (true multi-core; picklable req)

Process workers re-load RefineryData from path so PuLP problems are built in-process.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

from pims_admm_llm.models.data import RefineryData, load_crude_data

from .block_solvers import (
    BlockSolveRequest,
    BlockSolveResult,
    default_block_names,
    solve_block,
)


class ParallelBackend(str, Enum):
    SEQUENTIAL = "sequential"
    THREAD = "thread"
    PROCESS = "process"


def _worker_solve(
    data_path: Optional[str],
    request_dict: dict,
) -> dict:
    """Top-level function for ProcessPool (must be picklable)."""
    data = load_crude_data(data_path) if data_path else load_crude_data()
    req = BlockSolveRequest(**request_dict)
    result = solve_block(data, req)
    return result.to_dict()


def _result_from_dict(d: dict) -> BlockSolveResult:
    return BlockSolveResult(
        block_name=d["block_name"],
        status=d["status"],
        local_objective=d["local_objective"],
        linking_flows=d["linking_flows"],
        primal=d["primal"],
        solve_time_s=d["solve_time_s"],
        warm_started=d["warm_started"],
        message=d.get("message", ""),
    )


@dataclass
class ParallelSolveBatch:
    """Result of solving a set of blocks once."""

    results: List[BlockSolveResult]
    wall_time_s: float
    backend: str
    max_workers: int
    sum_block_time_s: float

    @property
    def speedup_vs_sum(self) -> float:
        """Theoretical parallel efficiency: sum(block times) / wall."""
        if self.wall_time_s <= 0:
            return 0.0
        return self.sum_block_time_s / self.wall_time_s


class ParallelBlockExecutor:
    """Solve many block subproblems concurrently."""

    def __init__(
        self,
        data: RefineryData,
        backend: ParallelBackend | str = ParallelBackend.THREAD,
        max_workers: Optional[int] = None,
        data_path: Optional[str] = None,
    ):
        if isinstance(backend, str):
            backend = ParallelBackend(backend)
        self.data = data
        self.backend = backend
        self.max_workers = max_workers or min(8, (os.cpu_count() or 2))
        self.data_path = data_path

    def solve_many(
        self,
        requests: Sequence[BlockSolveRequest],
    ) -> ParallelSolveBatch:
        if not requests:
            return ParallelSolveBatch([], 0.0, self.backend.value, self.max_workers, 0.0)

        if self.backend == ParallelBackend.SEQUENTIAL:
            return self._run_sequential(requests)
        if self.backend == ParallelBackend.THREAD:
            return self._run_thread(requests)
        if self.backend == ParallelBackend.PROCESS:
            return self._run_process(requests)
        raise ValueError(f"Unknown backend: {self.backend}")

    def solve_blocks(
        self,
        block_names: Sequence[str] | None = None,
        prices: Mapping[str, float] | None = None,
        consensus: Mapping[str, float] | None = None,
        rho: float = 0.0,
        warm_starts: Mapping[str, Mapping[str, float]] | None = None,
        time_limit_s: float = 30.0,
    ) -> ParallelSolveBatch:
        """Convenience: same prices/consensus for all named blocks."""
        prices = dict(prices or {})
        consensus = dict(consensus or {})
        warm_starts = warm_starts or {}
        names = list(block_names or default_block_names())
        requests = [
            BlockSolveRequest(
                block_name=n,
                prices=prices,
                consensus=consensus,
                rho=rho,
                warm_start=dict(warm_starts.get(n, {})),
                time_limit_s=time_limit_s,
                data_path=self.data_path,
            )
            for n in names
        ]
        return self.solve_many(requests)

    def _run_sequential(
        self, requests: Sequence[BlockSolveRequest]
    ) -> ParallelSolveBatch:
        t0 = time.perf_counter()
        results: List[BlockSolveResult] = []
        for req in requests:
            results.append(solve_block(self.data, req))
        t1 = time.perf_counter()
        sum_t = sum(r.solve_time_s for r in results)
        return ParallelSolveBatch(
            results=results,
            wall_time_s=t1 - t0,
            backend=ParallelBackend.SEQUENTIAL.value,
            max_workers=1,
            sum_block_time_s=sum_t,
        )

    def _run_thread(
        self, requests: Sequence[BlockSolveRequest]
    ) -> ParallelSolveBatch:
        t0 = time.perf_counter()
        results: List[BlockSolveResult] = []
        workers = min(self.max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(solve_block, self.data, req): req for req in requests}
            for fut in as_completed(futs):
                results.append(fut.result())
        t1 = time.perf_counter()
        # Preserve request order
        by_name = {r.block_name: r for r in results}
        ordered = [by_name[req.block_name] for req in requests if req.block_name in by_name]
        # If duplicate names, fall back to completion order
        if len(ordered) != len(results):
            ordered = results
        sum_t = sum(r.solve_time_s for r in ordered)
        return ParallelSolveBatch(
            results=ordered,
            wall_time_s=t1 - t0,
            backend=ParallelBackend.THREAD.value,
            max_workers=workers,
            sum_block_time_s=sum_t,
        )

    def _run_process(
        self, requests: Sequence[BlockSolveRequest]
    ) -> ParallelSolveBatch:
        t0 = time.perf_counter()
        workers = min(self.max_workers, len(requests))
        payloads = []
        for req in requests:
            d = {
                "block_name": req.block_name,
                "prices": dict(req.prices),
                "consensus": dict(req.consensus),
                "rho": req.rho,
                "warm_start": dict(req.warm_start),
                "time_limit_s": req.time_limit_s,
                "msg": req.msg,
                "data_path": req.data_path or self.data_path,
            }
            payloads.append((self.data_path or req.data_path, d))

        raw_results: List[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_solve, p[0], p[1]) for p in payloads]
            for fut in as_completed(futs):
                raw_results.append(fut.result())
        t1 = time.perf_counter()

        results = [_result_from_dict(d) for d in raw_results]
        by_name = {r.block_name: r for r in results}
        ordered = [
            by_name[req.block_name]
            for req in requests
            if req.block_name in by_name
        ]
        if len(ordered) != len(results):
            ordered = results
        sum_t = sum(r.solve_time_s for r in ordered)
        return ParallelSolveBatch(
            results=ordered,
            wall_time_s=t1 - t0,
            backend=ParallelBackend.PROCESS.value,
            max_workers=workers,
            sum_block_time_s=sum_t,
        )


def collect_warm_starts(results: Sequence[BlockSolveResult]) -> Dict[str, Dict[str, float]]:
    """Map block_name → primal for next-iteration warm start."""
    return {r.block_name: dict(r.primal) for r in results}
