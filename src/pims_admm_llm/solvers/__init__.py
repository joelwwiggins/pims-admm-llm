"""PuLP helpers, block solvers, parallel runners, and timing benchmarks."""

from .block_solvers import (
    BlockSolveRequest,
    BlockSolveResult,
    solve_block,
    build_cdu_block,
    build_blender_block,
    LINKING_STREAMS,
)
from .parallel import ParallelBlockExecutor, ParallelBackend
from .benchmark import run_parallel_benchmark, BenchmarkReport

__all__ = [
    "BlockSolveRequest",
    "BlockSolveResult",
    "solve_block",
    "build_cdu_block",
    "build_blender_block",
    "LINKING_STREAMS",
    "ParallelBlockExecutor",
    "ParallelBackend",
    "run_parallel_benchmark",
    "BenchmarkReport",
]
