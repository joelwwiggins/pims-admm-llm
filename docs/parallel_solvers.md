# Parallel subproblem solvers (Worker 4)

## Purpose

Execute ADMM / DW-style **block LPs in parallel**, benchmark wall time against a **monolithic** PuLP solve, and support **warm-start** of primal values across iterations.

## Layout

```
src/pims_admm_llm/solvers/
  block_solvers.py   # CDU / Blender LPs + λ, z, ρ (L1 ADMM) + warm-start
  parallel.py        # ParallelBlockExecutor: sequential | thread | process
  benchmark.py       # Timing suite + BenchmarkReport
demos/bench_parallel.py
```

## Public API (for ADMM coordinator)

```python
from pims_admm_llm.models.data import load_crude_data
from pims_admm_llm.solvers import (
    ParallelBlockExecutor,
    ParallelBackend,
    collect_warm_starts,  # from parallel
)

data = load_crude_data("data/synthetic_crudes.json")
ex = ParallelBlockExecutor(
    data,
    backend=ParallelBackend.THREAD,  # or PROCESS / SEQUENTIAL
    max_workers=4,
    data_path="data/synthetic_crudes.json",  # required for process backend
)

prices = {"naphtha": 85, "distillate": 90, "gasoil": 75, "residue": 45}  # λ
z = {"naphtha": 30, "distillate": 30, "gasoil": 30, "residue": 30}      # consensus
batch = ex.solve_blocks(prices=prices, consensus=z, rho=1.0)
warm = {r.block_name: r.primal for r in batch.results}
batch2 = ex.solve_blocks(prices=prices, consensus=z, rho=1.0, warm_starts=warm)
```

Each `BlockSolveResult` exposes:

- `linking_flows` — intermediate prod (CDU) or use (Blender) for dual residual
- `primal` — full var map for warm-start
- `local_objective`, `status`, `solve_time_s`, `warm_started`

## Math notes

- Hard constraints stay in PuLP/CBC per block.
- Dual prices λ enter the local linear objective.
- Quadratic ADMM term is **L1-approximated**: `ρ·|x−z|` via split variables (stays LP).
- Warm-start: `var.setInitialValue` + CBC `warmStart=True`. On pure LPs this is often a wash; still the right hook for larger / MIP blocks.

## Benchmark (measured on Orin Nano, 2026-07-08)

| Mode | N blocks | Wall | Notes |
|------|----------|------|--------|
| Monolithic | 1 | ~30–45 ms | Optimal, obj ≈ 1406 |
| Sequential blocks | 16 | ~284 ms | baseline |
| Thread pool (4) | 16 | ~171 ms | **~1.66× vs sequential** |
| Process pool (4) | 16 | ~204 ms | ~1.39× (spawn overhead) |
| Thread pool (4) | 4 | ~40 ms | overhead can erase gains on tiny LPs |

Run:

```bash
cd ~/projects/pims-admm-llm && source .venv/bin/activate
PYTHONPATH=src python -m demos.bench_parallel --blocks 16 --iters 10 --workers 4
```

## Integration tips for Worker 3 (ADMM)

1. Each ADMM iteration: broadcast λ, z → `executor.solve_blocks(...)`.
2. Residual: `r_s = x_CDU[s] - x_Blender[s]` (or vs consensus).
3. Dual update: `λ ← λ + ρ · r` (standard).
4. Pass previous `primal` maps as `warm_starts`.
5. Prefer **thread** backend while blocks are small; switch to **process** if block LPs grow or CBC threads fight the GIL less favorably.
