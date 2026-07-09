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

## Scale-up: when parallel wins (W6)

Toy **single-period mono CBC** is often faster than a parallel ADMM wave — the plant LP is tiny and thread-pool overhead dominates. Parallel wins when work is **replicated across independent periods / sites** (multi-period portfolio) or when unit blocks grow (larger crude slate, MIP blocks).

### What we measure

`demos/bench_scaleup.py` scales by multi-period copies (`n`) and optional crude-slate expansion (`--crudes`):

| Column | Meaning |
|--------|---------|
| `n` | Independent multi-period / multi-site copies |
| `mono_s` | Wall time of **n sequential** `solve_full_plant` (CBC) |
| `parallel_s` | Wall time of **n concurrent** `solve_all_plant_blocks` (ThreadPool) |
| `speedup` | `mono_s / parallel_s` (>1 ⇒ parallel portfolio beats sequential mono) |

Also reports sequential block baseline and in-period FCC∥Coker unit fan-out.

### Measured (Orin Nano, 6 cores, workers=4, crude_factor=1, 2026-07-08)

| n | mono_s | parallel_s | speedup |
|---|--------|------------|---------|
| 1 | ~0.008 | ~0.020 | **0.40×** (mono wins) |
| 2 | ~0.020 | ~0.026 | 0.79× |
| 4 | ~0.035 | ~0.034 | **~1.02×** (crossover) |
| 8 | ~0.077 | ~0.088 | ~0.87× (noise / contention) |
| 16 | ~0.146 | ~0.136 | **~1.07×** |
| 32 | ~0.301 | ~0.271 | **~1.11×** |

JSON: `demos/output/bench_scaleup.json`.

### Rule of thumb

| Situation | Prefer |
|-----------|--------|
| Single small plant LP (n=1 toy) | **Mono CBC** |
| Multi-period / multi-site portfolio with n ≳ workers | **Parallel block waves** |
| Large crude slate or MIP unit blocks | **Parallel** (amortizes pool overhead) |
| ADMM iterations with warm-start | Thread pool + `warm_starts` (see W4) |

Narrative for stakeholders: parallel ADMM is sold on **scale, agents, duals, explainability** — not always beating mono on a 30 ms toy. At portfolio scale the wall-clock story flips.

Run:

```bash
cd ~/projects/pims-admm-llm && source .venv/bin/activate
PYTHONPATH=src python -m demos.bench_scaleup --ns 1,2,4,8,16,32 --workers 4
PYTHONPATH=src python -m demos.bench_scaleup --ns 1,2,4,8,16 --workers 4 --crudes 3
```
