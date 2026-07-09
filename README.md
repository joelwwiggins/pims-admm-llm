# pims-admm-llm

**Open-source Aspen PIMS-style replacement demo**

Crude data + LP distillation / process-unit submodels + **block-angular ADMM** for shadow pricing, **parallel** subproblem solves, and **LLM multi-agents** for faster, more accurate planning-style LP solutions than classic monolithic PIMS-style runs.

## Why this exists

Commercial refinery planning (PIMS / RPMS / similar) still centers on large linear programs with block-angular structure:

- Local blocks: CDU/yields, tanks, blenders, utilities, …
- Linking constraints: intermediate balances, inventory, shared capacity

This project demonstrates:

1. **Same math guarantees** as a full LP (feasibility + dual/shadow prices at convergence)
2. **ADMM coordination** (modern, parallel-friendly alternative/complement to Dantzig–Wolfe)
3. **One agent per block** (LLM wrapper + real LP solver) so nonlinear yield notes, soft business rules, and warm-starts sit *on top of* hard constraints
4. **Shadow prices that scale like PIMS** (marginal value of streams, capacity, crude flexibility) for make-buy-sell decisions

## Quick start

```bash
cd ~/projects/pims-admm-llm   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m demos.run_demo
```

## Architecture (short)

```
                 ┌─────────────────────────┐
                 │   Master / ADMM loop    │
                 │  duals λ = shadow prices│
                 └───────────┬─────────────┘
           λ, z  ↓           ↑  x_i proposals
     ┌───────────┴───────────┴───────────┐
     │                                   │
 ┌───▼───┐  ┌───────┐  ┌────────┐  ┌─────▼────┐
 │  CDU  │  │ Tanks │  │Blender │  │ Utilities│
 │ LLM+LP│  │LLM+LP │  │ LLM+LP │  │  LLM+LP  │
 └───┬───┘  └───┬───┘  └───┬────┘  └─────┬────┘
     └──────────┴──────────┴─────────────┘
              linking streams (consensus z)
```

- **Hard constraints**: always enforced by PuLP/CBC (or Gurobi if licensed) inside each block.
- **ADMM**: updates duals λ and consensus z; λ are the economic shadow prices.
- **LLM layer**: optional suggestions (nonlinearity, uncertainty, business notes) that never bypass the solver.

## Comparison to PIMS

| Aspect | Classic PIMS-style | This demo |
|--------|--------------------|-----------|
| Model form | Monolithic LP | Block-angular + ADMM |
| Shadow prices | Duals of full LP | ADMM duals λ (match at convergence) |
| Parallelism | Limited | Natural per-block parallel solves |
| Nonlinearity | Linear yield vectors | LLM / surrogate suggestions on top of LP |
| Agents | N/A | One sub-agent per block + master |

## Repo layout

```
src/pims_admm_llm/
  models/          # crude, CDU, blender, linking data
  admm/            # ADMM coordinator, dual updates
  agents/          # LLM sub-agent + master prompts / stubs
  solvers/         # PuLP helpers, parallel runners
demos/
  run_demo.py      # monolithic vs ADMM + shadow price report
docs/
  story.md                   # non-math stakeholder narrative + carousel
  architecture.md            # planners/managers architecture
  admm-vs-dantzig-wolfe.md   # one-pager: ADMM vs DW for coordination
data/
  synthetic_crudes.json
```

## Stakeholder docs

- [docs/story.md](docs/story.md) — Smart Refinery Planning Team narrative + 6-slide carousel (non-math)
- [docs/architecture.md](docs/architecture.md) — planners/managers architecture
- [docs/admm-vs-dantzig-wolfe.md](docs/admm-vs-dantzig-wolfe.md) — ADMM vs Dantzig–Wolfe one-pager

## Status

MVP target: runnable toy refinery (crude → CDU → intermediates → blender) with ADMM duals comparable to a monolithic PuLP solve, timing report, and LLM agent stubs.

Kanban board: `pims-admm-llm-20260708`  
Backup: `/home/joel/backups/pims-admm-llm-20260708-191130`

## License

MIT
