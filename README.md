# pims-admm-llm

**Open-source Aspen PIMS-style replacement demo**

Crude data + LP distillation / process-unit submodels + **block-angular ADMM** for shadow pricing, **parallel** subproblem solves, and **LLM multi-agents** for faster, more accurate planning-style LP solutions than classic monolithic PIMS-style runs.

## Why this exists

Commercial refinery planning (PIMS / RPMS / similar) still centers on large linear programs with block-angular structure:

- Local blocks: CDU/yields, tanks, FCC, coker, reformer, blenders, utilities, …
- Linking constraints: intermediate balances, inventory, shared capacity

This project demonstrates:

1. **Same math guarantees** as a full LP (feasibility + dual/shadow prices at convergence)
2. **ADMM coordination** (modern, parallel-friendly alternative/complement to Dantzig–Wolfe)
3. **One agent per block** (LLM wrapper + real LP solver) so nonlinear yield notes, soft business rules, and warm-starts sit *on top of* hard constraints
4. **Shadow prices that scale like PIMS** (marginal value of streams, capacity, crude flexibility) for make-buy-sell decisions

## Quick start (CLI demos)

```bash
cd ~/projects/pims-admm-llm   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"      # optional: pytest path

export PYTHONPATH=src
python -m demos.run_demo
# Full plant (assays + dual recovery):
python -m demos.run_full_plant_demo
# Multi-period inventory smoke:
python -m demos.run_multi_period_demo
# Pure-ADMM research path (free λ; honest L∞ — not dual recovery):
python -m demos.run_full_plant_demo --pure-admm
```

## API + UI quickstart (flowsheet)

HYSYS-style PFD UI (Svelte 5 + `@xyflow/svelte`) talks to a FastAPI backend that maps canvas graphs to routing overlays and full-plant LP / ADMM.

**1. API** (repo root, port **8008**):

```bash
source .venv/bin/activate
pip install -r requirements-api.txt
export PYTHONPATH=src
uvicorn api.main:app --reload --port 8008
```

- Health: http://127.0.0.1:8008/health  
- OpenAPI: http://127.0.0.1:8008/docs  
- Details: [`api/README.md`](api/README.md)

**2. UI** (Vite dev server, port **5173**):

```bash
cd ui
npm install
npm run dev    # http://127.0.0.1:5173
```

- Details: [`ui/README.md`](ui/README.md)  
- Toolbar **Run** → `POST /api/graph` (real solve when plant imports succeed; `stub_only` fallback)  
- Recovery path selector: `mono-oracle` (default) | `pure-admm`

## Architecture (short)

```
                 ┌─────────────────────────┐
                 │   Master / ADMM loop    │
                 │  duals λ = shadow prices│
                 │  ρ, residuals, max_iter │
                 └───────────┬─────────────┘
           λ, z  ↓           ↑  x_i proposals
     ┌───────────┴───────────┴───────────┐
     │  CDU · Tanks · FCC · Coker ·      │
     │  Reformer · Blender · Utilities   │
     └─────────────────┬─────────────────┘
                       │ linking streams
```

**Material path (Wave3+):** crude → CDU → **decision arcs** (gasoil: FCC \| diesel \| sell; resid: coker \| FO; naphtha by chemical defaults) → blender, with optional tank balances and **Wave4** full unit yield streams + feed poolers.  
Source of truth: [`data/routing.json`](data/routing.json); one-pager: [`docs/routing.md`](docs/routing.md). Residuals / free disposal: [`docs/residuals.md`](docs/residuals.md).

- **Hard constraints**: always enforced by PuLP/CBC (or Gurobi if licensed) inside each block.
- **ADMM**: updates duals λ and consensus z; λ are the economic shadow prices; demos report ρ and residuals.
- **LLM layer**: optional suggestions (nonlinearity, uncertainty, business notes) that never bypass the solver (**advisory only**).

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
  models/          # CDU, tanks, FCC, coker, reformer, blender, full plant, unit_specs
  admm/            # ADMM coordinator, pure_plant_admm, residuals, dual recovery
  agents/          # LLM sub-agent + master prompts / stubs
  solvers/         # PuLP helpers, parallel runners
api/               # FastAPI flowsheet backend (graph → LP/ADMM)
ui/                # SvelteFlow / HYSYS-style PFD
demos/
  run_demo.py              # legacy monolithic vs ADMM + shadow price report
  run_full_plant_demo.py   # full plant assays + dual recovery
  run_multi_period_demo.py # multi-period inventory smoke
docs/
  portfolio.md               # 10-min portfolio walkthrough
  story.md                   # non-math stakeholder narrative + carousel
  architecture.md            # planners/managers architecture
  routing.md                 # Wave3 arc-flow superstructure one-pager
  residuals.md               # free-disposal / multi-stream ||r|| vs dual L∞
  pure_admm_floor.md         # pure-ADMM structural dual floor
  quality_blender.md         # delta-base RON+S MVP
  admm-vs-dantzig-wolfe.md   # one-pager: ADMM vs DW for coordination
  parallel_solvers.md        # parallel blocks + scale-up
data/
  routing.json               # superstructure source of truth (Wave3+4)
  assays/crudes.json
  synthetic_crudes.json
.github/workflows/ci.yml     # pytest on Python 3.11
```

## Stakeholder docs

- [docs/portfolio.md](docs/portfolio.md) — ~10 min interview/ops walkthrough + smoke script
- [docs/story.md](docs/story.md) — Smart Refinery Planning Team narrative + 6-slide carousel (non-math)
- [docs/architecture.md](docs/architecture.md) — planners/managers architecture
- [docs/routing.md](docs/routing.md) — Wave3 routing superstructure (decision arcs, chemical defaults, tanks, RON+S blender)
- [docs/residuals.md](docs/residuals.md) — mono-oracle ||r||, free-disposal shortage residual, dual L∞ honesty
- [docs/pure_admm_floor.md](docs/pure_admm_floor.md) — pure-ADMM structural dual floor (not dual recovery)
- [docs/quality_blender.md](docs/quality_blender.md) — delta-base / index quality MVP
- [docs/admm-vs-dantzig-wolfe.md](docs/admm-vs-dantzig-wolfe.md) — ADMM vs Dantzig–Wolfe one-pager
- [docs/parallel_solvers.md](docs/parallel_solvers.md) — parallel block solves + scale-up honesty

## Status

**Tip / active line: Wave5** (branch `wave5/all-phases`; residual tracker [issue #2](https://github.com/joelwwiggins/pims-admm-llm/issues/2)).  
`main` includes merged Wave3–4 (PR #3 @ `83bf360` and history through Wave4 unit streams).

| Wave | Delivered |
|------|-----------|
| **1 (MVP)** | Toy refinery (crude → CDU → intermediates → blender) with ADMM duals comparable to monolithic PuLP, timing report, LLM agent stubs |
| **2** | Full plant assays (`data/assays/`), conversion units + staging, dual recovery demo — `python -m demos.run_full_plant_demo` |
| **3** | Arc-flow **superstructure** routing (decision arcs, chemical defaults, optional tanks / multi-period inventory), quality blender (**delta-base / optional index RON + S**), ADMM demos report **ρ / primal residual / dual residual / max_iter** with mono-oracle dual recovery labeled, graph API + SvelteFlow UI scaffold, LLM **advisory only**. Source of truth: `data/routing.json` |
| **4** | Full **unit yield streams** (FCC dry gas/LPG/naphtha/LCO/slurry/coke + coker/reformer slate), **feed poolers** + direct bypass arcs, **process conditions** (ROT, C/O, …) wired into yields, HYSYS-style multi-port PFD + process inspector |
| **5 (in progress)** | Residual work only — tracked in **[#2](https://github.com/joelwwiggins/pims-admm-llm/issues/2)**: process-pool / MIP scale-up (honest parallel wall-clock), full recursive multi-level quality, optional pure-ADMM L∞ dual polish. Keep labels: `dual_recovery_path=mono-oracle` vs `pure-admm`; no silent mono dual injection on pure path. Residual semantics: [docs/residuals.md](docs/residuals.md) |

**CI:** GitHub Actions runs `pytest` on **Python 3.11** (`.github/workflows/ci.yml`).

```bash
PYTHONPATH=src python -m pytest -q
# optional one-command portfolio smoke:
bash demos/run_portfolio_smoke.sh
```

Kanban board (Wave5): `pims-admm-llm-wave5-all-phases-20260709`  
Prior waves: `pims-admm-llm-wave3-20260709`, `pims-admm-llm-wave4-unit-streams-20260709`  
Backups under `/home/joel/backups/pims-admm-llm-wave*`.

## License

MIT
