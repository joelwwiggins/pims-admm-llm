# Portfolio: PIMS-class planning in ~10 minutes

*Stakeholder walkthrough for demos, interviews, and ops briefings.*  
*Companions: [story.md](story.md) (non-math narrative) · [architecture.md](architecture.md) · [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md) · [routing.md](routing.md)*

**Runtime:** ~10 minutes talk + live smoke (`demos/run_portfolio_smoke.sh`).

---

## 0. One-line pitch

**Open-source Aspen PIMS-style replacement demo:** crude assays → block-angular plant LP → **ADMM coordination** with honest dual recovery → parallel unit agents (LLM advisory only) → **HYSYS-style snap-together UI**.

You get the same economic products planners already trust — **feasible plan + shadow prices for make-buy-sell** — with a department-shaped architecture that parallelizes and explains itself.

---

## 1. The problem (2 min)

Every planning cycle the refinery must answer:

| Decision | Why it hurts |
|----------|----------------|
| Which crudes, how much? | Crude flexibility is money; wrong slate locks margin |
| How hard to run CDU / conversion? | Capacity duals tell you what to expand |
| Where do intermediates go? | Gasoil → FCC vs diesel vs sell; resid → coker vs FO |
| What products under specs? | RON, sulfur, demand caps — quality binds margin |
| Make / buy / sell at the boundary? | Needs **shadow prices**, not just a production slate |

Classic commercial tools (PIMS / RPMS / similar) put this in **one giant monolithic LP**. That works, but:

1. **Hard to parallelize** — one big solve, limited “department” ownership  
2. **Opaque re-plans** — hard to explain *why* the plan moved  
3. **Awkward “almost linear” reality** — yields, soft business rules, operator judgment sit outside the model  
4. **Shadow prices are gold** — but only if you can get them fast enough for the meeting and **trust how they were recovered**

**What we keep from PIMS:** feasibility under hard constraints, margin objective, marginal values for make-buy-sell.  
**What we change:** structure the model like the plant (blocks + linking streams), coordinate with ADMM, label dual honesty, and put a flowsheet UI on top.

---

## 2. Block-angular ADMM (3 min)

### Plant as departments, not one black box

Refinery planning LPs are naturally **block-angular**:

- **Local blocks:** CDU, tanks, FCC, coker, reformer, HDT, blender, utilities  
- **Linking constraints:** intermediate balances, inventory, shared capacity, decision arcs  

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

### Wave3 material story (arc-flow superstructure)

Not a fixed pipe map — a **decision network** (`data/routing.json`):

```
crude → CDU → decision arcs
   gasoil:  FCC | diesel pool | sell*
   resid:   coker | FO*
   naphtha: chemistry defaults (heavy SR → reformer;
            FCC/coker naph → gasoline/HDT/FO — not reformer)
         → blender (delta-base RON + S MVP)
```

\* economically gated; often closed by default until what-if.

### How one coordination cycle feels

1. **Boss (master)** broadcasts prices λ (and consensus targets z) on linking streams  
2. **Each block** solves its own small LP under those prices — **in parallel**  
3. Master updates λ / z from imbalances; reports **ρ, primal residual, dual residual, max_iter**  
4. Stop when the plant balances → plan + shadow prices  

Hard physics and commercial limits stay in **PuLP/CBC** (or Gurobi).  
**LLM agents are advisory only** — notes, soft suggestions, warm-starts; never override mass balance or capacity.

### Why ADMM (vs classic Dantzig–Wolfe)

| Need | ADMM fit |
|------|----------|
| Explicit λ every round | Natural shadow-price product |
| Parallel block solves | Excellent |
| One LLM + LP agent per unit | Local augmented solve is natural |
| No growing column pool in MVP | Simpler ops |

At LP convergence, λ has the **same economic meaning** as PIMS duals on linking balances. Early iterations are directional only — use **converged** values for commercial decisions. Full comparison: [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md).

### Toy honesty on wall-clock

On a *tiny* single-period plant, mono CBC can still beat parallel ADMM on wall time — pool overhead wins. Parallel wins at **portfolio scale** (multi-period / multi-site waves) or larger unit MIPs. Narrative: **scale, agents, duals, explainability** — not “always faster on a 30 ms toy.” See [parallel_solvers.md](parallel_solvers.md).

---

## 3. Mono-oracle honesty (2–3 min)

Duals are the commercial product. We **never** blur recovery paths.

| Path | What it is | What you may claim |
|------|------------|--------------------|
| **`mono-oracle` (default)** | Recover a feasible plan; take duals from the monolithic balances on that plan | Dual L∞ gap ≈ **0 by construction** vs mono bal_* — **labeled dual recovery** |
| **`pure-admm`** | Free λ via consensus + market-clearing updates (`λ ← λ − αρ(prod−use)`, λ≥0, shortage residual) | Path **ran**; residuals controlled; L∞ vs mono reported honestly — **not** dual recovery |

### Talking points for stakeholders

1. **Mono LP remains plan truth** for objective and unit feeds when paths disagree.  
2. Default demos and UI use **`dual_recovery_path=mono-oracle`** so shadow tables are defensible.  
3. Pure-ADMM is the research / multi-agent λ path — always print  
   `lambda_vs_mono_Linf`, shortage residual, and the honesty string from the solver.  
4. Early ADMM λ is **directional**; commercial make-buy-sell uses converged / recovered duals.  
5. VERDICT lines in demos always **name the path** — never claim “ADMM duals match mono” without saying *which* recovery.

### Live proof commands

```bash
# Default mono-oracle dual recovery
PYTHONPATH=src python -m demos.run_full_plant_demo
# Free λ path (honest L∞; not dual recovery)
PYTHONPATH=src python -m demos.run_full_plant_demo --pure-admm
```

Look for:

- `dual_recovery_path: mono-oracle` (or `pure-admm`)  
- `honesty: ...` on the report  
- `VERDICT: PASS — ... path=mono-oracle` (or pure-admm wording that **denies** dual recovery)

---

## 4. UI — HYSYS-style flowsheet (2 min)

Snap-together **process flow diagram** for the Wave3 plant: Svelte 5 + `@xyflow/svelte` frontend, FastAPI backend.

| Zone | Role |
|------|------|
| **Toolbar** | Run / Validate / Reset; recovery path (`mono-oracle` \| `pure-admm`) |
| **Left dock** | Unit palette + solve results |
| **Canvas** | Multi-port PFD nodes + stream edges |
| **Right inspector** | Unit / stream properties (General, Streams, Yields, Process, Composition) |

### Run locally

```bash
# API — port 8008
source .venv/bin/activate
export PYTHONPATH=src
uvicorn api.main:app --reload --port 8008

# UI — Vite proxies /api and /health → :8008
cd ui && npm install && npm run dev   # http://127.0.0.1:5173
```

### Contract (lockstep)

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | liveness |
| GET | `/api/routing` | units/arcs from `data/routing.json` + palette |
| POST | `/api/connect` | edge allowed? score + reason |
| POST | `/api/graph` | graph → LP/ADMM (`recovery_path`, `run_admm`, `stub_only`) |

Chemical defaults in the template match routing: gasoil→FCC, resid→coker, heavy SR→reformer, FCC/coker naph→**HDT/gasoline** (not reformer). Inactive nodes skip clusters. Details: `ui/README.md`, `api/README.md`.

---

## 5. 10-minute live agenda (suggested)

| Min | Segment | Show |
|-----|---------|------|
| 0–2 | Problem | One giant LP vs department-shaped planning |
| 2–5 | Block-angular ADMM | Architecture sketch + routing swings |
| 5–7 | Mono-oracle honesty | Full-plant dual table + VERDICT path label |
| 7–9 | UI | Palette → connect → Run → obj / duals / splits |
| 9–10 | Close | Smoke script green; next steps |

**Close line:** *Same math guarantees planners expect from PIMS — feasibility and shadow prices — delivered as a multi-agent plant with honest dual labeling and a flowsheet you can rearrange.*

---

## 6. Smoke the portfolio (1 command)

From repo root:

```bash
./demos/run_portfolio_smoke.sh
```

Activates `.venv`, runs `pytest -q`, toy mono-vs-ADMM demo, full-plant dual-recovery demo, and prints **VERDICT** lines for each step.

Manual equivalents:

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m pytest tests/ -q
python -m demos.run_demo
python -m demos.run_full_plant_demo
```

---

## 7. Related assets

| Asset | Path |
|-------|------|
| Non-math story + carousel | [story.md](story.md) |
| Architecture | [architecture.md](architecture.md) |
| Routing superstructure | [routing.md](routing.md) · `data/routing.json` |
| ADMM vs DW | [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md) |
| Shadow prices | [shadow_prices.md](shadow_prices.md) |
| Parallel / scale-up | [parallel_solvers.md](parallel_solvers.md) |
| Quality blender MVP | [quality_blender.md](quality_blender.md) |
| Toy demo | `python -m demos.run_demo` |
| Full plant demo | `python -m demos.run_full_plant_demo` |
| Portfolio smoke | `demos/run_portfolio_smoke.sh` |
| Explainer video | `demos/video/` (~4–5 min) |

---

## 8. Status snapshot (for portfolio decks)

- **MVP:** synthetic crude → CDU → intermediates → blender; mono vs ADMM within tolerance  
- **Wave2:** assay-driven full plant + dual recovery path  
- **Wave3:** arc-flow superstructure, quality blender MVP, flowsheet UI + API  
- **Wave3b:** graph→LP, pure-ADMM λ path (labeled), multi-period tanks smoke  
- **Wave4:** unit yield streams, feed poolers, process conditions on inspector  
- **Invariant:** LLM advisory only; hard constraints in the solver; dual recovery path always labeled  

*This is a demo-grade open-source replacement path — not a certified Aspen product.*
