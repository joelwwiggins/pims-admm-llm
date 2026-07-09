# Architecture — Smart Refinery Planning Team

*For planners and managers (light math only)*

This document explains **how the system is put together**, what each part owns, and how decisions flow from crude data to a defensible plan and shadow prices.  
For the non-technical story, see [story.md](story.md). For method choice, see [ADMM vs Dantzig–Wolfe](admm-vs-dantzig-wolfe.md). For hard routes, see [routing.md](routing.md).

---

## 1. Purpose

Build a **PIMS-style replacement demo** that:

1. Loads **crude / yield / product** data (assays today; PIMS export later).
2. Models the refinery as **block-angular** pieces (CDU, tanks, FCC, coker, reformer, blender, …) linked by intermediate streams.
3. Coordinates those pieces with **ADMM** so duals **λ** are **shadow prices** usable for make-buy-sell.
4. Runs block solves **in parallel**.
5. Wraps each block (and the master) with an optional **LLM agent** for soft intelligence — never for hard constraint override.
6. Proves against a **monolithic LP** on the same data: same feasibility class, comparable objective and duals at convergence, better wall-clock path when blocks grow.

---

## 2. System diagram

```
                 ┌─────────────────────────────┐
                 │  Master / ADMM coordinator  │
                 │  • consensus targets z      │
                 │  • duals λ = shadow prices  │
                 │  • stop when balanced       │
                 └─────────────┬───────────────┘
              λ, z ↓           ↑  proposals x_i + notes
     ┌─────────────┴───────────┴──────────────────────────┐
     │         block agents (LLM + LP each)               │
     │  CDU · Tanks · FCC · Coker · Reformer · Blender    │
     │              (+ Utilities scaffold)                │
     └────────────────────┬───────────────────────────────┘
                          │ linking streams (consensus z)
     cdu_naphtha · cdu_distillate · cdu_gasoil · cdu_resid
     fcc_naphtha · fcc_lco · fcc_slurry · coker_naphtha
     coker_gasoil · reformate
```

### Plant material flow (routing)

Hard rules (see [routing.md](routing.md) and `data/routing.json`):

- CDU gasoil → **tank** → FCC  
- FCC naphtha → **tank** → Reformer  
- CDU resid → **tank** → Coker  
- Coker naphtha → **tank** → Reformer  

```
 Crude
   │
   ▼
  CDU ── naphtha / distillate ──────────────────────────────► BLENDER
   │
   ├─ gasoil ─► TANK_GASOIL ─► FCC ─┬─ naphtha ─► TANK ─┐
   │                                ├─ LCO ─────────────┼──► BLENDER
   │                                └─ slurry ──────────┤
   │                                                    │
   └─ resid ──► TANK_RESID ──► COKER ┬─ naphtha ─► TANK ┤
                                     └─ gasoil ─────────┼──► BLENDER
                                                        │
                         REFORMER ◄── (FCC + coker naphtha tanks)
                            │
                            └─ reformate ──────────────────► BLENDER
```

**Path in one line:**  
`crude → CDU → (gasoil → tank → FCC, resid → tank → coker) → naphthas → tanks → reformer → blender`

**Invariant:** Hard physics and commercial limits live **inside the LP**. The LLM only annotates, warm-starts, or proposes soft adjustments that the solver still has to accept under constraints.

---

## 3. Layers (outside-in)

| Layer | Responsibility | Failure mode if broken |
|-------|----------------|------------------------|
| **Data** | Crudes, assays, yields, capacities, product prices/demand, routing | Garbage in → wrong plan |
| **Block models** | Local LP for each unit + clear linking variables | Infeasible or unbalanced plant |
| **ADMM coordinator** | Prices λ, consensus z, iteration, dual report | Slow / non-convergent / wrong shadow prices |
| **Parallel runners** | Concurrent block solves, timing, warm-start | Serial bottleneck |
| **LLM agents** | Structured JSON proposals + human notes | Noise only if solver still rules |
| **Demo / report** | Monolithic vs ADMM, shadow price table, timings | Cannot prove value to stakeholders |

---

## 4. Block model (what a “department” is)

Each block is a **small optimization problem**:

- **Local decisions** — e.g. crude rates (CDU), FCC/coker charge, product rates (blender)
- **Local constraints** — capacity, yield vectors, blend specs, demand caps
- **Linking variables** — intermediate flows that must match other blocks
- **Local objective** — margin contribution **adjusted by prices λ** and (in ADMM) a penalty for drifting from consensus z

### MVP / full-plant blocks

| Block | Local decisions | Links to rest of plant |
|-------|-----------------|------------------------|
| **CDU** | Crude purchases / charge | Intermediate **production** (naphtha, distillate, gasoil, resid) |
| **Tanks** | Inventory / pass-through on staged streams | Gasoil before FCC; resid before coker; FCC & coker naphtha before reformer |
| **FCC** | Gasoil charge rate | Consumes tanked gasoil; produces naphtha, LCO, slurry |
| **Coker** | Resid charge rate | Consumes tanked resid; produces naphtha, gasoil |
| **Reformer** | Naphtha feed rate | Consumes tanked FCC + coker naphtha; produces reformate |
| **Blender** | Finished product slate | Intermediate **consumption** into gasoline / diesel / FO |

### Natural extensions

| Block | Role |
|-------|------|
| Utilities | Fuel gas, steam, power, emission caps |
| Hydrotreaters | Pre-treat gasoil / naphtha quality |
| Multi-period | Inventory + turnaround windows |

---

## 5. Coordination (what the Boss does)

### ADMM in planner language

Every round:

1. **Boss** publishes current **prices λ** and **targets z** for each intermediate.
2. **Blocks** (in parallel) each solve their local plan: maximize local margin **plus** price terms **minus** a penalty if they diverge from z.
3. **Boss** averages (or otherwise combines) the proposed linking flows into a new z, and **updates λ** from the remaining imbalance.
4. When imbalance is small enough (and duals stabilize), **stop**.

At a good stopping point:

- Intermediate supply ≈ demand (feasible plant balance)
- **λ** ≈ marginal value of each linking stream / capacity — same **economic role** as PIMS duals
- Objective comparable to a full one-shot LP on the same model

### Why not only “one big solve”?

| Concern | Monolithic PIMS-style | This architecture |
|---------|----------------------|-------------------|
| Re-run when one unit changes | Often full re-solve | Mostly that block + few ADMM rounds |
| Parallel hardware | Hard | Natural |
| Explaining “why” | Sparse duals + analyst skill | Duals **plus** per-block narratives |
| Nonlinear / soft rules | Yield vectors only | LLM/surrogate **on top of** LP |

See also the method comparison: [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md).

---

## 6. Shadow prices (make-buy-sell)

| Signal | Meaning for commercial / planning |
|--------|-------------------------------------|
| λ on intermediate balance | Value of **one more barrel** of that intermediate at the boundary |
| Dual on unit capacity | Value of **extra charge capacity** (CDU, FCC, coker, reformer, …) |
| Dual on product demand / quality | Marginal value of relaxing a commercial or quality limit |
| Crude reduced costs | Relative attractiveness of each crude vs the current slate |

**Linearity:** Like PIMS, these are **locally linear** — stable until the basis (active set of constraints) changes. Early ADMM iterations are directional; **use converged λ** for decisions.

See Worker 7 deliverables (shadow price report / full-plant dual recovery) for PIMS-style tables and sensitivity checks.

---

## 7. LLM multi-agent layer

| Agent | Inputs | Outputs (structured) |
|-------|--------|----------------------|
| Sub-agent (per block) | λ, z, local data, last solution | proposal, local_obj, reduced_costs, suggestion |
| Master agent | All proposals, residual, iteration | continue/stop, interpretation of λ, human summary |

**Rules of engagement:**

1. Solver owns feasibility.
2. LLM suggestions are **advisory** until encoded as optional soft constraints or column ideas and re-validated.
3. All agent I/O is **JSON** for audit and UI.

Stub mode runs without an API key so demos always work offline.

---

## 8. Repo map (where code lives)

```
pims-admm-llm/
  data/
    routing.json                 # hard plant routes (source of truth)
    assays/crudes.json           # PIMS-shaped assays + capacities
    synthetic_crudes.json        # legacy toy CDU→blender slate
  src/pims_admm_llm/
    models/                      # block & full-plant LP builders
    admm/                        # coordinator, dual updates, stop criteria
    solvers/                     # parallel runners, warm-start
    agents/                      # prompts + stub/real LLM wrappers
  demos/
    run_demo.py                  # legacy monolithic vs ADMM + reports
    run_full_plant_demo.py       # full plant mono + dual recovery
  docs/
    story.md                     # non-math narrative + 6-slide carousel
    architecture.md              # this file
    routing.md                   # plant routing one-pager
    admm-vs-dantzig-wolfe.md     # method one-pager
  tests/
```

---

## 9. End-to-end planning run (ops view)

```
1. Load crude assays + product prices/demand + routing.json
2. Build full-plant / block LPs (CDU, tanks, FCC, coker, reformer, blender)
3. Initialize λ = 0, z = reasonable intermediate guess
4. For iteration = 1 … N:
     a. Parallel solve all blocks with current λ, z
     b. (Optional) LLM notes per block
     c. Update z from proposals; update λ from imbalance
     d. If residual < tol and duals stable → break
5. Emit:
     - Global plan (crude, unit feeds, tanks, products)
     - Margin
     - Shadow price table
     - Timing vs monolithic
     - Agent notes for planner review
6. Human accepts, freezes, or re-runs with overrides
```

Demo path: `python -m demos.run_full_plant_demo` (monolithic full plant + dual recovery).

---

## 10. Trust, control, and safety

| Concern | How we address it |
|---------|-------------------|
| “AI hallucinates a plan” | Plan numbers come from **LP**; AI only comments |
| “Shadow prices fake” | Compare to **monolithic duals** on the same model |
| “We lose auditability” | JSON proposals + dual history + iteration log |
| “Production risk” | Demo scope; no live DCS; planner-in-the-loop |
| “Vendor lock-in” | Open stack: Python, PuLP/CBC (Gurobi optional), MIT license |

---

## 11. What “done” looks like for stakeholders

- [ ] Demo script runs offline on synthetic / assay crudes  
- [ ] Full plant routes honor tank-before-conversion hard rules  
- [ ] ADMM plan feasible and objective within tolerance of monolithic LP  
- [ ] Shadow prices reported in PIMS-like language (dual recovery demo)  
- [ ] Parallel timing report exists  
- [ ] Story + architecture + routing + ADMM/DW one-pager readable by non-math leadership  
- [ ] Clear path to load a real planning export later  

---

## Related reading

- [story.md](story.md) — Smart Refinery Planning Team narrative + 6-slide carousel  
- [routing.md](routing.md) — plant routing one-pager (from `data/routing.json`)  
- [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md) — method comparison one-pager  
- [../README.md](../README.md) — install and status  

---

*Kanban: board `pims-admm-llm-20260708` · Worker 7 / 8*
