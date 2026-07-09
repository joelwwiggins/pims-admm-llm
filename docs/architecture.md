# Architecture — Smart Refinery Planning Team

*For planners and managers (light math only)*

This document explains **how the system is put together**, what each part owns, and how decisions flow from crude data to a defensible plan and shadow prices.  
For the non-technical story, see [story.md](story.md). For method choice, see [ADMM vs Dantzig–Wolfe](admm-vs-dantzig-wolfe.md). For the Wave3 arc-flow superstructure, see [routing.md](routing.md).

---

## 1. Purpose

Build a **PIMS-style replacement demo** that:

1. Loads **crude / yield / product** data (assays today; PIMS export later).
2. Models the refinery as **block-angular** pieces (CDU, optional tanks, FCC, coker, reformer, blender, …) linked by intermediate streams and **decision arcs**.
3. Coordinates those pieces with **ADMM** so duals **λ** are **shadow prices** usable for make-buy-sell.
4. Runs block solves **in parallel**.
5. Wraps each block (and the master) with an optional **LLM agent** for soft intelligence — never for hard constraint override (**advisory only**).
6. Proves against a **monolithic LP** on the same data: same feasibility class, comparable objective and duals at convergence, better wall-clock path when blocks grow.

---

## 2. System diagram

```
                 ┌─────────────────────────────┐
                 │  Master / ADMM coordinator  │
                 │  • consensus targets z      │
                 │  • duals λ = shadow prices  │
                 │  • ρ, residuals, max_iter   │
                 │  • stop when balanced       │
                 └─────────────┬───────────────┘
              λ, z ↓           ↑  proposals x_i + notes
     ┌─────────────┴───────────┴──────────────────────────┐
     │         block agents (LLM + LP each)               │
     │  CDU · Tanks · FCC · Coker · Reformer · Blender    │
     │              (+ Utilities scaffold)                │
     └────────────────────┬───────────────────────────────┘
                          │ linking streams (consensus z)
     cdu_naphtha_light · cdu_naphtha_heavy · cdu_distillate
     cdu_gasoil · cdu_resid · fcc_naphtha · fcc_lco · fcc_slurry
     coker_naphtha · coker_gasoil · reformate
```

### Plant material flow (Wave3 superstructure)

Routing is **decision-arc based**, not hard single paths. Source of truth: [`data/routing.json`](../data/routing.json); narrative one-pager: [routing.md](routing.md).

**Chemical defaults (preferred chemistry, not hard laws):**

- Heavy SR naphtha → reformer; light SR → gasoline pool  
- FCC naphtha → gasoline (soft HDT) — **not** reformer default  
- Coker naphtha → HDT/gasoline or FO — **not** reformer default  
- Gasoil **swing**: FCC \| diesel pool \| sell (sell closed by default)  
- Resid **swing**: coker \| FO  

**Tanks:** single-period → optional pass-through balances; inventory/heels when multi-period or capacity binds.

**Blender MVP:** linear **RON + sulfur** on gasoline/diesel/FO pools.

```
 Crude
   │
   ▼
  CDU ── light SR naphtha / distillate ───────────────────► BLENDER
   │
   ├─ heavy SR naphtha ─► REFORMER ─ reformate ───────────► BLENDER
   │                   └─ (optional pool bypass)
   │
   ├─ gasoil ─► [ swing: FCC | diesel pool | sell* ]
   │              FCC ─┬─ naphtha ─► gasoline (not reformer default)
   │                   ├─ LCO ─────► diesel / FO
   │                   └─ slurry ──► FO
   │
   └─ resid ──► [ swing: coker | FO* ]
                  COKER ┬─ naphtha ─► HDT→gas | FO
                        └─ gasoil ──► diesel / FO
```

\* economically gated / optional arcs  

**Path in one line:**  
`crude → CDU → decision arcs (gasoil & resid swings; naphtha by chemistry) → blender`  
(with optional tank balances; multi-period inventory when enabled)

**Invariant:** Hard physics and commercial limits live **inside the LP**. The LLM only annotates, warm-starts, or proposes soft adjustments that the solver still has to accept under constraints. ADMM demos report **ρ, primal residual, dual residual, max_iter**, and label dual recovery (**mono-oracle** vs pure λ).

---

## 3. Layers (outside-in)

| Layer | Responsibility | Failure mode if broken |
|-------|----------------|------------------------|
| **Data** | Crudes, assays, yields, capacities, product prices/demand, routing arcs | Garbage in → wrong plan |
| **Block models** | Local LP for each unit + clear linking variables + arc flows | Infeasible or unbalanced plant |
| **ADMM coordinator** | Prices λ, consensus z, ρ, residuals, iteration, dual report | Slow / non-convergent / wrong shadow prices |
| **Parallel runners** | Concurrent block solves, timing, warm-start | Serial bottleneck |
| **LLM agents** | Structured JSON proposals + human notes (advisory) | Noise only if solver still rules |
| **Demo / report** | Monolithic vs ADMM, shadow price table, timings, residual metrics | Cannot prove value to stakeholders |

---

## 4. Block model (what a “department” is)

Each block is a **small optimization problem**:

- **Local decisions** — e.g. crude rates (CDU), FCC/coker charge, product rates (blender), arc flow fractions where owned
- **Local constraints** — capacity, yield vectors, blend specs (RON+S MVP), demand caps
- **Linking variables** — intermediate flows that must match other blocks
- **Local objective** — margin contribution **adjusted by prices λ** and (in ADMM) a penalty for drifting from consensus z

### MVP / full-plant blocks

| Block | Local decisions | Links to rest of plant |
|-------|-----------------|------------------------|
| **CDU** | Crude purchases / charge | Intermediate **production** (light/heavy naphtha, distillate, gasoil, resid) |
| **Tanks** | Optional balances / inventory | Single-period pass-through; multi-period heels when inventory_mode binds |
| **FCC** | Gasoil charge (swing-selected) | Consumes gasoil arc flow; produces naphtha → pool default, LCO, slurry |
| **Coker** | Resid charge (swing-selected) | Consumes resid arc flow; produces naphtha (HDT/gas or FO), gasoil |
| **Reformer** | Heavy SR naphtha feed (preferred) | Reformate to gasoline; FCC/coker naphtha reformer arcs non-default |
| **Blender** | Finished product slate + RON/S MVP | Intermediate **consumption** into gasoline / diesel / FO under linear quality |

### Natural extensions

| Block | Role |
|-------|------|
| Utilities | Fuel gas, steam, power, emission caps |
| Hydrotreaters | Soft HDT path for cracked naphthas (attribute/`via` today; full block later) |
| Multi-period | Inventory + turnaround windows (activates tank inventory mode) |
| Delta-base quality | Beyond linear RON+S MVP |

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

**Demo metrics (Wave3):** report **ρ**, **primal residual**, **dual residual**, **max_iter** (and iterations used). Dual recovery paths must be labeled (**mono-oracle** vs pure λ).

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

1. Solver owns feasibility (arcs, capacity, quality).
2. LLM suggestions are **advisory only** until encoded as optional soft constraints or column ideas and re-validated.
3. All agent I/O is **JSON** for audit and UI.

Stub mode runs without an API key so demos always work offline.

---

## 8. Repo map (where code lives)

```
pims-admm-llm/
  data/
    routing.json                 # Wave3 arc-flow superstructure (source of truth)
    assays/crudes.json           # PIMS-shaped assays + capacities
    synthetic_crudes.json        # legacy toy CDU→blender slate
  src/pims_admm_llm/
    models/                      # block & full-plant LP builders
    admm/                        # coordinator, dual updates, stop criteria
    solvers/                     # parallel runners, warm-start
    agents/                      # prompts + stub/real LLM wrappers (advisory only)
  demos/
    run_demo.py                  # legacy monolithic vs ADMM + reports
    run_full_plant_demo.py       # full plant mono + dual recovery (ρ, residuals)
  docs/
    story.md                     # non-math narrative + 6-slide carousel
    architecture.md              # this file
    routing.md                   # Wave3 superstructure one-pager
    admm-vs-dantzig-wolfe.md     # method one-pager
  tests/
```

---

## 9. End-to-end planning run (ops view)

```
1. Load crude assays + product prices/demand + routing.json (arcs + chemical_defaults)
2. Build full-plant / block LPs (CDU, optional tanks, FCC, coker, reformer, blender)
3. Initialize λ = 0, z = reasonable intermediate guess; set ρ, max_iter
4. For iteration = 1 … N:
     a. Parallel solve all blocks with current λ, z
     b. (Optional) LLM notes per block — advisory only
     c. Update z from proposals; update λ from imbalance
     d. Record primal/dual residuals; if residual < tol and duals stable → break
5. Emit:
     - Global plan (crude, unit feeds, arc flows / swings, products)
     - Margin
     - Shadow price table (mono-oracle vs pure λ labeled)
     - ρ, residuals, iterations
     - Timing vs monolithic
     - Agent notes for planner review
6. Human accepts, freezes, or re-runs with overrides (e.g. force_all_arcs_open)
```

Demo path: `python -m demos.run_full_plant_demo` (monolithic full plant + dual recovery).

---

## 10. Trust, control, and safety

| Concern | How we address it |
|---------|-------------------|
| “AI hallucinates a plan” | Plan numbers come from **LP**; AI only comments |
| “Shadow prices fake” | Compare to **monolithic duals** on the same model; label recovery path |
| “We lose auditability” | JSON proposals + dual history + iteration log + residual metrics |
| “Production risk” | Demo scope; no live DCS; planner-in-the-loop |
| “Vendor lock-in” | Open stack: Python, PuLP/CBC (Gurobi optional), MIT license |

---

## 11. What “done” looks like for stakeholders

- [ ] Demo script runs offline on synthetic / assay crudes  
- [ ] Full plant uses arc-flow superstructure + chemical defaults (`data/routing.json`)  
- [ ] ADMM plan feasible and objective within tolerance of monolithic LP  
- [ ] Demos report ρ, primal/dual residuals, max_iter; dual recovery labeled  
- [ ] Shadow prices reported in PIMS-like language (dual recovery demo)  
- [ ] Parallel timing report exists  
- [ ] Story + architecture + routing + ADMM/DW one-pager readable by non-math leadership  
- [ ] Clear path to load a real planning export later  

---

## Related reading

- [story.md](story.md) — Smart Refinery Planning Team narrative + 6-slide carousel  
- [routing.md](routing.md) — Wave3 arc-flow superstructure (from `data/routing.json`)  
- [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md) — method comparison one-pager  
- [../README.md](../README.md) — install and status  

---

*Wave3 · arc-flow routing · board `pims-admm-llm-wave3-20260709`*
