# Architecture — Smart Refinery Planning Team

*For planners and managers (light math only)*

This document explains **how the system is put together**, what each part owns, and how decisions flow from crude data to a defensible plan and shadow prices.  
For the non-technical story, see [story.md](story.md). For method choice, see [ADMM vs Dantzig–Wolfe](admm-vs-dantzig-wolfe.md).

---

## 1. Purpose

Build a **PIMS-style replacement demo** that:

1. Loads **crude / yield / product** data (synthetic today; PIMS export later).
2. Models the refinery as **block-angular** pieces (CDU, blender, …) linked by intermediate streams.
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
     ┌─────────────┴───────────┴──────────────┐
     │                                        │
 ┌───▼────┐  ┌────────┐  ┌─────────┐  ┌──────▼──────┐
 │  CDU   │  │ Tanks  │  │ Blender │  │  Utilities  │
 │ LLM+LP │  │ LLM+LP │  │ LLM+LP  │  │   LLM+LP    │
 └───┬────┘  └───┬────┘  └────┬────┘  └──────┬──────┘
     └───────────┴────────────┴──────────────┘
              linking streams (consensus z)
         naphtha · distillate · gasoil · residue
```

**Invariant:** Hard physics and commercial limits live **inside the LP**. The LLM only annotates, warm-starts, or proposes soft adjustments that the solver still has to accept under constraints.

---

## 3. Layers (outside-in)

| Layer | Responsibility | Failure mode if broken |
|-------|----------------|------------------------|
| **Data** | Crudes, yields, capacities, product prices/demand | Garbage in → wrong plan |
| **Block models** | Local LP for each unit + clear linking variables | Infeasible or unbalanced plant |
| **ADMM coordinator** | Prices λ, consensus z, iteration, dual report | Slow / non-convergent / wrong shadow prices |
| **Parallel runners** | Concurrent block solves, timing, warm-start | Serial bottleneck |
| **LLM agents** | Structured JSON proposals + human notes | Noise only if solver still rules |
| **Demo / report** | Monolithic vs ADMM, shadow price table, timings | Cannot prove value to stakeholders |

---

## 4. Block model (what a “department” is)

Each block is a **small optimization problem**:

- **Local decisions** — e.g. crude rates (CDU), product rates (blender)
- **Local constraints** — capacity, yield vectors, blend specs, demand caps
- **Linking variables** — intermediate flows that must match other blocks
- **Local objective** — margin contribution **adjusted by prices λ** and (in ADMM) a penalty for drifting from consensus z

### MVP blocks (implemented / in flight)

| Block | Local decisions | Links to rest of plant |
|-------|-----------------|------------------------|
| **CDU** | Crude purchases / charge | Intermediate **production** |
| **Blender** | Finished product slate | Intermediate **consumption** |

### Natural extensions

| Block | Role |
|-------|------|
| Tank farm | Inventory, period coupling, swing |
| Utilities | Fuel gas, steam, power, emission caps |
| FCC / hydrotreaters | Secondary conversion blocks |
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
| Dual on CDU capacity | Value of **extra charge capacity** |
| Dual on product demand / quality | Marginal value of relaxing a commercial or quality limit |
| Crude reduced costs | Relative attractiveness of each crude vs the current slate |

**Linearity:** Like PIMS, these are **locally linear** — stable until the basis (active set of constraints) changes. Early ADMM iterations are directional; **use converged λ** for decisions.

See Worker 7 deliverables (shadow price report) for PIMS-style tables and sensitivity checks.

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
  data/synthetic_crudes.json     # toy crude slate / prices / yields
  src/pims_admm_llm/
    models/                      # block & monolithic LP builders
    admm/                        # coordinator, dual updates, stop criteria
    solvers/                     # parallel runners, warm-start
    agents/                      # prompts + stub/real LLM wrappers
  demos/run_demo.py              # monolithic vs ADMM + reports
  docs/
    story.md                     # non-math narrative + 6-slide carousel
    architecture.md              # this file
    admm-vs-dantzig-wolfe.md     # method one-pager
  tests/
```

---

## 9. End-to-end planning run (ops view)

```
1. Load crude assays + product prices/demand
2. Build block LPs (+ optional monolithic twin for verification)
3. Initialize λ = 0, z = reasonable intermediate guess
4. For iteration = 1 … N:
     a. Parallel solve all blocks with current λ, z
     b. (Optional) LLM notes per block
     c. Update z from proposals; update λ from imbalance
     d. If residual < tol and duals stable → break
5. Emit:
     - Global plan (crude, intermediates, products)
     - Margin
     - Shadow price table
     - Timing vs monolithic
     - Agent notes for planner review
6. Human accepts, freezes, or re-runs with overrides
```

---

## 10. Trust, control, and safety

| Concern | How we address it |
|---------|-------------------|
| “AI hallucinates a plan” | Plan numbers come from **LP**; AI only comments |
| “Shadow prices fake” | Compare to **monolithic duals** on the same toy model |
| “We lose auditability” | JSON proposals + dual history + iteration log |
| “Production risk” | Demo scope; no live DCS; planner-in-the-loop |
| “Vendor lock-in” | Open stack: Python, PuLP/CBC (Gurobi optional), MIT license |

---

## 11. What “done” looks like for stakeholders

- [ ] Demo script runs offline on synthetic crudes  
- [ ] ADMM plan feasible and objective within tolerance of monolithic LP  
- [ ] Shadow prices reported in PIMS-like language  
- [ ] Parallel timing report exists  
- [ ] Story + architecture + ADMM/DW one-pager readable by non-math leadership  
- [ ] Clear path to load a real planning export later  

---

## Related reading

- [story.md](story.md) — Smart Refinery Planning Team narrative + 6-slide carousel  
- [admm-vs-dantzig-wolfe.md](admm-vs-dantzig-wolfe.md) — method comparison one-pager  
- [../README.md](../README.md) — install and status  

---

*Kanban: board `pims-admm-llm-20260708` · Worker 8*
