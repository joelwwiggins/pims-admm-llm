# ADMM vs Dantzig–Wolfe — One-Pager

*For planners, managers, and engineers choosing a coordination method*  
*Companion to [story.md](story.md) and [architecture.md](architecture.md)*

---

## The shared idea (no equations required)

Refinery planning models naturally split into **departments** (CDU, tanks, blenders, utilities) tied together by a few **connecting streams**.

Both methods avoid solving only one giant model every time:

| | **Dantzig–Wolfe (DW)** | **ADMM** |
|--|------------------------|----------|
| Mental model | Departments send **candidate plans (columns)**; a small boss problem picks the mix | Departments solve with **prices + stick-to-target** penalty; boss updates **prices and targets** |
| What the boss broadcasts | Prices (duals) from a restricted master | Duals **λ** and consensus targets **z** |
| What comes back | New extreme / improving proposals | Full local solutions under current λ, z |
| Shadow prices | Duals of the master linking constraints | Explicit dual variables **λ** updated each round |
| Parallel friendly | Good | Excellent (designed for distributed) |
| Implementation feel | Column pool, master LP growth | Fixed-size local problems + simple dual update |
| Classic home | Large LPs, generation of extreme points | Distributed energy, MPC, multi-block consensus |

**This project leads with ADMM** as the default coordinator, with the **same economic product**: shadow prices for make-buy-sell. DW ideas remain optional (column-style proposals inside a block or hybrid later).

---

## Side-by-side for your LLM multi-agent setup

| Aspect | Dantzig–Wolfe | ADMM | Better fit for one-agent-per-block + LLM |
|--------|---------------|------|------------------------------------------|
| Shadow prices / duals | Duals of restricted master | Explicit **λ** each round | **Tie** (both valid at convergence) |
| Complexity to implement | Manage growing columns | No column pool | **ADMM** |
| Parallel / distributed | Good | Excellent | **ADMM** |
| Numerical stability | Can tail-off / head-in | Quadratic penalty steadies | **ADMM** (often) |
| LLM agent friendliness | Works (propose columns) | Very natural (augmented local solve) | **ADMM** |
| Finite LP theory | Classic finite convergence | Practical convergence | DW edge on pure LP theory |
| Industry pedigree in refining | Decades of use | Rising (MPC / energy / distributed) | Context-dependent |

---

## ADMM in one paragraph (planner language)

Each department solves its own problem as if intermediate streams had a **market price** (λ) and a **preferred volume** (z). If a department wants more or less than the plant can agree on, the price of that stream moves — just like a real market clearing. After a few rounds, prices and volumes settle. Those settled prices are your **shadow prices**.

## Dantzig–Wolfe in one paragraph (planner language)

Each department, given prices, proposes its **best extreme plan**. The boss keeps a shortlist of those plans and chooses how to **mix** them so connecting streams balance. New good plans get added to the shortlist until nothing better exists. Duals of that small mix problem are your **shadow prices**.

---

## Why we chose ADMM first for pims-admm-llm

1. **Maps cleanly** to “one LLM + LP agent per block.”  
2. **No growing column database** in the MVP.  
3. **Natural parallelism** for Worker 4 timing benchmarks.  
4. Duals **λ** are first-class every iteration → easy PIMS-style reporting (Worker 7).  
5. Plays well with future **MPC-style** and multi-period extensions already common in industry ADMM papers.

DW remains a valid complementary story:

- Teach planners who already know column generation.  
- Optionally generate improving columns *inside* a block.  
- Hybrid prototypes later without changing the agent roles.

---

## Shadow prices: same job as PIMS?

| Question | Answer |
|----------|--------|
| Same economic meaning as PIMS duals? | **Yes** at convergence for linear models |
| Scale / “locally linear” like PIMS ranges? | **Yes** — piecewise constant until basis changes |
| Trust early iterations? | Directional only; use **converged** λ for commercial decisions |
| How we prove it | Toy **monolithic LP duals** vs ADMM λ on identical data |

---

## Decision cheat-sheet

| If you care most about… | Prefer |
|-------------------------|--------|
| Fast multi-agent prototype + parallel solves | **ADMM** (default here) |
| Pure LP column-generation theory / extreme points | **Dantzig–Wolfe** |
| Distributed energy / MPC style plant control | **ADMM** |
| Explaining to OR textbooks | Show **both**; same block-angular plant |
| Production MVP path in this repo | **ADMM** + optional DW notes |

---

## Where this lives in the product story

```
Smart Refinery Planning Team (story.md)
        │
        ▼
Architecture: blocks + master + LLM (architecture.md)
        │
        ▼
Coordination engine: ADMM (this page)  ·  optional DW later
        │
        ▼
Shadow prices λ → make-buy-sell tables → planner decisions
```

---

## Further reading inside the repo

- [story.md](story.md) — non-math narrative + 6-slide carousel  
- [architecture.md](architecture.md) — components and run loop  
- [../README.md](../README.md) — install and status  

---

*One-pager · Worker 8 · board `pims-admm-llm-20260708`*
