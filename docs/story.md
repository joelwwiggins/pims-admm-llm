# The Smart Refinery Planning Team

*A non-math story for planners, operators, and managers*

This project is not “replace the planner with a black box.”  
It is **a team of specialists that works the way a good refinery already works** — by department — with one boss who keeps the whole plant in balance.

---

## The problem you already know

Every planning cycle the refinery has to decide:

- Which crudes to buy (and how much)?
- How hard to run the crude unit and where to cut?
- What intermediates go to tanks, blenders, or sale?
- What finished products to make to hit specs and demand?
- How to keep steam, fuel gas, and other utilities in line?

In classic tools (PIMS / RPMS / similar), this is often one **giant** linear program: thousands of variables and constraints solved all at once. It works — but it is slow to re-run, hard to parallelize, awkward when yields are “almost linear,” and opaque when someone asks *why* the plan changed.

Shadow prices (marginal values) from that model are gold for **make-buy-sell**, crude evaluation, and “what if we had 5% more tank space?” — but only if you trust them and can get them fast enough to use in the meeting.

---

## Meet the team (not the math)

Imagine a standing planning crew:

| Role | Who they are | What they own |
|------|----------------|---------------|
| **Boss (Master Coordinator)** | Plant / planning manager persona | Linking streams, balance, stop/go, final plan |
| **CDU Agent** | Crude unit expert | Crude slate, yields, CDU capacity |
| **Tank Farm Agent** | Storage & timing expert | Optional intermediate balances (gasoil, resid, cracked naphthas); heels / inventory mainly when multi-period or capacity binds |
| **FCC Agent** | Cat cracker expert | Gasoil feed (when swing chooses FCC), capacity, naphtha / LCO / slurry yields |
| **Coker Agent** | Delayed coker expert | Resid feed (when swing chooses coker), capacity, naphtha / gasoil yields |
| **Reformer Agent** | Reforming expert | Heavy SR naphtha feed (preferred chemistry), reformate to gasoline |
| **Blender Agent** | Product quality & recipes | Specs (delta-base/index RON + S), blend recipes, product demand |
| **Utilities Agent** | Energy / fuel gas expert | Steam, fuel, power limits *(scaffold / future)* |

Each department expert is really **two things working together**:

1. A **reliable calculator** (LP solver) that never breaks hard rules (capacity, mass balance, min/max rates).
2. An **experienced operator brain** (LLM) that can notice soft things the linear model misses: “this crude is running richer in mid-cut than the yield vector says,” “if we ease the cut we free high-value naphtha,” “market wants more ULSD this week.”

The Boss does **not** re-solve the whole plant alone. He sets **price signals** (shadow prices / duals) for the streams that connect departments — naphtha, distillate, gasoil, residue, FCC and coker products, reformate — and asks each department for its best local plan under those prices.

---

## How molecules move (routing story — Wave3 superstructure)

Think of the plant as a **decision network**, not one black box and not one fixed pipe map. Crude comes in; each unit does one job; **decision arcs** choose where intermediates go under economics and **chemical defaults**. Optional tanks are balance / inventory nodes — not hard fences that force a single path.

1. **Crude → CDU.** The crude unit splits the barrel into light and heavy straight-run naphtha, distillate, gasoil, and resid.
2. **Naphtha chemistry (defaults, not hard laws).**
   - **Light SR** → gasoline pool.
   - **Heavy SR** → **reformer** (preferred) → reformate → gasoline; optional bypass to pool.
   - **FCC naphtha** → **gasoline** (soft HDT for sulfur) — **not** reformer by default.
   - **Coker naphtha** → soft **HDT → gasoline** or **FO** — **not** reformer by default.
3. **Gasoil swing.** CDU gasoil can go to **FCC**, to the **diesel / distillate pool**, or (optionally) **sell** as intermediate. The solver picks fractions; sell is closed by default.
4. **Resid swing.** CDU resid can go to the **coker** or to **fuel oil**. Conversion margin usually favors the coker; FO remains available.
5. **Tanks (optional philosophy).** In a **single-period** plan, tanks collapse to pure balances (no mandatory inventory). **Inventory, heels, and swing capacity** matter when you go multi-period or when tank limits bind.
6. **Blender quality (delta-base MVP).** Finished gasoline meets **delta-base RON + sulfur** (optional RON index); diesel uses soft-HDT linear S. Not full multi-level PIMS delta-base recursion — see `docs/quality_blender.md`. Reformate, cracked naphthas, and SR cuts meet in the gasoline pool under those specs.

So the story in one breath:  
**crude → CDU → decision arcs (gasoil: FCC \| diesel \| sell; resid: coker \| FO; naphthas by chemistry) → blender**, with tanks as optional staging when inventory matters.

That map lives in [`data/routing.json`](../data/routing.json) (source of truth) and the one-pager [routing.md](routing.md).

---

## How a planning “conversation” works (one cycle)

### Step 1 — The Boss sets the tone

> “Right now, extra naphtha is worth about $X/bbl at the boundary. Distillate is worth $Y. Plan your unit around that.”

Those numbers are not guesses. They come from the coordination math (ADMM duals λ). Economically they mean the same thing as PIMS-style **marginal values**: how much overall margin improves if you relax a linking balance by one barrel.

### Step 2 — Each department makes its plan (in parallel)

- **CDU** picks crude rates and produces intermediates under yield and capacity limits.
- **Tanks** (when used) balance gasoil, resid, and cracked naphthas — pass-through in single-period; inventory when multi-period / capacity binds.
- **FCC** takes gasoil that the **swing** sends its way and makes naphtha, LCO, and slurry (FCC naphtha prefers the gasoline pool).
- **Coker** takes resid that the swing sends its way and makes naphtha and gasoil (coker naphtha prefers HDT/gasoline or FO).
- **Reformer** upgrades **heavy SR naphtha** (chemical default) to reformate — not the default home for FCC/coker naphtha.
- **Blender** consumes intermediates into finished products under **RON + sulfur** MVP specs and demand caps.
- (Utilities, when live, do the same for energy.)

Each agent returns:

- A concrete proposal (rates, flows)
- Local economics
- Optional LLM note: one smart, human-readable suggestion the pure LP cannot see

They do this **at the same time** — no waiting for a single monolithic solve.

### Step 3 — The Boss reviews and adjusts

If everyone’s proposals already line up on the linking streams (supply ≈ demand at the boundaries), the Boss says:

> “We’re done. Here’s the global plan, the margin, and the shadow prices for make-buy-sell.”

If not, he updates the prices (and consensus targets) and they go around again. For a well-structured refinery model this is usually a **handful of quick rounds**, not an all-night batch.

---

## Why this feels like the plant — not a black box

- Each specialist only worries about **their** unit — same as real ops.
- The Boss keeps the **whole refinery** balanced — same as real planning.
- **Hard constraints stay with the solver** — feasibility is not optional.
- **LLM never overrides** mass balance or capacity; it only proposes smarter soft adjustments that the solver can accept or reject.
- Planners stay in control: review suggestions, freeze crudes, force products, re-price intermediates.

It feels more like a **collaborative team meeting** than a single opaque optimization run.

---

## What you get that looks like PIMS (and what improves)

| You already use in PIMS | What this team gives you |
|-------------------------|---------------------------|
| Plan (crude → products) | Same decision surface |
| Shadow / marginal values | Duals λ at convergence — same economic meaning |
| Make-buy-sell signals | Value of extra intermediate, capacity, crude flexibility |
| “What if” sensitivities | Locally linear around the plan (like PIMS ranges) |
| One big model | **Plus** parallel block solves + natural multi-agent interface |

Improvements the team is built for:

- **Faster re-plans** when one block changes (only that agent + a few coordination rounds)
- **Parallelism** by design (one block = one process / one agent)
- **Room for nonlinear reality** without throwing away LP guarantees
- **Explainability** in English: each agent can say *why* it proposed a change

---

## Carousel (6 slides) — for LinkedIn / ops briefings

Copy these into slides or a swipe deck.

### Slide 1/6 — Title & big idea

**Smart Refinery Planning Team**  
One boss. Specialist agents per unit. Real LP math + LLM judgment.  
*Faster, clearer plans than one giant black-box LP — without losing feasibility or shadow prices.*

### Slide 2/6 — Meet the team

- **Boss** — prices the connecting streams, keeps the plant balanced  
- **CDU** — crudes & yields  
- **Tanks** — optional balances / inventory (single-period pass-through vs multi-period)  
- **FCC** — gasoil cracker (when swing chooses FCC)  
- **Coker** — resid conversion (when swing chooses coker)  
- **Reformer** — heavy SR naphtha → reformate  
- **Blender** — specs & products (MVP: RON + S)  
- **Utilities** — energy balance  

Each expert = **solver (rules)** + **LLM (judgment)**

### Slide 3/6 — Step 1: Boss sets the tone

Sends **price signals** for naphtha, distillate, gasoil, residue, FCC/coker products, reformate:  
“Plan as if an extra barrel of X is worth λₓ.”  
Those λ values **are** the shadow prices used for make-buy-sell.

### Slide 4/6 — Step 2: Departments plan in parallel

Each block solves **its own small problem** under current prices.  
Returns proposal + optional operator-style note.  
No one waits on a 10,000-row monolithic solve.

### Slide 5/6 — Step 3: Review & adjust

Boss checks whether intermediate flows **agree** across blocks.  
If yes → final plan. If no → update prices and loop.  
Usually a few rounds. Humans can interrupt anytime.

### Slide 6/6 — The results

- Globally feasible plan  
- Margin you can defend  
- Shadow prices that behave like PIMS marginal values  
- Audit trail of each department’s proposal and notes  

**Next:** [Architecture for planners](architecture.md) · [ADMM vs Dantzig–Wolfe one-pager](admm-vs-dantzig-wolfe.md)

---

## 60-second elevator version

> We broke the planning model into the same pieces your plant already has — crude unit, optional intermediate tanks, FCC, coker, reformer, blender, utilities. Molecules follow an **arc-flow superstructure**: gasoil and resid **swing** by economics; naphtha paths follow **chemical defaults** (FCC/coker naphtha → gasoline or FO, not reformer; heavy SR → reformer). The blender enforces **delta-base RON + sulfur** (planning-grade; optional index). Each piece has a specialist agent with a real optimizer and an AI co-pilot (advisory only). A master coordinator sets prices on the streams that connect them, same idea as shadow prices in PIMS, and demos report ρ and residuals clearly. They talk for a few quick rounds until the plant balances. You get a solid plan, make-buy-sell values, and explanations — faster, and easier to extend when yields aren’t perfectly linear.

---

## Related docs

- [Architecture (planners / managers)](architecture.md) — components, data flow, who owns what  
- [Plant routing one-pager](routing.md) — Wave3 arc-flow superstructure from `data/routing.json`  
- [ADMM vs Dantzig–Wolfe one-pager](admm-vs-dantzig-wolfe.md) — why we lead with ADMM, when DW still matters  
- [README](../README.md) — install, demo, repo layout  

---

*Audience: planners, supervisors, commercial teams, leadership. Math details live in code under `src/pims_admm_llm/`.*
