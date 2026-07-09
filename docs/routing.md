# Plant routing — arc-flow superstructure (Wave3)

*Source of truth: [`data/routing.json`](../data/routing.json).*  
Routing is **economic and chemical**, not a single hard-wired path. The planner LP chooses **arc flows** (decision variables) within capacities, costs, and `default_open` preferences.

Wave2 “hard single-path” language is **deprecated**. Prefer this document and `routing.json` over older fixed-route wording in slides or notes.

---

## Design principles (20-year PIMS view)

1. **Decision arcs** — destinations for gasoil, resid, naphtha, LCO, etc. are optimization variables (`decision: true`), not fixed edges.
2. **Chemical defaults, not hard laws** — preferred chemistry is encoded as open/cheap arcs and closed/expensive non-preferred arcs; the solver may still open alternatives under economics or `force_all_arcs_open`.
3. **Tanks optional (philosophy)** — single-period mode collapses tanks to pure balances; inventory/heels only when multi-period or capacity binds.
4. **Quality blender MVP** — linear **RON + sulfur** pooling on product pools (`component_properties` + `product_quality_specs`). Full delta-base recursion is **not** in this wave.
5. **ADMM consensus** still prices linking streams; demos report **ρ, primal residual, dual residual, max_iter**, and label dual recovery (**mono-oracle** vs pure λ).
6. **LLM is advisory only** — never overrides arc balances, capacity, or quality constraints.

---

## Chemical defaults

| Stream | Preferred destination | Not the default |
|--------|----------------------|-----------------|
| `cdu_naphtha_light` | BLENDER (gasoline pool) | — |
| `cdu_naphtha_heavy` | REFORMER → reformate → gasoline | optional bypass to pool (`sr_heavy_to_gas`) |
| `cdu_gasoil` | **swing** FCC \| diesel pool \| sell | sell closed by default (`go_to_sell`) |
| `cdu_resid` | **swing** coker \| FO | FO costly vs conversion margin |
| `fcc_naphtha` | BLENDER gasoline (soft HDT attribute) | reformer arc closed / high cost |
| `coker_naphtha` | soft HDT → gasoline **or** FO | reformer arc closed / high cost |
| `reformate` | BLENDER gasoline | — |
| `fcc_lco` / `coker_gasoil` | diesel and/or FO swing | — |
| `fcc_slurry` | FO | — |

### Why naphtha chemistry matters

- **FCC naphtha** is already a high-octane cat-gasoline component. Soft HDT for sulfur is the natural path to the **gasoline pool**. Feeding it to the reformer is optional, non-preferred, and costly in the superstructure.
- **Coker naphtha** is olefinic and high-S. Preferred: **HDT → gasoline** or **FO**; **not** reformer by default.
- **Heavy SR naphtha** is the classic reformer feed (octane uplift). **Light SR** goes straight to the pool.
- Planning split of SR naphtha: `naphtha_split.light_frac_of_cdu_naphtha` / `heavy_frac_of_cdu_naphtha` in `routing.json`.

Encoded under `chemical_defaults` in the JSON; the LP does not treat them as hard single edges.

---

## Decision arcs (superstructure summary)

Production into tank balances is typically non-decision (bookkeeping). Downstream choices are decision arcs.

| Family | Arc ids | Role |
|--------|---------|------|
| Gasoil swing | `go_to_fcc`, `go_to_diesel`, `go_to_sell` | FCC feed vs distillate/diesel pool vs intermediate sell |
| Resid swing | `resid_to_coker`, `resid_to_fo` | delayed coker vs fuel oil |
| SR naphtha | `sr_light_to_gas`, `sr_heavy_to_reformer`, `sr_heavy_to_gas` | light → pool; heavy → reformer (preferred) or pool bypass |
| FCC naphtha | `fcc_naph_to_gas`, `fcc_naph_to_reformer` | gasoline default; reformer closed |
| Coker naphtha | `coker_naph_to_hdt_gas`, `coker_naph_to_fo`, `coker_naph_to_reformer` | HDT/gas or FO; reformer closed |
| Products | LCO, slurry, coker GO, reformate arcs | diesel / FO / gasoline pools |

- `default_open: false` means the arc is **available in the superstructure** but closed unless economics or `force_all_arcs_open=True` (e.g. on `solve_full_plant`) opens it for what-if studies.
- Arc `capacity`, `cost_usd_per_bbl`, and optional `sell_price_usd_per_bbl` / `product_pool` / `via` live on each arc object in JSON.

---

## Tank philosophy: single-period vs multi-period

| Mode | Behavior |
|------|----------|
| **Single-period (default)** | `tanks.mode = single_period_pass_optional`. Tanks collapse to **pure balances** — optional staging nodes, not mandatory 1:1 pass-through blocks. `inventory_mode=false` unless capacity/heels force inventory. |
| **Multi-period / capacity-bound** | Inventory, heels, and swing space matter. Tank capacities and time-coupling become first-class (future / when inventory_mode binds). |

**Tanks are not hard fences.** They are optional inventory/balance nodes so the same superstructure can grow into multi-period without rewriting chemistry. In single-period demos, expect “tank” labels mainly as balance points for gasoil, resid, FCC naphtha, and coker naphtha.

---

## Units

| Unit | Role |
|------|------|
| **CDU** | Crude distillation; SR cuts + gasoil/resid into swing network |
| **FCC** | Gasoil conversion (when arc flow chooses FCC) |
| **COKER** | Resid conversion (+ coke credit in objective when live) |
| **REFORMER** | Preferred heavy-SR path → reformate |
| **HYDROTREAT_NAPH** | Soft HDT path for cracked naphthas (`via` attribute on coker→gas arc) |
| **BLENDER** | Product pools + linear quality specs |
| **SELL** | Intermediate make/sell (optional; gasoil sell closed by default) |
| **TANK_*** | Optional inventory / balance nodes (see tank philosophy) |

---

## Flow sketch (defaults & swings — not forced single paths)

```
Crude → CDU
  ├─ light SR naphtha ───────────────────────► BLENDER (gasoline)
  ├─ heavy SR naphtha ─► REFORMER ─ reformate ► BLENDER
  │                    └─ (optional bypass) ─► BLENDER
  ├─ distillate ─────────────────────────────► BLENDER (diesel)
  ├─ gasoil ─► [ swing: FCC | diesel pool | sell* ]
  │              FCC → naphtha ─► gasoline (soft HDT; not reformer default)
  │                    LCO/slurry ─► diesel / FO
  └─ resid ─► [ swing: coker | FO* ]
                 coker → naphtha ─► HDT→gas | FO  (not reformer default)
                         gasoil ─► diesel / FO
```

\* optional / economically gated arcs  
Solver chooses **fractions** on decision arcs within capacities and costs.

---

## Quality blender MVP (RON + S)

| Product | Specs (MVP) | Pooling model |
|---------|-------------|----------------|
| **Gasoline** | `min_ron`, `max_sulfur_wt` | Linear volume blend of component RON and sulfur |
| **Diesel** | `max_sulfur_wt` | Linear S; soft HDT credit factors where modeled |
| **Fuel oil** | `max_sulfur_wt` | Linear S |

Component properties (`ron`, `sulfur_wt`) and product specs live in `routing.json`:

- `component_properties` — planning-grade RON/S by stream (FCC naphtha post soft-HDT assumption, raw vs HDT coker naphtha, reformate, etc.)
- `product_quality_specs` — e.g. gasoline min RON 87, max S 0.01 wt

**Out of scope this wave:** full delta-base / recursive property models, multi-property octane engines, nonlinear blending indices beyond linear RON+S MVP.

---

## Linking streams (ADMM)

Consensus / dual recovery operate on intermediate balances, including:

`cdu_naphtha`, `cdu_naphtha_light`, `cdu_naphtha_heavy`, `cdu_distillate`, `cdu_gasoil`, `cdu_resid`,  
`fcc_naphtha`, `fcc_lco`, `fcc_slurry`, `coker_naphtha`, `coker_gasoil`, `reformate`

### Demo reporting (Wave3)

ADMM and dual-recovery demos should report explicitly:

- **ρ** (penalty / step parameter)
- **primal residual**
- **dual residual**
- **max_iter** (and iterations used)
- dual recovery **label**: mono-oracle vs pure λ path

LLM notes remain advisory commentary only.

---

## Compatibility `routes` array

`routing.json` still carries a `routes` list for older readers. Treat it as a **compat projection** of the arc superstructure, not as a hard single-path specification. New code and docs should use **`arcs`** + **`chemical_defaults`** + **`tanks`**.

---

## Related

- [story.md](story.md) — stakeholder narrative (material flow updated for Wave3)
- [architecture.md](architecture.md) — layers, blocks, ADMM, repo map
- `demos/run_full_plant_demo.py` — mono + dual recovery + arc splits / residuals
- README — Wave3 status bullet

---

*Wave3 · arc-flow superstructure · source of truth `data/routing.json`*
