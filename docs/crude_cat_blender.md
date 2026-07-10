# Crude → CDU → Tanks (7d, bypass) → FCC → Tanks → Blender

**Board:** `pims-crude-cat-blender-20260710`  
**Plan task:** `t_251e3c9b`  
**Module:** `src/pims_admm_llm/models/crude_cat_blender.py`  
**Demo:** `PYTHONPATH=src python -m demos.run_crude_cat_blender_demo`  
**Smoke VERDICT (2026-07-09, WTI 100 kbd):** mono_obj≈1492.81 = admm_obj (gap_rel=0); mb_ok; gasoline≈50.04; fcc_mode=rot_low; h2_kscf≈2.39; fuel_gas_mmbtu≈13.36

## Goal

Planning-grade single case LP that a refinery LP engineer would recognize:

```
assay crude
     │ cut-point CDU (naphtha_ep / distillate_ep / gasoil_ep)
     ▼
SR naphtha ──► tank_naph (7d) ──┐
     │ bypass                   ├─► blender ─► gasoline (RON ≥ 87, S ≤ 0.01)
SR distillate ──────────────────┤      ▲ purchase naphtha / alkylate
     │ (sweet gasoil product)   │
SR gasoil ───► tank_go (7d) ──┐ │
     │ bypass                 ├─► FCC (base-delta / SOS1 modes)
SR resid ───► FO sell          │ │
                               │ └──► fcc_naphtha → blender
                               │      fcc_lco → sour gasoil sell
                               │      fcc_slurry → FO
                               │      dry_gas/lpg → fuel gas BTU sales
                               │      coke → regen credit
H2 purchase ───────────────────┘ (kscf / bbl FCC feed)
```

Deliverables: **mono LP** (plan truth) + **ADMM path with honest gap** + **mass-balance tests**.

## Economics (defaults in `DEFAULT_PRICES`)

| Stream / utility | Unit | Default |
|------------------|------|---------|
| gasoline | $/bbl | 105 |
| sweet_gasoil (SR distillate) | $/bbl | 95 |
| sour_gasoil (FCC LCO) | $/bbl | 78 |
| fuel_oil | $/bbl | 55 |
| crude | $/bbl | 70 |
| buy_naphtha | $/bbl | 92 |
| buy_alkylate | $/bbl | 110 |
| H2 | $/kscf | 8 |
| fuel gas | $/MMBTU | 3.5 |
| tank hold | $/bbl end inv | 0.05 |
| coke credit | $/bbl | 15 |

BTU factors (MMBTU/bbl liq-eq): dry_gas 3.8, LPG 3.5, CDU offgas 3.2.  
H2 use: 0.15 kscf/bbl FCC feed (planning).

## Acceptance criteria (all workers)

1. **Topology** — crude→CDU→tanks(7d+bypass)→FCC→blender; every product has an exit.
2. **Tanks** — capacity ≈ `tank_days × design_charge`; start heel; end ≤ cap; balance `start + in − out = end`; bypass arcs parallel to tanks; no free heel liquidation as pure profit (no-drawdown or hold cost).
3. **Product specs** — gasoline RON min + S max binding-capable; sweet gasoil = SR distillate; sour = FCC LCO; optional naphtha intermediate sell.
4. **Make-buy** — blender may purchase naphtha and alkylate (toggle `allow_purchases`).
5. **H2 buy** — cost proportional to FCC feed; reported in `purchases` / `utilities`.
6. **Fuel gas BTU sales** — light ends × BTU factors → MMBTU revenue.
7. **Base-delta FCC** — SOS1 process modes from `process_modes_fcc` / `build_fcc_base_delta`; chosen mode labeled in VERDICT.
8. **Mono = plan truth** — Optimal; `mass_balance.ok`; quality ok when gasoline > 0.
9. **ADMM parity** — real or labeled coordinated path; `obj_gap_rel < 2%` vs mono; never claim dual recovery if path is mono-equivalent.
10. **Tests** — pytest covers mass balance, specs, tanks/bypass, H2+BTU economics, mono/ADMM gap; demo exits 0 on smoke.

## Baseline status (plan-time inventory)

| Feature | Mono status | Residual |
|---------|-------------|----------|
| Cut-point CDU yields | **done** (`solve_cdu_from_cut_points`) | optional: live cut-point knobs in demo CLI |
| tank_naph + tank_go 7d + bypass | **done** | multi-period carry optional later; unit tests |
| FCC base-delta SOS1 | **done** | wire CDU base-delta modes if desired; document |
| Gasoline RON/S specs | **done** | optional diesel S specs on gasoils |
| Buy naphtha/alkylate | **done** | export purchase duals optional |
| H2 buy + FG BTU | **done** | refine factors from assay if needed |
| Mass balance in result | **done** | **need pytest** |
| ADMM | **stub** (re-solves mono, gap=0) | **real 2-block ADMM or honest label + residual** |
| `__init__` export | missing | export `solve_crude_cat_blender`, `compare_mono_admm` |
| pytest | missing | `tests/test_crude_cat_blender.py` |

## Worker decomposition (file ownership — avoid multi-writer races)

Workers **must not** thrash the whole of `crude_cat_blender.py` at once. Prefer **patch-only** sections named below; if a file is mid-edit by another worker, re-read before overwrite.

### W1 `t_7c2d6ebe` — tanks 7-day inventory + bypass (FCC/blender front)

**Own:** tank balance constraints + tank meta keys in `crude_cat_blender.py` (vars `naph_*`, `go_*`, `tank_*`); optional extract `models/tanks_case.py` if file thrash; **tests** `tests/test_crude_cat_tanks.py`.

**Do:**
- Verify/fix: capacity = `tank_days * design`, heel, bypass + through-tank paths both feasible.
- Prove bypass can carry 100% of stream when tank hold cost high.
- Prove tank path works when bypass forced 0.
- No multi-period rewrite of `multi_period.py` (different plant).

**Done when:** pytest tank tests pass; VERDICT tank dict shows start/end/cap/bypass/to/from.

### W2 `t_480a811b` — product specs + blender purchases

**Own:** RON/SULFUR tables, `gas_ron_min`/`gas_s_max`, buy vars, product slate keys; `tests/test_crude_cat_specs.py`.

**Do:**
- Gasoline RON/S constraints remain linear and binding-capable (tighten RON and show alkylate buy or cut crude).
- Products: gasoline, sweet_gasoil, sour_gasoil, fuel_oil, naphtha_intermediate.
- `allow_purchases=False` feasible path still Optimal with quality ok (or document infeasibility if too tight).

**Done when:** pytest specs + purchase toggle; quality flags in result.

### W3 `t_dff5ed60` — H2 purchases + fuel gas BTU/bbl light-ends sales

**Own:** `BTU_MMBTU_PER_BBL`, `H2_KSCF_PER_BBL_FCC`, objective utility terms, `utilities` dict; `tests/test_crude_cat_utilities.py`.

**Do:**
- H2 cost in obj = price × kscf; reported separately.
- FG revenue = $/MMBTU × (dry_gas·btu + lpg·btu + offgas·btu).
- Optional: surface dual/marginal note for H2 (mono dual of H2 balance if explicit).

**Done when:** pytest economics numbers match hand calc at fixed feeds; demo prints h2 + mmbtu.

### W4 `t_4e03aa1c` — base-delta CDU/FCC/blender wired into case LP

**Own:** FCC mode build path + any CDU mode hooks; docs cross-links in this file § Base-delta; avoid deep rewrites of `base_delta.py` / `cdu_fcc.py` unless bugfix.

**Do:**
- Keep single yield pipeline: assay cut-point CDU → gasoil props → `build_fcc_base_delta` → `process_modes_fcc`.
- Label chosen FCC mode + conditions in `process`.
- Export public API from `models/__init__.py`.
- Do **not** invent a third yield system.

**Done when:** import path clean; mode in VERDICT; no orphan products.

### W5 `t_ab5d76c8` — mono LP + ADMM + obj gap + mass balance tests

**Own:** `solve_crude_cat_blender_admm`, `compare_mono_admm`, `demos/run_crude_cat_blender_demo.py`, `tests/test_crude_cat_blender.py` (integration).

**Do:**
1. Keep mono as plan truth.
2. ADMM options (pick one, label clearly):
   - **A (preferred):** real 2-block ADMM — Block Plant (CDU+tanks+FCC) ‖ Block Blender; link naphtha feeds; ρ dual ascent; report λ, residuals, gap.
   - **B (honest fallback):** path `coordinated-mono-equivalent` with gap=0 and note that single-period fully coupled case does not need ADMM for primal; still report dual placeholders.
3. Mass balance tests: CDU vol sum vs crude; tank balances; FCC feed = tank_go out + bypass; gasoline = blend components.
4. Demo exit 0 when `mb_ok` and `gap_rel < 0.02`.

**Done when:** `pytest tests/test_crude_cat*.py -q` green; demo VERDICT line; raw obj/gap/mb in handoff.

## Verification commands (raw)

```bash
cd /home/joel/projects/pims-admm-llm
source .venv/bin/activate
export PYTHONPATH=src

# mono + ADMM compare
python -m demos.run_crude_cat_blender_demo --crude WTI --charge 100
python -m demos.run_crude_cat_blender_demo --crude Cold_Lake_Blend

# tests
pytest tests/test_crude_cat_blender.py tests/test_crude_cat_tanks.py \
       tests/test_crude_cat_specs.py tests/test_crude_cat_utilities.py -q

# full suite regression (do not break wave5)
pytest -q
```

## Honesty rules

- Mono objective is plan truth for economics.
- Do not claim pure-ADMM dual recovery unless free λ path is implemented and labeled `dual_recovery_path=pure-admm`.
- Cut points remain CDU operational handles (not free economic swings as primary story).
- No third-party PDFs in repo.
- Mass balance gate before adding reformer/HDT to this case.

## Out of scope (this board)

- Full plant `routing.json` rewrite / reformer / coker cascade on this case
- UI auto-wire for this case (optional later)
- Process-pool MIP integration
- Multi-period 7-day rolling horizon (use existing `multi_period.py` patterns only if a follow-up task is created)

## Suggested merge order

1. W1 tanks tests stabilize balances  
2. W2 + W3 (specs/utilities) can parallel if they only touch disjoint symbols  
3. W4 exports + mode labeling  
4. W5 ADMM + integration tests last (depends on stable mono API)

---

**Plan complete.** Workers execute residuals above; user decides when the board is done (no RALPH DONE).
