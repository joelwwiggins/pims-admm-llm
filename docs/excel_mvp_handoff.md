# Excel PIMS → ADMM MVP handoff (2026-07-11)

**Status:** Delivered and merged to `main`  
**PR:** https://github.com/joelwwiggins/pims-admm-llm/pull/5  
**Merge:** `ec66332`  
**Kanban board:** `pims-excel-admm-mvp-20260711`  
**Kanban backup:** `~/backups/pims-excel-mvp-20260711-*`

## What works

| Path | Command / endpoint |
|------|--------------------|
| CLI | `export PYTHONPATH=src && python -m demos.run_excel_pipeline_demo` |
| Template | `data/assays/crudes_template.xlsx` (Crudes / Products / Capacities / Intermediates) |
| Pipeline | `src/pims_admm_llm/models/excel_pipeline.py` |
| API | `uvicorn api.main:app --port 8008` |
| Template GET | `GET /api/excel/template` |
| Solve POST | `POST /api/excel/solve` (multipart `file`) |
| Results GET | `GET /api/excel/results?path=<basename.xlsx>` |
| Web try-it | http://127.0.0.1:8008/docs |
| Dashboard | http://127.0.0.1:9119 (Hermes) |

## Live VERDICT (template, post-polish)

```
Mono obj ≈ 3610.57
ADMM obj ≈ 3609.74  (ρ=8, dual_step=0.5, max_iter=120)
Gap ≈ 0.023%
Dual L∞ (online λ vs mono) ≈ 2.66
VERDICT: PASS
```

## Lean results workbook (authoritative sheet map)

Target **≤15 sheets**, one planner tab per unit. Source of truth: `tests/test_excel_pipeline.py` (`REQUIRED` / `BANNED_*`) and `write_results_excel`.

| Sheet | Role |
|-------|------|
| How_to_read | Planner guide (three-path + dual PRIMARY/SECONDARY + `tf_offline_units` / `tf_offline_priced` / `tf_offline_timing` / `tf_offline_admm_residual` / `tf_offline_admm_block_subproblem`) |
| Submodel_Index | Unit map + **OFFLINE_TF** readiness row (FCC+COKER+CDU kernels + **priced residual readiness** + **block-solve timing readiness** + **ADMM residual readiness** + **ADMM block subproblem readiness (synthetic λ,z,ρ; raw affine under box)** — **not** classic Case 1 solve; prices ≠ duals; timings ≠ Case 1 wall; synthetic residual/subproblem ≠ duals / ≠ pure-ADMM dual recovery) |
| Calc_Yields / Calc_Blend | Editable coefficients |
| Submodel_CDU / Submodel_Blender | Classic **live** 2-block solve tables |
| Submodel_FCC / Submodel_Coker | PIMS BASE/DELTA **export** matrices (not live ADMM blocks on this path) |
| Submodel_Linking | prod−use balances → duals |
| Calc_Check | Identity / feasibility checks + **honesty audits** (`form_classic_2block`, `dual_gate_online_only`, `offline_tf_not_on_case1`, `offline_tf_priced_not_duals`, `offline_tf_timing_not_case1`, `offline_tf_admm_residual_not_duals`, `offline_tf_admm_block_subproblem_not_duals`) |
| Summary / Rates / Shadows | Verdict, rates, mono vs online-λ vs recovered duals; Summary honesty strip includes form + dual_gate + offline_tf units/priced/timing/ADMM residual/block subproblem readiness notes |

`model.form == classic_2block_excel_path` — Excel solve remains CDU+Blender only; FCC/Coker tabs are teaching/export from `base_delta`. Offline TF kernels (FCC+COKER+CDU) plus priced residual + block-solve timing + ADMM residual + ADMM block subproblem harnesses are readiness-only; not on Case 1 solve; `dual_recovery_path=None` on TF surface; not wire shipped.

After dual-honesty packaging, TF offline multi-unit registry, priced residual, timing readiness, ADMM residual harness, and ADMM block subproblem maximizer: Index / Summary / Calc_Check **glance-lock** offline TF readiness package (units + priced + timing + ADMM residual + **block subproblem** + ladder through gate-criteria + **isolation-rewrite design**) + dual PRIMARY gate + classic form via static pure formatters (see `meta.planner_honesty` and `format_planner_honesty_package`). Isolation design packaging: design_present; rewrite_shipped=false; isolation checklist open; dual-ban — packaging ≠ rewrite shipped ≠ wire ≠ VERDICT. Still isolation-safe: no TF import on Excel path; still not wire.

## Honesty

- Primary ADMM shadows = **free online λ** (not recovered blender duals). Path label: `…+online_lambda_shadows` on Summary/`dual_recovery_path`.
- Default FO **$68** may idle coker on light WTI+Cold Lake (resid→FO). Multi-unit tests use FO **$50**.
- Pure-ADMM default **ρ=2.0** (ρ=1.2 collapsed FCC feed).
- Classic 2-block CDU/blender Excel path only (not full-plant Excel yet).

## Tests

`PYTHONPATH=src python -m pytest tests/ -q` → **165 passed**, 1 skipped (as of merge).

## Commits on main

1. `1acae7f` — Excel pipeline + CLI  
2. `030a0d0` — API upload endpoints  
3. `739ebf4` — Plant suite green  
4. `c86011f` — ρ / online-λ polish  
5. `ec66332` — merge PR #5  

## Open residual (board)

- `t_498f8cc3` — Svelte UI Excel tab (upload + results panel) — **implemented** (left dock Excel tab)

## Continue from dashboard

1. Open http://127.0.0.1:9119  
2. Kanban board: `pims-excel-admm-mvp-20260711`  
3. Project workdir: `/home/joel/projects/pims-admm-llm`  
4. UI: `cd ui && npm run dev` → http://127.0.0.1:5173 → left dock **Excel** tab  
5. API must be up: `uvicorn api.main:app --port 8008` (from repo root, `PYTHONPATH=src`)  
