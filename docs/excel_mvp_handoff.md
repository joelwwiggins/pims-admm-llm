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

## Honesty

- Primary ADMM shadows = **free online λ** (not recovered blender duals).
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

- `t_498f8cc3` — **NEXT:** Svelte UI Excel tab (upload + results panel)

## Continue from dashboard

1. Open http://127.0.0.1:9119  
2. Kanban board: `pims-excel-admm-mvp-20260711`  
3. Project workdir: `/home/joel/projects/pims-admm-llm`  
4. Optional next: claim `t_498f8cc3` for Svelte Excel tab  
