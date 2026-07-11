# Flowsheet API (wave3)

FastAPI backend for the SvelteFlow snap-together UI. `POST /api/graph` maps
canvas nodes/edges to a routing overlay and runs full-plant LP + ADMM.

## Run

From the **repo root** (`pims-admm-llm/`):

```bash
source .venv/bin/activate   # if present
pip install -r requirements-api.txt
# plant solver lives under src/
export PYTHONPATH=src
uvicorn api.main:app --reload --port 8008
```

- Health: http://127.0.0.1:8008/health
- OpenAPI: http://127.0.0.1:8008/docs

## Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `{ok, service, wave}` |
| GET | `/api/routing` | — | units, arcs from `data/routing.json`, palette defaults |
| POST | `/api/graph` | `{nodes, edges, recovery_path?, run_admm?, stub_only?}` | solve result or stub |
| POST | `/api/connect` | edge attempt + port attrs | `{allowed, score, reason}` |
| GET | `/api/excel/template` | — | PIMS-shaped `.xlsx` template download |
| POST | `/api/excel/solve` | multipart `file` (+ optional `return_xlsx`) | JSON mono+ADMM summary or results `.xlsx` |
| GET | `/api/excel/results?path=` | basename only | prior results workbook |

### Excel PIMS MVP

```bash
# template
curl -OJ http://127.0.0.1:8008/api/excel/template
# solve (JSON)
curl -F "file=@data/assays/crudes_template.xlsx" http://127.0.0.1:8008/api/excel/solve
# solve (return workbook)
curl -F "file=@data/assays/crudes_template.xlsx" \
  "http://127.0.0.1:8008/api/excel/solve?return_xlsx=true" -o results.xlsx
```

OpenAPI try-it UI: http://127.0.0.1:8008/docs → **POST /api/excel/solve**

### POST `/api/graph` response (real solve)

```json
{
  "ok": true,
  "clusters": [{"id": "cluster_process", "node_ids": ["cdu-1"]}],
  "admm_status": "mono-oracle",
  "objective": 2916.46,
  "unit_feeds": {"cdu_charge": 140.0, "fcc_feed": 40.4, "...": "..."},
  "products": {"gasoline": 35.9, "diesel": 62.8, "fuel_oil": 15.4},
  "routing_splits": {"go_frac_fcc": 1.0, "...": "..."},
  "duals": {"bal_gasoil": "..."},
  "rho": 0.35,
  "residuals": {"primal": 1e-8, "dual": 0.0},
  "dual_recovery_path": "mono-oracle",
  "message": "..."
}
```

- Inactive nodes (`data.active=false`) are skipped in clusters and routing overlay.
- Decision arcs open when graph edges match unit-type pairs (tanks expanded).
- `stub_only=true` or plant `ImportError` → `admm_status: "stub"` (no crash).
- `recovery_path`: `mono-oracle` (default) | `pure-admm`.

CORS is open (`*`) for local Vite dev.

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=src:. python -m pytest tests/test_api_graph.py -q
```
