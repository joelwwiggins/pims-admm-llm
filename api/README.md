# Flowsheet API (wave3)

Minimal FastAPI stubs for the SvelteFlow snap-together UI.

## Run

From the **repo root** (`pims-admm-llm/`):

```bash
# optional: use project venv
source .venv/bin/activate   # if present
pip install -r requirements-api.txt

uvicorn api.main:app --reload --port 8008
```

- Health: http://127.0.0.1:8008/health  
- OpenAPI: http://127.0.0.1:8008/docs  

## Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `{ok, service, wave}` |
| GET | `/api/routing` | — | units, arcs from `data/routing.json`, palette defaults |
| POST | `/api/graph` | `{nodes, edges}` | `{ok, clusters: [{id, node_ids}], admm_status: "stub", message}` |
| POST | `/api/connect` | edge attempt + port attrs | `{allowed, score, reason}` |

CORS is open (`*`) for local Vite dev.

## Notes

- ADMM / clustering are **stubs** — no solver import from `src/pims_admm_llm`.
- Connect validation uses a soft unit-type compatibility table, not full arc chemistry.
