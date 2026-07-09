# PIMS-ADMM Flowsheet UI (Wave3)

Svelte 5 + SvelteFlow (`@xyflow/svelte`) dock + canvas for snap-together units,
wired to the FastAPI graph solver (`POST /api/graph` → real LP/ADMM).

## Run

```bash
# Terminal 1 — API (repo root)
cd ~/projects/pims-admm-llm
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8008

# Terminal 2 — UI
cd ui
npm install
npm run dev   # http://127.0.0.1:5173  (proxies /api + /health → :8008)
```

## UI features

| Control | Purpose |
|---------|---------|
| **Solve graph** | `POST /api/graph` with nodes/edges + options |
| **recovery** | `mono-oracle` (default duals) or `pure-admm` (free λ) |
| **inventory mode** | single-period pass vs inventory balances |
| **run ADMM metrics** | include ADMM residual/λ block in response |
| **stub only** | skip LP (clusters only) |
| **Full plant** | load CDU/FCC/Coker/Reformer/HDT/Blender template |
| **Results panel** | obj, feeds, products, splits, ADMM ‖r‖/‖s‖/λ L∞ |

## Node data

Each unit node stores:

- `unitType`, `label`, `category`, `color`
- `active` (toggle — inactive nodes skipped by solver)
- `submodel`: `lp` \| `tensorflow` (stub for future surrogates)

## Notes

- Connect validation: `POST /api/connect` (local edge if API offline).
- Chemical defaults: FCC/coker naphtha prefer HDT/gasoline, **not** reformer.
- Vite proxy: see `vite.config.js`.
