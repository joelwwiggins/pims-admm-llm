# PIMS-ADMM flowsheet UI (Wave3)

Svelte 5 + SvelteFlow (`@xyflow/svelte`) dock + canvas for snap-together units.

## Run

```bash
# API (repo root)
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8008

# UI
cd ui
npm install
npm run dev
```

Env: `VITE_API_BASE=http://127.0.0.1:8008` (default).

## Features (MVP scaffold)

- Palette/dock drag units (process + supply-chain stubs)
- Custom node: active toggle + submodel `lp|tensorflow`
- `onConnect` → `POST /api/connect` compatibility check
- Graph changes → `POST /api/graph` ADMM rebuild **stub**
