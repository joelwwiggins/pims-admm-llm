# Wave3 Flowsheet UI

Minimal **Svelte 5 + [@xyflow/svelte](https://svelteflow.dev)** (SvelteFlow) canvas for snap-together refinery / supply-chain routing.

## Features (scaffold)

- Sidebar palette: `CDU`, `FCC`, `COKER`, `REFORMER`, `HDT_NAPH`, `BLENDER`, `TANK`, `SELL`, plus supply-chain stubs `warehouse`, `transport`
- Custom unit nodes: **label**, **active** toggle, **submodel** (`lp` | `tensorflow` stub)
- Drag from palette onto canvas (or click to add)
- `onconnect` → `POST /api/connect` validation (allowed / score / reason)
- **Submit graph** → `POST /api/graph` cluster + ADMM status stubs
- **Load routing** → `GET /api/routing` + `/health` (proxy to FastAPI)

## Setup

```bash
cd ui
npm install
npm run dev
```

Opens Vite on http://127.0.0.1:5173 with proxy:

| Path | Target |
|------|--------|
| `/api/*`, `/health` | `http://127.0.0.1:8008` |

## Backend

From repo root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8008
```

See [`../api/README.md`](../api/README.md).

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Preview production build |
