# PIMS-ADMM HYSYS-style PFD

Professional process-flow diagram UI (Svelte 5 + `@xyflow/svelte`) for the
wave3 arc-flow plant. Designed to feel like Aspen HYSYS: multi-port unit
blocks, piping streams, right-side property inspector.

## Run

```bash
# API (repo root)
cd ~/projects/pims-admm-llm
source .venv/bin/activate
uvicorn api.main:app --reload --port 8008

# UI
cd ui
npm install
npm run dev    # http://127.0.0.1:5173
```

## Layout

| Zone | Role |
|------|------|
| **Top toolbar** | Run / Validate / Reset PFD / recovery path / **Excel** |
| **Left dock** | Tabs: **Palette** (units + graph results) · **Excel** (template / upload / mono+ADMM) |
| **Canvas** | HYSYS-style PFD (nodes + stream edges) |
| **Right inspector** | Unit or stream properties (click to select) |

## Excel PIMS tab

Uses the MVP API (Vite proxies `/api` → `:8008`):

1. **Template** → `GET /api/excel/template`
2. **Upload** a PIMS-shaped `.xlsx` (Crudes / Products / Capacities)
3. **Solve Excel** → `POST /api/excel/solve` (multipart `file`)
4. Results panel: mono/ADMM obj, gap, dual L∞ (online λ), rates, shadows
5. **Download results .xlsx** → `GET /api/excel/results?path=<basename>`

Honesty: primary ADMM shadows = free online λ, not recovered blender duals.

## Custom components

- `lib/nodes/PfdUnitNode.svelte` — multi-handle PFD block, yield preview
- `lib/edges/StreamEdge.svelte` — thick piping + label + flow kbd
- `lib/InspectorPanel.svelte` — General / Streams / Yields / Composition
- `lib/data/plantTemplate.js` — full-plant mock (CDU→FCC/Coker/Reformer/HDT→Blender)

## Interaction

1. **Click unit** → inspector: tag, active, submodel, yields (editable)
2. **Click stream** → inspector: flow, T, P, VF, S, RON, composition table
3. **Drag palette** → drop new units
4. **Connect handles** → `POST /api/connect` validation
5. **Run** → `POST /api/graph` real LP/ADMM; Results panel shows obj/feeds/duals

## Chemical defaults (template)

- Gasoil → FCC; Resid → Coker
- Heavy SR naph → Reformer
- FCC / coker naph → **HDT** (not reformer)
- Reformate + HDT naph + LCO + coker GO → Blender
