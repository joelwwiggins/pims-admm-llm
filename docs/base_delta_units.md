# Base-delta unit submodels (incremental cascade)

Planning-grade **BASE + DELTA** LP submodels. Add units only after yields, optimization,
and **mass balances** pass on the enabled set.

## Enabled set (current)

| Units | Status |
|-------|--------|
| CDU + FCC | verified |
| CDU + FCC + **COKER** | verified (optional enable) |
| Reformer / HDT / full multi-unit | **not yet** — wait for balances |

## Pattern per unit

1. Product yield vector (every product has an **exit sink**)
2. BASE + DELTA (feed attrs + **process conditions**)
3. Stream **compositions** (API, S, CCR, RON, PNA, TBP, metals, …)
4. SOS1 **process modes** inside the unit block LP
5. **Auto-wire** feed + product edges when the unit is added to the flowsheet

## Coker

When `COKER` is active:

- Auto-wire: `cdu_resid → COKER` (feed)
- Resid swing remains: `resid_to_coker + resid_to_fo = cdu_resid`
- Products: dry gas, LPG, naphtha, gasoil, coke — each with exit
- Process modes: `rec_low` / `rec_mid` / `rec_high` (recycle + drum T)
- Coker naphtha auto-route prefers **HDT**, not reformer

## Mass-balance checks (hard gate)

`solve_cdu_fcc(...).mass_balance.ok` must be true:

- CDU liquids ≈ crude charge
- FCC feed = cdu_gasoil
- FCC liquids / coke match mode yields × feed
- If coker: resid split, coker feed, liquids + coke yields

## API / run

```bash
PYTHONPATH=src pytest tests/test_base_delta_cdu_fcc.py -q
PYTHONPATH=src python -m demos.run_cdu_fcc_demo
PYTHONPATH=src python -m demos.run_cdu_fcc_demo --coker
```

| Endpoint | Role |
|----------|------|
| `POST /api/auto_wire` | body `{active_units, existing_edges}` → new edges |
| `POST /api/base_delta/solve` | body `{active_units, enable_coker, ...}` → LP + mass_balance |
| `POST /api/connect` | property scorer (+ guesses) |

## Modules

| Path | Role |
|------|------|
| `models/stream_composition.py` | Property library |
| `models/base_delta.py` | CDU/FCC/COKER BASE+DELTA, modes, auto_wire |
| `models/auto_route.py` | Property destination guess |
| `models/cdu_fcc.py` | Cascade LP + mass_balance |

## Next unit rule

Do **not** add reformer (or more) until:

1. `mass_balance.ok` on CDU+FCC+COKER
2. Every product has exit + composition
3. Auto-wire covers feed when unit is dropped on the sheet


## UI auto-wire (SvelteFlow)

When a process unit is **clicked or dropped** from the palette (`CDU`, `FCC`, `COKER`, …):

1. UI POSTs `/api/auto_wire` with `active_units` on the canvas + existing edges
2. Response edges are mapped onto the PFD (`ui/src/lib/autoWire.js`)
3. Missing product sinks become lightweight terminal nodes (`term-fuel_gas`, …)
4. Structural feeds: `cdu_gasoil→FCC`, `cdu_resid→COKER` when those units exist

Toolbar:
- **Auto-wire** — re-run wire for current units
- **Base-δ** — `POST /api/base_delta/solve` (cascade LP + mass_balance)


## Assay → heart/swing CDU

Import real assays as ordered TBP cuts, then fractionate with a **heart + swing** LP:

| Piece | Path |
|-------|------|
| Detailed Cold Lake cuts | `data/assays/cold_lake_blend_clkbl23b.json` |
| Engine | `models/assay_swing.py` |
| Demo | `python -m demos.run_assay_cdu_demo --crude Cold_Lake_Blend` |
| API | `GET /api/assays`, `POST /api/cdu/assay` |

Hearts are fixed to a product; swing cuts on product boundaries are LP-allocated so effective cut points and blended product properties still **mass-balance** (vol + sulfur).
