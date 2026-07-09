# Base-delta unit submodels (CDU → FCC first)

Planning-grade **BASE + DELTA** LP submodels for conversion units, with:

1. **Every product yield has an exit stream** (default sink + alternates)
2. **Process conditions inside the unit block** (SOS1 severity / cut modes)
3. **Stream compositions** (API, S, CCR, RON, PNA, TBP markers, metals, …)
4. **Property-based auto-route** when the flowsheet has no drawn edge

Paper PDFs are **not** stored in the repo. This is the operational pattern used in industry PIMS-style unit vectors.

## Modules

| Path | Role |
|------|------|
| `models/stream_composition.py` | Property bag + library for CDU/FCC products |
| `models/base_delta.py` | BASE/DELTA builders, process modes, exit catalogs |
| `models/auto_route.py` | Guess destination from composition |
| `models/cdu_fcc.py` | Focused plant: crude → CDU → FCC |

## Unit contract

```
feed attrs + process conditions
        ↓
 BASE + Σ DELTA_j · (x_j − x_j0)
        ↓
 yield vector (every key has ProductExit)
        ↓
 compositions + auto_route if no edge
```

### CDU products / exits

| Stream | Default exit |
|--------|----------------|
| `cdu_offgas` | FUEL_GAS |
| `cdu_naphtha_light` | GASOLINE |
| `cdu_naphtha_heavy` | REFORMER |
| `cdu_distillate` | DIESEL |
| `cdu_gasoil` | FCC |
| `cdu_resid` | FO |

Process modes: `cuts_light` / `cuts_mid` / `cuts_heavy` (flash zone + cut points).

### FCC products / exits

| Stream | Default exit |
|--------|----------------|
| `fcc_dry_gas` | FUEL_GAS |
| `fcc_lpg` | LPG |
| `fcc_naphtha` | GASOLINE (not reformer) |
| `fcc_lco` | DIESEL |
| `fcc_slurry` | FO |
| `fcc_coke` | REGEN_HEAT |

Process modes: `rot_low` / `rot_mid` / `rot_high` (ROT + C/O couple).

## Run

```bash
cd ~/projects/pims-admm-llm && source .venv/bin/activate
PYTHONPATH=src pytest tests/test_base_delta_cdu_fcc.py -q
PYTHONPATH=src python -m demos.run_cdu_fcc_demo
```

## Auto-route rules (high level)

- Family affinity (light ends → fuel/LPG, gasoil → FCC, resid → coker/FO, solids → regen/coke sales)
- Property boosts: high RON + olefins **block** reformer default; high CCR/metals **penalize** FCC feed; SR heavy low RON **prefers** reformer
- `POST /api/connect` uses the same scorer when `stream` is set; response may include `guesses` + `best`

## Scope discipline

Do **not** expand to coker/reformer/full multi-unit until:

1. Stream composition library is trusted for CDU/FCC
2. Base-delta + process modes solve cleanly on CDU→FCC
3. Auto-exits cover every product without orphan production

Then copy the same `BaseDeltaModel` pattern per unit.
