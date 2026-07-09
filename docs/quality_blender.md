# Quality blender — delta-base / index + recursive multi-level v1

**Status:** Wave3b single-level MVP + Wave5 recursive multi-level v1 (optional)  
**Modules:**
- `src/pims_admm_llm/models/quality_blender.py` — fixed-assay delta-base / index
- `src/pims_admm_llm/models/quality_recursive.py` — tank quality → pool deltas  
**Wired in:** `full_plant.solve_full_plant` gasoline pool (fixed-assay default)  
**Optional path:** `resolve_gasoline_components(..., recursive_quality=True)` or
`solve_full_plant_with_recursive_quality(...)`  
**Config:** `data/routing.json` → `quality_model.gasoline` + `product_quality_specs.gasoline`

---

## Why delta-base

PIMS-style matrices rarely store absolute quality coefficients on every blend
column forever. They store a **base** quality and **component deltas** so that:

1. A base recipe / base stream can be refreshed without rewriting every column.
2. Specs appear as RHS adjustments against the base:  
   \(\sum_i \delta_i x_i \;\ge\; (Q_{\min} - Q_{\text{base}})\,V\)
3. Intermediate assays can later plug in as updated \(\delta\) vectors (full
   recursion) without changing row structure.

When \(\delta_i = Q_i - Q_{\text{base}}\), delta-base is **algebraically identical**
to linear volume pooling \(\sum Q_i x_i \ge Q_{\min} V\). The value of the form is
**matrix structure and extensibility**, not a different number on day one.

---

## What the single-level MVP does

| Property | Model | Constraint name |
|----------|--------|-----------------|
| Gasoline **RON** | `delta_base` (default) or `index` | `qual_gas_min_ron` |
| Gasoline **S** wt% | delta-base (always) | `qual_gas_max_s` |
| Diesel S | soft-HDT linear credits (unchanged) | `qual_diesel_max_s` |

### Default base

- `base_stream = reformate` (RON 100, S ≈ 0.0005 from `component_properties`)
- Components: reformate, light SR naphtha, heavy SR bypass, FCC naphtha, HDT coker naphtha

### Index mode (optional)

Optional Ethyl-style rational RON blending index:

```
BI(r) = (r − floor) / (ceiling − r)     # floor=0, ceiling=120 default
```

LP constraint averages BI by volume and inverts for reporting. Identity
`BI(r)=r` recovers linear / delta-base. Enable via
`quality_model.gasoline.model = "index"`.

### Result metadata

`FullPlantResult.meta["quality"]` includes model label, base stream, base RON/S,
spec limits, constraint names, and per-component **deltas**.

---

## Wave5 recursive multi-level quality v1

**Goal:** one intermediate planning step — recompute **tank quality** from
inflow volume-weighted properties (optional heel / multi-source mix), optional
soft-HDT transform, then rebuild **product-pool deltas** from updated component Q.

### Algebra

```
# Tank (heel optional)
V_tot = V_heel + Σ_j V_j
Q_tank = (V_heel·Q_heel + Σ_j V_j·Q_j) / V_tot

# Pool deltas (same row form as MVP)
δ_i = Q_i^{level} − Q_base
Σ δ_i x_i  ≷  (Q_spec − Q_base) · V
```

Identity case: single pure unit inflow, zero heel → `Q_tank ≡ Q_assay`.

### Default quality graph

```
leaf: reformate, cdu light/heavy
tank: tank_fcc_naph ← fcc_naphtha (+ heel_fcc_naph)
tank: tank_coker_naph ← coker_naphtha (+ heel)
transform: coker_naphtha_hdt ← soft HDT absolute targets
→ gasoline pool components → deltas vs reformate base
```

### API (`quality_recursive.py`)

| Function | Role |
|----------|------|
| `volume_weighted_quality` | Core blend algebra |
| `compute_tank_quality` | Inflows + optional heel/transform |
| `build_default_gasoline_quality_graph` | Plant-aligned single-step graph |
| `evaluate_recursive_quality` | Closed-form multi-level + pool deltas |
| `evaluate_from_plant_result` | Same from a `FullPlantResult` |
| `successive_recursive_refine` | Patch `routing.component_properties` |
| `resolve_gasoline_components` | Flag path (default fixed-assay) |
| `solve_full_plant_with_recursive_quality` | Optional successive-LP wrapper |
| `patch_routing_component_properties` | Non-mutating routing patch |

Default `solve_full_plant()` is **unchanged** (fixed-assay).

```python
from pims_admm_llm.models.quality_recursive import (
    TankInflow,
    evaluate_recursive_quality,
    resolve_gasoline_components,
    solve_full_plant_with_recursive_quality,
)

rec = evaluate_recursive_quality(
    routing,
    volumes={"fcc_naphtha": 40, "reformate": 30, "heel_fcc_naph": 10},
    multi_source_inflows={
        "tank_fcc_naph": [TankInflow("fcc_naphtha", 30), TankInflow("cdu_naphtha_light", 20)],
    },
)
print(rec.component_qualities["fcc_naphtha"].ron, rec.deltas)

comps, meta = resolve_gasoline_components(routing, recursive_quality=True)
res = solve_full_plant_with_recursive_quality(max_refine_steps=1)
```

```bash
PYTHONPATH=src python -m pytest tests/test_quality_recursive.py -q
```

**Honest scope:** planning-grade single intermediate step + volume-linear props
(+ absolute soft-HDT targets). Not full multi-level PIMS SLP, not yield response
to feed quality, not bilinear Q·x inside one CBC solve.

---

## Limitations vs full Aspen PIMS delta-base recursion

1. Default path is **single-level, fixed-assay** unless recursive helper runs.
2. Recursive v1 is **closed-form / open-loop volumes → tank Q → pool deltas**
   with optional successive re-solve — not multi-tank SLP in one CBC solve.
3. **No multi-property octane engine** (MON, R+M/2, RVP, aromatics, benzene, DI).
4. **Sulfur is volume-weighted wt%** (planning approximation; mass-basis needs ρ).
5. Diesel/FO still use soft-HDT linear S credits outside this module.
6. Index mode is a **planning index**, not a certified laboratory blending model.

---

## How to run / verify

```bash
source .venv/bin/activate
PYTHONPATH=src python -c "from pims_admm_llm.models.full_plant import solve_full_plant; r=solve_full_plant(); print(r.meta['quality']['model'], list(r.meta['quality']['deltas'])[:3])"
PYTHONPATH=src python -c "from pims_admm_llm.models.quality_recursive import resolve_gasoline_components, TankInflow; from pims_admm_llm.models.assay_loader import load_routing; c,m=resolve_gasoline_components(load_routing(), recursive_quality=True, multi_source_inflows={'tank_fcc_naph':[('fcc_naphtha',30),('cdu_naphtha_light',20)]}, volumes={'fcc_naphtha':30,'cdu_naphtha_light':20,'reformate':1}); print(m['recursive_quality'], c['fcc_naphtha'].ron)"
PYTHONPATH=src python -m pytest tests/test_quality_blender.py tests/test_quality_recursive.py tests/test_wave3b_issue1.py -q
```
