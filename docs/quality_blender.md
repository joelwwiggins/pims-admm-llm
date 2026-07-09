# Quality blender — delta-base / index MVP

**Status:** Wave3b planning-grade gasoline RON + sulfur  
**Module:** `src/pims_admm_llm/models/quality_blender.py`  
**Wired in:** `full_plant.solve_full_plant` gasoline pool  
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

## What this MVP does

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

## Limitations vs full Aspen PIMS delta-base recursion

This MVP is **single-level, fixed-assay, non-recursive**:

1. Component properties are **fixed** in `routing.json` (planning-grade assays).
2. **No multi-tank recursion**: intermediate tanks do not recompute quality from
   inflows and feed that quality into the next pool within the same LP.
3. **No multi-property octane engine** (MON, R+M/2, RVP, aromatics, benzene, DI).
4. **Sulfur is volume-weighted wt%** (planning approximation; mass-basis needs ρ).
5. Diesel/FO still use soft-HDT linear S credits outside this module.
6. Index mode is a **planning index**, not a certified laboratory blending model.

---

## How to run / verify

```bash
source .venv/bin/activate
PYTHONPATH=src python -c "from pims_admm_llm.models.full_plant import solve_full_plant; r=solve_full_plant(); print(r.meta['quality']['model'], r.meta['quality']['deltas'])"
PYTHONPATH=src python -m pytest tests/test_quality_blender.py tests/test_wave3b_issue1.py -q
```
