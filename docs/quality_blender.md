# Quality blender — linear vs delta-base (Wave3b)

## Linear pooling (implemented)

For gasoline RON / S and diesel S:

```
sum_i q_i * x_i  >=  min_q * product   (or <= for max S)
```

Component qualities from `data/routing.json` `component_properties`.

## Delta-base form (planning MVP)

Algebraically for a chosen base component `b` (default: reformate):

```
q = base_q + sum_i (q_i - base_q) * (x_i / sum x)
```

Multiplying by total volume recovers the same linear constraint. PIMS full
delta-base recursion (index blending, nonlinear octane, multi-spec cascades)
is **not** implemented.

### API / solver flag

`solve_full_plant` reports quality duals under either interpretation; the LP
constraints are linear pooling. Future: `quality_mode="delta_base"` will only
change reporting (base + deltas table), not feasible region, until nonlinear
indexes are added.

## Status

- Linear RON + S: **done**
- Delta-base reporting table: optional next
- Full PIMS delta-base recursion: open
