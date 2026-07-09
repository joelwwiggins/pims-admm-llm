# Shadow prices & economic interpretation (PIMS-style)

This document is the Worker 7 deliverable: how duals from the monolithic LP
(and ADMM λ on linking streams) map to **make-buy-sell** language used by
refinery planners.

## What a shadow price is

For a maximize-margin LP, the dual (shadow price) on a constraint is the
**instantaneous rate of change of optimal margin** if you relax that
constraint by one unit — holding the current basis fixed.

| Constraint family | Unit of relaxation | Planner language |
|-------------------|--------------------|------------------|
| Intermediate balance / yield | +1 bbl stream | Stream marginal value (MV) |
| CDU capacity | +1 kbd charge | Value of debottleneck / swing capacity |
| Tank hold cap | +1 kbd ullage | Value of tank farm space |
| Crude supply | +1 bbl availability | Crude flexibility / purchase value |
| Product demand | +1 kbd outlet | Value of finding more sales |

This is the same economic object Aspen PIMS reports as marginal values / DJs
(within a basis neighborhood).

## Dual extraction in this repo

Named constraints in `build_monolithic_lp`:

- `cdu_capacity`
- `crude_supply_<crude>`
- `product_demand_<product>`
- `tank_<intermediate>`  (`prod - use ≤ tank_cap`)
- `tank_farm_total` (optional)
- `yield_*`, `balance_*`, `blend_use_*`

ADMM: `ADMMResult.shadow_prices` = λ on intermediate consensus variables.
At good convergence these should track monolithic stream MVs.

## Make-buy-sell rules of thumb

1. **Positive intermediate MV** → buy/make more of that stream (spot purchase,
   yield shift, cut-point move) if landed cost < MV.
2. **Positive CDU dual** → expand/swing capacity if unit cost of capacity < dual.
3. **Positive crude supply dual** → that crude is binding; chase incremental barrels.
4. **Zero dual** → non-binding; money spent expanding that limit is wasted *at this plan*.
5. **ADMM λ** → internal transfer price between CDU and Blender agents.

## Scale / linearity

Shadow prices are **piecewise constant**. Locally:

```
Δobjective ≈ shadow_price × ΔRHS
```

`run_linearity_checks` re-solves after small RHS moves (CDU +0.5/+1, binding
crude +1, free intermediate gift) and flags PASS/FAIL. Large moves can change
the basis — PIMS has the same limitation; always re-optimize for big cases.

## How to run

```bash
cd ~/projects/pims-admm-llm
source .venv/bin/activate
PYTHONPATH=src python demos/shadow_price_report.py
PYTHONPATH=src python demos/shadow_price_report.py --with-admm --json /tmp/mv.json
```

Programmatic:

```python
from pims_admm_llm.reporting import build_shadow_price_report, format_report_text
report = build_shadow_price_report()
print(format_report_text(report))
```

## Relation to ADMM vs Dantzig–Wolfe

Both produce duals on linking constraints with the same economic meaning as a
full PIMS solve **at convergence**. ADMM updates λ each iteration via the
augmented residual; DW takes duals of the restricted master. This project uses
ADMM as the multi-agent coordinator; Worker 7 reports and validates the prices.
