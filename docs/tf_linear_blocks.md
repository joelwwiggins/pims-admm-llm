# TensorFlow linear blocks (optional, offline)

**Status:** scaffold only (optional dependency + import isolation).  
**Not** on the Excel Case 1 / PuLP ADMM solve path.

## Install

Core install never requires TensorFlow:

```bash
pip install -e .
```

Optional TF extra (when a wheel is available for your platform):

```bash
pip install -e ".[tf]"
```

On Jetson / environments without a TF wheel, leave TF uninstalled. Case 1 Excel
smoke (`python -m demos.run_excel_pipeline_demo`) must stay green.

## Contract

| Claim | Reality |
|-------|---------|
| Formula (future blocks) | Exact affine copy of base_delta: `y = y0 + D @ (x − x0)` |
| Postprocess / clamps | Outside any TF graph (numpy/Python) |
| Excel Case 1 solver | **No** — stays `classic_2block_excel_path` (CBC + package ADMM) |
| ADMM dual recovery | **No** — `dual_recovery_path` on this surface is always `None` |
| Learned / neural weights | **No** — coefficients come from base_delta only |
| EMRPS / pure research floor | Validation-only elsewhere; not this module |

## API (current cycle)

```python
from pims_admm_llm.models.tf_linear_blocks import tf_available, honesty_metadata

tf_available()       # False when TF not installed
honesty_metadata()   # solver=False, dual_recovery_path=None, on_excel_case1_path=False
```

Forward TF matvec blocks land in a later cycle behind parity tests. Do not wire
this module into `excel_pipeline` or the ADMM coordinator without dual L∞ proof
and an explicit form label change.
