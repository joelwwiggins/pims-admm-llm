# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear FCC kernel + parity (optional TF) + Excel coeff honesty.  
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
| Formula | Exact affine copy of base_delta: `y_raw = y0 + D @ (x − x0)` |
| Postprocess / clamps | Outside any TF graph (`postprocess_fcc_yields` / numpy) |
| Excel Case 1 solver | **No** — stays `classic_2block_excel_path` (CBC + package ADMM) |
| ADMM dual recovery | **No** — `dual_recovery_path` on this surface is always `None` |
| Learned / neural weights | **No** — coefficients come from base_delta only |
| Excel Submodel_FCC | MB_* BASE/D_* match the same affine package (always-on check) |
| Excel Submodel_Coker | MB_* BASE/D_* match the same affine package via `excel_coker_matrix_matches_affine` (always-on; pre-postprocess) |
| EMRPS / pure research floor | Validation-only elsewhere; not this module |

## API

```python
from pims_admm_llm.models.tf_linear_blocks import (
    tf_available,
    honesty_metadata,
    affine_coeffs_from_base_delta,
    numpy_affine_forward,
    pack_driver_vector,
    apply_fcc_postprocess,
    tf_linear_fcc,          # requires TensorFlow
    excel_fcc_matrix_matches_affine,
    excel_coker_matrix_matches_affine,  # always-on, no TF
)
from pims_admm_llm.models.base_delta import build_fcc_base_delta

coeffs = affine_coeffs_from_base_delta(build_fcc_base_delta())
x = pack_driver_vector(coeffs, feed={"api": 24.0}, conditions={"riser_outlet_temp_f": 990.0})
y_raw = numpy_affine_forward(coeffs, x, clamp_products=True)
y_full = apply_fcc_postprocess(y_raw, products=coeffs.products)

# Optional TF path (ImportError if TF missing)
if tf_available():
    block = tf_linear_fcc()
    y_tf = block.forward(x, clamp_products=True)

honesty_metadata()  # solver=False, dual_recovery_path=None, on_excel_case1_path=False
excel_fcc_matrix_matches_affine()   # Submodel_FCC export ↔ y0/D
excel_coker_matrix_matches_affine()  # Submodel_Coker export ↔ y0/D (E7/E8)
```

## Honesty boundary (planner-facing)

1. **Excel Submodel_FCC / Submodel_Coker** — base_delta **export** of BASE / D_* (pre-postprocess).
2. **Offline TF / numpy affine** — exact linear copy of the same coeffs; optional dep (FCC factory today; Coker excel gate locks the export).
3. **Case 1 solve** — still CDU+Blender package ADMM; duals labeled with online λ; **not** TF duals.

Do not wire this module into `excel_pipeline` or the ADMM coordinator without dual L∞
proof and an explicit form label change.
