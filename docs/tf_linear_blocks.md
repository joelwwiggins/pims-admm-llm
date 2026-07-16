# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear **FCC + Coker** offline kernels + parity (optional TF) + Excel coeff honesty.  
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
| Postprocess / clamps | Outside any TF graph (`postprocess_fcc_yields` / `postprocess_coker_yields`) |
| Excel Case 1 solver | **No** — stays `classic_2block_excel_path` (CBC + package ADMM) |
| ADMM dual recovery | **No** — `dual_recovery_path` on this surface is always `None` |
| Learned / neural weights | **No** — coefficients come from base_delta only |
| Excel Submodel_FCC | MB_* BASE/D_* match the same affine package (always-on check) |
| Excel Submodel_Coker | MB_* BASE/D_* match the same affine package via `excel_coker_matrix_matches_affine` (always-on; pre-postprocess) |
| Coker renorm honesty | Raw affine ≠ full `evaluate()` **even at reference** (renorm always engages) |
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
    apply_coker_postprocess,
    tf_linear_fcc,            # requires TensorFlow
    tf_linear_coker,          # requires TensorFlow
    excel_fcc_matrix_matches_affine,
    excel_coker_matrix_matches_affine,  # always-on, no TF
)
from pims_admm_llm.models.base_delta import (
    build_fcc_base_delta,
    build_coker_base_delta,
    postprocess_fcc_yields,
    postprocess_coker_yields,
)

# FCC
coeffs_f = affine_coeffs_from_base_delta(build_fcc_base_delta())
x_f = pack_driver_vector(coeffs_f, feed={"api": 24.0}, conditions={"riser_outlet_temp_f": 990.0})
y_raw_f = numpy_affine_forward(coeffs_f, x_f, clamp_products=True)
y_full_f = apply_fcc_postprocess(y_raw_f, products=coeffs_f.products)

# Coker (5×6); renorm always engages
coeffs_c = affine_coeffs_from_base_delta(build_coker_base_delta())
x_c = pack_driver_vector(coeffs_c, feed={"api": 12.0}, conditions={"recycle_ratio": 0.15})
y_raw_c = numpy_affine_forward(coeffs_c, x_c, clamp_products=True)
y_full_c = apply_coker_postprocess(y_raw_c, products=coeffs_c.products)

# Optional TF path (ImportError if TF missing)
if tf_available():
    y_tf_f = tf_linear_fcc().forward(x_f, clamp_products=True)
    y_tf_c = tf_linear_coker().forward(x_c, clamp_products=True)

meta = honesty_metadata()
# solver=False, dual_recovery_path=None, on_excel_case1_path=False, units=["FCC","COKER"]
excel_fcc_matrix_matches_affine()   # Submodel_FCC export ↔ y0/D
excel_coker_matrix_matches_affine()  # Submodel_Coker export ↔ y0/D (E7/E8)
```

## Honesty boundary (planner-facing)

1. **Excel Submodel_FCC / Submodel_Coker** — base_delta **export** of BASE / D_* (pre-postprocess).
2. **Offline TF / numpy affine** — exact linear copy of the same coeffs; optional dep (`tf_linear_fcc` / `tf_linear_coker`).
3. **Case 1 solve** — still CDU+Blender package ADMM; duals labeled with online λ; **not** TF duals.

Coker-specific: postprocess renorm is **outside** the affine export. Summing MB_* BASE
liquids + coke is **not** the same as full `evaluate()` totals (renorm always engages at reference).

Do not wire this module into `excel_pipeline` or the ADMM coordinator without dual L∞
proof and an explicit form label change.

## Critics checklist (before claiming “done”)

- [ ] `solver=False` and `dual_recovery_path=None` on honesty_metadata / block
- [ ] Not on Excel Case 1 path (`on_excel_case1_path=False`; form still classic_2block)
- [ ] No learned weights — base_delta coeffs only
- [ ] Postprocess outside TF; Coker L_div (raw ≠ evaluate at ref) tested
- [ ] Isolation: excel_pipeline / models `__init__` do not import tf_linear_blocks / tensorflow
- [ ] FCC E1/E10 and Coker E7/E8 excel match gates still green
