# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear **FCC + Coker + CDU** offline kernels + parity (optional TF) + Excel coeff honesty (FCC/Coker only).  
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
| Postprocess / clamps | Outside any TF graph (`postprocess_fcc_yields` / `postprocess_coker_yields` / `postprocess_cdu_yields`) |
| Excel Case 1 solver | **No** — stays `classic_2block_excel_path` (CBC + package ADMM) |
| ADMM dual recovery | **No** — `dual_recovery_path` on this surface is always `None` |
| Learned / neural weights | **No** — coefficients come from base_delta only |
| Excel Submodel_FCC | MB_* BASE/D_* match the same affine package (always-on check) |
| Excel Submodel_Coker | MB_* BASE/D_* match the same affine package via `excel_coker_matrix_matches_affine` (always-on; pre-postprocess) |
| Excel Submodel_CDU | **Classic TECH+A** yield/recipe export — **not** a PIMS MB_* matrix twin; **no** `excel_cdu_matrix_matches_affine` |
| Coker renorm honesty | Raw affine ≠ full `evaluate()` **even at reference** (renorm always engages) |
| CDU renorm honesty | Often **identity at reference** (liquids already sum≈1); cut/API offsets can engage renorm → raw ≠ evaluate |
| CDU nested drivers | `cut_points_f.*` flattened into x0 (same as pack/evaluate) |
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
    apply_cdu_postprocess,
    tf_linear_fcc,            # requires TensorFlow
    tf_linear_coker,          # requires TensorFlow
    tf_linear_cdu,            # requires TensorFlow
    excel_fcc_matrix_matches_affine,
    excel_coker_matrix_matches_affine,  # always-on, no TF
)
from pims_admm_llm.models.base_delta import (
    build_fcc_base_delta,
    build_coker_base_delta,
    build_cdu_base_delta,
    postprocess_fcc_yields,
    postprocess_coker_yields,
    postprocess_cdu_yields,
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

# CDU (6×8); nested cut_points_f.* in x0; Submodel_CDU is TECH+A not MB_*
coeffs_d = affine_coeffs_from_base_delta(build_cdu_base_delta())
x_d = pack_driver_vector(
    coeffs_d,
    feed={"api": 30.0},
    conditions={"cut_points_f": {"naphtha_ep": 400.0, "distillate_ep": 700.0, "gasoil_ep": 1030.0}},
)
y_raw_d = numpy_affine_forward(coeffs_d, x_d, clamp_products=True)
y_full_d = apply_cdu_postprocess(y_raw_d, products=coeffs_d.products)

# Optional TF path (ImportError if TF missing)
if tf_available():
    y_tf_f = tf_linear_fcc().forward(x_f, clamp_products=True)
    y_tf_c = tf_linear_coker().forward(x_c, clamp_products=True)
    y_tf_d = tf_linear_cdu().forward(x_d, clamp_products=True)

meta = honesty_metadata()
# solver=False, dual_recovery_path=None, on_excel_case1_path=False, units=["FCC","COKER","CDU"]
excel_fcc_matrix_matches_affine()   # Submodel_FCC export ↔ y0/D
excel_coker_matrix_matches_affine()  # Submodel_Coker export ↔ y0/D (E7/E8)
# Do NOT invent excel_cdu_matrix_matches_affine — Submodel_CDU is classic TECH+A.
```

## Honesty boundary (planner-facing)

1. **Excel Submodel_FCC / Submodel_Coker** — base_delta **export** of BASE / D_* (pre-postprocess MB_*).
2. **Excel Submodel_CDU** — classic mono/ADMM **TECH + A** yield/recipe export (Case 1 solve) — **not** Aspen How-To 07 MB_* matrix.
3. **Offline TF / numpy affine** — exact linear copy of base_delta coeffs; optional dep (`tf_linear_fcc` / `tf_linear_coker` / `tf_linear_cdu`).
4. **Case 1 solve** — still CDU+Blender package ADMM; duals labeled with online λ; **not** TF duals.

Coker-specific: postprocess renorm is **outside** the affine export. Summing MB_* BASE
liquids + coke is **not** the same as full `evaluate()` totals (renorm always engages at reference).

CDU-specific: nested `cut_points_f.*` drivers must appear in affine `x0` (same flatten as
pack/evaluate). Liquid renorm + offgas clamp sit outside raw affine; renorm is often
identity at exact reference.

Do not wire this module into `excel_pipeline` or the ADMM coordinator without dual L∞
proof and an explicit form label change.

**Case 1 dual ownership (not this module):** PRIMARY economic shadows = free online λ
(`online_lambda` in `dual_recovery_path`); SECONDARY = recovered blender duals (may
diverge). VERDICT dual gate is online L∞ only. TF surface never owns Case 1 duals
(`dual_recovery_path=None`). See `docs/shadow_prices.md` § Case 1 Excel dual honesty.

## Critics checklist (before claiming “done”)

- [ ] `solver=False` and `dual_recovery_path=None` on honesty_metadata / block
- [ ] Not on Excel Case 1 path (`on_excel_case1_path=False`; form still classic_2block)
- [ ] No learned weights — base_delta coeffs only
- [ ] Postprocess outside TF; Coker L_div (raw ≠ evaluate at ref) tested
- [ ] CDU nested x0 honesty + L0/L1; no fake `excel_cdu_matrix_matches_affine`
- [ ] Isolation: excel_pipeline / models `__init__` do not import tf_linear_blocks / tensorflow
- [ ] FCC E1/E10 and Coker E7/E8 excel match gates still green
- [ ] How_to includes `fcc_three_path` + `coker_three_path` + `cdu_three_path`
