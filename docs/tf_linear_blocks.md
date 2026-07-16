# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear **FCC + Coker + CDU** offline kernels + multi-unit registry + wiring-readiness parity harness + **offline priced residual / local box direction harness** + **cached multi-unit block-solve timing / readiness harness** + Excel coeff honesty (FCC/Coker only).  
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
| Multi-unit registry | `offline_unit_registry` / `offline_units_status` / `multi_unit_parity_report` — readiness only |
| Priced residual (goal 5) | `multi_unit_priced_residual_report` / `default_offline_prices` / `local_box_direction` — economics readiness; prices **not** duals |
| Block-solve timing (goal 5) | `multi_unit_block_solve_timing_report` / `offline_block_solve_readiness_report` / `cached_offline_unit_coeffs` — microsecond-class readiness; **not** Case 1 wall time; **not** duals |
| EMRPS / pure research floor | Validation-only elsewhere; not this module |

## Multi-unit offline registry API

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_unit_registry,
    offline_unit_coeffs,
    build_offline_unit,       # requires TensorFlow
    offline_units_status,
    multi_unit_parity_report,
    UNITS,                    # ("FCC", "COKER", "CDU")
)

reg = offline_unit_registry()
# ordered OfflineUnitDescriptor: unit, builder/factory/postprocess names,
# excel_match_name (None for CDU), renorm_note, shape

status = offline_units_status()
# solver=False, dual_recovery_path=None, on_excel_case1_path=False,
# tf_available, per_unit shapes — never claims dual recovery or Case 1 ownership

report = multi_unit_parity_report(atol=1e-9)
# always-on numpy: pack@ref≡x0; affine+postprocess≡evaluate at ref + mild offset
# optional TF raw≡numpy raw when tf_available(); skipped otherwise
# report["ok"] requires numeric checks + honesty fields
assert report["ok"]
assert report["dual_recovery_path"] is None

coeffs = offline_unit_coeffs("FCC")  # always-on AffineCoeffs
# block = build_offline_unit("CDU")  # ImportError if TF missing
```

## Offline priced residual / local box direction (goal 5 readiness)

Always-on numpy surface. Proves exact-linear blocks track **economics** under
simple synthetic product prices, and optionally a tiny closed-form local box
step on drivers. Still offline, still dual-ban, still not on Case 1.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    default_offline_prices,
    pack_price_vector,
    priced_residual_for_unit,
    multi_unit_priced_residual_report,
    local_box_direction,
)

prices = default_offline_prices("FCC")  # synthetic_offline_demo — NOT duals
report = multi_unit_priced_residual_report()  # FCC+COKER+CDU at ref + mild offset
assert report["ok"]
assert report["kind"] == "offline_priced_residual"
assert report["dual_recovery_path"] is None
assert report["on_excel_case1_path"] is False
assert report["price_source"] == "synthetic_offline_demo"
# Coker: raw priced value may ≠ full evaluate even at reference (renorm)

box = local_box_direction("COKER", delta=0.5)
# x_star = x0 + δ * sign(D.T @ p); postprocess outside LP
assert box["dual_recovery_path"] is None
assert box["kind"] == "offline_local_box_direction"
```

Honesty table (priced surface):

| Field | Value |
|-------|-------|
| `kind` | `offline_priced_residual` / `offline_local_box_direction` |
| `solver` | `False` |
| `dual_recovery_path` | `None` |
| `on_excel_case1_path` | `False` |
| `price_source` | `synthetic_offline_demo` (not Case 1 blender / not online λ) |
| Local box gradients | **Not** ADMM λ / **not** Case 1 shadows |

## Offline cached block-solve timing / readiness (goal 5 residual)

Always-on numpy surface. Proves **microsecond-class** affine block direction under
a default-ref coeff cache, with optional local-box step timing and optional
composition of parity + priced `ok` flags. Still offline, still dual-ban, still
not on Case 1. **No hard µs SLA** — report structure + honesty + finite positive
timings; host load may vary.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    cached_offline_unit_coeffs,
    clear_offline_unit_coeffs_cache,
    multi_unit_block_solve_timing_report,
    offline_block_solve_readiness_report,
)

# Default-ref coeffs memoized once per unit (custom refs use offline_unit_coeffs)
c = cached_offline_unit_coeffs("FCC")
assert cached_offline_unit_coeffs("FCC") is c

timing = multi_unit_block_solve_timing_report(
    n_repeats=500, warmup=5, include_box=True, include_composition=False
)
assert timing["ok"]
assert timing["kind"] == "offline_block_solve_timing"
assert timing["dual_recovery_path"] is None
assert timing["on_excel_case1_path"] is False
assert timing["cached_coeffs"] is True
# per-unit: affine.median_us / mean_us; optional box.*; optional tf arm (skipped if no TF)

ready = offline_block_solve_readiness_report(n_repeats=200)
assert ready["kind"] == "offline_block_solve_readiness"
assert ready["parity_ok"] and ready["priced_ok"] and ready["timings_ok"]
assert ready["ready_for_wire_discussion"] is True  # structural only — wire still deferred
assert ready["dual_recovery_path"] is None
```

Honesty table (timing / readiness surface):

| Field | Value |
|-------|-------|
| `kind` | `offline_block_solve_timing` / `offline_block_solve_readiness` |
| `solver` | `False` |
| `dual_recovery_path` | `None` |
| `on_excel_case1_path` | `False` |
| Timings | **Offline readiness** — not Case 1 wall time; not ADMM duals / shadows |
| `ready_for_wire_discussion` | Structural only (parity + priced + timing + honesty); **not** wire shipped |

Planner-facing How_to keys `tf_offline_units` / `tf_offline_priced` / `tf_offline_timing`
(static text in `excel_pipeline`, **no** import of this module) state the same honesty:
FCC+COKER+CDU offline exact-linear + priced residual + block-solve timing readiness
available; **not** on Case 1 solve; duals remain PRIMARY online-λ / SECONDARY recovered.
Index / Summary / Calc_Check also glance-lock that readiness package via pure static
formatters (`format_planner_honesty_package`) — still offline-only, still not wire shipped.

## Per-unit affine API

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

## Before wiring TF into ADMM / Case 1 (pre-wire checklist)

This is a **gate list only** — do **not** implement the wire from this doc alone.

- [ ] `multi_unit_parity_report()` aggregate `ok` (always-on numpy; TF arm green if installed)
- [ ] `multi_unit_priced_residual_report()` aggregate `ok` (always-on economics residual; dual_recovery_path=None; prices not duals)
- [ ] Offline block-solve timing / readiness report green (cached affine; dual-ban intact; not Case 1 wall time) — `multi_unit_block_solve_timing_report` / `offline_block_solve_readiness_report`
- [ ] Dual honesty PRIMARY online λ still gates VERDICT (online L∞ ≤15); do not retune ρ solely to shrink recovered dual L∞
- [ ] Explicit form label change plan: `classic_2block_excel_path` → a named TF-aware form when wire lands (never silent form reuse)
- [ ] Isolation tests (`test_tf_import_isolation.py`) must be **rewritten with** the wire — not silently broken or deleted
- [ ] TF never claims dual recovery without an online-λ proof path (`dual_recovery_path` must stay labeled honestly)
- [ ] Excel lean ≤15 sheets preserved; no EMRPS on hot path; no reformer/HDT kernel as wire side-effect
- [ ] Case 1 demo VERDICT still PASS (gap ≤0.5%, dual L∞ online ≤15) with or without TF installed
- [ ] Local box direction (if used) never treated as Case 1 shadows / online λ
- [ ] Timings / local box gradients never treated as Case 1 shadows / online λ / pure-ADMM duals

## Critics checklist (before claiming “done”)

- [ ] `solver=False` and `dual_recovery_path=None` on honesty_metadata / block / offline_units_status
- [ ] Not on Excel Case 1 path (`on_excel_case1_path=False`; form still classic_2block)
- [ ] No learned weights — base_delta coeffs only
- [ ] Postprocess outside TF; Coker L_div (raw ≠ evaluate at ref) tested
- [ ] CDU nested x0 honesty + L0/L1; no fake `excel_cdu_matrix_matches_affine`
- [ ] Isolation: excel_pipeline / models `__init__` do not import tf_linear_blocks / tensorflow
- [ ] FCC E1/E10 and Coker E7/E8 excel match gates still green
- [ ] How_to includes `fcc_three_path` + `coker_three_path` + `cdu_three_path` + `tf_offline_units`
- [ ] `offline_unit_registry` lists exactly FCC/COKER/CDU; `multi_unit_parity_report` ok without TF
- [ ] `multi_unit_priced_residual_report` ok without TF; Coker raw≠full priced honesty preserved
- [ ] Pre-wire dual-L∞ proof checklist present (this doc); wire not claimed as shipped
- [ ] Priced residual pre-wire gate present; local box gradients not claimed as duals
- [ ] Timing / readiness report honesty: dual_recovery_path=None; on_excel_case1_path=False; no flaky absolute µs hard-fail; timings not duals / not Case 1 wall time
- [ ] `cached_offline_unit_coeffs` default-ref only; custom refs never silently reuse default cache
