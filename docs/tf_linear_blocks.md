# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear **FCC + Coker + CDU** offline kernels + multi-unit registry + wiring-readiness parity harness + **offline priced residual / local box direction harness** + **cached multi-unit block-solve timing / readiness harness** + **offline multi-unit ADMM-style consensus residual harness** + **offline multi-unit ADMM block subproblem maximizer (raw affine under box)** + **offline multi-round ADMM coordination harness (subproblem → z → λ under synthetic λ,z,ρ)** + **offline multi-block plant-linking ADMM harness (synthetic linking topology + shared λ/z + incidence)** + Excel coeff honesty (FCC/Coker only).  
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
| ADMM residual (goal 5 pre-wire) | `multi_unit_admm_residual_report` / `admm_residual_for_unit` — consensus `r=y−z` + L1 augmented local under synthetic λ,z,ρ; **not** dual recovery; **not** Case 1 |
| ADMM block subproblem (goal 5 pre-wire) | `multi_unit_admm_block_subproblem_report` / `admm_block_subproblem_for_unit` — maximize L1-augmented local on **raw affine** under driver box + synthetic λ,z,ρ; **not** dual recovery; **not** Case 1; **not** wire |
| ADMM multi-round coordination (goal 5 pre-wire) | `multi_unit_admm_coordination_report` / `admm_coordination_round_for_unit` — subproblem → raw z consensus → λ ascent under synthetic λ,z,ρ; per-unit synthetic loops; **not** plant linking; **not** dual recovery; **not** Case 1; **not** wire |
| ADMM plant-linking multi-block (goal 5 pre-wire) | `multi_block_plant_linking_admm_report` / `plant_linking_admm_round` / `offline_plant_linking_topology` — shared λ/z on synthetic (default) **or plant-named** linking streams + per-unit incidence; compose subproblem; **not** full plant mass balance; **not** dual recovery; **not** Case 1; **not** wire |
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
# Additive checklist (does not redefine ready_for_wire_discussion):
assert ready.get("admm_residual_ok") is True
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
| `admm_residual_ok` | Additive pre-wire checklist only; **not** wire shipped |

## Offline multi-unit ADMM-style consensus residual (goal 5 pre-wire bridge)

Always-on numpy surface. Proves affine blocks can report **ADMM-shaped** consensus
residual and an L1 augmented local objective under **synthetic** λ, z, ρ — still
offline, still dual-ban, still **not** on Case 1, still **not** wire shipped, still
**not** pure-ADMM dual recovery.

Primary formula (plant ADMM L1 spirit from `blocks.py` language only):

`augmented_local = λ · y_full − ρ ‖y_full − z‖₁`

Default z = postprocess(affine @ reference). Residual space = postprocess (document
raw vs full for Coker renorm honesty). No absolute residual magnitude SLAs.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    admm_residual_for_unit,
    multi_unit_admm_residual_report,
)

report = multi_unit_admm_residual_report(rho=1.0, x_mode="offset")
assert report["ok"]
assert report["kind"] == "offline_admm_block_residual"
assert report["dual_recovery_path"] is None
assert report["on_excel_case1_path"] is False
assert report["solver"] is False
assert report["not_a_solve"] is True
assert report["price_source"] == "synthetic_offline_demo"
# per unit: y_full, z, consensus_residual, r_l1/r_l2/r_linf, augmented_local, formula

row = admm_residual_for_unit("COKER", x_mode="ref")
# at ref with default z: penalty ≈ 0 on full path; raw may ≠ full (renorm)
assert row["r_l1"] <= 1e-9
assert row["raw_vs_full_residual_l1_gap"] >= 0.0
```

Honesty table (ADMM residual surface):

| Field | Value |
|-------|-------|
| `kind` | `offline_admm_block_residual` |
| `solver` | `False` |
| `dual_recovery_path` | `None` |
| `on_excel_case1_path` | `False` |
| `price_source` / `lam_source` / `z_source` / `rho_source` | `synthetic_offline_demo` — **not** Case 1 PRIMARY online λ / **not** SECONDARY recovered |
| Primary formula | L1: `lambda_dot_y - rho * \|\|y_full - z\|\|_1` |
| L2 fields | Diagnostic only (`augmented_local_l2_diagnostic`) |
| Wire | **Not shipped** |

Planner-facing How_to keys `tf_offline_units` / `tf_offline_priced` / `tf_offline_timing`
/ `tf_offline_admm_residual` / `tf_offline_admm_block_subproblem` (static text in
`excel_pipeline`, **no** import of this module) state the same honesty: FCC+COKER+CDU
offline exact-linear + priced residual + block-solve timing + ADMM residual + ADMM
**block subproblem** readiness available; **not** on Case 1 solve; duals remain PRIMARY
online-λ / SECONDARY recovered. Index / Summary / Calc_Check also glance-lock the
readiness package (units + priced + timing + **ADMM residual** + **ADMM block subproblem**,
synthetic λ/z/ρ / x_star ≠ Case 1 duals) via pure static formatters
(`format_planner_honesty_package` /
`meta.planner_honesty.offline_tf_admm_residual_ready` +
`offline_tf_admm_block_subproblem_ready`) — still offline-only, still not wire shipped;
no live residual/subproblem call on Excel write path.

## Offline multi-unit ADMM block subproblem (goal 5 pre-wire maximizer)

Always-on numpy surface. **Maximizes** the L1-augmented local objective under an
independent driver box on **raw affine** (not residual *evaluation* only; not
postprocess optimand). Still offline, still dual-ban, still **not** on Case 1, still
**not** wire shipped, still **not** pure-ADMM dual recovery. **No PuLP/CBC** on this path.

Primary optimand (matches `local_box_direction` raw honesty):

`augmented_local_raw = λ · y_raw − ρ ‖y_raw − z‖₁`

with `y_raw = clamp(y0 + D @ (x − x0))` and `x ∈ [x0−δ, x0+δ]`.

Default z = full postprocess(affine @ reference) while optimand is raw — labeled
(`optimand_space=raw_affine`, `z_source`, full fields diagnostic only). Coker renorm:
raw ≠ full expected. Method: coordinate-ascent with exact 1-D piecewise-linear
maximizers + multi-start from `{x0, priced corner under λ}`; `optimality_note` does
**not** claim multi-D global optimality.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    admm_block_subproblem_for_unit,
    multi_unit_admm_block_subproblem_report,
)

report = multi_unit_admm_block_subproblem_report(rho=1.0, delta=0.5)
assert report["ok"]
assert report["kind"] == "offline_admm_block_subproblem"
assert report["dual_recovery_path"] is None
assert report["on_excel_case1_path"] is False
assert report["solver"] is False
assert report["optimand_space"] == "raw_affine"
assert report["price_source"] == "synthetic_offline_demo"
# per unit: x_star, y_raw_star, augmented_local_raw ≥ ref, formula raw-L1

row = admm_block_subproblem_for_unit("COKER", delta=0.5)
assert row["not_worse_than_ref"] is True
assert row["optimand_space"] == "raw_affine"
# full postprocess fields are diagnostic only
_ = row["augmented_local_full_diagnostic"]
```

Honesty table (ADMM block subproblem surface):

| Field | Value |
|-------|-------|
| `kind` | `offline_admm_block_subproblem` |
| `solver` | `False` |
| `dual_recovery_path` | `None` |
| `on_excel_case1_path` | `False` |
| `optimand_space` | `raw_affine` (full postprocess = diagnostic only) |
| `price_source` / `lam_source` / `z_source` / `rho_source` | synthetic offline — **not** Case 1 PRIMARY online λ / **not** SECONDARY recovered |
| Primary formula | L1 raw: `lambda_dot_y_raw - rho * \|\|y_raw - z\|\|_1` |
| Method | `coordinate_ascent_exact_1d_pl` + multi-start; see `optimality_note` |
| Wire | **Not shipped** |
| Backend | Always-on numpy — **not** PuLP/CBC |

## Offline multi-round ADMM coordination (goal 5 pre-wire loop)

Always-on numpy surface. Runs a **small number of ADMM-style rounds** over
FCC+COKER+CDU by **composing** the existing block subproblem maximizer:

1. **x / y step** — `admm_block_subproblem_for_unit` under current synthetic λ, z, ρ, δ
2. **residual** — `r = y_raw − z_pre` (pre-update; never after free `z←y`)
3. **z consensus** — raw-space update `z ← (1−β)z + β y_raw` (default β=1 copy)
4. **λ dual ascent** — `λ ← λ + α·ρ·r` (α=`dual_step`, default 1.0)

Per-unit **independent** product-space loops (registry order). **Not** a plant
linking-stream coordinator. Still offline, dual-ban, **not** Case 1, **not** wire,
**not** pure-ADMM dual recovery. No absolute residual-must-converge hard-fail.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    admm_coordination_round_for_unit,
    multi_unit_admm_coordination_report,
)

report = multi_unit_admm_coordination_report(
    n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
)
assert report["ok"]
assert report["kind"] == "offline_admm_coordination"
assert report["dual_recovery_path"] is None
assert report["on_excel_case1_path"] is False
assert report["solver"] is False
assert report["optimand_space"] == "raw_affine"
assert report["z_update_space"] == "raw_affine"
assert report["not_plant_linking_coordinator"] is True
assert report["coordination_lambda_is_not_case1_online_lambda"] is True
assert len(report["trajectory"]) == 3
# trajectory residuals are finite; no residual-must-vanish SLA
```

Honesty table (ADMM coordination surface):

| Field | Value |
|-------|-------|
| `kind` | `offline_admm_coordination` |
| `solver` | `False` |
| `dual_recovery_path` | `None` |
| `on_excel_case1_path` | `False` |
| `optimand_space` / `z_update_space` | `raw_affine` |
| `coordination_scope` | `per_unit_synthetic_offline` (not plant linking) |
| `price_source` / `lam_source` / `z_source` / `rho_source` | synthetic offline — **not** Case 1 PRIMARY online λ / **not** SECONDARY recovered |
| Dual ascent residual | **z_pre** (`r = y_raw − z_pre`) — never post-z zero theater |
| Wire | **Not shipped** |
| Backend | Always-on numpy compose of subproblem maximizer — **not** PuLP/CBC |

## Offline multi-block plant-linking ADMM (goal 5 pre-wire)

Always-on numpy surface. Runs ADMM-style rounds over FCC+COKER+CDU with **shared**
λ/z on a plant linking-stream space, mapped to each unit via explicit incidence
(unit products are name-disjoint — no product-name intersection theater). Composes
existing `admm_block_subproblem_for_unit`:

1. **Map** — `prices_unit = A^T λ_link`, `z_unit = A^T z_link` (0/1 selection incidence)
2. **x / y step** — existing subproblem maximizer under (λ_u, z_u, ρ, δ)
3. **Lift + residual** — `y_link_u = A_u y_raw`; `r_link = sum_u y_link_u − z_link_pre`
   (pre-z-update; never post free z←y zero theater)
4. **z consensus** — linking-space update `z ← (1−β)z + β y_link_total`
5. **λ dual ascent** — `λ ← λ + α·ρ·r_link`

### Topology modes

| Mode | `topology_source` | Streams | Incidence | `linking_space` |
|------|-------------------|---------|-----------|-----------------|
| `synthetic` (**default**) | `synthetic_offline_demo` | family: `light_ends`, `naphtha`, … | product → family stream | `synthetic_linking_streams` |
| `plant_named` | `plant_named_offline_demo` | plant product names: `fcc_naphtha`, `cdu_gasoil`, … | **identity** product → product | `plant_named_linking_streams` |

Plant-named offline demo **≠** full plant mass balance / live plant_blocks cascade /
Case 1 CDU↔Blender links. Shared plant-linking λ **≠** Case 1 PRIMARY online λ /
SECONDARY recovered. Existing `multi_unit_admm_coordination_report` remains a
separate surface with `not_plant_linking_coordinator=True`. No absolute
residual-must-converge hard-fail. Additive readiness flags:

- `admm_plant_linking_ok` — synthetic default mode
- `admm_plant_named_linking_ok` — plant-named mode

Neither redefines `ready_for_wire_discussion` (still parity∧priced∧timings∧honesty).

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_plant_linking_topology,
    offline_plant_named_linking_topology,
    plant_linking_admm_round,
    multi_block_plant_linking_admm_report,
    multi_block_plant_named_linking_admm_report,
)

# Default remains synthetic (bit-stable for existing plant-linking tests)
topo = offline_plant_linking_topology()
assert topo["topology_source"] == "synthetic_offline_demo"
assert topo["not_full_plant_mass_balance"] is True

report = multi_block_plant_linking_admm_report(
    n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
)
assert report["ok"]
assert report["kind"] == "offline_admm_plant_linking"
assert report["dual_recovery_path"] is None
assert report["on_excel_case1_path"] is False
assert report["solver"] is False
assert report["not_full_plant_mass_balance"] is True
assert report["plant_linking_lambda_is_not_case1_online_lambda"] is True
assert report["not_wire_shipped"] is True
assert len(report["trajectory"]) == 3
# trajectory residuals are finite; no residual-must-vanish SLA

# Plant-named mode (identity incidence; plant-style stream names)
topo_n = offline_plant_named_linking_topology()
assert topo_n["topology_source"] == "plant_named_offline_demo"
assert "fcc_naphtha" in topo_n["streams"]
assert "light_ends" not in topo_n["streams"]

named = multi_block_plant_named_linking_admm_report(
    n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
)
assert named["ok"]
assert named["topology_source"] == "plant_named_offline_demo"
assert named["linking_space"] == "plant_named_linking_streams"
assert named["dual_recovery_path"] is None
assert named["not_full_plant_mass_balance"] is True
assert named["not_wire_shipped"] is True
# plant-named offline demo ≠ full plant MB / live cascade / Case 1 duals
```

Honesty table (ADMM plant-linking surface — both modes):

| Field | Synthetic default | Plant-named mode |
|-------|-------------------|------------------|
| `kind` | `offline_admm_plant_linking` | same |
| `solver` | `False` | same |
| `dual_recovery_path` | `None` | same |
| `on_excel_case1_path` | `False` | same |
| `optimand_space` | `raw_affine` (unit level; reuses subproblem) | same |
| `linking_space` / `z_update_space` | `synthetic_linking_streams` | `plant_named_linking_streams` |
| `topology_source` / `plant_linking_scope` | `synthetic_offline_demo` | `plant_named_offline_demo` |
| `not_full_plant_mass_balance` | `True` | same |
| `plant_linking_lambda_is_not_case1_online_lambda` | `True` | same |
| Dual ascent residual | **pre-z** linking residual (`r_link = sum y_link − z_pre`) | same |
| Wire / full plant MB / live cascade | **Not shipped** | **Not shipped** |
| Backend | Always-on numpy compose of subproblem — **not** PuLP/CBC | same |

Excel How_to / Index / Summary / meta / Calc_Check / demo also glance-lock multi-block
plant-linking readiness **statically** (`tf_offline_admm_plant_linking` How_to topic;
`meta.planner_honesty.offline_tf_admm_plant_linking_ready`;
`offline_tf_admm_plant_linking_not_duals` Calc_Check row; demo readiness bit
`admm_plant_linking`). Existence packaging only — **not** a live
`multi_block_plant_linking_admm_report` call from the Excel write path; **not** Case 1;
**not** full plant mass balance; **not** wire; plant-linking λ ≠ Case 1 duals. Per-unit
coordination surface remains distinct (`not_plant_linking_coordinator=True`).

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
- [x] `multi_unit_admm_residual_report()` ok (synthetic λ,z,ρ; dual-ban; not Case 1; not pure-ADMM dual recovery; not wire shipped)
- [x] `multi_unit_admm_block_subproblem_report()` ok (raw affine L1 maximizer under box; dual-ban; not Case 1; not pure-ADMM dual recovery; not wire shipped; not PuLP)
- [x] `multi_unit_admm_coordination_report()` ok (multi-round subproblem → raw z → λ under synthetic λ,z,ρ; dual-ban; per-unit synthetic scope; not plant linking; not Case 1; not pure-ADMM dual recovery; not wire shipped; no residual-must-vanish hard-fail)
- [x] `multi_block_plant_linking_admm_report()` ok (synthetic linking-stream topology + shared λ/z + per-unit incidence; compose subproblem; dual-ban; not full plant mass balance; plant-linking λ ≠ Case 1 online λ; not Case 1; not pure-ADMM dual recovery; not wire shipped; no residual-must-vanish hard-fail)
- [x] Plant-named linking topology mode ok (`mode="plant_named"` / `multi_block_plant_named_linking_admm_report`; identity incidence; `topology_source=plant_named_offline_demo`; dual-ban; not full plant MB; not live cascade; not Case 1; not wire; synthetic default still green)
- [ ] Dual honesty PRIMARY online λ still gates VERDICT (online L∞ ≤15); do not retune ρ solely to shrink recovered dual L∞
- [ ] Explicit form label change plan: `classic_2block_excel_path` → a named TF-aware form when wire lands (never silent form reuse)
- [ ] Isolation tests (`test_tf_import_isolation.py`) must be **rewritten with** the wire — not silently broken or deleted
- [ ] TF never claims dual recovery without an online-λ proof path (`dual_recovery_path` must stay labeled honestly)
- [ ] Excel lean ≤15 sheets preserved; no EMRPS on hot path; no reformer/HDT kernel as wire side-effect
- [ ] Case 1 demo VERDICT still PASS (gap ≤0.5%, dual L∞ online ≤15) with or without TF installed
- [ ] Local box direction (if used) never treated as Case 1 shadows / online λ
- [ ] Timings / local box gradients never treated as Case 1 shadows / online λ / pure-ADMM duals
- [ ] Synthetic λ / z / ρ / ADMM residuals / subproblem x_star never treated as Case 1 online λ or recovered duals
- [ ] ADMM block subproblem optimand stays raw_affine; full postprocess diagnostic only; no PuLP offline backend

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
- [ ] `multi_unit_admm_residual_report` ok without TF; honesty locks; L1 formula; Coker raw≠full residual honesty; synthetic λ ≠ Case 1 online λ
- [ ] `multi_unit_admm_block_subproblem_report` ok without TF; raw optimand; maximizer ≥ ref; delta=0 ⇒ x_star≈x0; dual-ban; not wire; Coker raw≠full diagnostic
- [ ] `multi_unit_admm_coordination_report` ok without TF; honesty locks; trajectory finite; z_pre dual ascent; per-unit synthetic scope; no residual-must-vanish SLA; not wire
- [ ] `multi_block_plant_linking_admm_report` ok without TF; synthetic topology (default) + plant-named mode; shared linking λ/z; pre-z linking residual dual ascent; compose subproblem; not full plant MB; plant-linking λ ≠ Case 1 duals; dual_recovery_path=None; no residual-must-vanish SLA; existing coordination still `not_plant_linking_coordinator=True`; additive `admm_plant_named_linking_ok` does not redefine `ready_for_wire_discussion`
