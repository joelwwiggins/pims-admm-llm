# TensorFlow linear blocks (optional, offline)

**Status:** exact-linear **FCC + Coker + CDU** offline kernels + multi-unit registry + wiring-readiness parity harness + **offline priced residual / local box direction harness** + **cached multi-unit block-solve timing / readiness harness** + **offline multi-unit ADMM-style consensus residual harness** + **offline multi-unit ADMM block subproblem maximizer (raw affine under box)** + **offline multi-round ADMM coordination harness (subproblem → z → λ under synthetic λ,z,ρ)** + **offline multi-block plant-linking ADMM harness (synthetic linking topology + shared λ/z + incidence)** + **offline dual-honest wire-preflight report (compose gates + machine-readable wire_blockers; wire_shipped=False)** + **offline Case-1-shaped CDU↔Blender linking skeleton (dual-ban; wire_shipped=False; linear_quality_pooling)** + **offline Case-1 dual-space / form-label contract (planned form registry without flip; stream map; dual_linf unproven checklist)** + **offline Case-1 dual-space L∞ probe / dual_linf proof-prep (stream-aligned numeric L∞; dual_linf still unproven; not VERDICT)** + Excel coeff honesty (FCC/Coker only).  
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
| Wire preflight (goal 5 honesty residual) | `offline_wire_preflight_report` / `offline_wire_blocker_catalog` — compose readiness + additive ADMM gates + machine-readable `wire_blockers`; `wire_shipped=False`; **not** dual recovery; **not** Case 1; **not** wire shipped; does **not** redefine `ready_for_wire_discussion` |
| Case-1-shaped CDU↔Blender skeleton (goal 5 residual) | `offline_case1_shaped_cdu_blender_linking_report` — CDU affine + blender `linear_quality_pooling` residual under synthetic λ,z,ρ on Case 1 intermediates; dual-ban; `wire_shipped=False`; skeleton ≠ wire; **not** form flip; does **not** clear `DEFAULT_WIRE_BLOCKERS` |
| Case-1 dual-space / form-label contract (goal 5+3 residual) | `offline_case1_dual_space_form_contract_report` — planned TF-aware form registry **without** flipping Case 1; dual-space stream map (Case 1 intermediates ↔ skeleton λ); `dual_linf_under_wire=unproven` + open checklist; dual-ban; `wire_shipped=False`; does **not** clear blockers; does **not** redefine ready |
| Case-1 dual-space L∞ probe (goal 5+3 residual) | `offline_case1_dual_space_linf_probe_report` — stream-aligned numeric L∞ between fixture/supplied Case 1 PRIMARY online λ and Case-1-shaped skeleton λ; dual_linf_under_wire stays **unproven**; checklist `online_linf_gate_under_tf_path` open; probe ≠ wire proof; probe ≠ VERDICT gate; dual-ban; `wire_shipped=False`; does **not** clear blockers; does **not** redefine ready |
| Case-1 dual-space L∞ live-λ bridge (goal 5+3 residual) | `offline_case1_dual_space_linf_live_lambda_bridge_report` — pure extract/normalize this-run Case 1 PRIMARY online λ (+ optional SECONDARY) into existing probe; `live_lambda_source` labeled; dual-ban; dual_linf unproven; bridge ≠ VERDICT; bridge ≠ wire proof; no excel_pipeline on TF hot path |
| Case-1 dual-space L∞ live-λ-seeded warm-start (goal 5+3 residual) | `offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report` — seed Case-1-shaped skeleton λ0 from live/caller PRIMARY (source labeled); run N skeleton rounds; post-round stream L∞ + `linf_at_seed` seed-identity diagnostic; dual_linf unproven **always**; warm-start ≠ VERDICT; warm-start ≠ wire proof; seed identity ≠ proof; no excel_pipeline on TF hot path |
| Case-1 honest blender pooling path (goal 5+3 residual) | `offline_case1_honest_blender_pooling_path_report` — formalizes `linear_quality_pooling` as dual-honest Case-1 blender path; checklist `honest_pooling_path_present` (not bare open; not closed_via_affine); no BLENDER UNITS; `no_blender_offline_affine_kernel` still true; dual_linf unproven; pooling ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; no excel_pipeline on TF hot path |
| Case-1 online_linf_gate flip-criteria contract (goal 3+5 residual) | `offline_case1_online_linf_gate_criteria_contract_report` — machine-readable flip criteria for checklist `online_linf_gate_under_tf_path` (**gate stays open**); `gate_flip_allowed_today=False`; `criteria_met_today=False`; dual_linf unproven; contract ≠ gate flip ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; no excel_pipeline on TF hot path |
| Case-1 isolation-rewrite design-only contract (goal 5+3 residual) | `offline_case1_isolation_rewrite_design_contract_report` — formalizes isolation rewrite WITH dual-honest wire (rewrite-not-delete); `isolation_rewrite_design_present=True`; **`isolation_rewrite_shipped=False`**; checklist `isolation_rewrite_with_wire` **stays open**; dual_linf unproven; online_linf_gate open; design ≠ rewrite shipped ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; isolation suite unchanged; no excel_pipeline on TF hot path |
| Case-1 dual-honest wire-ship acceptance design contract (goal 5+3 residual) | `offline_case1_wire_ship_acceptance_design_contract_report` — machine-readable criteria for when wire *may* ship; `design_present=True`; **`wire_ship_allowed_today=False`**; **`wire_shipped=False`**; criteria_met_today=False; dual_linf unproven; form classic; isolation rewrite not shipped; online_linf_gate open; design ≠ ship allow ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; blockers remain; no excel_pipeline on TF hot path |
| Case-1 dual-honest TF-aware path design contract (goal 5+3 residual) | `offline_case1_dual_honest_tf_aware_path_design_contract_report` — machine-readable *path shape* for future dual-honest TF Case 1 wire (CDU affine + blender `linear_quality_pooling`; form_planned; dual_recovery planned-vs-today; feature flag reserved false); **`path_design_present=True`**; **`path_shipped=False`**; **`dual_honest_tf_aware_path_present` ship-met remains False**; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; design ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; blockers remain; no excel_pipeline on TF hot path. **Excel packaging twin:** static How_to/Index/Summary/meta/Calc_Check/demo expose path design present + path_shipped=false + ship-met=false without shipping path/wire or importing `tf_linear_blocks` on excel path |
| Case-1 dual_honest_tf_aware_path_present ship-met flip criteria contract (goal 5+3 residual) | `offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report` — machine-readable *when path counts as present-for-ship*; **`criteria_present=True`**; **`ship_met_allowed_today=False`**; **`criteria_met_today=False`**; **`dual_honest_tf_aware_path_present` ship-met remains False**; **`path_design_present=True`**; path_shipped=False; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; criteria ≠ ship-met ≠ path shipped ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; blockers remain; no excel_pipeline on TF hot path. Path design formalizes *what*; this formalizes *when present-for-ship* without shipping. **Excel packaging twin:** static How_to/Index/Summary/meta/Calc_Check/demo expose criteria_present + ship_met_allowed=false + ship-met=false without shipping path/wire or importing `tf_linear_blocks` on excel path |
| Case-1 form_label_change_shipped flip criteria contract (goal 5+3 residual) | `offline_case1_form_label_change_shipped_criteria_contract_report` — machine-readable *when form_label_change_shipped may become True*; **`criteria_present=True`**; **`form_label_ship_allowed_today=False`**; **`criteria_met_today=False`**; **`form_label_change_shipped` remains False**; form remains classic; path_design_present=True; path_shipped=False; ship-met False; wire_shipped=False; dual_linf unproven; isolation rewrite not shipped; online_linf_gate open; form registration ≠ form_label shipped ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; blockers remain (incl. form_label_change_required); no excel_pipeline on TF hot path. Form registration formalizes *what*; this formalizes *when shipped* without flipping form. |
| Case-1 isolation-rewrite ship-met / flip criteria contract (goal 5+3 residual) | `offline_case1_isolation_rewrite_shipped_criteria_contract_report` — machine-readable *when isolation_rewrite_with_wire / isolation_rewrite_shipped may become met/True*; **`criteria_present=True`**; **`isolation_ship_allowed_today=False`**; **`criteria_met_today=False`**; **`isolation_rewrite_shipped` remains False**; checklist `isolation_rewrite_with_wire` **stays open**; rewrite-not-delete; form classic; path_shipped=False; ship-met False; form_label_change_shipped False; wire_shipped=False; dual_linf unproven; online_linf_gate open; isolation design ≠ isolation rewrite shipped ≠ form ship ≠ path ship ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag; blockers remain (incl. isolation_rewrite_required); isolation suite unchanged; no excel_pipeline on TF hot path. Isolation design formalizes *what*; this formalizes *when shipped* without shipping rewrite. **Excel packaging twin:** static How_to/Index/Summary/meta/Calc_Check/demo expose criteria_present + isolation_ship_allowed=false + isolation_rewrite_shipped=false + checklist open without shipping rewrite/wire or importing `tf_linear_blocks` on excel path |
| Case-1 dual-honest multi-blocker wire bundle design contract (goal 5+3 residual) | `offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report` — machine-readable *what must land together* for `SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT` (`dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change`); **`bundle_design_present=True`**; **`bundle_shipped=False`**; **`bundle_ship_allowed_today=False`**; criteria_met_today=False; wire_shipped=False; isolation_rewrite_shipped=False; form classic; form_label_change_shipped=False; path_shipped=False; ship-met False; dual_linf unproven; online_linf_gate open; dual_recovery_path=None; feature flag reserved false; optional order_hint is **not** an executor (no auto-wire); distinct from wire-ship acceptance design (unordered when-ship); dual-ban; additive readiness flag; blockers remain; isolation suite unchanged; no excel_pipeline on TF hot path. Individual design/criteria formalize each co-req; this formalizes the multi-blocker *bundle* without shipping wire/bundle. **Excel packaging twin:** static How_to/Index/Summary/meta/Calc_Check/demo expose bundle_design_present + bundle_shipped=false + bundle_ship_allowed_today=false + order_hint not executor without shipping wire/bundle or importing `tf_linear_blocks` on excel path. |
| Case-1 dual-honest multi-blocker wire bundle ship-met / flip criteria contract (goal 5+3 residual) | `offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report` — machine-readable *when* `bundle_shipped` / `bundle_ship_allowed_today` may become True; **`criteria_present=True`**; **`bundle_shipped=False`**; **`bundle_ship_allowed_today=False`**; **`criteria_met_today=False`**; wire_shipped=False; isolation_rewrite_shipped=False; isolation checklist open; form classic; form_label_change_shipped=False; path_shipped=False; path ship-met False; dual_linf unproven; online_linf_gate open; dual_recovery_path=None; feature flag reserved false; order_hint **not** executor; dual-ban; additive readiness flag; blockers remain; isolation suite unchanged; no excel_pipeline on TF hot path. Bundle design formalizes *what*; this formalizes *when* without shipping wire/bundle. **Excel packaging twin of *criteria*:** static How_to/Index/Summary/meta/Calc_Check/demo expose criteria_present + bundle_shipped=false + bundle_ship_allowed_today=false + criteria_met_today=false without shipping wire/bundle or importing `tf_linear_blocks` on excel path. |
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
`admm_plant_linking`) **and** plant-named linking readiness
(`tf_offline_admm_plant_named_linking` How_to topic;
`offline_tf_admm_plant_named_linking_ready`;
`offline_tf_admm_plant_named_linking_not_duals`; demo bit `admm_plant_named_linking`;
`topology_source=plant_named_offline_demo`) **and** offline wire-preflight readiness
(`tf_offline_wire_preflight` How_to topic; `offline_tf_wire_preflight_ready`;
`offline_tf_wire_preflight_not_duals` / `offline_tf_wire_not_shipped`;
`wire_shipped=False`; static mirror of `DEFAULT_WIRE_BLOCKERS` honesty ids) **and**
offline Case-1-shaped CDU↔Blender skeleton readiness
(`tf_offline_case1_shaped_linking` How_to topic; `offline_tf_case1_shaped_linking_ready`;
`offline_tf_case1_shaped_linking_not_duals` / `offline_tf_case1_shaped_not_wire`;
`blender_surface=linear_quality_pooling`; streams naphtha/distillate/gasoil/residue;
`wire_shipped=False`; skeleton λ ≠ Case 1 duals; skeleton ≠ package-ADMM wire) **and**
offline Case-1 dual-space / form-label contract readiness
(`tf_offline_case1_dual_space_form_contract` How_to topic;
`offline_tf_case1_dual_space_form_contract_ready`;
`offline_tf_case1_dual_space_form_contract_not_duals` /
`offline_tf_case1_dual_space_form_contract_not_wire`;
`form_current=classic_2block_excel_path` vs planned
`tf_affine_cdu_blender_shaped_excel_path` registered only; streams aligned;
`dual_linf_under_wire=unproven` + open checklist; dual-ban; `wire_shipped=False`;
does **not** clear blockers). Existence
packaging only — **not** a live
`multi_block_plant_linking_admm_report` / `multi_block_plant_named_linking_admm_report`
/ `offline_wire_preflight_report` / `offline_case1_shaped_cdu_blender_linking_report`
/ `offline_case1_dual_space_form_contract_report`
call from the Excel write path; **not** Case 1; **not** full plant mass balance; **not**
wire shipped; plant-linking / plant-named / preflight / skeleton / contract λ ≠ Case 1 duals.
`ready_for_wire_discussion` remains structural only
(parity∧priced∧timings∧honesty) — packaging surfaces blockers so structural ready ≠
"wire tomorrow". Per-unit coordination surface remains distinct
(`not_plant_linking_coordinator=True`). Synthetic plant-linking packaging remains
present alongside plant-named. Case-1-shaped + dual-space/form contract packaging do
**not** clear `DEFAULT_WIRE_BLOCKERS` / Excel `_OFFLINE_WIRE_BLOCKER_IDS`.

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


## Offline dual-honest wire preflight (goal 5 residual)

Always-on numpy surface. Answers: **what is already green on the offline ladder**,
and **what still blocks a dual-honest wire** — without shipping wire.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_wire_preflight_report,
    offline_wire_blocker_catalog,
    multi_unit_wire_preflight_report,  # alias
)

cat = offline_wire_blocker_catalog()
assert cat["wire_shipped"] is False
assert "isolation_rewrite_required" in cat["wire_blockers"]
assert "form_label_change_required" in cat["wire_blockers"]
assert "dual_linf_under_wire_unproven" in cat["wire_blockers"]

rep = offline_wire_preflight_report()
assert rep["kind"] == "offline_wire_preflight"
assert rep["dual_recovery_path"] is None
assert rep["wire_shipped"] is False
assert rep["blockers_documented"] is True
assert rep["preflight_ok"] is True
# Structural ready meaning UNCHANGED (parity∧priced∧timings∧honesty only)
assert rep["ready_for_wire_discussion"] is True  # green ladder; not "wire tomorrow"
assert "wire_not_shipped" in rep["wire_blockers"]
# preflight / plant-linking λ ≠ Case 1 PRIMARY online λ / SECONDARY recovered
assert rep["preflight_lambda_is_not_case1_online_lambda"] is True
assert rep["not_full_plant_mass_balance"] is True
assert rep["not_pure_admm_dual_recovery"] is True
```

| Field | Meaning |
|-------|---------|
| `ready_for_wire_discussion` | **Unchanged** structural readiness (parity∧priced∧timings∧honesty) — **not** redefined by blockers or preflight |
| `preflight_ok` / `blockers_documented` | Compose healthy + honesty locks + non-empty true-at-HEAD blockers |
| `wire_shipped` | Always `False` on this surface |
| `wire_blockers` | Stable honesty ids (isolation rewrite, form label, dual L∞ under wire unproven, Case 1 CDU+Blender shape, no blender kernel, wire_not_shipped, affine≠plant_blocks feed LP) |
| `ok` | Preflight surface healthy — **never** means wire is shipped |

**Hard locks:** no Case 1 form flip; no isolation rewrite; no live excel→tf preflight call;
no ρ retune; no dual recovery claim; no full plant MB claim; no PuLP on preflight path.


## Offline Case-1-shaped CDU↔Blender linking skeleton (goal 5 residual)

Always-on numpy surface. Models **Case 1 package shape** (CDU producer ↔ Blender
consumer on classic intermediates `naphtha` / `distillate` / `gasoil` / `residue`)
without wiring TF into Case 1 or flipping form.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_case1_shaped_cdu_blender_linking_report,
    multi_block_case1_shaped_linking_admm_report,  # alias
    CASE1_SHAPED_LINKING_STREAMS,
    CASE1_SHAPED_BLENDER_SURFACE,
    DEFAULT_WIRE_BLOCKERS,
)

rep = offline_case1_shaped_cdu_blender_linking_report(n_rounds=3, rho=1.0, delta=0.5)
assert rep["kind"] == "offline_case1_shaped_cdu_blender_linking"
assert rep["dual_recovery_path"] is None
assert rep["wire_shipped"] is False
assert rep["case1_form_unchanged"] is True
assert rep["linking_lambda_is_not_case1_online_lambda"] is True
assert rep["blender_surface"] == "linear_quality_pooling"
assert rep["blender_is_base_delta_affine_unit"] is False
assert set(rep["streams"]) == set(CASE1_SHAPED_LINKING_STREAMS)
assert "BLENDER" not in rep["units_affine_unchanged"]  # still FCC/COKER/CDU
# skeleton does NOT clear wire blockers
assert "case1_is_cdu_blender_package_admm" in DEFAULT_WIRE_BLOCKERS
assert "no_blender_offline_affine_kernel" in DEFAULT_WIRE_BLOCKERS
# ok = honesty + finite trajectory — NOT residual-must-vanish
assert rep["ok"] is True
```

| Field | Meaning |
|-------|---------|
| `kind` | `offline_case1_shaped_cdu_blender_linking` (≠ plant-linking) |
| `dual_recovery_path` | Always `None` |
| `wire_shipped` | Always `False` (skeleton ≠ wire) |
| `blender_surface` | `linear_quality_pooling` — **not** `base_delta_affine_unit` |
| `linking_lambda_is_not_case1_online_lambda` | Skeleton λ ≠ Case 1 PRIMARY online λ / SECONDARY recovered |
| `case1_form_unchanged` | Case 1 still `classic_2block_excel_path` outside this surface |
| `ok` | Honesty + finite trajectory + structure — **not** residual-must-vanish |

**Hard locks:** no Case 1 form flip; no isolation rewrite; no live excel→tf skeleton
call as primary; no ρ retune; no dual recovery claim; no full plant MB claim; no PuLP
on offline hot path; no silent `BLENDER` in `UNITS`; no invent
`excel_cdu_matrix_matches_affine` / `excel_blender_matrix_matches_affine`;
`DEFAULT_WIRE_BLOCKERS` stay true (skeleton ≠ package-ADMM wire; linear pooling ≠
affine kernel). Additive readiness flag `admm_case1_shaped_linking_ok` does **not**
redefine `ready_for_wire_discussion`.

**Dual honesty cross-link:** Case 1 duals remain PRIMARY free online λ /
SECONDARY recovered blender on the Excel path. This skeleton has
`dual_recovery_path=None` and `linking_lambda_is_not_case1_online_lambda=True`.

## Offline Case-1 dual-space / form-label contract (goal 5 + goal 3 residual)

Always-on numpy surface that makes the softest pre-wire blockers
(`form_label_change_required`, `dual_linf_under_wire_unproven`) machine-checkable
**prep** without shipping wire or flipping Case 1 form.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_case1_dual_space_form_contract_report,
    multi_unit_case1_dual_space_form_contract_report,  # alias
    CASE1_FORM_CURRENT,
    CASE1_PLANNED_TF_AWARE_FORM,
    CASE1_DUAL_LINF_UNDER_WIRE_STATUS,
)

rep = offline_case1_dual_space_form_contract_report()
assert rep["kind"] == "offline_case1_dual_space_form_contract"
assert rep["ok"] is True
assert rep["form_current"] == CASE1_FORM_CURRENT == "classic_2block_excel_path"
assert rep["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
assert rep["form_planned"] != rep["form_current"]
assert rep["form_unchanged"] is True
assert rep["stream_alignment_ok"] is True
assert set(rep["streams"]) == {"naphtha", "distillate", "gasoil", "residue"}
assert rep["dual_linf_under_wire_status"] == CASE1_DUAL_LINF_UNDER_WIRE_STATUS == "unproven"
assert rep["dual_recovery_path"] is None
assert rep["wire_shipped"] is False
assert rep["blockers_still_documented"] is True
```

| Field | Meaning |
|-------|---------|
| `form_current` / `form_planned` | Classic Case 1 form vs registered TF-aware future label (distinct; no flip) |
| `stream_alignment_ok` | Case 1 intermediates name-set equals Case-1-shaped skeleton λ slots |
| `package_dual_gate` | `online_lambda` (PRIMARY VERDICT gate) |
| `package_dual_secondary` | `recovered_blender` (SECONDARY; not gate) |
| `dual_linf_under_wire_status` | Always `unproven` at HEAD; open checklist items remain open |
| `wire_shipped` | Always `False`; contract does **not** clear `DEFAULT_WIRE_BLOCKERS` |
| `ok` | Honesty ∧ form contract ∧ stream alignment ∧ dual_linf unproven ∧ blockers documented — **not** wire |

Additive readiness flag `admm_case1_dual_space_form_contract_ok` does **not**
redefine `ready_for_wire_discussion` (still parity∧priced∧timings∧honesty only).

**Dual honesty cross-link:** Case 1 duals remain PRIMARY free online λ /
SECONDARY recovered blender on the Excel path. This contract surface has
`dual_recovery_path=None`; skeleton λ ≠ Case 1 PRIMARY/SECONDARY duals;
registering a planned form is **not** form flip and **not** dual L∞ proof under wire.

## Offline Case-1 dual-space L∞ probe / dual_linf proof-prep (goal 5 + goal 3 residual)

Always-on numpy surface that makes dual-L∞-under-wire *numeric prep* concrete
without claiming proof and without shipping wire. Stream-aligned L∞ between
fixture/supplied Case 1 PRIMARY online λ and Case-1-shaped skeleton `final_lam`.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_case1_dual_space_linf_probe_report,
    case1_dual_space_linf_probe,  # alias
    case1_primary_online_lambda_fixture,
    case1_dual_space_stream_aligned_linf,
    CASE1_DUAL_LINF_UNDER_WIRE_STATUS,
)

rep = offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
assert rep["kind"] == "offline_case1_dual_space_linf_probe"
assert rep["probe_ok"] is True
assert rep["stream_alignment_ok"] is True
assert rep["finite_linf"] is True
assert rep["dual_linf_under_wire_status"] == CASE1_DUAL_LINF_UNDER_WIRE_STATUS == "unproven"
assert rep["online_linf_gate_under_tf_path"] == "open"
assert rep["dual_recovery_path"] is None
assert rep["wire_shipped"] is False
assert rep["probe_is_not_verdict_gate"] is True
assert rep["probe_is_not_dual_linf_under_wire_proof"] is True
assert rep["skeleton_lambda_is_not_case1_online_lambda"] is True
# Even L∞≈0 on identical synthetic vectors keeps status unproven:
z = {"naphtha": 0.0, "distillate": 0.0, "gasoil": 0.0, "residue": 0.0}
zero = offline_case1_dual_space_linf_probe_report(
    case1_primary_online_lambda=z, skeleton_lambda=z
)
assert zero["linf"] == 0.0
assert zero["dual_linf_under_wire_status"] == "unproven"
```

| Field | Meaning |
|-------|---------|
| `linf` / `per_stream_abs` | Stream-aligned max-abs / per-stream gaps (naphtha/distillate/gasoil/residue) |
| `dual_vector_face` | Default `raw_online_duals` (negative fixture); economic shadow face optional |
| `dual_linf_under_wire_status` | Always `unproven` — probe available ≠ proven under wire |
| `online_linf_gate_under_tf_path` | Remains `open` after probe ships |
| `probe_ok` | Honesty ∧ aligned ∧ finite ∧ dual-ban ∧ blockers documented — **not** `linf≤15` |
| `probe_is_not_verdict_gate` | Probe L∞ is never Case 1 VERDICT hard-fail |
| `wire_shipped` | Always `False`; does **not** clear `DEFAULT_WIRE_BLOCKERS` |

Additive readiness flag `admm_case1_dual_space_linf_probe_ok` does **not**
redefine `ready_for_wire_discussion` (still parity∧priced∧timings∧honesty only).

**Dual honesty:** Case 1 duals remain PRIMARY free online λ / SECONDARY recovered
blender on the Excel path. This probe surface has `dual_recovery_path=None`;
skeleton λ ≠ Case 1 PRIMARY/SECONDARY duals as dual recovery; numeric L∞ prep is
**not** dual L∞ under wire proof and **not** a VERDICT gate change.

## Offline Case-1 dual-space L∞ live-λ bridge (goal 5 + goal 3 residual)

Always-on numpy capture/bridge so dual-L∞ prep can consume **this-run** Case 1
PRIMARY online λ (and optional SECONDARY recovered diagnostic) instead of
fixture-only defaults. Composes the existing probe — does **not** re-implement
L∞ math. Does **not** import `excel_pipeline` on the TF hot path.

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_case1_dual_space_linf_live_lambda_bridge_report,
    extract_case1_primary_online_lambda,
    case1_primary_online_lambda_from_mapping,
)

# From a Case 1 comparison package (e.g. demo report / results JSON shape):
bridge = offline_case1_dual_space_linf_live_lambda_bridge_report(
    case1_package=report,  # has report["admm"]["online_duals"]
    skeleton_n_rounds=1,
)
assert bridge["kind"] == "offline_case1_dual_space_linf_live_lambda_bridge"
assert bridge["live_lambda_source"] in (
    "caller_supplied", "package_extract", "fixture"
)
assert bridge["dual_recovery_path"] is None
assert bridge["dual_linf_under_wire_status"] == "unproven"
assert bridge["online_linf_gate_under_tf_path"] == "open"
assert bridge["wire_shipped"] is False
assert bridge["bridge_is_not_verdict_gate"] is True
# bridge_ok never requires linf<=15; never VERDICT:
assert "NOT linf<=15" in bridge["ok_criteria"] or "not" in bridge["ok_criteria"].lower()
```

| Field | Meaning |
|-------|---------|
| `live_lambda_source` | `caller_supplied` \| `package_extract` \| `fixture` — always labeled |
| `bridge_ok` | extract ∧ source documented ∧ probe honesty — **not** `linf≤15` |
| `dual_linf_under_wire_status` | Always `unproven` — bridge ≠ proof under wire |
| `online_linf_gate_under_tf_path` | Remains `open` |
| `fixture_is_not_live` | Fixture fallback never claimed as this-run live duals |
| `secondary_recovered_is_not_gate` | Optional SECONDARY face is diagnostic only |

Additive readiness flag `admm_case1_dual_space_linf_live_lambda_bridge_ok` does
**not** redefine `ready_for_wire_discussion`. Excel static packaging of live-λ
bridge readiness exists (`tf_offline_case1_dual_space_linf_live_lambda_bridge`
How_to + Index/Summary/meta/Calc_Check/demo; source-must-be-labeled;
`offline_tf_case1_dual_space_linf_live_lambda_bridge_ready`; dual_linf unproven;
bridge ≠ VERDICT; bridge ≠ wire proof; dual-ban; wire_shipped=False; no live
excel→tf bridge call). Demo may still print a post-solve diagnostic line only —
never gates VERDICT, never writes live L∞ into the workbook.

**Dual honesty:** Case 1 duals still owned by Excel PRIMARY online λ path.
Extracted vectors are **probe inputs only** (`dual_recovery_path=None` on TF
surface). Bridge L∞ is diagnostic prep, not dual L∞ under wire proof.

## Offline Case-1 dual-space L∞ live-λ-seeded warm-start / dual_linf proof-prep

Always-on compose (no TF, no PuLP, no excel_pipeline on hot path):

```python
from pims_admm_llm.models.tf_linear_blocks import (
    offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report,
)

warm = offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
    case1_primary_online_lambda={"naphtha": -1.0, "distillate": -2.0,
                                 "gasoil": -3.0, "residue": -4.0},
    n_rounds=2,
)
assert warm["kind"] == "offline_case1_dual_space_linf_live_lambda_seeded_warmstart"
assert warm["seed_policy"] == "lambda0_from_live_primary_online"
assert warm["z0_policy"] == "unchanged_default_skeleton_z"
assert warm["live_lambda_source"] in (
    "caller_supplied", "package_extract", "fixture"
)
assert warm["dual_recovery_path"] is None
assert warm["wire_shipped"] is False
assert warm["dual_linf_under_wire_status"] == "unproven"  # even if L∞ 0 or ≤15
assert warm["warmstart_is_not_verdict_gate"] is True
assert warm["seed_identity_linf_is_not_proof"] is True
# warmstart_ok never requires linf<=15
```

| Field | Meaning |
|-------|---------|
| `live_lambda_source` | `caller_supplied` \| `package_extract` \| `fixture` — always labeled |
| `seed_policy` / `z0_policy` | Documented seed (λ0 from live PRIMARY; z0 default skeleton — no plant MB) |
| `linf_at_seed` | Seed-identity L∞ (often ~0); **not** dual L∞ under wire proof |
| `linf_post_rounds` / `linf` | Primary proof-prep metric after N seeded skeleton rounds |
| `warmstart_ok` | extract ∧ source ∧ seed ∧ rounds ∧ dual-ban — **not** linf≤15; **not** VERDICT |
| `dual_linf_under_wire_status` | Always `unproven` on this surface |
| `fixture_is_not_live` | Fixture fallback never claimed as this-run live duals |

Additive readiness flag
`admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok` does **not**
redefine `ready_for_wire_discussion`. Excel static packaging of warm-start
readiness is present (isolation-safe How_to/Index/Summary/meta/Calc_Check twin;
dual_linf still unproven; not VERDICT; not wire; seed identity ≠ proof; source +
seed_policy labeled; no live excel→tf warm-start call). Demo may print a
post-VERDICT diagnostic line only — never gates VERDICT, never writes live
warm-start L∞ into the workbook.

**Dual honesty:** Case 1 duals still owned by Excel PRIMARY online λ path.
Seeded / post-round skeleton λ are **probe inputs only** (`dual_recovery_path=None`).
Warm-start L∞ is dual_linf **proof-prep**, not dual L∞ under wire proof.
Seed identity L∞≈0 is never dual L∞ under wire proof.

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
- [x] Excel static packaging mentions plant-named mode (`tf_offline_admm_plant_named_linking` How_to + Index/Summary/meta/Calc_Check/demo; dual_recovery_path=None; not full plant MB; not wire; synthetic plant-linking packaging still present; no live excel→tf plant-named call)
- [x] `offline_wire_preflight_report()` documents machine-readable `wire_blockers` (isolation rewrite, form label change, dual L∞ under wire unproven, Case 1 CDU+Blender shape, no blender affine kernel, wire_not_shipped, …); `wire_shipped=False`; dual-ban; does **not** redefine `ready_for_wire_discussion`; preflight ≠ wire shipped
- [x] `offline_case1_shaped_cdu_blender_linking_report()` ok (Case-1-shaped CDU↔Blender offline skeleton; dual-ban; wire_shipped=False; blender linear_quality_pooling ≠ affine kernel; skeleton λ ≠ Case 1 duals; not form flip; does **not** clear wire_blockers; no residual-must-vanish; not full plant MB) — **still not wire**
- [x] `offline_case1_dual_space_form_contract_report()` ok (planned TF-aware form registered and **distinct** from classic; form_unchanged; stream map naphtha/distillate/gasoil/residue ↔ skeleton λ; dual_linf_under_wire=unproven + open checklist; dual-ban; wire_shipped=False; blockers still documented; does **not** redefine ready; does **not** clear blockers) — **still not wire / not form flip / not dual L∞ proven**
- [x] Excel static packaging of dual-space/form contract (`tf_offline_case1_dual_space_form_contract` How_to + Index/Summary/meta/Calc_Check/demo; form_current classic vs form_planned registered; dual_linf unproven; dual_recovery_path=None; wire_shipped=False; blockers non-empty; no live excel→tf contract call) — **still not wire / not form flip**
- [x] `offline_case1_dual_space_linf_probe_report()` ok (stream-aligned numeric L∞ fixture/supplied Case 1 PRIMARY online λ vs skeleton final_lam; dual_linf_under_wire stays unproven; checklist online_linf_gate_under_tf_path open; probe ≠ wire proof; probe ≠ VERDICT; dual-ban; wire_shipped=False; blockers still documented; does **not** redefine ready) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] Excel static packaging of dual-space L∞ probe readiness (`tf_offline_case1_dual_space_linf_probe` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_dual_space_linf_probe_ready`; dual_linf unproven; online_linf_gate open; probe ≠ VERDICT; probe ≠ wire proof; dual_recovery_path=None; wire_shipped=False; blockers non-empty; no live excel→tf probe call) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] `offline_case1_dual_space_linf_live_lambda_bridge_report()` ok (pure extract/normalize this-run Case 1 PRIMARY online λ → existing probe; `live_lambda_source` labeled; dual-ban; dual_linf unproven; online_linf_gate open; bridge_ok ≠ linf≤15; bridge ≠ VERDICT; bridge ≠ wire proof; dual_recovery_path=None; wire_shipped=False; blockers still documented; does **not** redefine ready; no excel_pipeline on TF hot path) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] Excel static packaging of dual-space L∞ live-λ bridge readiness (`tf_offline_case1_dual_space_linf_live_lambda_bridge` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_dual_space_linf_live_lambda_bridge_ready`; live_lambda_source must be labeled; dual_linf unproven; online_linf_gate open; bridge ≠ VERDICT; bridge ≠ wire proof; dual_recovery_path=None; wire_shipped=False; blockers non-empty; no live excel→tf bridge call) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] `offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report()` ok (seed Case-1-shaped skeleton λ0 from live/caller PRIMARY; source labeled; seed_policy + z0_policy documented; N skeleton rounds; post-round stream L∞ + linf_at_seed seed-identity-not-proof; dual-ban; dual_linf unproven **always** even if L∞ 0 or ≤15; online_linf_gate open; warmstart_ok ≠ linf≤15; warm-start ≠ VERDICT; warm-start ≠ wire proof; dual_recovery_path=None; wire_shipped=False; blockers still documented; does **not** redefine ready; no excel_pipeline on TF hot path) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] `offline_case1_honest_blender_pooling_path_report()` ok (linear_quality_pooling documented; dual-ban; not affine UNITS; checklist blender item = honest_pooling_path_present; no_blender_offline_affine_kernel still true; dual_linf unproven; not wire; not form flip; additive readiness flag does not redefine ready; no excel_pipeline on TF hot path) — **still not dual L∞ proven under wire / not form flip / not wire / not affine kernel shipped**
- [x] `offline_case1_online_linf_gate_criteria_contract_report()` ok (machine-readable flip criteria for `online_linf_gate_under_tf_path`; **gate stays open**; `gate_flip_allowed_today=False`; `criteria_met_today=False`; dual_linf unproven; dual-ban; contract ≠ gate flip ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof; linf≤15 not required for ok and not flip criterion today; blockers still documented; form classic; UNITS FCC/COKER/CDU; additive readiness flag does not redefine ready; no excel_pipeline on TF hot path) — **still not dual L∞ proven under wire / not form flip / not wire / not gate flipped**
- [x] Excel static packaging of dual-space L∞ live-λ-seeded warm-start readiness (`tf_offline_case1_dual_space_linf_live_lambda_seeded_warmstart` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_dual_space_linf_live_lambda_seeded_warmstart_ready`; seed_policy + z0_policy; live_lambda_source must be labeled; seed identity ≠ proof; dual_linf unproven; online_linf_gate open; warm-start ≠ VERDICT; warm-start ≠ wire proof; dual_recovery_path=None; wire_shipped=False; blockers non-empty; no live excel→tf warm-start call) — **still not dual L∞ proven under wire / not form flip / not wire**
- [x] Excel static packaging of honest Case-1 blender pooling path readiness (`tf_offline_case1_honest_blender_pooling_path` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_honest_blender_pooling_path_ready`; linear_quality_pooling; checklist honest_pooling_path_present; open-ids realigned (blender no longer open); not affine kernel; dual-ban; dual_linf unproven; not VERDICT; not wire; no_blender_offline_affine_kernel still true; no live excel→tf pooling call) — **still not dual L∞ proven under wire / not form flip / not wire / not affine kernel shipped**
- [x] Excel static packaging of online_linf_gate flip-criteria contract readiness (`tf_offline_case1_online_linf_gate_criteria_contract` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_online_linf_gate_criteria_contract_ready`; gate stays open; gate_flip_allowed_today=false; criteria_met_today=false; dual-ban; dual_linf unproven; not VERDICT; not wire; not gate flip; no live excel→tf criteria call; Index ≤1439) — **still not dual L∞ proven under wire / not form flip / not wire / not gate flipped**
- [x] `offline_case1_isolation_rewrite_design_contract_report()` ok (design_present; **rewrite_shipped=False**; isolation checklist stays **open**; isolation_rewrite_required still in blockers; dual_linf unproven; online_linf_gate open; gate_flip/criteria_met false; dual-ban; design ≠ rewrite shipped ≠ wire ≠ VERDICT; additive readiness flag does not redefine ready; isolation suite still classic gates; no excel_pipeline on TF hot path) — **still not isolation rewrite shipped / not dual L∞ proven under wire / not form flip / not wire**
- [x] Excel static packaging twin of isolation-rewrite design contract readiness (`tf_offline_case1_isolation_rewrite_design_contract` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_isolation_rewrite_design_contract_ready`; design_present; rewrite_shipped=false; isolation checklist open; dual-ban; dual_linf unproven; not VERDICT; not wire; not rewrite shipped; isolation_rewrite_required still in blockers; no live excel→tf design call; Index ≤1439) — **still not isolation rewrite shipped / not dual L∞ proven under wire / not form flip / not wire**
- [x] `offline_case1_wire_ship_acceptance_design_contract_report()` ok (design_present; **wire_ship_allowed_today=False**; **wire_shipped=False**; wire_ship_criteria_met_today=False; dual_linf unproven; form classic; isolation rewrite not shipped; isolation checklist open; online_linf_gate open; gate_flip/criteria_met false; dual-ban; design ≠ ship allow ≠ wire ≠ VERDICT; full DEFAULT_WIRE_BLOCKERS remain; additive readiness flag does not redefine ready; isolation suite still classic gates; no excel_pipeline on TF hot path) — **still not wire shipped / not ship allow / not dual L∞ proven under wire / not form flip / not isolation rewrite shipped**
- [x] Excel static packaging twin of wire-ship acceptance design contract readiness (`tf_offline_case1_wire_ship_acceptance_design_contract` How_to + Index/Summary/meta/Calc_Check/demo; `offline_tf_case1_wire_ship_acceptance_design_contract_ready`; design_present; wire_ship_allowed_today=false; wire_shipped=false; dual-ban; dual_linf unproven; not VERDICT; not wire; not ship allow; isolation rewrite not shipped; blockers still documented; no live excel→tf design call; Index ≤1439) — **still not wire shipped / not ship allow / not dual L∞ proven under wire / not form flip / not isolation rewrite shipped**
- [x] `offline_case1_dual_honest_tf_aware_path_design_contract_report()` ok (path_design_present; **path_shipped=False**; **dual_honest_tf_aware_path_present ship-met remains False**; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; design ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag does not redefine ready; no excel_pipeline on TF hot path) — **still not path shipped / not ship-met / not wire**
- [x] Excel static packaging twin of path design contract readiness (`tf_offline_case1_dual_honest_tf_aware_path_design_contract` How_to + Index/Summary/meta/Calc_Check/demo; path design present; path_shipped=false; ship-met=false; dual-ban; dual_linf unproven; not VERDICT; not wire; Index ≤1439) — **still not path shipped / not ship-met / not wire**
- [x] `offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report()` ok (criteria_present; **ship_met_allowed_today=False**; **criteria_met_today=False**; **dual_honest_tf_aware_path_present ship-met remains False**; path_design_present=True; path_shipped=False; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; criteria ≠ ship-met ≠ path shipped ≠ wire ≠ VERDICT; dual-ban; additive readiness flag does not redefine ready; blockers remain; no excel_pipeline on TF hot path) — **still not ship-met / not path shipped / not wire**
- [x] `offline_case1_form_label_change_shipped_criteria_contract_report()` ok (criteria_present; **form_label_ship_allowed_today=False**; **criteria_met_today=False**; **form_label_change_shipped remains False**; form classic; path_design_present=True; path_shipped=False; ship-met False; wire_shipped=False; dual_linf unproven; isolation rewrite not shipped; online_linf_gate open; form registration ≠ form_label shipped ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag does not redefine ready; blockers remain incl. form_label_change_required; checklist form_label open; no excel_pipeline on TF hot path) — **still not form_label shipped / not form flip / not path shipped / not ship-met / not wire**
- [x] `offline_case1_isolation_rewrite_shipped_criteria_contract_report()` ok (criteria_present; **isolation_ship_allowed_today=False**; **criteria_met_today=False**; **isolation_rewrite_shipped remains False**; checklist isolation_rewrite_with_wire **open**; rewrite-not-delete; form classic; path_shipped=False; ship-met False; form_label_change_shipped False; wire_shipped=False; dual_linf unproven; online_linf_gate open; isolation design ≠ isolation rewrite shipped ≠ form ship ≠ path ship ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; additive readiness flag does not redefine ready; blockers remain incl. isolation_rewrite_required; isolation suite unchanged; no excel_pipeline on TF hot path) — **still not isolation rewrite shipped / not form flip / not path shipped / not ship-met / not wire**
- [x] Excel static packaging twin of isolation-rewrite ship-met / flip criteria (`tf_offline_case1_isolation_rewrite_shipped_criteria_contract` How_to + Index/Summary/meta/Calc_Check/demo; criteria_present; isolation_ship_allowed=false; isolation_rewrite_shipped=false; checklist open; rewrite-not-delete; dual-ban; dual_linf unproven; not VERDICT; not wire; not isolation rewrite shipped; Index ≤1439; no live excel→tf criteria call) — **still not isolation rewrite shipped / not form flip / not path shipped / not ship-met / not wire**
- [x] `offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report()` ok (bundle_design_present; **bundle_shipped=False**; **bundle_ship_allowed_today=False**; criteria_met_today=False; wire_shipped=False; isolation_rewrite_shipped=False; form classic; dual_linf unproven; order_hint not executor; dual-ban; additive readiness flag does not redefine ready; blockers remain; no excel_pipeline on TF hot path) — **still not bundle shipped / not wire / not VERDICT**
- [x] Excel static packaging twin of multi-blocker wire bundle design (`tf_offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract` How_to + Index/Summary/meta/Calc_Check/demo; bundle_design_present; bundle_shipped=false; bundle_ship_allowed=false; dual-ban; dual_linf unproven; not VERDICT; not wire; Index ≤1439; no live excel→tf design call) — **still not bundle shipped / not wire / not VERDICT**
- [x] Excel static packaging twin of multi-blocker wire bundle ship-met / flip criteria (`tf_offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract` How_to + Index design+crit ≤1439 + Summary/meta/Calc_Check/demo; criteria_present; bundle_shipped=false; bundle_ship_allowed=false; criteria_met=false; dual-ban; dual_linf unproven; not VERDICT; not wire; not bundle ship; no live excel→tf criteria call) — **still not bundle shipped / not wire / not isolation rewrite shipped / not form flip / not path ship / not VERDICT**
- [x] `offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()` ok (criteria_present; **bundle_shipped=False**; **bundle_ship_allowed_today=False**; **criteria_met_today=False**; wire_shipped=False; isolation_rewrite_shipped=False; isolation checklist open; form classic; form_label_change_shipped False; path_shipped=False; path ship-met False; dual_linf unproven; online_linf_gate open; order_hint not executor; dual-ban; additive readiness flag does not redefine ready; blockers remain; isolation suite unchanged; no excel_pipeline on TF hot path) — **still not bundle shipped / not wire / not isolation rewrite shipped / not form flip / not path ship / not VERDICT** Excel packaging twin of *criteria* shipped: `tf_offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract` How_to + Index design+crit ≤1439 + Summary/meta/Calc_Check/demo; criteria_present; ship/allow/met false; dual-ban; dual_linf unproven; not VERDICT; not wire; not bundle ship; Index ≤1439; no live excel→tf criteria call
- [x] `offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()` ok (scaffold_present / execution_scaffold_present; **path_shipped=False**; **dual_honest_tf_aware_path_present ship-met False**; **wire_shipped=False**; **bundle_shipped=False**; bundle_ship_allowed_today=False; criteria_met_today=False; isolation_rewrite_shipped=False; isolation checklist open; form classic; form_label_change_shipped=False; dual_linf unproven; online_linf_gate open; feature flag reserved False; dual_recovery_path=None; order_hint not executor; callable compose CDU offline affine + blender linear_quality_pooling + Case-1 streams + optional labeled λ + diagnostic-only dual-space residual; dual-ban; additive readiness `admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok` does not redefine ready; blockers remain; isolation suite unchanged; no excel_pipeline on TF hot path) — **still not path shipped / not ship-met / not wire / not bundle ship / not isolation rewrite shipped / not form flip / not dual L∞ proven / not VERDICT**
- [x] Excel static packaging twin of dual-honest TF-aware path execution scaffold (`tf_offline_case1_dual_honest_tf_aware_path_execution_scaffold` How_to + Index deep-compress ≤1439 + Summary/meta/Calc_Check/demo; `offline_tf_case1_dual_honest_tf_aware_path_execution_scaffold_ready`; scaffold_present / execution_scaffold_present packaging true; path/wire/bundle/isolation/form ship hard false; dual-ban; dual_linf unproven; dual_recovery_path=None; no live excel→tf scaffold call; Index ≤1439) — **still not path shipped / not ship-met / not wire / not bundle ship / not isolation rewrite shipped / not form flip / not dual L∞ proven / not VERDICT**
- [x] `offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()` ok (rehearsal_present / wire_rehearsal_present; scaffold_present / execution_scaffold_present linked; machine-readable co-req status matrix for SUGGESTED_NEXT_WAVE; **path_shipped=False**; **dual_honest_tf_aware_path_present ship-met False**; **wire_shipped=False**; **bundle_shipped=False**; bundle_ship_allowed_today=False; criteria_met_today=False; isolation_rewrite_shipped=False; isolation checklist open; form classic; form_label_change_shipped=False; dual_linf unproven; online_linf_gate open; feature flag reserved False; dual_recovery_path=None; order_hint not executor; dual-ban; additive readiness `admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok` does not redefine ready; blockers remain; isolation suite unchanged; no excel_pipeline/pulp/tensorflow on rehearsal hot path) — **still not path shipped / not ship-met / not wire / not bundle ship / not isolation rewrite shipped / not form flip / not dual L∞ proven / not VERDICT**.
- [x] Excel static packaging twin of dual-honest multi-blocker wire rehearsal (`tf_offline_case1_dual_honest_multi_blocker_wire_rehearsal` How_to + Index scaffold+rehearse ≤1439 + Summary/meta/Calc_Check/demo; `offline_tf_case1_dual_honest_multi_blocker_wire_rehearsal_ready`; rehearsal_present / wire_rehearsal_present packaging true; scaffold linked; co-req dry-run static labels; path/wire/bundle/isolation/form ship hard false; dual-ban; dual_linf unproven; dual_recovery_path=None; no live excel→tf rehearsal call; Index ≤1439) — **still not path shipped / not ship-met / not wire / not bundle ship / not isolation rewrite shipped / not form flip / not dual L∞ proven / not VERDICT**. Next residual after packaging: multi-blocker full wire *execution* long-term (not form flip alone / not isolation rewrite-as-ship alone / not design/criteria/scaffold/rehearsal re-ship).

- [ ] Dual honesty PRIMARY online λ still gates VERDICT (online L∞ ≤15); do not retune ρ solely to shrink recovered dual L∞
- [ ] Explicit form label change **shipped** (not merely registered / criteria-formalized): `classic_2block_excel_path` → `tf_affine_cdu_blender_shaped_excel_path` when wire lands (never silent form reuse). Planned form is **registered** by the dual-space/form contract; flip criteria for *when shipped* are formalized by the form_label_change_shipped criteria contract above (form_label_ship_allowed_today still False).
- [ ] Isolation tests (`test_tf_import_isolation.py`) must be **rewritten with** the wire — not silently broken or deleted (design-only contract documents rewrite-not-delete; ship-met flip criteria formalize *when* rewrite may count as shipped; rewrite itself still not shipped)
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
- [ ] Pre-wire dual-L∞ proof checklist present (this doc + `offline_case1_dual_space_form_contract_report`); wire not claimed as shipped
- [ ] Priced residual pre-wire gate present; local box gradients not claimed as duals
- [ ] Timing / readiness report honesty: dual_recovery_path=None; on_excel_case1_path=False; no flaky absolute µs hard-fail; timings not duals / not Case 1 wall time
- [ ] `cached_offline_unit_coeffs` default-ref only; custom refs never silently reuse default cache
- [ ] `multi_unit_admm_residual_report` ok without TF; honesty locks; L1 formula; Coker raw≠full residual honesty; synthetic λ ≠ Case 1 online λ
- [ ] `multi_unit_admm_block_subproblem_report` ok without TF; raw optimand; maximizer ≥ ref; delta=0 ⇒ x_star≈x0; dual-ban; not wire; Coker raw≠full diagnostic
- [ ] `multi_unit_admm_coordination_report` ok without TF; honesty locks; trajectory finite; z_pre dual ascent; per-unit synthetic scope; no residual-must-vanish SLA; not wire
- [ ] `multi_block_plant_linking_admm_report` ok without TF; synthetic topology (default) + plant-named mode; shared linking λ/z; pre-z linking residual dual ascent; compose subproblem; not full plant MB; plant-linking λ ≠ Case 1 duals; dual_recovery_path=None; no residual-must-vanish SLA; existing coordination still `not_plant_linking_coordinator=True`; additive `admm_plant_named_linking_ok` does not redefine `ready_for_wire_discussion`
- [ ] `offline_wire_preflight_report` ok without TF; wire_blockers non-empty with critical honesty ids; wire_shipped=False; dual_recovery_path=None; ready_for_wire_discussion meaning unchanged; preflight_ok/blockers_documented separate from ready; not full plant MB; not wire shipped
- [ ] `offline_case1_shaped_cdu_blender_linking_report` ok without TF; dual-ban; wire_shipped=False; blender_surface=linear_quality_pooling; Case 1 intermediate streams; UNITS still FCC/COKER/CDU; blockers still true; additive `admm_case1_shaped_linking_ok` does not redefine ready; not residual-must-vanish; not full plant MB; not wire
- [ ] `offline_case1_dual_space_form_contract_report` ok without TF; form current classic + planned distinct; stream_alignment_ok; dual_linf unproven + open checklist; dual_recovery_path=None; wire_shipped=False; blockers still documented; additive `admm_case1_dual_space_form_contract_ok` does not redefine ready; not form flip; not dual L∞ proven; not wire
- [ ] `offline_case1_dual_space_linf_probe_report` ok without TF; stream-aligned finite L∞; dual_linf unproven; checklist online_linf_gate open; probe_ok ≠ linf≤15; probe ≠ VERDICT; dual_recovery_path=None; wire_shipped=False; blockers still documented; additive `admm_case1_dual_space_linf_probe_ok` does not redefine ready; not dual L∞ proven under wire; not wire
- [ ] Excel static dual-space/form contract packaging present (`tf_offline_case1_dual_space_form_contract`; `offline_tf_case1_dual_space_form_contract_ready`; Calc_Check not-duals/not-wire); isolation-safe (no live excel→tf); Case 1 form unchanged; blockers non-empty; lean ≤15
- [ ] Excel static dual-space L∞ probe packaging present (`tf_offline_case1_dual_space_linf_probe`; `offline_tf_case1_dual_space_linf_probe_ready`; Calc_Check not-duals/not-wire/not-verdict); isolation-safe (no live excel→tf probe); dual_linf unproven; online_linf_gate open; Case 1 form unchanged; blockers non-empty; lean ≤15
- [ ] `offline_case1_honest_blender_pooling_path_report` ok without TF; blender_surface=linear_quality_pooling; checklist honest_pooling_path_present (not bare open; not closed_via_affine); dual_linf unproven; dual_recovery_path=None; wire_shipped=False; no_blender_offline_affine_kernel still true; UNITS FCC/COKER/CDU; additive `admm_case1_honest_blender_pooling_path_ok` does not redefine ready; not form flip; not dual L∞ proven under wire; not wire; not affine kernel shipped
- [ ] `offline_case1_online_linf_gate_criteria_contract_report` ok without TF; flip-criteria map present (required + required_under_wire_only); online_linf_gate stays open; gate_flip_allowed_today=False; criteria_met_today=False; dual_linf unproven; dual_recovery_path=None; wire_shipped=False; blockers still documented; form classic; UNITS FCC/COKER/CDU; additive `admm_case1_online_linf_gate_criteria_contract_ok` does not redefine ready; contract ≠ gate flip ≠ wire ≠ VERDICT; not dual L∞ proven under wire
- [ ] `offline_case1_isolation_rewrite_design_contract_report` ok without TF; design_present; rewrite_shipped=False; isolation_rewrite checklist stays open; isolation_rewrite_required still in blockers; dual_linf unproven; online_linf_gate open; gate_flip/criteria_met false; dual_recovery_path=None; wire_shipped=False; form classic; UNITS FCC/COKER/CDU; additive `admm_case1_isolation_rewrite_design_contract_ok` does not redefine ready; design ≠ rewrite shipped ≠ wire ≠ VERDICT; isolation suite behavior unchanged
- [ ] `offline_case1_wire_ship_acceptance_design_contract_report` ok without TF; design_present; wire_ship_allowed_today=False; wire_ship_criteria_met_today=False; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; isolation checklist open; online_linf_gate open; gate_flip/criteria_met false; dual_recovery_path=None; blockers still documented (incl. isolation_rewrite_required, no_blender, wire_not_shipped); UNITS FCC/COKER/CDU; additive `admm_case1_wire_ship_acceptance_design_contract_ok` does not redefine ready; design ≠ ship allow ≠ wire shipped ≠ VERDICT; isolation suite behavior unchanged
- [ ] `offline_case1_dual_honest_tf_aware_path_design_contract_report` ok without TF; path_design_present; path_shipped=False; dual_honest_tf_aware_path_present ship-met False; wire_shipped=False; dual_linf unproven; form classic; dual_recovery_path=None; blockers still documented; UNITS FCC/COKER/CDU; additive `admm_case1_dual_honest_tf_aware_path_design_contract_ok` does not redefine ready; design ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT
- [ ] `offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report` ok without TF; criteria map present; ship_met_allowed_today=False; criteria_met_today=False; dual_honest_tf_aware_path_present ship-met False; path_design_present=True; path_shipped=False; wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not shipped; online_linf_gate open; dual_recovery_path=None; blockers still documented; UNITS FCC/COKER/CDU; additive `admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok` does not redefine ready; criteria ≠ ship-met ≠ path shipped ≠ wire ≠ VERDICT; isolation suite behavior unchanged
- [ ] `offline_case1_form_label_change_shipped_criteria_contract_report` ok without TF; criteria map present; form_label_ship_allowed_today=False; criteria_met_today=False; form_label_change_shipped False; form classic; path_design_present=True; path_shipped=False; ship-met False; wire_shipped=False; dual_linf unproven; isolation rewrite not shipped; online_linf_gate open; dual_recovery_path=None; blockers still documented incl. form_label_change_required; checklist form_label open; UNITS FCC/COKER/CDU; additive `admm_case1_form_label_change_shipped_criteria_contract_ok` does not redefine ready; criteria ≠ form_label shipped ≠ form flip ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; isolation suite behavior unchanged
- [ ] `offline_case1_isolation_rewrite_shipped_criteria_contract_report` ok without TF; criteria map present; isolation_ship_allowed_today=False; criteria_met_today=False; isolation_rewrite_shipped False; isolation checklist open; rewrite-not-delete; form classic; path_shipped=False; ship-met False; form_label_change_shipped False; wire_shipped=False; dual_linf unproven; online_linf_gate open; dual_recovery_path=None; blockers still documented incl. isolation_rewrite_required; UNITS FCC/COKER/CDU; additive `admm_case1_isolation_rewrite_shipped_criteria_contract_ok` does not redefine ready; criteria ≠ isolation rewrite shipped ≠ form flip ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT; isolation suite behavior unchanged
- [ ] `offline_case1_dual_honest_tf_aware_path_execution_scaffold_report` ok without TF; scaffold_present; path_shipped=False; path present ship-met False; wire_shipped=False; bundle_shipped=False; isolation_rewrite_shipped=False; form classic; form_label_change_shipped=False; dual_linf unproven; dual_recovery_path=None; feature flag false; UNITS FCC/COKER/CDU; blockers still documented; additive `admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok` does not redefine ready; scaffold ≠ path ship ≠ wire ≠ bundle ship ≠ isolation rewrite shipped ≠ form ship ≠ VERDICT; Excel packaging twin present (How_to + Index ≤1439 + meta/Calc_Check/demo; ship flags hard false); isolation suite unchanged
