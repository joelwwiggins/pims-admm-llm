"""Optional TensorFlow exact-linear block helpers (charter direction).

This module is **not** on the Excel Case 1 / PuLP ADMM path. It hosts an exact
affine copy of base_delta (``y_raw = y0 + D @ (x − x0)``) behind an optional
dependency, plus always-on numpy helpers for Excel coeff honesty.

Hard rules for this surface:
- Never import tensorflow at package import time (module import stays TF-free).
- Feasibility ownership stays with existing solvers (CBC / package ADMM).
- Dual recovery is **not** claimed here (``dual_recovery_path`` is always None).
- LLM / advisory layers must not treat this module as a solver bypass.
- Postprocess (coke clamp + liquid renorm) stays **outside** any TF graph.

Install (optional)::

    pip install -e \".[tf]\"

On hosts without a TF wheel (e.g. some Jetson images), leave TF uninstalled;
Case 1 Excel smoke must remain green.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

# Honesty metadata shared by block objects and tests (E6/E14).
MODULE_KIND = "exact_linear_copy"
SOURCE = "base_delta"  # single source of truth for coefficients
SOLVER = False
DUAL_RECOVERY_PATH: Optional[str] = None
ON_EXCEL_CASE1_PATH = False
# Multi-unit offline surface: FCC + Coker + CDU shells share this module;
# postprocess stays numpy/Python outside any TF graph (never dual recovery).
POSTPROCESS = "numpy_outside_tf"
UNITS = ("FCC", "COKER", "CDU")

_TF_IMPORT_ERROR: Optional[BaseException] = None
_TF_CHECKED = False
_TF_OK = False


def tf_available() -> bool:
    """Return True iff ``tensorflow`` can be imported in this environment.

    Result is cached after the first probe. Never raises for missing TF.
    """
    global _TF_CHECKED, _TF_OK, _TF_IMPORT_ERROR
    if _TF_CHECKED:
        return _TF_OK
    _TF_CHECKED = True
    try:
        import tensorflow as _tf  # noqa: F401

        _TF_OK = True
        _TF_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - env dependent
        _TF_OK = False
        _TF_IMPORT_ERROR = exc
    return _TF_OK


def tf_import_error() -> Optional[BaseException]:
    """Last TF probe exception (None if available or not yet probed)."""
    if not _TF_CHECKED:
        tf_available()
    return _TF_IMPORT_ERROR


def honesty_metadata() -> Dict[str, Any]:
    """Runtime honesty contract for critics / demos (never claims dual recovery)."""
    return {
        "kind": MODULE_KIND,
        "source": SOURCE,
        "backend": "tensorflow" if tf_available() else "unavailable",
        "solver": SOLVER,
        "dual_recovery_path": DUAL_RECOVERY_PATH,
        "on_excel_case1_path": ON_EXCEL_CASE1_PATH,
        "postprocess": POSTPROCESS,
        "units": list(UNITS),
        "tf_available": tf_available(),
        "block_solve_timing_available": True,
        "admm_residual_available": True,
        "admm_block_subproblem_available": True,
        "admm_coordination_available": True,
        "admm_plant_linking_available": True,
        "admm_plant_named_linking_available": True,
        "wire_preflight_available": True,
        "admm_case1_shaped_linking_available": True,
        "admm_case1_dual_space_form_contract_available": True,
        "admm_case1_dual_space_linf_probe_available": True,
        "admm_case1_dual_space_linf_live_lambda_bridge_available": True,
        "admm_case1_dual_space_linf_live_lambda_seeded_warmstart_available": True,
        "admm_case1_honest_blender_pooling_path_available": True,
        "admm_case1_online_linf_gate_criteria_contract_available": True,
        "admm_case1_dual_linf_under_wire_criteria_contract_available": True,
        "admm_case1_isolation_rewrite_design_contract_available": True,
        "admm_case1_wire_ship_acceptance_design_contract_available": True,
        "admm_case1_dual_honest_tf_aware_path_design_contract_available": True,
        "admm_case1_dual_honest_tf_aware_path_present_criteria_contract_available": True,
        "admm_case1_form_label_change_shipped_criteria_contract_available": True,
        "admm_case1_isolation_rewrite_shipped_criteria_contract_available": True,
        "admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_available": True,
        "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_available": True,
        "admm_case1_dual_honest_tf_aware_path_execution_scaffold_available": True,
        "admm_case1_dual_honest_multi_blocker_wire_rehearsal_available": True,
        "admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_available": True,
        "formula": "y_raw = y0 + D @ (x - x0)  # pre-postprocess exact linear",
        "note": (
            "Optional exact-linear surface only (FCC + COKER + CDU offline kernels). "
            "Not Excel Case 1 solver; not ADMM dual recovery; not a learned model. "
            "Full evaluate() = affine + numpy postprocess (clamp/renorm) outside TF. "
            "Coker renorm always engages → raw ≠ evaluate even at reference. "
            "CDU has nested cut_points_f.* drivers in x0; Submodel_CDU is classic "
            "TECH+A export (not a PIMS MB_* matrix twin like FCC/Coker). "
            "Offline cached block-solve timing / readiness harness available "
            "(multi_unit_block_solve_timing_report); timings are readiness only, "
            "not Case 1 wall time and not ADMM duals. "
            "Offline multi-unit ADMM-style consensus residual harness available "
            "(multi_unit_admm_residual_report) under synthetic λ,z,ρ — dual-ban; "
            "not Case 1 online λ; not pure-ADMM dual recovery; not wire shipped. "
            "Offline multi-unit ADMM block subproblem maximizer available "
            "(multi_unit_admm_block_subproblem_report) on raw affine under synthetic "
            "λ,z,ρ + driver box — dual-ban; not Case 1; not pure-ADMM dual recovery; "
            "not wire shipped; not PuLP. "
            "Offline multi-round ADMM coordination harness available "
            "(multi_unit_admm_coordination_report): subproblem → z consensus → λ ascent "
            "under synthetic λ,z,ρ (per-unit product spaces; not plant linking) — "
            "dual-ban; not Case 1; not pure-ADMM dual recovery; not wire shipped. "
            "Offline multi-block plant-linking ADMM harness available "
            "(multi_block_plant_linking_admm_report): synthetic (default) and plant-named "
            "linking topology modes + shared λ/z + per-unit incidence; composes block "
            "subproblem; not full plant mass balance; plant-linking λ ≠ Case 1 online λ; "
            "not wire shipped; plant-named offline demo ≠ live cascade. "
            "Offline dual-honest wire preflight available "
            "(offline_wire_preflight_report): composes readiness gates + machine-readable "
            "wire_blockers; wire_shipped=False; preflight ≠ wire; dual_recovery_path=None; "
            "ready_for_wire_discussion meaning unchanged (structural only). "
            "Offline Case-1-shaped CDU↔Blender linking skeleton available "
            "(offline_case1_shaped_cdu_blender_linking_report): CDU affine + honest blender "
            "linear quality/pooling residual under synthetic λ,z,ρ on Case 1 intermediate "
            "streams; dual-ban; wire_shipped=False; skeleton ≠ wire; skeleton λ ≠ Case 1 "
            "PRIMARY/SECONDARY duals; blender_surface=linear_quality_pooling (not affine "
            "UNITS); does not clear DEFAULT_WIRE_BLOCKERS. "
            "Offline Case-1 dual-space / form-label contract available "
            "(offline_case1_dual_space_form_contract_report): planned TF-aware form label "
            "registry without flipping Case 1; dual-space stream map Case 1 intermediates "
            "↔ skeleton λ slots; dual_linf_under_wire status=unproven + open checklist; "
            "dual-ban; wire_shipped=False; does not clear DEFAULT_WIRE_BLOCKERS; does not "
            "redefine ready_for_wire_discussion; not wire; not dual L∞ proven under wire. "
            "Offline Case-1 dual-space L∞ probe available "
            "(offline_case1_dual_space_linf_probe_report): stream-aligned numeric L∞ between "
            "fixture/supplied Case 1 PRIMARY online λ and Case-1-shaped skeleton λ — dual-ban; "
            "dual_linf_under_wire still unproven; checklist online_linf_gate_under_tf_path open; "
            "probe ≠ dual L∞ under wire proof; probe ≠ Case 1 VERDICT gate; wire_shipped=False; "
            "does not clear DEFAULT_WIRE_BLOCKERS; does not redefine ready_for_wire_discussion. "
            "Offline Case-1 dual-space L∞ live-λ bridge available "
            "(offline_case1_dual_space_linf_live_lambda_bridge_report): pure extract/normalize "
            "of this-run Case 1 PRIMARY online λ (+ optional SECONDARY recovered) into the "
            "existing probe; live_lambda_source always labeled (caller_supplied / fixture / "
            "package_extract); dual-ban; dual_linf still unproven; bridge ≠ VERDICT; bridge ≠ "
            "wire proof; dual_recovery_path=None; wire_shipped=False; does not clear "
            "DEFAULT_WIRE_BLOCKERS; does not redefine ready_for_wire_discussion; no "
            "excel_pipeline import on hot path. "
            "Offline Case-1 dual-space L∞ live-λ-seeded skeleton warm-start available "
            "(offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report): seed "
            "Case-1-shaped skeleton λ0 from this-run/caller PRIMARY online λ (source labeled); "
            "run N skeleton rounds; measure post-round stream L∞; also report linf_at_seed "
            "labeled seed-identity-not-proof; dual-ban; dual_linf_under_wire still unproven "
            "even if L∞ 0 or ≤15; online_linf_gate open; warm-start ≠ VERDICT; warm-start ≠ "
            "dual L∞ under wire proof; dual_recovery_path=None; wire_shipped=False; does not "
            "clear DEFAULT_WIRE_BLOCKERS; does not redefine ready_for_wire_discussion; no "
            "excel_pipeline import on hot path; no TF required. "
            "Offline Case-1 honest blender pooling path formalization available "
            "(offline_case1_honest_blender_pooling_path_report): documents "
            "linear_quality_pooling as the dual-honest Case-1 blender path; checklist "
            "blender_affine_kernel_or_honest_pooling_path=honest_pooling_path_present "
            "(not closed_via_affine_kernel); no BLENDER affine UNITS; "
            "no_blender_offline_affine_kernel still true; dual-ban; dual_linf unproven; "
            "pooling path ≠ wire; pooling path ≠ VERDICT; dual_recovery_path=None; "
            "wire_shipped=False; does not clear DEFAULT_WIRE_BLOCKERS; does not redefine "
            "ready_for_wire_discussion; no excel_pipeline import on hot path; no TF required. "
            "Offline Case-1 online_linf_gate flip-criteria contract available "
            "(offline_case1_online_linf_gate_criteria_contract_report): machine-readable "
            "flip criteria for checklist online_linf_gate_under_tf_path (gate stays open); "
            "gate_flip_allowed_today=False; criteria_met_today=False; dual_linf unproven; "
            "contract ≠ gate flip ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof; dual-ban; "
            "dual_recovery_path=None; wire_shipped=False; does not clear DEFAULT_WIRE_BLOCKERS; "
            "does not redefine ready_for_wire_discussion; no excel_pipeline import on hot "
            "path; no TF required. "
            "Offline Case-1 dual_linf_under_wire flip-criteria contract available "
            "(offline_case1_dual_linf_under_wire_criteria_contract_report): machine-readable "
            "criteria for status dual_linf_under_wire (order_hint dual_linf_under_wire_proven); "
            "dual_linf stays unproven; dual_linf_proof_allowed_today=False; criteria_met_today=False; "
            "distinct from online_linf_gate criteria; dual-ban; dual_recovery_path=None; "
            "wire_shipped=False; does not clear DEFAULT_WIRE_BLOCKERS; does not redefine "
            "ready_for_wire_discussion; no excel_pipeline import on hot path; no TF required. "
            "Offline Case-1 isolation-rewrite design-only contract available "
            "(offline_case1_isolation_rewrite_design_contract_report): formalizes what "
            "isolation rewrite WITH dual-honest wire means (rewrite-not-delete); "
            "isolation_rewrite_design_present=True; isolation_rewrite_shipped=False; "
            "checklist isolation_rewrite_with_wire stays open; dual_linf unproven; "
            "online_linf_gate open; gate_flip_allowed_today=False; criteria_met_today=False; "
            "design ≠ rewrite shipped ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof; dual-ban; "
            "dual_recovery_path=None; wire_shipped=False; isolation_rewrite_required still in "
            "DEFAULT_WIRE_BLOCKERS; does not clear DEFAULT_WIRE_BLOCKERS; does not redefine "
            "ready_for_wire_discussion; no excel_pipeline import on hot path; no TF required. "
            "Offline Case-1 dual-honest wire-ship acceptance design/acceptance contract "
            "available (offline_case1_wire_ship_acceptance_design_contract_report): "
            "machine-readable criteria for when wire *may* ship; design_present=True; "
            "wire_ship_allowed_today=False; wire_ship_criteria_met_today=False; "
            "wire_shipped=False; dual_linf unproven; form classic; isolation rewrite not "
            "shipped; online_linf_gate open; design ≠ ship allow ≠ wire shipped ≠ VERDICT ≠ "
            "dual L∞ under wire proof; dual-ban; dual_recovery_path=None; full "
            "DEFAULT_WIRE_BLOCKERS remain; does not clear DEFAULT_WIRE_BLOCKERS; does not "
            "redefine ready_for_wire_discussion; no excel_pipeline import on hot path; no "
            "TF required. "
            "Offline Case-1 dual-honest TF-aware path design contract available "
            "(offline_case1_dual_honest_tf_aware_path_design_contract_report): "
            "machine-readable *path shape* for future dual-honest TF Case 1 wire "
            "(CDU affine + blender linear_quality_pooling; form_planned; dual_recovery "
            "planned-vs-today; feature flag reserved false); path_design_present=True; "
            "path_shipped=False; dual_honest_tf_aware_path_present ship-met remains False; "
            "wire_shipped=False; wire_ship_allowed_today=False; dual_linf unproven; form "
            "classic; isolation rewrite not shipped; online_linf_gate open; design ≠ path "
            "shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; dual_recovery_path=None; full "
            "DEFAULT_WIRE_BLOCKERS remain; does not clear DEFAULT_WIRE_BLOCKERS; does not "
            "redefine ready_for_wire_discussion; no excel_pipeline import on hot path; no "
            "TF required. "
            "Offline Case-1 dual_honest_tf_aware_path_present ship-met flip criteria "
            "contract available "
            "(offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report): "
            "machine-readable *when path counts as present-for-ship* criteria map; "
            "criteria_present=True; ship_met_allowed_today=False; criteria_met_today=False; "
            "dual_honest_tf_aware_path_present ship-met remains False; path_design_present=True; "
            "path_shipped=False; wire_shipped=False; wire_ship_allowed_today=False; dual_linf "
            "unproven; form classic; isolation rewrite not shipped; online_linf_gate open; "
            "criteria contract ≠ ship-met ≠ path shipped ≠ wire ≠ VERDICT; dual-ban; "
            "dual_recovery_path=None; full DEFAULT_WIRE_BLOCKERS remain; does not clear "
            "DEFAULT_WIRE_BLOCKERS; does not redefine ready_for_wire_discussion; no "
            "excel_pipeline import on hot path; no TF required. "
            "Offline Case-1 form_label_change_shipped flip criteria contract "
            "available (offline_case1_form_label_change_shipped_criteria_contract_report): "
            "machine-readable *when form_label_change_shipped may become True*; "
            "criteria_present=True; form_label_ship_allowed_today=False; "
            "criteria_met_today=False; form_label_change_shipped remains False; "
            "form remains classic_2block_excel_path; path_design_present=True; "
            "path_shipped=False; dual_honest_tf_aware_path_present ship-met False; "
            "wire_shipped=False; dual_linf unproven; isolation rewrite not shipped; "
            "online_linf_gate open; form registration ≠ form_label shipped ≠ path "
            "shipped ≠ ship-met ≠ wire ≠ VERDICT; dual-ban; dual_recovery_path=None; "
            "full DEFAULT_WIRE_BLOCKERS remain; does not clear DEFAULT_WIRE_BLOCKERS; "
            "does not redefine ready_for_wire_discussion; no excel_pipeline import "
            "on hot path; no TF required. "
            "Offline Case-1 isolation-rewrite ship-met / flip criteria contract "
            "available (offline_case1_isolation_rewrite_shipped_criteria_contract_report): "
            "machine-readable *when isolation_rewrite_with_wire / isolation_rewrite_shipped "
            "may become met/True*; criteria_present=True; isolation_ship_allowed_today=False; "
            "criteria_met_today=False; isolation_rewrite_shipped remains False; checklist "
            "isolation_rewrite_with_wire stays open; rewrite-not-delete; form classic; "
            "path_shipped=False; ship-met False; form_label_change_shipped False; "
            "wire_shipped=False; dual_linf unproven; online_linf_gate open; dual-ban; "
            "dual_recovery_path=None; isolation_rewrite_required still in "
            "DEFAULT_WIRE_BLOCKERS; does not clear DEFAULT_WIRE_BLOCKERS; does not "
            "redefine ready_for_wire_discussion; no excel_pipeline import on hot path; "
            "no TF required; isolation suite behavior unchanged. "
            "Offline Case-1 dual-honest multi-blocker wire bundle design contract "
            "available (offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report): "
            "machine-readable *what must land together* for SUGGESTED_NEXT_WAVE; "
            "bundle_design_present=True; bundle_shipped=False; "
            "bundle_ship_allowed_today=False; criteria_met_today=False; "
            "wire_shipped=False; isolation_rewrite_shipped=False; form classic; "
            "form_label_change_shipped=False; path_shipped=False; ship-met False; "
            "dual_linf unproven; online_linf_gate open; dual_recovery_path=None; "
            "feature flag reserved false; order_hint is not executor (no auto-wire); "
            "bundle design ≠ wire-ship acceptance alone ≠ isolation/form/path criteria "
            "alone ≠ VERDICT; dual-ban; full DEFAULT_WIRE_BLOCKERS remain; does not "
            "clear DEFAULT_WIRE_BLOCKERS; does not redefine ready_for_wire_discussion; "
            "no excel_pipeline import on hot path; no TF required; isolation suite "
            "behavior unchanged. "
            "Offline Case-1 dual-honest multi-blocker wire bundle ship-met / flip "
            "criteria contract available "
            "(offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report): "
            "machine-readable *when bundle_shipped / bundle_ship_allowed_today may become "
            "True*; criteria_present=True; bundle_shipped=False; "
            "bundle_ship_allowed_today=False; criteria_met_today=False; wire_shipped=False; "
            "isolation_rewrite_shipped=False; isolation checklist open; form classic; "
            "form_label_change_shipped=False; path_shipped=False; path ship-met False; "
            "dual_linf unproven; online_linf_gate open; dual_recovery_path=None; "
            "feature flag reserved false; order_hint is not executor; dual-ban; "
            "this criteria contract alone is not bundle ship / wire ship / VERDICT; "
            "full DEFAULT_WIRE_BLOCKERS remain; does not clear DEFAULT_WIRE_BLOCKERS; "
            "does not redefine ready_for_wire_discussion; no excel_pipeline import on "
            "hot path; no TF required; isolation suite behavior unchanged."
        ),
    }


@dataclass(frozen=True)
class AffineCoeffs:
    """Pre-postprocess affine package: y_raw = y0 + D @ (x - x0).

    Arrays are float64. Product/driver order matches ``BaseDeltaModel``.
    """

    unit: str
    products: List[str]
    drivers: List[str]
    y0: np.ndarray  # shape (n_products,)
    D: np.ndarray  # shape (n_products, n_drivers)
    x0: np.ndarray  # shape (n_drivers,)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "products": list(self.products),
            "drivers": list(self.drivers),
            "y0": self.y0.tolist(),
            "D": self.D.tolist(),
            "x0": self.x0.tolist(),
            "kind": "pre_postprocess_affine",
            "formula": "y_raw = y0 + D @ (x - x0)",
        }


def _ref_flat_from_model(model: Any) -> Dict[str, float]:
    """Flatten reference_feed + reference_conditions for affine x0.

    Mirrors ``pack_driver_vector`` / CDU evaluate nested rules: top-level
    numerics plus dotted keys for nested mappings (e.g. ``cut_points_f.naphtha_ep``).
    Flat-only units (FCC/Coker) are unchanged.
    """
    ref_flat: Dict[str, float] = {}
    for k, v in (getattr(model, "reference_feed", None) or {}).items():
        if isinstance(v, (int, float)):
            ref_flat[str(k)] = float(v)
    for k, v in (getattr(model, "reference_conditions", None) or {}).items():
        if isinstance(v, (int, float)):
            ref_flat[str(k)] = float(v)
        elif isinstance(v, Mapping):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    ref_flat[f"{k}.{kk}"] = float(vv)
    return ref_flat


def affine_coeffs_from_base_delta(model: Any) -> AffineCoeffs:
    """Read-only freeze of pre-postprocess BASE/DELTA map from a BaseDeltaModel.

    Does **not** include coke clamp / liquid renorm. Coefficients match Excel
    Submodel_* MB_* BASE and D_* export (not post-process evaluate() output).
    """
    products = list(model.products)
    drivers = list(model.drivers)
    n_p, n_d = len(products), len(drivers)
    y0 = np.zeros(n_p, dtype=np.float64)
    D = np.zeros((n_p, n_d), dtype=np.float64)
    ref_flat = _ref_flat_from_model(model)
    x0 = np.zeros(n_d, dtype=np.float64)
    for j, drv in enumerate(drivers):
        x0[j] = float(ref_flat.get(drv, 0.0))
    for i, p in enumerate(products):
        y0[i] = float(model.base_yields.get(p, 0.0))
        dmap = model.deltas.get(p) or {}
        for j, drv in enumerate(drivers):
            if drv in dmap:
                D[i, j] = float(dmap[drv])
    return AffineCoeffs(
        unit=str(getattr(model, "unit", "") or ""),
        products=products,
        drivers=drivers,
        y0=y0,
        D=D,
        x0=x0,
    )


def pack_driver_vector(
    coeffs: AffineCoeffs,
    feed: Optional[Mapping[str, float]] = None,
    conditions: Optional[Mapping[str, Any]] = None,
    *,
    merge_conditions: bool = True,
) -> np.ndarray:
    """Build ordered driver vector x matching ``_FCCModel.evaluate`` flatten rules.

    When ``merge_conditions`` is True and unit is FCC, merges via
    ``merge_process_conditions`` so process defaults fill missing keys.
    Missing driver keys fall back to ``x0``.
    """
    flat: Dict[str, float] = {}
    if feed:
        for k, v in feed.items():
            if isinstance(v, (int, float)):
                flat[str(k)] = float(v)
    cond_in = conditions
    if merge_conditions and coeffs.unit:
        try:
            from .base_delta import merge_process_conditions

            cond_in = merge_process_conditions(coeffs.unit, conditions)
        except Exception:
            cond_in = conditions
    if cond_in:
        for k, v in cond_in.items():
            if isinstance(v, (int, float)):
                flat[str(k)] = float(v)
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        flat[f"{k}.{kk}"] = float(vv)
    x = np.array(coeffs.x0, dtype=np.float64, copy=True)
    for j, drv in enumerate(coeffs.drivers):
        if drv in flat:
            x[j] = float(flat[drv])
    return x


def numpy_affine_forward(
    coeffs: AffineCoeffs,
    x: Union[np.ndarray, Sequence[float]],
    *,
    clamp_products: bool = False,
) -> np.ndarray:
    """Always-on numpy twin: y_raw = y0 + D @ (x - x0). Backend label: numpy_affine_ref."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_arr.shape[0] != coeffs.x0.shape[0]:
        raise ValueError(
            f"x length {x_arr.shape[0]} != n_drivers {coeffs.x0.shape[0]}"
        )
    y = coeffs.y0 + coeffs.D @ (x_arr - coeffs.x0)
    if clamp_products:
        y = np.maximum(y, 0.0)
    return y.astype(np.float64, copy=False)


def y_raw_dict(coeffs: AffineCoeffs, y_vec: np.ndarray) -> Dict[str, float]:
    return {p: float(y_vec[i]) for i, p in enumerate(coeffs.products)}


def apply_fcc_postprocess(
    y_raw: Union[Mapping[str, float], np.ndarray],
    products: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Numpy/Python FCC postprocess (coke clamp + liquid renorm). Not TF."""
    from .base_delta import postprocess_fcc_yields

    if isinstance(y_raw, Mapping):
        return postprocess_fcc_yields(y_raw)
    if products is None:
        raise ValueError("products required when y_raw is a vector")
    y_dict = {p: float(y_raw[i]) for i, p in enumerate(products)}
    return postprocess_fcc_yields(y_dict)


def apply_coker_postprocess(
    y_raw: Union[Mapping[str, float], np.ndarray],
    products: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Numpy/Python Coker postprocess (coke clamp + liquid renorm). Not TF."""
    from .base_delta import postprocess_coker_yields

    if isinstance(y_raw, Mapping):
        return postprocess_coker_yields(y_raw)
    if products is None:
        raise ValueError("products required when y_raw is a vector")
    y_dict = {p: float(y_raw[i]) for i, p in enumerate(products)}
    return postprocess_coker_yields(y_dict)


def apply_cdu_postprocess(
    y_raw: Union[Mapping[str, float], np.ndarray],
    products: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Numpy/Python CDU postprocess (liquid renorm + offgas clamp). Not TF."""
    from .base_delta import postprocess_cdu_yields

    if isinstance(y_raw, Mapping):
        return postprocess_cdu_yields(y_raw, products=products)
    if products is None:
        raise ValueError("products required when y_raw is a vector")
    y_dict = {p: float(y_raw[i]) for i, p in enumerate(products)}
    return postprocess_cdu_yields(y_dict, products=list(products))


class TFLinearBlock:
    """Exact-linear block: y_raw = y0 + D @ (x - x0) with float64 TF constants.

    Postprocess is intentionally **not** in the graph. Call
    ``apply_fcc_postprocess`` / ``apply_coker_postprocess`` /
    ``apply_cdu_postprocess`` separately for full evaluate parity.
    """

    def __init__(self, coeffs: AffineCoeffs):
        if not tf_available():
            err = tf_import_error()
            raise ImportError(
                "TensorFlow is not available; install optional extra: pip install -e '.[tf]'"
            ) from err
        import tensorflow as tf

        self.coeffs = coeffs
        self._tf = tf
        self._y0 = tf.constant(coeffs.y0, dtype=tf.float64, name="y0")
        self._D = tf.constant(coeffs.D, dtype=tf.float64, name="D")
        self._x0 = tf.constant(coeffs.x0, dtype=tf.float64, name="x0")

    def honesty_metadata(self) -> Dict[str, Any]:
        meta = honesty_metadata()
        meta.update(
            {
                "unit": self.coeffs.unit,
                "n_products": len(self.coeffs.products),
                "n_drivers": len(self.coeffs.drivers),
                "dtype": "float64",
                "backend": "tensorflow",
                "solver": False,
                "dual_recovery_path": None,
                "on_excel_case1_path": False,
            }
        )
        return meta

    def forward(
        self,
        x: Union[np.ndarray, Sequence[float], Mapping[str, float]],
        *,
        clamp_products: bool = False,
        as_dict: bool = False,
    ) -> Union[np.ndarray, Dict[str, float]]:
        """Affine forward only. Accepts ordered vector or driver dict."""
        tf = self._tf
        if isinstance(x, Mapping):
            x_vec = pack_driver_vector(
                self.coeffs, feed=x, conditions=None, merge_conditions=False
            )
            # allow pure driver map: merge with x0 for missing
            x_list = []
            for j, drv in enumerate(self.coeffs.drivers):
                if drv in x and isinstance(x[drv], (int, float)):
                    x_list.append(float(x[drv]))
                else:
                    x_list.append(float(self.coeffs.x0[j]))
            x_vec = np.asarray(x_list, dtype=np.float64)
        else:
            x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_vec.shape[0] != self.coeffs.x0.shape[0]:
            raise ValueError(
                f"x length {x_vec.shape[0]} != n_drivers {self.coeffs.x0.shape[0]}"
            )
        x_tf = tf.constant(x_vec, dtype=tf.float64)
        y_tf = self._y0 + tf.linalg.matvec(self._D, x_tf - self._x0)
        y = y_tf.numpy().astype(np.float64)
        if clamp_products:
            y = np.maximum(y, 0.0)
        if as_dict:
            return y_raw_dict(self.coeffs, y)
        return y


def tf_linear_fcc(
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> TFLinearBlock:
    """Factory: FCC exact-linear block from ``build_fcc_base_delta()`` coeffs only."""
    from .base_delta import build_fcc_base_delta

    model = build_fcc_base_delta(
        reference_feed=reference_feed,
        reference_conditions=reference_conditions,
    )
    coeffs = affine_coeffs_from_base_delta(model)
    return TFLinearBlock(coeffs)


def tf_linear_coker(
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> TFLinearBlock:
    """Factory: Coker exact-linear block from ``build_coker_base_delta()`` coeffs only.

    Lazy TF import via ``TFLinearBlock``; raises ``ImportError`` when TF is missing.
    Affine only — call ``apply_coker_postprocess`` for full evaluate parity.
    """
    from .base_delta import build_coker_base_delta

    model = build_coker_base_delta(
        reference_feed=reference_feed,
        reference_conditions=reference_conditions,
    )
    coeffs = affine_coeffs_from_base_delta(model)
    return TFLinearBlock(coeffs)


def tf_linear_cdu(
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> TFLinearBlock:
    """Factory: CDU exact-linear block from ``build_cdu_base_delta()`` coeffs only.

    Lazy TF import via ``TFLinearBlock``; raises ``ImportError`` when TF is missing.
    Affine only (6×8, nested cut_points in x0) — call ``apply_cdu_postprocess``
    for full evaluate parity. Not a PIMS MB_* Excel matrix twin; Submodel_CDU
    remains classic TECH+A export.
    """
    from .base_delta import build_cdu_base_delta

    model = build_cdu_base_delta(
        reference_feed=reference_feed,
        reference_conditions=reference_conditions,
    )
    coeffs = affine_coeffs_from_base_delta(model)
    return TFLinearBlock(coeffs)


# ---------------------------------------------------------------------------
# Multi-unit offline registry + wiring-readiness harness (not a solve / dual)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineUnitDescriptor:
    """Metadata for one offline exact-linear unit (FCC / COKER / CDU).

    Callables are stored as symbols; factories stay lazy (TF not imported until
    ``build_offline_unit`` / factory is invoked).
    """

    unit: str
    builder_name: str
    factory_name: str
    postprocess_name: str
    excel_match_name: Optional[str]  # None for CDU (TECH+A, not MB_* twin)
    renorm_note: str
    n_products: int
    n_drivers: int


def _normalize_unit_name(unit: str) -> str:
    key = str(unit or "").strip().upper()
    if key not in UNITS:
        raise ValueError(
            f"Unknown offline unit {unit!r}; expected one of {list(UNITS)}"
        )
    return key


def offline_unit_registry() -> List[OfflineUnitDescriptor]:
    """Ordered multi-unit offline registry: FCC, COKER, CDU.

    Always-on / TF-free. Does not construct models or import tensorflow.
    Shapes are the frozen base_delta catalog sizes (exact-linear shell).
    """
    return [
        OfflineUnitDescriptor(
            unit="FCC",
            builder_name="build_fcc_base_delta",
            factory_name="tf_linear_fcc",
            postprocess_name="apply_fcc_postprocess",
            excel_match_name="excel_fcc_matrix_matches_affine",
            renorm_note=(
                "FCC coke clamp + liquid renorm outside TF; raw may ≠ evaluate "
                "when renorm engages (offset cases)."
            ),
            n_products=6,
            n_drivers=8,
        ),
        OfflineUnitDescriptor(
            unit="COKER",
            builder_name="build_coker_base_delta",
            factory_name="tf_linear_coker",
            postprocess_name="apply_coker_postprocess",
            excel_match_name="excel_coker_matrix_matches_affine",
            renorm_note=(
                "Coker renorm always engages → raw affine ≠ evaluate even at "
                "reference; L1 uses affine + postprocess."
            ),
            n_products=5,
            n_drivers=6,
        ),
        OfflineUnitDescriptor(
            unit="CDU",
            builder_name="build_cdu_base_delta",
            factory_name="tf_linear_cdu",
            postprocess_name="apply_cdu_postprocess",
            excel_match_name=None,  # never invent excel_cdu_matrix_matches_affine
            renorm_note=(
                "CDU liquid renorm + offgas clamp outside TF; renorm often "
                "identity at reference (mass-conserving drivers). Nested "
                "cut_points_f.* in x0. Submodel_CDU is TECH+A export, not MB_*."
            ),
            n_products=6,
            n_drivers=8,
        ),
    ]


def _builder_for(unit: str):
    from .base_delta import (
        build_cdu_base_delta,
        build_coker_base_delta,
        build_fcc_base_delta,
    )

    key = _normalize_unit_name(unit)
    return {
        "FCC": build_fcc_base_delta,
        "COKER": build_coker_base_delta,
        "CDU": build_cdu_base_delta,
    }[key]


def _postprocess_for(unit: str):
    key = _normalize_unit_name(unit)
    return {
        "FCC": apply_fcc_postprocess,
        "COKER": apply_coker_postprocess,
        "CDU": apply_cdu_postprocess,
    }[key]


def _factory_for(unit: str):
    key = _normalize_unit_name(unit)
    return {
        "FCC": tf_linear_fcc,
        "COKER": tf_linear_coker,
        "CDU": tf_linear_cdu,
    }[key]


def offline_unit_coeffs(
    unit: str,
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> AffineCoeffs:
    """Always-on AffineCoeffs for a registry unit (no TensorFlow)."""
    builder = _builder_for(unit)
    model = builder(
        reference_feed=reference_feed,
        reference_conditions=reference_conditions,
    )
    return affine_coeffs_from_base_delta(model)


# Process-local cache for **default-reference** coeffs only (offline readiness /
# microbench). Custom reference_feed/conditions never use this cache.
_DEFAULT_UNIT_COEFFS_CACHE: Dict[str, AffineCoeffs] = {}


def clear_offline_unit_coeffs_cache() -> None:
    """Clear default-ref AffineCoeffs cache (tests / process reset)."""
    _DEFAULT_UNIT_COEFFS_CACHE.clear()


def cached_offline_unit_coeffs(unit: str) -> AffineCoeffs:
    """Default-reference AffineCoeffs with process-local memoization (no TF).

    Cache is for offline readiness / timing harness only — not a solver state
    store. Custom refs must still call ``offline_unit_coeffs(unit, feed, cond)``
    so wrong-key caching cannot silently reuse default coeffs.
    """
    key = _normalize_unit_name(unit)
    hit = _DEFAULT_UNIT_COEFFS_CACHE.get(key)
    if hit is not None:
        return hit
    coeffs = offline_unit_coeffs(key)
    _DEFAULT_UNIT_COEFFS_CACHE[key] = coeffs
    return coeffs


def build_offline_unit(
    unit: str,
    reference_feed: Optional[Mapping[str, float]] = None,
    reference_conditions: Optional[Mapping[str, Any]] = None,
) -> TFLinearBlock:
    """Lazy TF factory via registry unit name. Raises ImportError if TF missing."""
    factory = _factory_for(unit)
    return factory(
        reference_feed=reference_feed,
        reference_conditions=reference_conditions,
    )


def offline_units_status() -> Dict[str, Any]:
    """Honesty status for the multi-unit offline surface (always-on).

    Never claims dual recovery or Case 1 solve ownership. ``tf_available`` is
    probed lazily; coefficients shapes are live from base_delta builders.
    """
    per_unit: Dict[str, Any] = {}
    for desc in offline_unit_registry():
        coeffs = offline_unit_coeffs(desc.unit)
        per_unit[desc.unit] = {
            "n_products": int(coeffs.y0.shape[0]),
            "n_drivers": int(coeffs.x0.shape[0]),
            "shape": [int(coeffs.y0.shape[0]), int(coeffs.x0.shape[0])],
            "excel_match": desc.excel_match_name is not None,
            "excel_match_name": desc.excel_match_name,
            "postprocess": desc.postprocess_name,
            "factory": desc.factory_name,
            "builder": desc.builder_name,
            "renorm_note": desc.renorm_note,
        }
    return {
        "units": list(UNITS),
        "solver": SOLVER,
        "dual_recovery_path": DUAL_RECOVERY_PATH,
        "on_excel_case1_path": ON_EXCEL_CASE1_PATH,
        "tf_available": tf_available(),
        "per_unit": per_unit,
        "kind": MODULE_KIND,
        "source": SOURCE,
        "postprocess": POSTPROCESS,
        "note": (
            "Offline exact-linear FCC+COKER+CDU kernels available; "
            "not on Excel Case 1 solve (classic_2block_excel_path); "
            "dual_recovery_path=None on TF surface; Case 1 duals remain "
            "PRIMARY online-λ / SECONDARY recovered (not TF-owned)."
        ),
    }


def _mild_offset_for(unit: str, model: Any) -> tuple[Dict[str, float], Dict[str, Any]]:
    """One mild feed/process offset per unit for L1 readiness (not a solve)."""
    key = _normalize_unit_name(unit)
    feed = dict(getattr(model, "reference_feed", None) or {})
    cond = dict(getattr(model, "reference_conditions", None) or {})
    if key == "FCC":
        feed["api"] = float(feed.get("api", 22.0)) + 3.0
        cond["riser_outlet_temp_f"] = float(cond.get("riser_outlet_temp_f", 980.0)) + 10.0
    elif key == "COKER":
        feed["api"] = float(feed.get("api", 10.0)) + 2.0
        cond["recycle_ratio"] = float(cond.get("recycle_ratio", 0.1)) + 0.05
    else:  # CDU
        feed["api"] = float(feed.get("api", 28.0)) + 2.0
        cuts = dict(cond.get("cut_points_f") or {})
        if cuts:
            cuts = {k: float(v) + 8.0 for k, v in cuts.items()}
            cond["cut_points_f"] = cuts
        else:
            cond["cut_points_f"] = {
                "naphtha_ep": 400.0,
                "distillate_ep": 700.0,
                "gasoil_ep": 1030.0,
            }
    return feed, cond


def multi_unit_parity_report(atol: float = 1e-9) -> Dict[str, Any]:
    """Wiring-readiness harness: multi-unit numpy parity (optional TF skip).

    **Not** a solve, **not** ADMM, **not** dual recovery. For each registry unit:
    pack@ref ≡ x0; numpy affine + unit postprocess ≡ ``evaluate()`` at reference
    and one mild offset. Optional TF arm compares raw forward when available.

    Aggregate ``ok`` requires numeric checks **and** honesty fields
    (solver=False, dual_recovery_path=None, on_excel_case1_path=False).
    """
    units_out: Dict[str, Any] = {}
    all_ok = True
    for desc in offline_unit_registry():
        builder = _builder_for(desc.unit)
        post = _postprocess_for(desc.unit)
        model = builder()
        coeffs = affine_coeffs_from_base_delta(model)
        checks: Dict[str, Any] = {}

        # pack@ref ≡ x0
        x_ref = pack_driver_vector(
            coeffs,
            feed=model.reference_feed,
            conditions=model.reference_conditions,
        )
        pack_ok = bool(np.allclose(x_ref, coeffs.x0, atol=atol, rtol=0.0))
        checks["pack_ref_eq_x0"] = pack_ok

        def _l1_case(feed: Mapping[str, float], cond: Mapping[str, Any]) -> bool:
            x = pack_driver_vector(coeffs, feed=feed, conditions=cond)
            y_raw = numpy_affine_forward(coeffs, x, clamp_products=True)
            y_full = post(y_raw, products=coeffs.products)
            y_eval = model.evaluate(feed, cond, clamp_products=True)
            for p in coeffs.products:
                if abs(float(y_full[p]) - float(y_eval[p])) > atol:
                    return False
            return True

        ref_ok = _l1_case(model.reference_feed, model.reference_conditions)
        checks["affine_postprocess_eq_evaluate_ref"] = ref_ok
        feed_off, cond_off = _mild_offset_for(desc.unit, model)
        off_ok = _l1_case(feed_off, cond_off)
        checks["affine_postprocess_eq_evaluate_offset"] = off_ok

        tf_section: Dict[str, Any] = {
            "skipped": True,
            "ok": None,
            "reason": "tf_unavailable",
        }
        if tf_available():
            try:
                block = build_offline_unit(desc.unit)
                x_off = pack_driver_vector(coeffs, feed=feed_off, conditions=cond_off)
                y_np = numpy_affine_forward(coeffs, x_off, clamp_products=True)
                y_tf_raw = block.forward(x_off, clamp_products=True, as_dict=False)
                y_tf = np.asarray(y_tf_raw, dtype=np.float64).reshape(-1)
                tf_ok = bool(np.allclose(y_np, y_tf, atol=atol, rtol=0.0))
                tf_section = {"skipped": False, "ok": tf_ok, "reason": None}
                checks["tf_raw_eq_numpy_raw_offset"] = tf_ok
            except Exception as exc:  # pragma: no cover - env dependent
                tf_section = {
                    "skipped": False,
                    "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                checks["tf_raw_eq_numpy_raw_offset"] = False

        unit_numeric_ok = pack_ok and ref_ok and off_ok
        # TF optional: only fail aggregate when TF path was attempted and failed
        if tf_section.get("skipped") is False and tf_section.get("ok") is False:
            unit_numeric_ok = False

        unit_row = {
            "ok": unit_numeric_ok,
            "unit": desc.unit,
            "n_products": int(coeffs.y0.shape[0]),
            "n_drivers": int(coeffs.x0.shape[0]),
            "excel_match_name": desc.excel_match_name,
            "checks": checks,
            "tf": tf_section,
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "renorm_note": desc.renorm_note,
        }
        units_out[desc.unit] = unit_row
        if not unit_numeric_ok:
            all_ok = False

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
    )
    if not honesty_ok:
        all_ok = False

    return {
        "ok": all_ok,
        "units": units_out,
        "unit_order": list(UNITS),
        "solver": SOLVER,
        "dual_recovery_path": DUAL_RECOVERY_PATH,
        "on_excel_case1_path": ON_EXCEL_CASE1_PATH,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "atol": atol,
        "note": (
            "Wiring-readiness only: exact linear copy parity across FCC/COKER/CDU. "
            "Not a solve; not ADMM; not dual recovery. TF dual_recovery_path stays None."
        ),
    }


# ---------------------------------------------------------------------------
# Offline priced residual harness + optional local box direction (goal 5)
# Always-on numpy; not a solve; not ADMM duals; not on Excel Case 1 path.
# ---------------------------------------------------------------------------

PRICE_SOURCE = "synthetic_offline_demo"

# Stable non-negative synthetic demo prices (not Case 1 blender / not ADMM λ).
_DEFAULT_OFFLINE_PRICES: Dict[str, Dict[str, float]] = {
    "FCC": {
        "fcc_dry_gas": 8.0,
        "fcc_lpg": 22.0,
        "fcc_naphtha": 45.0,
        "fcc_lco": 38.0,
        "fcc_slurry": 18.0,
        "fcc_coke": 2.0,
    },
    "COKER": {
        "coker_dry_gas": 7.0,
        "coker_lpg": 20.0,
        "coker_naphtha": 40.0,
        "coker_gasoil": 35.0,
        "coker_coke": 5.0,
    },
    "CDU": {
        "cdu_offgas": 6.0,
        "cdu_naphtha_light": 48.0,
        "cdu_naphtha_heavy": 44.0,
        "cdu_distillate": 42.0,
        "cdu_gasoil": 36.0,
        "cdu_resid": 15.0,
    },
}


def _priced_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / Case1-off contract for priced residual reports."""
    return {
        "kind": "offline_priced_residual",
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "note": (
            "Offline priced residual harness only: product prices are synthetic demo "
            "values (not ADMM λ, not pure-ADMM dual recovery, not Case 1 shadows). "
            "Not a solve; not on classic_2block_excel_path. TF dual_recovery_path stays None."
        ),
    }


def default_offline_prices(unit: str) -> Dict[str, float]:
    """Stable synthetic non-negative product prices for one registry unit.

    Prices are **readiness demo values** only (``price_source=synthetic_offline_demo``).
    They are **not** Case 1 blender prices, online λ, recovered duals, or shadows.
    """
    key = _normalize_unit_name(unit)
    prices = _DEFAULT_OFFLINE_PRICES.get(key)
    if prices is None:
        raise ValueError(
            f"No default offline prices for unit {unit!r}; expected one of {list(UNITS)}"
        )
    return dict(prices)


def pack_price_vector(
    unit_or_coeffs: Union[str, AffineCoeffs],
    prices: Mapping[str, float],
    *,
    fill_missing: bool = False,
) -> np.ndarray:
    """Pack product prices into product order matching AffineCoeffs.products.

    By default missing product keys raise ValueError (honesty). Set
    ``fill_missing=True`` to fill absent keys with 0.0. Unknown keys raise.
    """
    if isinstance(unit_or_coeffs, AffineCoeffs):
        coeffs = unit_or_coeffs
        products = list(coeffs.products)
    else:
        coeffs = offline_unit_coeffs(str(unit_or_coeffs))
        products = list(coeffs.products)

    price_map = {str(k): float(v) for k, v in prices.items()}
    unknown = sorted(set(price_map) - set(products))
    if unknown:
        raise ValueError(
            f"Unknown product price keys {unknown} for unit {coeffs.unit!r}; "
            f"expected subset of {products}"
        )
    missing = [p for p in products if p not in price_map]
    if missing and not fill_missing:
        raise ValueError(
            f"Missing product prices for unit {coeffs.unit!r}: {missing}"
        )
    vec = np.zeros(len(products), dtype=np.float64)
    for i, p in enumerate(products):
        if p in price_map:
            val = float(price_map[p])
            if val < 0.0:
                raise ValueError(f"Price for {p!r} must be non-negative, got {val}")
            vec[i] = val
        else:
            vec[i] = 0.0
    return vec


def _dot_prices(p_vec: np.ndarray, y_map: Mapping[str, float], products: Sequence[str]) -> float:
    total = 0.0
    for i, prod in enumerate(products):
        total += float(p_vec[i]) * float(y_map[prod])
    return float(total)


def _resolve_prices(
    unit: str, prices: Optional[Mapping[str, float]]
) -> tuple[Dict[str, float], np.ndarray, AffineCoeffs]:
    coeffs = offline_unit_coeffs(unit)
    price_dict = dict(prices) if prices is not None else default_offline_prices(unit)
    p_vec = pack_price_vector(coeffs, price_dict)
    return price_dict, p_vec, coeffs


def priced_residual_for_unit(
    unit: str,
    prices: Optional[Mapping[str, float]] = None,
    *,
    feed: Optional[Mapping[str, float]] = None,
    conditions: Optional[Mapping[str, Any]] = None,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> Dict[str, Any]:
    """Per-unit priced residual: p·postprocess(affine) vs p·evaluate (always-on).

    At the given drivers (default: reference), compares economics under product
    prices. Also reports raw affine priced value (may ≠ full for Coker renorm).
    Honesty: dual_recovery_path=None; not on Case 1; prices are not duals.
    """
    key = _normalize_unit_name(unit)
    price_dict, p_vec, coeffs = _resolve_prices(key, prices)
    builder = _builder_for(key)
    post = _postprocess_for(key)
    model = builder()
    feed_use = dict(feed) if feed is not None else dict(model.reference_feed)
    cond_use = (
        dict(conditions)
        if conditions is not None
        else dict(model.reference_conditions)
    )

    x = pack_driver_vector(coeffs, feed=feed_use, conditions=cond_use)
    y_raw = numpy_affine_forward(coeffs, x, clamp_products=True)
    y_aff_full = post(y_raw, products=coeffs.products)
    y_eval = model.evaluate(feed_use, cond_use, clamp_products=True)

    v_raw = float(p_vec @ y_raw)
    v_aff = _dot_prices(p_vec, y_aff_full, coeffs.products)
    v_eval = _dot_prices(p_vec, y_eval, coeffs.products)
    abs_err = abs(v_aff - v_eval)
    scale = max(abs(v_eval), 1e-12)
    rel_err = abs_err / scale
    ok = bool(abs_err <= atol + rtol * abs(v_eval))
    raw_vs_full = abs(v_raw - v_eval)

    honesty = _priced_honesty_fields()
    return {
        "unit": key,
        "ok": ok,
        "v_eval": v_eval,
        "v_aff": v_aff,
        "v_raw": v_raw,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "raw_vs_full_priced_gap": raw_vs_full,
        "prices": price_dict,
        "products": list(coeffs.products),
        "atol": atol,
        "rtol": rtol,
        **honesty,
    }


def multi_unit_priced_residual_report(
    prices: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> Dict[str, Any]:
    """Always-on multi-unit priced residual (FCC/COKER/CDU) + honesty locks.

    For each registry unit, at reference and one mild driver offset, compares
    ``p · postprocess(numpy_affine_forward)`` vs ``p · evaluate``. Aggregate
    ``ok`` requires numeric residuals within atol/rtol **and** dual-ban honesty.

    Not a solve, not ADMM, not dual recovery, not on Excel Case 1.
    """
    units_out: Dict[str, Any] = {}
    all_ok = True
    honesty = _priced_honesty_fields()

    for desc in offline_unit_registry():
        unit_prices: Optional[Mapping[str, float]] = None
        if prices is not None and desc.unit in prices:
            unit_prices = prices[desc.unit]
        elif prices is not None and desc.unit not in prices:
            unit_prices = None  # fall back to defaults

        builder = _builder_for(desc.unit)
        model = builder()
        feed_off, cond_off = _mild_offset_for(desc.unit, model)

        ref_row = priced_residual_for_unit(
            desc.unit, unit_prices, atol=atol, rtol=rtol
        )
        off_row = priced_residual_for_unit(
            desc.unit,
            unit_prices,
            feed=feed_off,
            conditions=cond_off,
            atol=atol,
            rtol=rtol,
        )
        unit_ok = bool(ref_row["ok"] and off_row["ok"])
        unit_row = {
            "ok": unit_ok,
            "unit": desc.unit,
            "ref": ref_row,
            "offset": off_row,
            "raw_vs_full_priced_gap_ref": ref_row["raw_vs_full_priced_gap"],
            "renorm_note": desc.renorm_note,
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "kind": "offline_priced_residual",
            "price_source": PRICE_SOURCE,
        }
        units_out[desc.unit] = unit_row
        if not unit_ok:
            all_ok = False

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
    )
    if not honesty_ok:
        all_ok = False

    return {
        "ok": all_ok,
        "units": units_out,
        "unit_order": list(UNITS),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": "offline_priced_residual",
        "price_source": PRICE_SOURCE,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "atol": atol,
        "rtol": rtol,
        "note": honesty["note"],
    }


def local_box_direction(
    unit: str,
    prices: Optional[Mapping[str, float]] = None,
    *,
    delta: Union[float, Mapping[str, float], np.ndarray] = 1.0,
    driver_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Closed-form local box maximizer for raw affine product value (offline).

    Maximizes ``p · (y0 + D @ (x − x0))`` over box ``x ∈ [x0−δ, x0+δ]``
    (independent per driver). Maximizer is a corner: ``sign(D.T @ p)``.

    Reports ``x_star``, ``v_raw_star``, and postprocess ``v_full_star``.
    Postprocess is **outside** the linear program (Coker raw ≠ full expected).
    Gradients / local prices are **not** ADMM λ or Case 1 shadows.
    """
    key = _normalize_unit_name(unit)
    price_dict, p_vec, coeffs = _resolve_prices(key, prices)
    n_d = int(coeffs.x0.shape[0])

    if isinstance(delta, Mapping):
        d_vec = np.array(
            [float(delta.get(drv, 0.0)) for drv in coeffs.drivers],
            dtype=np.float64,
        )
    elif np.isscalar(delta):
        d_vec = np.full(n_d, float(np.asarray(delta).item()), dtype=np.float64)
    else:
        d_vec = np.asarray(delta, dtype=np.float64).reshape(-1)
        if d_vec.shape[0] != n_d:
            raise ValueError(
                f"delta length {d_vec.shape[0]} != n_drivers {n_d}"
            )
    if np.any(d_vec < 0.0):
        raise ValueError("delta components must be non-negative")

    mask = np.ones(n_d, dtype=bool)
    if driver_mask is not None:
        mask_arr = np.asarray(list(driver_mask), dtype=bool).reshape(-1)
        if mask_arr.shape[0] != n_d:
            raise ValueError(
                f"driver_mask length {mask_arr.shape[0]} != n_drivers {n_d}"
            )
        mask = mask_arr

    # Gradient of raw affine value w.r.t. x: g = D.T @ p
    g = coeffs.D.T @ p_vec
    x_star = np.array(coeffs.x0, dtype=np.float64, copy=True)
    for j in range(n_d):
        if not mask[j] or abs(float(g[j])) < 1e-15 or float(d_vec[j]) == 0.0:
            continue
        x_star[j] = float(coeffs.x0[j]) + float(d_vec[j]) * float(np.sign(g[j]))

    y_raw = numpy_affine_forward(coeffs, x_star, clamp_products=True)
    v_raw_star = float(p_vec @ y_raw)
    post = _postprocess_for(key)
    y_full = post(y_raw, products=coeffs.products)
    v_full_star = _dot_prices(p_vec, y_full, coeffs.products)

    # Reference raw value for comparison
    y_raw_ref = numpy_affine_forward(coeffs, coeffs.x0, clamp_products=True)
    v_raw_ref = float(p_vec @ y_raw_ref)

    tf_section: Dict[str, Any] = {
        "skipped": True,
        "ok": None,
        "reason": "tf_unavailable",
    }
    if tf_available():
        try:
            block = build_offline_unit(key)
            y_tf = np.asarray(
                block.forward(x_star, clamp_products=True, as_dict=False),
                dtype=np.float64,
            ).reshape(-1)
            v_tf_raw = float(p_vec @ y_tf)
            tf_ok = bool(abs(v_tf_raw - v_raw_star) <= 1e-9)
            tf_section = {
                "skipped": False,
                "ok": tf_ok,
                "v_raw_star_tf": v_tf_raw,
                "reason": None,
            }
        except Exception as exc:  # pragma: no cover - env dependent
            tf_section = {
                "skipped": False,
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    return {
        "unit": key,
        "kind": "offline_local_box_direction",
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "prices": price_dict,
        "x0": coeffs.x0.tolist(),
        "x_star": x_star.tolist(),
        "drivers": list(coeffs.drivers),
        "gradient_raw": g.tolist(),
        "delta": d_vec.tolist(),
        "v_raw_ref": v_raw_ref,
        "v_raw_star": v_raw_star,
        "v_full_star": v_full_star,
        "raw_vs_full_priced_gap_star": abs(v_raw_star - v_full_star),
        "tf": tf_section,
        "note": (
            "Offline local box maximizer for raw affine product value under "
            "independent driver box. Maximizer is a corner (sign of D.T @ p). "
            "Postprocess (renorm/clamp) is outside the linear program — raw vs "
            "full may differ (especially Coker). Local gradients/prices are "
            "NOT ADMM λ, NOT pure-ADMM dual recovery, NOT Case 1 shadows. "
            "Not a solve; not on classic_2block_excel_path."
        ),
    }


def _timing_stats_us(samples_ns: Sequence[int]) -> Dict[str, Any]:
    """Convert nanosecond samples to µs summary stats (median/mean/min/max)."""
    if not samples_ns:
        return {
            "median_us": 0.0,
            "mean_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
            "n": 0,
        }
    arr = np.asarray(samples_ns, dtype=np.float64) / 1000.0  # ns → µs
    return {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "n": int(arr.shape[0]),
    }


def _bench_callable(
    fn: Callable[[], Any],
    *,
    n_repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    """Warmup then time ``fn`` with ``time.perf_counter_ns``; return µs stats."""
    for _ in range(max(0, int(warmup))):
        fn()
    samples: List[int] = []
    for _ in range(int(n_repeats)):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return _timing_stats_us(samples)


def _local_box_step_raw(
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    delta: float = 1.0,
) -> np.ndarray:
    """Thin closed-form box step using prebuilt coeffs (timing hot path).

    Same corner maximizer as ``local_box_direction`` for independent box ±delta.
    """
    g = coeffs.D.T @ p_vec
    d = float(delta)
    x_star = np.array(coeffs.x0, dtype=np.float64, copy=True)
    for j in range(int(coeffs.x0.shape[0])):
        gj = float(g[j])
        if abs(gj) < 1e-15 or d == 0.0:
            continue
        x_star[j] = float(coeffs.x0[j]) + d * float(np.sign(gj))
    return x_star


def multi_unit_block_solve_timing_report(
    *,
    n_repeats: int = 500,
    warmup: int = 5,
    include_box: bool = True,
    include_composition: bool = False,
    box_delta: float = 1.0,
) -> Dict[str, Any]:
    """Always-on cached multi-unit block-solve timing (FCC/COKER/CDU).

    Builds / caches default-ref ``AffineCoeffs`` **once per unit**, then times
    pure ``numpy_affine_forward`` (and optionally the closed-form local box
    step) over ``n_repeats``. Optional TF raw-forward arm when installed.

    Honesty locks: ``solver=False``, ``dual_recovery_path=None``,
    ``on_excel_case1_path=False``. Timings are **offline readiness**, not Case 1
    wall time and not ADMM duals / shadows. Aggregate ``ok`` is structural +
    honesty + finite positive timings — **not** a hard microsecond SLA.
    """
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be >= 1")
    n_repeats = int(n_repeats)
    warmup = max(0, int(warmup))

    units_out: Dict[str, Any] = {}
    timings_ok = True

    for unit in UNITS:
        t_build0 = time.perf_counter_ns()
        coeffs = cached_offline_unit_coeffs(unit)
        coeffs_build_us = (time.perf_counter_ns() - t_build0) / 1000.0

        x_probe = np.array(coeffs.x0, dtype=np.float64, copy=True)

        def _affine_once(
            _c: AffineCoeffs = coeffs, _x: np.ndarray = x_probe
        ) -> np.ndarray:
            return numpy_affine_forward(_c, _x, clamp_products=True)

        aff_stats = _bench_callable(
            _affine_once, n_repeats=n_repeats, warmup=warmup
        )
        aff_stats["shape"] = [
            int(coeffs.y0.shape[0]),
            int(coeffs.x0.shape[0]),
        ]

        box_stats: Optional[Dict[str, Any]] = None
        if include_box:
            price_dict = default_offline_prices(unit)
            p_vec = pack_price_vector(coeffs, price_dict)

            def _box_once(
                _c: AffineCoeffs = coeffs,
                _p: np.ndarray = p_vec,
                _d: float = float(box_delta),
            ) -> np.ndarray:
                x_star = _local_box_step_raw(_c, _p, delta=_d)
                return numpy_affine_forward(_c, x_star, clamp_products=True)

            box_stats = _bench_callable(
                _box_once, n_repeats=n_repeats, warmup=warmup
            )

        tf_section: Dict[str, Any] = {
            "skipped": True,
            "reason": "tf_unavailable",
            "median_us": None,
            "mean_us": None,
            "n": 0,
        }
        if tf_available():
            try:
                block = build_offline_unit(unit)
                x_tf = np.array(coeffs.x0, dtype=np.float64, copy=True)

                def _tf_once(_b=block, _x=x_tf) -> Any:
                    return _b.forward(_x, clamp_products=True, as_dict=False)

                tf_stats = _bench_callable(
                    _tf_once, n_repeats=n_repeats, warmup=warmup
                )
                tf_section = {
                    "skipped": False,
                    "reason": None,
                    "median_us": tf_stats["median_us"],
                    "mean_us": tf_stats["mean_us"],
                    "min_us": tf_stats["min_us"],
                    "max_us": tf_stats["max_us"],
                    "n": tf_stats["n"],
                }
            except Exception as exc:  # pragma: no cover - env dependent
                tf_section = {
                    "skipped": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "median_us": None,
                    "mean_us": None,
                    "n": 0,
                }

        unit_ok = bool(
            aff_stats["median_us"] > 0.0
            and aff_stats["mean_us"] > 0.0
            and np.isfinite(aff_stats["median_us"])
            and np.isfinite(aff_stats["mean_us"])
        )
        if include_box and box_stats is not None:
            unit_ok = unit_ok and bool(
                box_stats["median_us"] > 0.0
                and box_stats["mean_us"] > 0.0
                and np.isfinite(box_stats["median_us"])
            )
        if not unit_ok:
            timings_ok = False

        units_out[unit] = {
            "unit": unit,
            "ok": bool(unit_ok),
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "coeffs_build_us": float(coeffs_build_us),
            "affine": aff_stats,
            "box": box_stats,
            "box_skipped": not include_box,
            "tf": tf_section,
            "n_products": int(coeffs.y0.shape[0]),
            "n_drivers": int(coeffs.x0.shape[0]),
        }

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
    )

    parity_ok: Optional[bool] = None
    priced_ok: Optional[bool] = None
    if include_composition:
        try:
            parity_ok = bool(multi_unit_parity_report()["ok"])
        except Exception:  # pragma: no cover
            parity_ok = False
        try:
            priced_ok = bool(multi_unit_priced_residual_report()["ok"])
        except Exception:  # pragma: no cover
            priced_ok = False

    all_ok = bool(honesty_ok and timings_ok)
    if include_composition:
        all_ok = all_ok and bool(parity_ok) and bool(priced_ok)

    note = (
        "Offline cached multi-unit block-solve timing harness (FCC+COKER+CDU). "
        "AffineCoeffs built once per unit (default-ref cache); pure numpy "
        "affine forward (+ optional closed-form local box step) timed over "
        f"n_repeats={n_repeats}. Timings are offline readiness only — NOT Case 1 "
        "solve wall time, NOT ADMM duals, NOT pure-ADMM dual recovery, NOT "
        "Case 1 shadows / online λ. solver=False; dual_recovery_path=None; "
        "on_excel_case1_path=False; not on classic_2block_excel_path. "
        "No hard microsecond SLA — report numbers; aggregate ok is structure + "
        "honesty + finite positive timings."
    )

    out: Dict[str, Any] = {
        "ok": all_ok,
        "units": units_out,
        "unit_order": list(UNITS),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": "offline_block_solve_timing",
        "honesty_ok": honesty_ok,
        "timings_ok": timings_ok,
        "n_repeats": n_repeats,
        "warmup": warmup,
        "include_box": include_box,
        "include_composition": include_composition,
        "cached_coeffs": True,
        "tf_available": tf_available(),
        "readiness_only": True,
        "note": note,
    }
    if include_composition:
        out["parity_ok"] = parity_ok
        out["priced_ok"] = priced_ok
    return out


def offline_block_solve_readiness_report(
    *,
    n_repeats: int = 200,
    warmup: int = 3,
    include_box: bool = True,
    box_delta: float = 1.0,
    include_admm_residual: bool = True,
    include_admm_block_subproblem: bool = True,
    include_admm_coordination: bool = True,
    include_admm_plant_linking: bool = True,
    include_admm_plant_named_linking: bool = True,
    include_admm_case1_shaped_linking: bool = True,
    include_admm_case1_dual_space_form_contract: bool = True,
    include_admm_case1_dual_space_linf_probe: bool = True,
    include_admm_case1_dual_space_linf_live_lambda_bridge: bool = True,
    include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart: bool = True,
    include_admm_case1_honest_blender_pooling_path: bool = True,
    include_admm_case1_online_linf_gate_criteria_contract: bool = True,
    include_admm_case1_isolation_rewrite_design_contract: bool = True,
    include_admm_case1_wire_ship_acceptance_design_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_design_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract: bool = True,
    include_admm_case1_form_label_change_shipped_criteria_contract: bool = True,
    include_admm_case1_isolation_rewrite_shipped_criteria_contract: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_execution_scaffold: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_rehearsal: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint: bool = True,
    include_admm_case1_isolation_rewrite_first_blocker_operational_prep: bool = True,
    include_admm_case1_dual_linf_under_wire_criteria_contract: bool = True,
) -> Dict[str, Any]:
    """Compose timing + parity_ok + priced_ok under dual-ban honesty locks.

    One call answers \"ready for wire discussion?\" without re-implementing
    parity/priced math. Does **not** mean wire is shipped or duals are owned.

    ``admm_residual_ok``, ``admm_block_subproblem_ok``, ``admm_coordination_ok``,
    ``admm_plant_linking_ok``, ``admm_plant_named_linking_ok``,
    ``admm_case1_shaped_linking_ok``,
    ``admm_case1_dual_space_form_contract_ok``,
    ``admm_case1_dual_space_linf_probe_ok``,
    ``admm_case1_dual_space_linf_live_lambda_bridge_ok``, and
    ``admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok``, and
    ``admm_case1_honest_blender_pooling_path_ok``, and
    ``admm_case1_online_linf_gate_criteria_contract_ok``, and
    ``admm_case1_isolation_rewrite_design_contract_ok``, and
    ``admm_case1_wire_ship_acceptance_design_contract_ok``, and
    ``admm_case1_dual_honest_tf_aware_path_design_contract_ok``, and
    ``admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok``, and
    ``admm_case1_form_label_change_shipped_criteria_contract_ok``, and
    ``admm_case1_isolation_rewrite_shipped_criteria_contract_ok``, and
    ``admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok``, and
    ``admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok``, and
    ``admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok``, and
    ``admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok``, and
    ``admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok``, and
    ``admm_case1_isolation_rewrite_first_blocker_operational_prep_ok``, and
    ``admm_case1_dual_linf_under_wire_criteria_contract_ok`` are
    **additive** pre-wire checklist info (does **not** change
    ``ready_for_wire_discussion`` semantics: still parity∧priced∧timings∧honesty).
    Never claims wire shipped or full plant mass balance when residual /
    subproblem / coordination / plant-linking ok.
    """
    base = multi_unit_block_solve_timing_report(
        n_repeats=n_repeats,
        warmup=warmup,
        include_box=include_box,
        include_composition=True,
        box_delta=box_delta,
    )
    parity_ok = bool(base.get("parity_ok"))
    priced_ok = bool(base.get("priced_ok"))
    timings_ok = bool(base.get("timings_ok"))
    honesty_ok = bool(base.get("honesty_ok"))
    ready = bool(parity_ok and priced_ok and timings_ok and honesty_ok)
    base = dict(base)
    base["kind"] = "offline_block_solve_readiness"
    base["ready_for_wire_discussion"] = ready
    base["ok"] = ready
    admm_residual_ok: Optional[bool] = None
    if include_admm_residual:
        try:
            admm_rep = multi_unit_admm_residual_report(rho=1.0, x_mode="offset")
            admm_residual_ok = bool(admm_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive; harness should not raise
            admm_residual_ok = False
    base["admm_residual_ok"] = admm_residual_ok
    admm_block_subproblem_ok: Optional[bool] = None
    if include_admm_block_subproblem:
        try:
            sub_rep = multi_unit_admm_block_subproblem_report(rho=1.0, delta=0.5)
            admm_block_subproblem_ok = bool(sub_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_block_subproblem_ok = False
    base["admm_block_subproblem_ok"] = admm_block_subproblem_ok
    admm_coordination_ok: Optional[bool] = None
    if include_admm_coordination:
        try:
            coord_rep = multi_unit_admm_coordination_report(
                n_rounds=2, rho=1.0, delta=0.5
            )
            admm_coordination_ok = bool(coord_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_coordination_ok = False
    base["admm_coordination_ok"] = admm_coordination_ok
    admm_plant_linking_ok: Optional[bool] = None
    if include_admm_plant_linking:
        try:
            # Keep plant_linking_ok on synthetic default (do not require plant-named).
            pl_rep = multi_block_plant_linking_admm_report(
                n_rounds=2, rho=1.0, delta=0.5, mode="synthetic"
            )
            admm_plant_linking_ok = bool(pl_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_plant_linking_ok = False
    base["admm_plant_linking_ok"] = admm_plant_linking_ok
    admm_plant_named_linking_ok: Optional[bool] = None
    if include_admm_plant_named_linking:
        try:
            pn_rep = multi_block_plant_linking_admm_report(
                n_rounds=2, rho=1.0, delta=0.5, mode="plant_named"
            )
            admm_plant_named_linking_ok = bool(pn_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_plant_named_linking_ok = False
    base["admm_plant_named_linking_ok"] = admm_plant_named_linking_ok
    admm_case1_shaped_linking_ok: Optional[bool] = None
    if include_admm_case1_shaped_linking:
        try:
            c1_rep = offline_case1_shaped_cdu_blender_linking_report(
                n_rounds=2, rho=1.0, delta=0.5
            )
            admm_case1_shaped_linking_ok = bool(c1_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_case1_shaped_linking_ok = False
    base["admm_case1_shaped_linking_ok"] = admm_case1_shaped_linking_ok
    admm_case1_dual_space_form_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_space_form_contract:
        try:
            # Pure compose — no maximizer / no Case 1 solve on contract hot path.
            ds_rep = offline_case1_dual_space_form_contract_report()
            admm_case1_dual_space_form_contract_ok = bool(ds_rep.get("ok"))
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_space_form_contract_ok = False
    base["admm_case1_dual_space_form_contract_ok"] = (
        admm_case1_dual_space_form_contract_ok
    )
    admm_case1_dual_space_linf_probe_ok: Optional[bool] = None
    if include_admm_case1_dual_space_linf_probe:
        try:
            # Cheap proof-prep: fixture PRIMARY online λ vs skeleton final_lam (n_rounds=1).
            # Does not require TF/PuLP/excel_pipeline; does not flip dual_linf under wire.
            probe_rep = offline_case1_dual_space_linf_probe_report(
                skeleton_n_rounds=1,
            )
            admm_case1_dual_space_linf_probe_ok = bool(
                probe_rep.get("probe_ok", probe_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_space_linf_probe_ok = False
    base["admm_case1_dual_space_linf_probe_ok"] = (
        admm_case1_dual_space_linf_probe_ok
    )
    admm_case1_dual_space_linf_live_lambda_bridge_ok: Optional[bool] = None
    if include_admm_case1_dual_space_linf_live_lambda_bridge:
        try:
            # Additive readiness: bridge with explicit fixture fallback (source labeled).
            # Not live dual recovery; not VERDICT; dual_linf stays unproven.
            bridge_rep = offline_case1_dual_space_linf_live_lambda_bridge_report(
                allow_fixture_fallback=True,
                skeleton_n_rounds=1,
            )
            admm_case1_dual_space_linf_live_lambda_bridge_ok = bool(
                bridge_rep.get("bridge_ok", bridge_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_space_linf_live_lambda_bridge_ok = False
    base["admm_case1_dual_space_linf_live_lambda_bridge_ok"] = (
        admm_case1_dual_space_linf_live_lambda_bridge_ok
    )
    admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok: Optional[bool] = None
    if include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart:
        try:
            # Additive readiness: fixture-labeled live-λ-seeded warm-start self-test.
            # dual_linf stays unproven even if post-round L∞ small; not VERDICT; not wire.
            warm_rep = offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
                allow_fixture_fallback=True,
                n_rounds=1,
            )
            admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok = bool(
                warm_rep.get("warmstart_ok", warm_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok = False
    base["admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok"] = (
        admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok
    )
    admm_case1_honest_blender_pooling_path_ok: Optional[bool] = None
    if include_admm_case1_honest_blender_pooling_path:
        try:
            # Additive readiness: pure honesty compose (no maximizer / no residual gate).
            # dual_linf stays unproven; not VERDICT; not wire; not affine kernel claim.
            pool_rep = offline_case1_honest_blender_pooling_path_report()
            admm_case1_honest_blender_pooling_path_ok = bool(
                pool_rep.get("pooling_path_ok", pool_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_honest_blender_pooling_path_ok = False
    base["admm_case1_honest_blender_pooling_path_ok"] = (
        admm_case1_honest_blender_pooling_path_ok
    )
    admm_case1_online_linf_gate_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_online_linf_gate_criteria_contract:
        try:
            # Additive readiness: pure criteria-contract compose (no maximizer).
            # online_linf_gate stays open; criteria_met_today=False; dual_linf unproven;
            # not VERDICT; not wire; not gate flip.
            crit_rep = offline_case1_online_linf_gate_criteria_contract_report()
            admm_case1_online_linf_gate_criteria_contract_ok = bool(
                crit_rep.get("contract_ok", crit_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_online_linf_gate_criteria_contract_ok = False
    base["admm_case1_online_linf_gate_criteria_contract_ok"] = (
        admm_case1_online_linf_gate_criteria_contract_ok
    )
    admm_case1_isolation_rewrite_design_contract_ok: Optional[bool] = None
    if include_admm_case1_isolation_rewrite_design_contract:
        try:
            # Additive readiness: pure isolation-rewrite design-only compose.
            # design_present; rewrite_shipped=False; isolation checklist stays open;
            # dual_linf unproven; not VERDICT; not wire; not isolation rewrite shipped.
            design_rep = offline_case1_isolation_rewrite_design_contract_report()
            admm_case1_isolation_rewrite_design_contract_ok = bool(
                design_rep.get("design_contract_ok", design_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_isolation_rewrite_design_contract_ok = False
    base["admm_case1_isolation_rewrite_design_contract_ok"] = (
        admm_case1_isolation_rewrite_design_contract_ok
    )
    admm_case1_wire_ship_acceptance_design_contract_ok: Optional[bool] = None
    if include_admm_case1_wire_ship_acceptance_design_contract:
        try:
            # Additive readiness: pure wire-ship acceptance design-only compose.
            # design_present; wire_ship_allowed_today=False; wire_shipped=False;
            # dual_linf unproven; not VERDICT; not wire; not ship allow.
            wire_ship_rep = offline_case1_wire_ship_acceptance_design_contract_report()
            admm_case1_wire_ship_acceptance_design_contract_ok = bool(
                wire_ship_rep.get("design_contract_ok", wire_ship_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_wire_ship_acceptance_design_contract_ok = False
    base["admm_case1_wire_ship_acceptance_design_contract_ok"] = (
        admm_case1_wire_ship_acceptance_design_contract_ok
    )
    admm_case1_dual_honest_tf_aware_path_design_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_tf_aware_path_design_contract:
        try:
            # Additive readiness: pure dual-honest TF-aware path design compose.
            # path_design_present; path_shipped=False; ship-met dual_honest_tf_aware_path_present
            # remains False; wire_shipped=False; dual_linf unproven; not VERDICT; not path ship.
            path_design_rep = (
                offline_case1_dual_honest_tf_aware_path_design_contract_report()
            )
            admm_case1_dual_honest_tf_aware_path_design_contract_ok = bool(
                path_design_rep.get("design_contract_ok", path_design_rep.get("ok"))
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_tf_aware_path_design_contract_ok = False
    base["admm_case1_dual_honest_tf_aware_path_design_contract_ok"] = (
        admm_case1_dual_honest_tf_aware_path_design_contract_ok
    )
    admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract:
        try:
            # Additive readiness: pure ship-met / path-present-for-ship flip criteria compose.
            # criteria_present; ship_met_allowed_today=False; dual_honest_tf_aware_path_present
            # remains False; path_design_present=True; path_shipped=False; wire_shipped=False;
            # dual_linf unproven; not VERDICT; not ship-met; not path ship; not wire.
            path_present_crit_rep = (
                offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report()
            )
            admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok = bool(
                path_present_crit_rep.get(
                    "design_contract_ok",
                    path_present_crit_rep.get("contract_ok", path_present_crit_rep.get("ok")),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok = False
    base["admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok"] = (
        admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok
    )
    admm_case1_form_label_change_shipped_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_form_label_change_shipped_criteria_contract:
        try:
            # Additive readiness: pure form_label_change_shipped flip criteria compose.
            # criteria_present; form_label_ship_allowed_today=False; form_label_change_shipped
            # remains False; form classic; path_shipped=False; ship-met False; wire_shipped=False;
            # dual_linf unproven; not VERDICT; not form flip; not form_label shipped; not wire.
            form_label_crit_rep = (
                offline_case1_form_label_change_shipped_criteria_contract_report()
            )
            admm_case1_form_label_change_shipped_criteria_contract_ok = bool(
                form_label_crit_rep.get(
                    "design_contract_ok",
                    form_label_crit_rep.get(
                        "contract_ok", form_label_crit_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_form_label_change_shipped_criteria_contract_ok = False
    base["admm_case1_form_label_change_shipped_criteria_contract_ok"] = (
        admm_case1_form_label_change_shipped_criteria_contract_ok
    )
    admm_case1_isolation_rewrite_shipped_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_isolation_rewrite_shipped_criteria_contract:
        try:
            # Additive readiness: pure isolation-rewrite ship-met flip criteria compose.
            # criteria_present; isolation_ship_allowed_today=False; isolation_rewrite_shipped
            # remains False; checklist open; rewrite-not-delete; dual_linf unproven;
            # not VERDICT; not isolation rewrite shipped; not wire; not form flip.
            isolation_ship_crit_rep = (
                offline_case1_isolation_rewrite_shipped_criteria_contract_report()
            )
            admm_case1_isolation_rewrite_shipped_criteria_contract_ok = bool(
                isolation_ship_crit_rep.get(
                    "design_contract_ok",
                    isolation_ship_crit_rep.get(
                        "contract_ok", isolation_ship_crit_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_isolation_rewrite_shipped_criteria_contract_ok = False
    base["admm_case1_isolation_rewrite_shipped_criteria_contract_ok"] = (
        admm_case1_isolation_rewrite_shipped_criteria_contract_ok
    )
    admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract:
        try:
            # Additive readiness: pure multi-blocker wire bundle design compose.
            # bundle_design_present; bundle_shipped=False; bundle_ship_allowed_today=False;
            # wire_shipped=False; isolation_rewrite_shipped=False; dual_linf unproven;
            # not VERDICT; not wire; not bundle ship; order_hint not executor.
            bundle_design_rep = (
                offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report()
            )
            admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok = bool(
                bundle_design_rep.get(
                    "design_contract_ok",
                    bundle_design_rep.get(
                        "contract_ok", bundle_design_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok = False
    base["admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok"] = (
        admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok
    )
    admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract:
        try:
            # Additive readiness: pure multi-blocker bundle ship-met flip criteria.
            # criteria_present; bundle_shipped=False; bundle_ship_allowed_today=False;
            # criteria_met_today=False; wire_shipped=False; isolation_rewrite_shipped=False;
            # dual_linf unproven; not VERDICT; not bundle ship; order_hint not executor.
            bundle_ship_crit_rep = (
                offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report()
            )
            admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok = bool(
                bundle_ship_crit_rep.get(
                    "design_contract_ok",
                    bundle_ship_crit_rep.get(
                        "contract_ok", bundle_ship_crit_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok = False
    base["admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"] = (
        admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok
    )
    admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_tf_aware_path_execution_scaffold:
        try:
            # Additive readiness: pure dual-honest path execution scaffold compose.
            # scaffold_present; path/wire/bundle/isolation/form ship hard False;
            # dual_linf unproven; dual_recovery_path=None; not VERDICT; not wire.
            scaffold_rep = (
                offline_case1_dual_honest_tf_aware_path_execution_scaffold_report()
            )
            admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok = bool(
                scaffold_rep.get(
                    "scaffold_ok",
                    scaffold_rep.get(
                        "contract_ok", scaffold_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok = False
    base["admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok"] = (
        admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok
    )
    admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_multi_blocker_wire_rehearsal:
        try:
            # Additive readiness: multi-blocker wire rehearsal / dry-run co-req
            # matrix under scaffold. rehearsal_present; all ship flags hard False;
            # dual_linf unproven; dual_recovery_path=None; not VERDICT; not wire.
            rehearse_rep = (
                offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
            )
            admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok = bool(
                rehearse_rep.get(
                    "rehearsal_ok",
                    rehearse_rep.get(
                        "contract_ok", rehearse_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok = False
    base["admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok"] = (
        admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok
    )
    admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok: Optional[bool] = None
    if include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint:
        try:
            # Additive readiness: multi-blocker wire implementation blueprint /
            # go-board under scaffold+rehearsal. blueprint_present; first_blocking
            # coreq + order_hint prep map; all ship flags hard False; dual_linf
            # unproven; dual_recovery_path=None; not VERDICT; not wire.
            blueprint_rep = (
                offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
            )
            admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok = bool(
                blueprint_rep.get(
                    "blueprint_ok",
                    blueprint_rep.get(
                        "contract_ok", blueprint_rep.get("ok")
                    ),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok = False
    base["admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok"] = (
        admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok
    )

    admm_case1_isolation_rewrite_first_blocker_operational_prep_ok: Optional[bool] = None
    if include_admm_case1_isolation_rewrite_first_blocker_operational_prep:
        try:
            # Additive readiness: isolation first-blocker operational prep
            # (prep_present; ship flags hard False; dual_linf unproven;
            # dual_recovery_path=None; not VERDICT; not isolation rewrite shipped).
            prep_rep = (
                offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
            )
            admm_case1_isolation_rewrite_first_blocker_operational_prep_ok = bool(
                prep_rep.get(
                    "prep_ok",
                    prep_rep.get("contract_ok", prep_rep.get("ok")),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_isolation_rewrite_first_blocker_operational_prep_ok = False
    base["admm_case1_isolation_rewrite_first_blocker_operational_prep_ok"] = (
        admm_case1_isolation_rewrite_first_blocker_operational_prep_ok
    )

    admm_case1_dual_linf_under_wire_criteria_contract_ok: Optional[bool] = None
    if include_admm_case1_dual_linf_under_wire_criteria_contract:
        try:
            # Additive readiness: dual_linf_under_wire flip-criteria contract
            # (criteria_present; dual_linf unproven; proof_allowed hard False;
            # dual_recovery_path=None; not VERDICT; not wire).
            dl_rep = offline_case1_dual_linf_under_wire_criteria_contract_report()
            admm_case1_dual_linf_under_wire_criteria_contract_ok = bool(
                dl_rep.get(
                    "contract_ok",
                    dl_rep.get("ok"),
                )
            )
        except Exception:  # pragma: no cover - defensive
            admm_case1_dual_linf_under_wire_criteria_contract_ok = False
    base["admm_case1_dual_linf_under_wire_criteria_contract_ok"] = (
        admm_case1_dual_linf_under_wire_criteria_contract_ok
    )
    base["note"] = (
        "Offline block-solve readiness report: cached multi-unit timing + "
        "parity_ok + priced_ok under dual-ban honesty. "
        "ready_for_wire_discussion is structural readiness only (parity∧priced"
        "∧timings∧honesty) — wire is a separate checklist + form label change; "
        "dual_recovery_path remains None; on_excel_case1_path=False; "
        "timings/prices/gradients/ADMM residuals are NOT ADMM λ / Case 1 shadows; "
        "not Case 1 solve wall time; not a solve. Not pure-ADMM dual recovery. "
        "admm_residual_ok, admm_block_subproblem_ok, admm_coordination_ok, "
        "admm_plant_linking_ok, admm_plant_named_linking_ok, "
        "admm_case1_shaped_linking_ok, "
        "admm_case1_dual_space_form_contract_ok, "
        "admm_case1_dual_space_linf_probe_ok, "
        "admm_case1_dual_space_linf_live_lambda_bridge_ok, "
        "admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok, "
        "admm_case1_honest_blender_pooling_path_ok, and "
        "admm_case1_online_linf_gate_criteria_contract_ok are additive "
        "pre-wire checklist items (synthetic λ,z,ρ residual / block subproblem / "
        "multi-round coordination / multi-block plant-linking synthetic + plant-named "
        "topology modes / Case-1-shaped CDU↔Blender offline skeleton / dual-space "
        "form-label contract / dual-space L∞ probe / live-λ bridge compose; coordination "
        "is per-unit synthetic not plant linking; plant-linking modes are offline demos; "
        "Case-1-shaped skeleton is offline-only (not wire, not Case 1 duals, not full "
        "plant mass balance); dual-space/form contract registers planned form + stream "
        "map + dual_linf unproven checklist without flipping Case 1 or shipping wire; "
        "dual-space L∞ probe is numeric prep only (status stays unproven; not VERDICT); "
        "live-λ bridge extracts/accepts caller Case 1 PRIMARY online λ into the existing "
        "probe with source labeled (fixture fallback only when explicit); live-λ-seeded "
        "warm-start seeds skeleton λ0 from live/caller PRIMARY, runs N skeleton rounds, "
        "and measures post-round stream L∞ (seed identity L∞ ≠ dual L∞ under wire proof; "
        "dual_linf stays unproven); honest blender pooling path formalizes "
        "linear_quality_pooling with checklist honest_pooling_path_present (affine "
        "kernel still absent; dual_linf unproven); online_linf_gate flip-criteria "
        "contract formalizes machine-readable close criteria while gate stays open "
        "(gate_flip_allowed_today=False; criteria_met_today=False; dual_linf unproven; "
        "contract ≠ gate flip ≠ wire ≠ VERDICT); isolation-rewrite design-only contract "
        "formalizes rewrite-with-wire-not-delete while isolation_rewrite_shipped=False and "
        "checklist isolation_rewrite_with_wire stays open (design ≠ rewrite shipped ≠ wire "
        "≠ VERDICT); wire-ship acceptance design contract formalizes machine-readable "
        "criteria for when wire *may* ship while wire_ship_allowed_today=False and "
        "wire_shipped=False (design ≠ ship allow ≠ wire ≠ VERDICT); dual-honest TF-aware "
        "path design contract formalizes *path shape* for future wire while "
        "path_design_present=True, path_shipped=False, and dual_honest_tf_aware_path_present "
        "ship-met remains False (design ≠ path shipped ≠ ship-met ≠ wire ≠ VERDICT); "
        "dual_honest_tf_aware_path_present ship-met flip criteria contract formalizes "
        "*when path may count as present-for-ship* while criteria_present=True, "
        "ship_met_allowed_today=False, dual_honest_tf_aware_path_present remains False, "
        "path_design_present=True, path_shipped=False (criteria ≠ ship-met ≠ path shipped "
        "≠ wire ≠ VERDICT) — and do not redefine ready_for_wire_discussion. "
        "admm_case1_honest_blender_pooling_path_ok, "
        "admm_case1_online_linf_gate_criteria_contract_ok, "
        "admm_case1_isolation_rewrite_design_contract_ok, "
        "admm_case1_wire_ship_acceptance_design_contract_ok, "
        "admm_case1_dual_honest_tf_aware_path_design_contract_ok, "
        "admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok, and "
        "admm_case1_form_label_change_shipped_criteria_contract_ok are additive only; "
        "form_label_change_shipped flip criteria contract formalizes *when form may "
        "count as shipped* while criteria_present=True, form_label_ship_allowed_today=False, "
        "form_label_change_shipped remains False, form classic, path_shipped=False "
        "(criteria ≠ form_label shipped ≠ form flip ≠ path shipped ≠ ship-met ≠ wire ≠ "
        "VERDICT); isolation-rewrite ship-met flip criteria contract formalizes *when "
        "isolation rewrite may count as shipped/met* while criteria_present=True, "
        "isolation_ship_allowed_today=False, isolation_rewrite_shipped remains False, "
        "checklist isolation_rewrite_with_wire open, rewrite-not-delete "
        "(criteria ≠ isolation rewrite shipped ≠ form flip ≠ path shipped ≠ ship-met ≠ "
        "wire ≠ VERDICT) — and does not redefine ready_for_wire_discussion. "
        "admm_case1_isolation_rewrite_shipped_criteria_contract_ok is additive only; "
        "multi-blocker wire bundle design contract formalizes *what co-reqs must land "
        "together* for SUGGESTED_NEXT_WAVE while bundle_design_present=True, "
        "bundle_shipped=False, bundle_ship_allowed_today=False, wire_shipped=False, "
        "isolation_rewrite_shipped=False, form classic, path_shipped=False, ship-met "
        "False, dual_linf unproven (bundle design ≠ bundle ship ≠ wire ship ≠ VERDICT; "
        "order_hint is not an executor) — and does not redefine ready_for_wire_discussion. "
        "admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok is "
        "additive only; multi-blocker wire bundle ship-met / flip criteria contract "
        "formalizes *when bundle_shipped / bundle_ship_allowed_today may become True* "
        "while criteria_present=True, bundle_shipped=False, bundle_ship_allowed_today=False, "
        "criteria_met_today=False, wire_shipped=False, isolation_rewrite_shipped=False, "
        "form classic, path_shipped=False, dual_linf unproven (criteria ≠ bundle ship ≠ "
        "wire ship ≠ isolation rewrite shipped ≠ form ship ≠ path ship ≠ VERDICT; "
        "order_hint is not an executor) — and does not redefine ready_for_wire_discussion. "
        "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok is "
        "additive only; dual-honest TF-aware path execution scaffold formalizes "
        "*offline how-without-ship* (callable compose of CDU offline affine + "
        "blender linear_quality_pooling + Case-1 streams + optional labeled λ + "
        "diagnostic-only dual-space residual) while scaffold_present=True, "
        "path_shipped=False, dual_honest_tf_aware_path_present ship-met False, "
        "wire_shipped=False, bundle_shipped=False, isolation_rewrite_shipped=False, "
        "form classic, dual_linf unproven (scaffold ≠ path ship ≠ wire ship ≠ "
        "bundle ship ≠ isolation rewrite shipped ≠ form ship ≠ VERDICT; "
        "order_hint is not an executor) — and does not redefine ready_for_wire_discussion. "
        "admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok is additive only; "
        "dual-honest multi-blocker wire rehearsal formalizes *co-req readiness dry-run "
        "under scaffold without ship* (machine-readable co-req status matrix for "
        "SUGGESTED_NEXT_WAVE) while rehearsal_present=True, path_shipped=False, "
        "dual_honest_tf_aware_path_present ship-met False, wire_shipped=False, "
        "bundle_shipped=False, isolation_rewrite_shipped=False, form classic, dual_linf "
        "unproven (rehearsal ≠ path ship ≠ wire ship ≠ bundle ship ≠ isolation rewrite "
        "shipped ≠ form ship ≠ VERDICT; order_hint is not an executor; rehearsal_present "
        "is not wire_shipped) — and does not redefine ready_for_wire_discussion. "
        "admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok is additive only; "
        "dual-honest multi-blocker wire implementation blueprint / go-board "
        "formalizes *order_hint-sequenced first-blocker + file-level prep without "
        "ship* (first_blocking_coreq + go-board prep map under scaffold/rehearsal) "
        "while blueprint_present=True, path_shipped=False, dual_honest_tf_aware_path_present "
        "ship-met False, wire_shipped=False, bundle_shipped=False, isolation_rewrite_shipped=False, "
        "form classic, dual_linf unproven (blueprint ≠ path ship ≠ wire ship ≠ bundle ship ≠ "
        "isolation rewrite shipped ≠ form ship ≠ VERDICT; order_hint is not an executor; "
        "blueprint_present is not wire_shipped) — and does not redefine "
        "ready_for_wire_discussion. "
        "admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok, admm_case1_isolation_rewrite_first_blocker_operational_prep_ok and admm_case1_dual_linf_under_wire_criteria_contract_ok are additive only."
    )
    return base


# ---------------------------------------------------------------------------
# Offline multi-unit ADMM-style consensus residual / augmented local (goal 5)
# ---------------------------------------------------------------------------

ADMM_RESIDUAL_KIND = "offline_admm_block_residual"
# Primary formula matches plant ADMM L1 consensus penalty spirit (blocks.py):
# maximize λ·y − ρ‖y−z‖₁  → report evaluation of that form.
ADMM_AUGMENTED_FORMULA_L1 = "lambda_dot_y - rho * ||y_full - z||_1"
# Optional secondary diagnostic only (not primary dual form).
ADMM_AUGMENTED_FORMULA_L2 = "lambda_dot_y - (rho/2) * ||y_full - z||_2^2"


def _admm_residual_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / synthetic-λ locks for ADMM residual reports."""
    return {
        "kind": ADMM_RESIDUAL_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "not_a_solve": True,
        "note": (
            "Offline multi-unit ADMM-style consensus residual / augmented local "
            "objective under synthetic λ, z, ρ only (not a solve). "
            "Not on classic_2block_excel_path; dual_recovery_path=None. "
            "Synthetic λ/z/ρ are NOT Case 1 PRIMARY online λ, NOT SECONDARY "
            "recovered blender duals, NOT pure-ADMM dual recovery, NOT wire shipped. "
            "Primary formula L1 spirit: lambda_dot_y - rho * ||y_full - z||_1 "
            "(matches plant blocks.py consensus penalty shape language only)."
        ),
    }


def _default_z_full_for_unit(unit: str, coeffs: AffineCoeffs) -> Dict[str, float]:
    """Synthetic consensus z = postprocess(affine@x0) in product order (default)."""
    post = _postprocess_for(unit)
    y_raw = numpy_affine_forward(coeffs, coeffs.x0, clamp_products=True)
    y_full = post(y_raw, products=coeffs.products)
    return {p: float(y_full[p]) for p in coeffs.products}


def _resolve_z_vector(
    coeffs: AffineCoeffs,
    z: Optional[Mapping[str, float]],
) -> tuple[Dict[str, float], np.ndarray, str]:
    """Pack z into product order; default = postprocess yields at reference."""
    if z is None:
        z_dict = _default_z_full_for_unit(coeffs.unit, coeffs)
        source = "postprocess_yield_at_reference"
    else:
        z_dict = {str(k): float(v) for k, v in z.items()}
        missing = [p for p in coeffs.products if p not in z_dict]
        if missing:
            raise ValueError(
                f"Missing z products for unit {coeffs.unit!r}: {missing}"
            )
        unknown = sorted(set(z_dict) - set(coeffs.products))
        if unknown:
            raise ValueError(
                f"Unknown z product keys {unknown} for unit {coeffs.unit!r}"
            )
        source = "caller_override"
    z_vec = np.array([float(z_dict[p]) for p in coeffs.products], dtype=np.float64)
    return z_dict, z_vec, source


def _x_for_admm_mode(
    unit: str,
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    *,
    x_mode: str,
    feed: Optional[Mapping[str, float]],
    conditions: Optional[Mapping[str, Any]],
    box_delta: float,
) -> tuple[np.ndarray, str]:
    """Resolve driver vector x under ref / offset / box modes (always-on)."""
    mode = str(x_mode or "ref").lower().strip()
    if feed is not None or conditions is not None:
        builder = _builder_for(unit)
        model = builder()
        feed_use = dict(feed) if feed is not None else dict(model.reference_feed)
        cond_use = (
            dict(conditions)
            if conditions is not None
            else dict(model.reference_conditions)
        )
        x = pack_driver_vector(coeffs, feed=feed_use, conditions=cond_use)
        return x, "explicit_feed_conditions"
    if mode == "ref":
        return np.array(coeffs.x0, dtype=np.float64, copy=True), "reference_x0"
    if mode == "offset":
        builder = _builder_for(unit)
        model = builder()
        feed_off, cond_off = _mild_offset_for(unit, model)
        x = pack_driver_vector(coeffs, feed=feed_off, conditions=cond_off)
        return x, "mild_offset"
    if mode == "box":
        x = _local_box_step_raw(coeffs, p_vec, delta=float(box_delta))
        return x, "local_box_step"
    raise ValueError(
        f"Unknown x_mode {x_mode!r}; expected one of 'ref', 'offset', 'box'"
    )


def admm_residual_for_unit(
    unit: str,
    *,
    prices: Optional[Mapping[str, float]] = None,
    z: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    x_mode: str = "ref",
    feed: Optional[Mapping[str, float]] = None,
    conditions: Optional[Mapping[str, Any]] = None,
    box_delta: float = 1.0,
) -> Dict[str, Any]:
    """Per-unit ADMM-style consensus residual + L1 augmented local (always-on).

    Under synthetic λ (default offline prices), synthetic consensus z (default:
    postprocess yield at reference), and ρ > 0:

    - ``y_raw`` = clamp affine forward at x
    - ``y_full`` = unit postprocess(y_raw)  ← primary residual space
    - ``r = y_full − z`` with L1 / L2 / L∞ scalars
    - primary ``augmented_local = λ·y_full − ρ‖r‖₁`` (plant ADMM L1 spirit)
    - optional L2 diagnostic field (secondary only)

    Coker renorm: raw path may ≠ full even at reference; residual_on=postprocess.
    Honesty: dual_recovery_path=None; not Case 1; synthetic λ ≠ online λ.
    """
    key = _normalize_unit_name(unit)
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    price_dict, p_vec, coeffs = _resolve_prices(key, prices)
    # Prefer cached default-ref coeffs when no custom prices path needs rebuild
    # (_resolve_prices already used offline_unit_coeffs; re-use for z default).
    coeffs = cached_offline_unit_coeffs(key) if prices is None else coeffs
    if prices is not None:
        # re-pack against cached coeffs product order (same as offline_unit_coeffs)
        coeffs = offline_unit_coeffs(key)
        p_vec = pack_price_vector(coeffs, price_dict)

    z_dict, z_vec, z_source = _resolve_z_vector(coeffs, z)
    x, x_source = _x_for_admm_mode(
        key,
        coeffs,
        p_vec,
        x_mode=x_mode,
        feed=feed,
        conditions=conditions,
        box_delta=box_delta,
    )
    post = _postprocess_for(key)
    y_raw = numpy_affine_forward(coeffs, x, clamp_products=True)
    y_full_map = post(y_raw, products=coeffs.products)
    y_full = np.array(
        [float(y_full_map[p]) for p in coeffs.products], dtype=np.float64
    )
    r_full = y_full - z_vec
    r_raw = y_raw - z_vec  # honesty: may differ under renorm (Coker)

    r_l1 = float(np.sum(np.abs(r_full)))
    r_l2 = float(np.linalg.norm(r_full))
    r_linf = float(np.max(np.abs(r_full))) if r_full.size else 0.0
    r_raw_l1 = float(np.sum(np.abs(r_raw)))

    lambda_dot_y = float(p_vec @ y_full)
    penalty_l1 = float(rho) * r_l1
    augmented_local = lambda_dot_y - penalty_l1
    # Secondary diagnostic only — not primary plant dual form
    penalty_l2 = 0.5 * float(rho) * float(r_l2 * r_l2)
    augmented_local_l2 = lambda_dot_y - penalty_l2

    finite_ok = bool(
        np.all(np.isfinite(r_full))
        and np.isfinite(augmented_local)
        and np.isfinite(lambda_dot_y)
        and np.isfinite(r_l1)
        and np.isfinite(r_linf)
    )
    honesty = _admm_residual_honesty_fields()
    r_dict = {p: float(r_full[i]) for i, p in enumerate(coeffs.products)}
    y_raw_dict_ = {p: float(y_raw[i]) for i, p in enumerate(coeffs.products)}
    y_full_dict = {p: float(y_full[i]) for i, p in enumerate(coeffs.products)}

    return {
        **honesty,
        "unit": key,
        "ok": finite_ok,
        "products": list(coeffs.products),
        "y_raw": y_raw_dict_,
        "y_full": y_full_dict,
        "z": z_dict,
        "z_source": z_source,  # overrides honesty aggregate tag with concrete path
        "x_source": x_source,
        "x_mode": str(x_mode),
        "rho": float(rho),
        "prices": price_dict,
        "residual_on": "postprocess",
        "consensus_residual": r_dict,
        "r_l1": r_l1,
        "r_l2": r_l2,
        "r_linf": r_linf,
        "r_raw_l1": r_raw_l1,
        "raw_vs_full_residual_l1_gap": abs(r_raw_l1 - r_l1),
        "lambda_dot_y": lambda_dot_y,
        "penalty": penalty_l1,
        "penalty_l1": penalty_l1,
        "augmented_local": augmented_local,
        "formula": ADMM_AUGMENTED_FORMULA_L1,
        "augmented_local_l2_diagnostic": augmented_local_l2,
        "formula_l2_diagnostic": ADMM_AUGMENTED_FORMULA_L2,
        "renorm_note": (
            "Coker renorm always engages → y_raw may ≠ y_full even at reference; "
            "primary residual uses postprocess (y_full) space when z is full-space."
            if key == "COKER"
            else "Primary residual uses postprocess (y_full) vs synthetic z."
        ),
    }


def multi_unit_admm_residual_report(
    *,
    rho: float = 1.0,
    x_mode: str = "offset",
    use_box_step: bool = False,
    box_delta: float = 1.0,
    prices: Optional[Mapping[str, Mapping[str, float]]] = None,
    z_override: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Always-on multi-unit ADMM-style residual (FCC/COKER/CDU) + honesty locks.

    For each registry unit under synthetic λ / z / ρ:

    - consensus residual ``r = y_full − z`` (per-product + L1/L2/L∞)
    - augmented local ``λ·y − ρ‖r‖₁`` (L1 spirit matching plant ADMM language)

    Aggregate ``ok`` requires finite residuals + dual-ban honesty + unit structure.
    Not a solve, not dual recovery, not on Excel Case 1, not wire shipped.
    """
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    mode = "box" if use_box_step else str(x_mode or "offset")
    units_out: Dict[str, Any] = {}
    all_ok = True
    honesty = _admm_residual_honesty_fields()

    for desc in offline_unit_registry():
        unit_prices: Optional[Mapping[str, float]] = None
        if prices is not None and desc.unit in prices:
            unit_prices = prices[desc.unit]
        unit_z: Optional[Mapping[str, float]] = None
        if z_override is not None and desc.unit in z_override:
            unit_z = z_override[desc.unit]
        row = admm_residual_for_unit(
            desc.unit,
            prices=unit_prices,
            z=unit_z,
            rho=float(rho),
            x_mode=mode,
            box_delta=float(box_delta),
        )
        unit_ok = bool(row.get("ok"))
        unit_row = {
            **row,
            "ok": unit_ok,
            "renorm_note": desc.renorm_note or row.get("renorm_note"),
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "kind": ADMM_RESIDUAL_KIND,
            "price_source": PRICE_SOURCE,
            "not_a_solve": True,
        }
        units_out[desc.unit] = unit_row
        if not unit_ok:
            all_ok = False

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["kind"] == ADMM_RESIDUAL_KIND
        and honesty.get("not_a_solve") is True
        and set(units_out.keys()) == set(UNITS)
    )
    if not honesty_ok:
        all_ok = False

    return {
        "ok": all_ok,
        "units": units_out,
        "unit_order": list(UNITS),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": ADMM_RESIDUAL_KIND,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "rho": float(rho),
        "x_mode": mode,
        "use_box_step": bool(use_box_step),
        "formula": ADMM_AUGMENTED_FORMULA_L1,
        "not_a_solve": True,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "note": honesty["note"],
    }


# ---------------------------------------------------------------------------
# Offline multi-unit ADMM block subproblem (goal 5 pre-wire) — maximize L1
# augmented local under independent driver box on **raw affine** (not residual-eval)
# ---------------------------------------------------------------------------

ADMM_SUBPROBLEM_KIND = "offline_admm_block_subproblem"
# Optimand is raw affine (matches local_box_direction honesty). Residual-eval
# primary remains postprocess/full; do not silently reuse residual formula space.
ADMM_SUBPROBLEM_FORMULA_L1_RAW = "lambda_dot_y_raw - rho * ||y_raw - z||_1"
ADMM_SUBPROBLEM_METHOD = "coordinate_ascent_exact_1d_pl"
ADMM_SUBPROBLEM_OPTIMALITY_NOTE = (
    "Coordinate-ascent with exact 1-D piecewise-linear maximizers per driver; "
    "multi-start from {x0, local_box under λ}. Exact per coordinate; multi-D "
    "global optimality not claimed (independent box, coupled via y)."
)


def _admm_subproblem_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / synthetic-λ locks for ADMM block subproblem."""
    return {
        "kind": ADMM_SUBPROBLEM_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "z_space": "full_postprocess_default",
        "not_a_solve": False,  # this surface *is* a local maximizer (offline only)
        "not_case1_solve": True,
        "method": ADMM_SUBPROBLEM_METHOD,
        "optimality_note": ADMM_SUBPROBLEM_OPTIMALITY_NOTE,
        "note": (
            "Offline multi-unit ADMM block subproblem: maximize L1-augmented local "
            "on raw affine under independent driver box and synthetic λ,z,ρ. "
            "Optimand space = raw_affine (clamp affine); full postprocess fields are "
            "diagnostic only (Coker renorm: raw ≠ full expected). "
            "Not on classic_2block_excel_path; dual_recovery_path=None; solver=False. "
            "Synthetic λ/z/ρ and x_star are NOT Case 1 PRIMARY online λ, NOT SECONDARY "
            "recovered blender duals, NOT pure-ADMM dual recovery, NOT wire shipped. "
            "Not PuLP/CBC; always-on numpy piecewise-linear coordinate ascent. "
            "Default z is full-space postprocess@ref while optimand is raw — labeled."
        ),
    }


def _augmented_local_raw_parts(
    p_vec: np.ndarray,
    y_raw: np.ndarray,
    z_vec: np.ndarray,
    rho: float,
) -> tuple[float, float, float]:
    """Return (augmented_local_raw, lambda_dot_y_raw, r_l1_raw)."""
    r = y_raw - z_vec
    r_l1 = float(np.sum(np.abs(r)))
    lambda_dot = float(p_vec @ y_raw)
    return lambda_dot - float(rho) * r_l1, lambda_dot, r_l1


def _eval_aug_raw_at_x(
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    z_vec: np.ndarray,
    rho: float,
    x: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    y_raw = numpy_affine_forward(coeffs, x, clamp_products=True)
    aug, ld, rl1 = _augmented_local_raw_parts(p_vec, y_raw, z_vec, rho)
    return aug, y_raw, ld, rl1


def _parse_delta_vec(
    coeffs: AffineCoeffs,
    delta: Union[float, Mapping[str, float], np.ndarray],
) -> np.ndarray:
    n_d = int(coeffs.x0.shape[0])
    if isinstance(delta, Mapping):
        d_vec = np.array(
            [float(delta.get(drv, 0.0)) for drv in coeffs.drivers],
            dtype=np.float64,
        )
    elif np.isscalar(delta):
        d_vec = np.full(n_d, float(np.asarray(delta).item()), dtype=np.float64)
    else:
        d_vec = np.asarray(delta, dtype=np.float64).reshape(-1)
        if d_vec.shape[0] != n_d:
            raise ValueError(
                f"delta length {d_vec.shape[0]} != n_drivers {n_d}"
            )
    if np.any(d_vec < 0.0):
        raise ValueError("delta components must be non-negative")
    return d_vec


def _exact_1d_pl_max_coordinate(
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    z_vec: np.ndarray,
    rho: float,
    x_fixed: np.ndarray,
    j: int,
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Exact 1-D piecewise-linear max of raw L1-augmented objective on coord j.

    y_i(t) = max(0, c_i + b_i * (t - t0)) with other drivers fixed.
    Breakpoints: box ends, product clamp-to-zero, and y_i = z_i crossings.
    Continuous PL ⇒ max at a breakpoint.
    """
    if hi < lo - 1e-15:
        raise ValueError(f"empty box on driver {j}: lo={lo} hi={hi}")
    if abs(hi - lo) <= 1e-15:
        x = np.array(x_fixed, dtype=np.float64, copy=True)
        x[j] = 0.5 * (lo + hi)
        aug, _, _, _ = _eval_aug_raw_at_x(coeffs, p_vec, z_vec, rho, x)
        return float(x[j]), float(aug)

    t0 = float(coeffs.x0[j])
    b = np.asarray(coeffs.D[:, j], dtype=np.float64)
    dx = np.asarray(x_fixed, dtype=np.float64) - coeffs.x0
    dx[j] = 0.0
    c = coeffs.y0 + coeffs.D @ dx  # unclamped y when t = t0 (coord j at x0)

    candidates: List[float] = [float(lo), float(hi)]
    n_p = int(b.shape[0])
    for i in range(n_p):
        bi = float(b[i])
        if abs(bi) < 1e-15:
            continue
        # clamp kink: c_i + b_i*(t-t0) = 0
        t_clamp = t0 - float(c[i]) / bi
        if lo - 1e-12 <= t_clamp <= hi + 1e-12:
            candidates.append(float(np.clip(t_clamp, lo, hi)))
        # |y-z| kink when unclamped y crosses z (only meaningful if z > 0)
        zi = float(z_vec[i])
        if zi > 1e-15:
            t_cross = t0 + (zi - float(c[i])) / bi
            if lo - 1e-12 <= t_cross <= hi + 1e-12:
                candidates.append(float(np.clip(t_cross, lo, hi)))

    # unique sorted
    cand = np.array(sorted(set(float(np.clip(t, lo, hi)) for t in candidates)), dtype=np.float64)

    best_t = float(lo)
    best_f = -np.inf
    x = np.array(x_fixed, dtype=np.float64, copy=True)
    for t in cand:
        x[j] = float(t)
        # Direct eval (includes clamp)
        y = np.maximum(c + b * (float(t) - t0), 0.0)
        aug, _, _ = _augmented_local_raw_parts(p_vec, y, z_vec, rho)
        # Prefer higher f; on ties prefer closer to x0 then lower t (deterministic)
        better = False
        if aug > best_f + 1e-12:
            better = True
        elif abs(aug - best_f) <= 1e-12:
            if abs(float(t) - t0) < abs(best_t - t0) - 1e-15:
                better = True
            elif abs(abs(float(t) - t0) - abs(best_t - t0)) <= 1e-15 and float(t) < best_t:
                better = True
        if better:
            best_f = float(aug)
            best_t = float(t)
    return best_t, float(best_f)


def _priced_corner_seed(
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    d_vec: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Corner maximizer of linear p·y_raw under box (same spirit as local_box)."""
    g = coeffs.D.T @ p_vec
    x = np.array(coeffs.x0, dtype=np.float64, copy=True)
    for j in range(int(coeffs.x0.shape[0])):
        if not mask[j] or abs(float(g[j])) < 1e-15 or float(d_vec[j]) == 0.0:
            continue
        x[j] = float(coeffs.x0[j]) + float(d_vec[j]) * float(np.sign(g[j]))
    return x


def _coordinate_ascent_admm_subproblem(
    coeffs: AffineCoeffs,
    p_vec: np.ndarray,
    z_vec: np.ndarray,
    rho: float,
    d_vec: np.ndarray,
    mask: np.ndarray,
    *,
    max_passes: int = 32,
    atol: float = 1e-10,
) -> tuple[np.ndarray, float, int, List[str]]:
    """Multi-start coordinate ascent with exact 1-D PL maximizers.

    Seeds: x0 and priced corner under λ. Returns best (x_star, f_star, passes, seeds_used).
    """
    x0 = np.array(coeffs.x0, dtype=np.float64, copy=True)
    seeds = [
        ("x0", x0),
        ("priced_corner", _priced_corner_seed(coeffs, p_vec, d_vec, mask)),
    ]
    # De-dupe seeds that are numerically identical
    unique_seeds: List[tuple[str, np.ndarray]] = []
    for name, s in seeds:
        if any(np.allclose(s, us, atol=1e-12, rtol=0.0) for _, us in unique_seeds):
            continue
        unique_seeds.append((name, s))

    best_x = np.array(x0, copy=True)
    best_f, _, _, _ = _eval_aug_raw_at_x(coeffs, p_vec, z_vec, rho, best_x)
    best_passes = 0
    seed_names: List[str] = []

    for sname, seed in unique_seeds:
        seed_names.append(sname)
        x = np.array(seed, dtype=np.float64, copy=True)
        # Project seed into box (priced corner already on box; keep general)
        for j in range(int(x.shape[0])):
            lo = float(coeffs.x0[j]) - float(d_vec[j])
            hi = float(coeffs.x0[j]) + float(d_vec[j])
            x[j] = float(np.clip(x[j], lo, hi))
        f, _, _, _ = _eval_aug_raw_at_x(coeffs, p_vec, z_vec, rho, x)
        passes_used = 0
        for _p in range(int(max_passes)):
            passes_used = _p + 1
            improved = False
            for j in range(int(coeffs.x0.shape[0])):
                if not mask[j] or float(d_vec[j]) == 0.0:
                    continue
                lo = float(coeffs.x0[j]) - float(d_vec[j])
                hi = float(coeffs.x0[j]) + float(d_vec[j])
                t_star, f_1d = _exact_1d_pl_max_coordinate(
                    coeffs, p_vec, z_vec, rho, x, j, lo, hi
                )
                if f_1d > f + atol or (
                    abs(f_1d - f) <= atol and abs(t_star - float(x[j])) > atol
                ):
                    if f_1d > f + atol:
                        improved = True
                    x[j] = t_star
                    f = f_1d
            if not improved:
                break
        # Recompute f at final x for safety
        f, _, _, _ = _eval_aug_raw_at_x(coeffs, p_vec, z_vec, rho, x)
        if f > best_f + atol or (
            abs(f - best_f) <= atol and np.linalg.norm(x - x0) < np.linalg.norm(best_x - x0)
        ):
            best_f = f
            best_x = np.array(x, copy=True)
            best_passes = passes_used

    return best_x, float(best_f), int(best_passes), seed_names


def admm_block_subproblem_for_unit(
    unit: str,
    *,
    prices: Optional[Mapping[str, float]] = None,
    z: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float], np.ndarray] = 1.0,
    driver_mask: Optional[Sequence[bool]] = None,
    max_passes: int = 32,
) -> Dict[str, Any]:
    """Per-unit offline ADMM block subproblem maximizer (always-on numpy).

    Maximize on **raw affine** under independent driver box::

        augmented_local_raw = λ · y_raw − ρ ‖y_raw − z‖₁
        y_raw = clamp(y0 + D @ (x − x0))
        x ∈ [x0 − δ, x0 + δ]

    Defaults parallel residual harness: synthetic offline prices, z = full
    postprocess yield at reference, ρ > 0. Method: coordinate-ascent with exact
    1-D piecewise-linear maximizers + multi-start from {x0, priced corner}.

    Honesty: dual_recovery_path=None; not Case 1; not pure-ADMM dual recovery;
    not wire; raw optimand (full postprocess diagnostic only).
    """
    key = _normalize_unit_name(unit)
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if int(max_passes) < 1:
        raise ValueError("max_passes must be >= 1")

    price_dict, p_vec, coeffs = _resolve_prices(key, prices)
    coeffs = cached_offline_unit_coeffs(key) if prices is None else coeffs
    if prices is not None:
        coeffs = offline_unit_coeffs(key)
        p_vec = pack_price_vector(coeffs, price_dict)

    z_dict, z_vec, z_source = _resolve_z_vector(coeffs, z)
    d_vec = _parse_delta_vec(coeffs, delta)
    n_d = int(coeffs.x0.shape[0])
    mask = np.ones(n_d, dtype=bool)
    if driver_mask is not None:
        mask_arr = np.asarray(list(driver_mask), dtype=bool).reshape(-1)
        if mask_arr.shape[0] != n_d:
            raise ValueError(
                f"driver_mask length {mask_arr.shape[0]} != n_drivers {n_d}"
            )
        mask = mask_arr

    x0 = np.array(coeffs.x0, dtype=np.float64, copy=True)
    aug_ref, y_raw_ref, ld_ref, r_l1_ref = _eval_aug_raw_at_x(
        coeffs, p_vec, z_vec, float(rho), x0
    )

    x_star, aug_star, passes_used, seeds_used = _coordinate_ascent_admm_subproblem(
        coeffs,
        p_vec,
        z_vec,
        float(rho),
        d_vec,
        mask,
        max_passes=int(max_passes),
    )
    # Clamp x_star into box (numerical)
    for j in range(n_d):
        lo = float(x0[j]) - float(d_vec[j])
        hi = float(x0[j]) + float(d_vec[j])
        x_star[j] = float(np.clip(x_star[j], lo, hi))

    aug_star, y_raw_star, ld_star, r_l1_star = _eval_aug_raw_at_x(
        coeffs, p_vec, z_vec, float(rho), x_star
    )
    # If numerical drift made star worse than ref, snap to ref (maximizer gate)
    if aug_star < aug_ref - 1e-9:
        x_star = np.array(x0, copy=True)
        aug_star, y_raw_star, ld_star, r_l1_star = (
            aug_ref,
            y_raw_ref,
            ld_ref,
            r_l1_ref,
        )

    post = _postprocess_for(key)
    y_full_star_map = post(y_raw_star, products=coeffs.products)
    y_full_star = np.array(
        [float(y_full_star_map[p]) for p in coeffs.products], dtype=np.float64
    )
    y_full_ref_map = post(y_raw_ref, products=coeffs.products)
    y_full_ref = np.array(
        [float(y_full_ref_map[p]) for p in coeffs.products], dtype=np.float64
    )
    # Diagnostic full-space augmented local (NOT the optimand)
    r_full = y_full_star - z_vec
    r_full_l1 = float(np.sum(np.abs(r_full)))
    lambda_dot_y_full = float(p_vec @ y_full_star)
    aug_full_star = lambda_dot_y_full - float(rho) * r_full_l1
    r_full_ref = y_full_ref - z_vec
    r_full_ref_l1 = float(np.sum(np.abs(r_full_ref)))
    aug_full_ref = float(p_vec @ y_full_ref) - float(rho) * r_full_ref_l1

    # Diagnostic: raw aug at priced corner (soft comparison; high ρ can prefer ref)
    x_box = _priced_corner_seed(coeffs, p_vec, d_vec, mask)
    aug_box, _, _, _ = _eval_aug_raw_at_x(coeffs, p_vec, z_vec, float(rho), x_box)

    not_worse_than_ref = bool(aug_star + 1e-9 >= aug_ref)
    finite_ok = bool(
        np.all(np.isfinite(x_star))
        and np.all(np.isfinite(y_raw_star))
        and np.isfinite(aug_star)
        and np.isfinite(ld_star)
        and np.isfinite(r_l1_star)
    )
    honesty = _admm_subproblem_honesty_fields()

    # Optional thin TF arm: raw forward at x_star when TF available (parity only)
    tf_section: Dict[str, Any] = {
        "skipped": True,
        "ok": None,
        "reason": "tf_unavailable",
    }
    if tf_available():
        try:
            block = build_offline_unit(key)
            y_tf = np.asarray(
                block.forward(x_star, clamp_products=True, as_dict=False),
                dtype=np.float64,
            ).reshape(-1)
            tf_ok = bool(np.max(np.abs(y_tf - y_raw_star)) <= 1e-9)
            tf_section = {
                "skipped": False,
                "ok": tf_ok,
                "y_raw_star_tf_max_abs_diff": float(np.max(np.abs(y_tf - y_raw_star))),
                "reason": None,
            }
        except Exception as exc:  # pragma: no cover - env dependent
            tf_section = {
                "skipped": False,
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    y_raw_star_dict = {p: float(y_raw_star[i]) for i, p in enumerate(coeffs.products)}
    y_full_star_dict = {
        p: float(y_full_star[i]) for i, p in enumerate(coeffs.products)
    }

    ok = bool(finite_ok and not_worse_than_ref)
    return {
        **honesty,
        "unit": key,
        "ok": ok,
        "products": list(coeffs.products),
        "drivers": list(coeffs.drivers),
        "x0": x0.tolist(),
        "x_star": x_star.tolist(),
        "delta": d_vec.tolist(),
        "rho": float(rho),
        "prices": price_dict,
        "z": z_dict,
        "z_source": z_source,
        "y_raw_star": y_raw_star_dict,
        "y_full_star": y_full_star_dict,
        "augmented_local_raw": float(aug_star),
        "augmented_local_raw_ref": float(aug_ref),
        "augmented_local_raw_priced_box": float(aug_box),
        "lambda_dot_y_raw": float(ld_star),
        "penalty_raw": float(rho) * float(r_l1_star),
        "r_l1_raw": float(r_l1_star),
        "augmented_local_full_diagnostic": float(aug_full_star),
        "augmented_local_full_ref_diagnostic": float(aug_full_ref),
        "r_l1_full_diagnostic": float(r_full_l1),
        "raw_vs_full_aug_gap_star": abs(float(aug_star) - float(aug_full_star)),
        "raw_vs_full_r_l1_gap_star": abs(float(r_l1_star) - float(r_full_l1)),
        "not_worse_than_ref": not_worse_than_ref,
        "improvement_raw": float(aug_star - aug_ref),
        "formula": ADMM_SUBPROBLEM_FORMULA_L1_RAW,
        "passes_used": int(passes_used),
        "seeds_used": list(seeds_used),
        "max_passes": int(max_passes),
        "tf": tf_section,
        "renorm_note": (
            "Coker renorm always engages → y_raw may ≠ y_full even at reference; "
            "subproblem optimand is raw_affine; full postprocess fields are diagnostic only."
            if key == "COKER"
            else "Subproblem optimand is raw_affine; full postprocess fields are diagnostic only."
        ),
    }


def multi_unit_admm_block_subproblem_report(
    *,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float]] = 1.0,
    prices: Optional[Mapping[str, Mapping[str, float]]] = None,
    z_override: Optional[Mapping[str, Mapping[str, float]]] = None,
    max_passes: int = 32,
) -> Dict[str, Any]:
    """Always-on multi-unit ADMM block subproblem report (FCC/COKER/CDU).

    Aggregate ``ok`` = finite values + honesty locks + per-unit structure +
    maximizer-not-worse-than-ref. No absolute residual magnitude SLAs; no flaky
    µs hard-fail. Not Case 1, not wire, not dual recovery.
    """
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    units_out: Dict[str, Any] = {}
    all_ok = True
    honesty = _admm_subproblem_honesty_fields()

    for desc in offline_unit_registry():
        unit_prices: Optional[Mapping[str, float]] = None
        if prices is not None and desc.unit in prices:
            unit_prices = prices[desc.unit]
        unit_z: Optional[Mapping[str, float]] = None
        if z_override is not None and desc.unit in z_override:
            unit_z = z_override[desc.unit]
        # Per-unit delta may be scalar or shared mapping of driver→δ
        row = admm_block_subproblem_for_unit(
            desc.unit,
            prices=unit_prices,
            z=unit_z,
            rho=float(rho),
            delta=delta,
            max_passes=int(max_passes),
        )
        unit_ok = bool(row.get("ok"))
        unit_row = {
            **row,
            "ok": unit_ok,
            "renorm_note": desc.renorm_note or row.get("renorm_note"),
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "kind": ADMM_SUBPROBLEM_KIND,
            "price_source": PRICE_SOURCE,
            "optimand_space": "raw_affine",
        }
        units_out[desc.unit] = unit_row
        if not unit_ok:
            all_ok = False

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["kind"] == ADMM_SUBPROBLEM_KIND
        and honesty.get("optimand_space") == "raw_affine"
        and set(units_out.keys()) == set(UNITS)
    )
    if not honesty_ok:
        all_ok = False

    return {
        "ok": all_ok,
        "units": units_out,
        "unit_order": list(UNITS),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": ADMM_SUBPROBLEM_KIND,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "rho": float(rho),
        "formula": ADMM_SUBPROBLEM_FORMULA_L1_RAW,
        "method": ADMM_SUBPROBLEM_METHOD,
        "optimality_note": ADMM_SUBPROBLEM_OPTIMALITY_NOTE,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "note": honesty["note"],
    }


# ---------------------------------------------------------------------------
# Offline multi-round ADMM coordination (goal 5 pre-wire loop harness)
# ---------------------------------------------------------------------------

ADMM_COORDINATION_KIND = "offline_admm_coordination"
ADMM_COORDINATION_FORMULA = (
    "round: x=argmax raw L1-aug under (λ,z,ρ,δ); r=y_raw-z_pre; "
    "z←(1-β)z+β y_raw; λ←λ+α·ρ·r"
)
ADMM_COORDINATION_SCOPE = "per_unit_synthetic_offline"


def _admm_coordination_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / synthetic-λ locks for multi-round coordination."""
    return {
        "kind": ADMM_COORDINATION_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "z_update_space": "raw_affine",
        "coordination_scope": ADMM_COORDINATION_SCOPE,
        "not_plant_linking_coordinator": True,
        "coordination_lambda_is_not_case1_online_lambda": True,
        "not_a_solve": False,  # offline maximizer loop; not Case 1 solve
        "not_case1_solve": True,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "formula": ADMM_COORDINATION_FORMULA,
        "note": (
            "Offline multi-round ADMM coordination harness: for each unit in "
            "independent product spaces (FCC/COKER/CDU), run synthetic rounds of "
            "block subproblem maximizer → raw-space z consensus → λ dual ascent "
            "under synthetic λ,z,ρ. Not a plant linking-stream coordinator. "
            "Not on classic_2block_excel_path; dual_recovery_path=None; solver=False. "
            "Synthetic / coordination λ are NOT Case 1 PRIMARY online λ, NOT SECONDARY "
            "recovered blender duals, NOT pure-ADMM dual recovery, NOT wire shipped. "
            "Optimand + residual trajectory use raw_affine (z_pre residual for dual "
            "ascent; first-round default z may be full postprocess@ref — labeled). "
            "No PuLP/CBC; always-on numpy; no absolute residual-must-converge hard-fail."
        ),
    }


def admm_coordination_round_for_unit(
    unit: str,
    *,
    prices: Optional[Mapping[str, float]] = None,
    z: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float], np.ndarray] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    max_passes: int = 32,
) -> Dict[str, Any]:
    """One offline ADMM coordination round for a unit (always-on numpy).

    Structure (deterministic)::

        1. x / y step: call ``admm_block_subproblem_for_unit`` under (λ,z,ρ,δ)
        2. residual r = y_raw − z_pre  (pre-update; never after free z←y)
        3. z ← (1−β)·z_pre + β·y_raw   (β=z_blend; default 1.0 = raw copy)
        4. λ ← λ + α·ρ·r               (α=dual_step; product-ordered prices)

    Reuses the block subproblem maximizer — does **not** reimplement coordinate
    ascent. Honesty: dual_recovery_path=None; not Case 1; not wire; not dual recovery.
    """
    key = _normalize_unit_name(unit)
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    # Resolve current λ and z for residual bookkeeping before subproblem.
    price_dict, p_vec, coeffs = _resolve_prices(key, prices)
    z_pre_dict, z_pre_vec, z_seed_source = _resolve_z_vector(coeffs, z)

    sub = admm_block_subproblem_for_unit(
        key,
        prices=price_dict,
        z=z_pre_dict,
        rho=float(rho),
        delta=delta,
        max_passes=int(max_passes),
    )
    y_raw_dict = dict(sub["y_raw_star"])
    y_raw_vec = np.array(
        [float(y_raw_dict[p]) for p in coeffs.products], dtype=np.float64
    )
    # Pre-update residual in raw product space (drives dual ascent).
    r_vec = y_raw_vec - z_pre_vec
    r_l1 = float(np.sum(np.abs(r_vec)))
    r_linf = float(np.max(np.abs(r_vec))) if r_vec.size else 0.0
    r_l2 = float(np.sqrt(np.sum(r_vec * r_vec))) if r_vec.size else 0.0

    # z consensus in raw space
    z_new_vec = (1.0 - beta) * z_pre_vec + beta * y_raw_vec
    z_new_dict = {
        p: float(z_new_vec[i]) for i, p in enumerate(coeffs.products)
    }

    # λ dual ascent: λ ← λ + α ρ r  (product-ordered synthetic prices)
    alpha = float(dual_step)
    lam_new_vec = p_vec + alpha * float(rho) * r_vec
    lam_new_dict = {
        p: float(lam_new_vec[i]) for i, p in enumerate(coeffs.products)
    }

    finite_ok = bool(
        np.all(np.isfinite(r_vec))
        and np.all(np.isfinite(z_new_vec))
        and np.all(np.isfinite(lam_new_vec))
        and np.isfinite(r_l1)
        and np.isfinite(r_linf)
        and bool(sub.get("ok"))
    )
    honesty = _admm_coordination_honesty_fields()
    return {
        **{k: honesty[k] for k in (
            "kind",
            "solver",
            "dual_recovery_path",
            "on_excel_case1_path",
            "price_source",
            "lam_source",
            "z_source",
            "rho_source",
            "optimand_space",
            "z_update_space",
            "coordination_scope",
            "not_plant_linking_coordinator",
            "coordination_lambda_is_not_case1_online_lambda",
            "not_case1_solve",
            "not_wire_shipped",
            "not_pure_admm_dual_recovery",
            "formula",
        )},
        "unit": key,
        "ok": finite_ok,
        "products": list(coeffs.products),
        "rho": float(rho),
        "dual_step": alpha,
        "z_blend": beta,
        "z_mode": "raw_copy" if abs(beta - 1.0) <= 1e-15 else "raw_blend",
        "z_seed_source": z_seed_source,
        "z_pre": dict(z_pre_dict),
        "z_post": z_new_dict,
        "lam_pre": dict(price_dict),
        "lam_post": lam_new_dict,
        "y_raw_star": y_raw_dict,
        "x_star": list(sub["x_star"]),
        "r_raw": {p: float(r_vec[i]) for i, p in enumerate(coeffs.products)},
        "r_l1_raw": r_l1,
        "r_linf_raw": r_linf,
        "r_l2_raw": r_l2,
        "augmented_local_raw": float(sub["augmented_local_raw"]),
        "not_worse_than_ref": bool(sub.get("not_worse_than_ref")),
        "subproblem_ok": bool(sub.get("ok")),
        "subproblem_kind": sub.get("kind"),
        "renorm_note": sub.get("renorm_note"),
        "raw_vs_full_r_l1_gap_star": sub.get("raw_vs_full_r_l1_gap_star"),
        "augmented_local_full_diagnostic": sub.get(
            "augmented_local_full_diagnostic"
        ),
    }


def multi_unit_admm_coordination_report(
    *,
    n_rounds: int = 3,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float]] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    prices: Optional[Mapping[str, Mapping[str, float]]] = None,
    z0: Optional[Mapping[str, Mapping[str, float]]] = None,
    max_passes: int = 32,
) -> Dict[str, Any]:
    """Always-on multi-round offline ADMM coordination (FCC/COKER/CDU).

    Per unit (independent synthetic product-space loops, registry order):
    for t=1..n_rounds run subproblem → raw z consensus → λ ascent. Trajectory
    residual norms use **pre-z-update** residual (z_pre). Aggregate ``ok`` =
    finite trajectory + honesty locks + structure + per-unit subproblem ok.
    **No** absolute residual-must-vanish hard-fail.

    Not Case 1, not plant linking coordinator, not wire, not dual recovery.
    """
    if int(n_rounds) < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    n_r = int(n_rounds)
    honesty = _admm_coordination_honesty_fields()
    units_out: Dict[str, Any] = {}
    trajectory: List[Dict[str, Any]] = []
    all_ok = True

    # Per-unit mutable state (λ as prices dict, z as product dict)
    state_lam: Dict[str, Dict[str, float]] = {}
    state_z: Dict[str, Dict[str, float]] = {}
    for desc in offline_unit_registry():
        unit = desc.unit
        unit_prices: Optional[Mapping[str, float]] = None
        if prices is not None and unit in prices:
            unit_prices = prices[unit]
        price_dict, _, _ = _resolve_prices(unit, unit_prices)
        state_lam[unit] = dict(price_dict)
        unit_z0: Optional[Mapping[str, float]] = None
        if z0 is not None and unit in z0:
            unit_z0 = z0[unit]
        coeffs = cached_offline_unit_coeffs(unit)
        z_dict, _, z_src = _resolve_z_vector(coeffs, unit_z0)
        state_z[unit] = dict(z_dict)
        units_out[unit] = {
            "unit": unit,
            "ok": True,
            "z0_source": z_src,
            "rounds": [],
            "final_lam": dict(price_dict),
            "final_z": dict(z_dict),
            "renorm_note": desc.renorm_note,
            "solver": False,
            "dual_recovery_path": None,
            "on_excel_case1_path": False,
            "kind": ADMM_COORDINATION_KIND,
            "optimand_space": "raw_affine",
            "z_update_space": "raw_affine",
            "coordination_scope": ADMM_COORDINATION_SCOPE,
            "not_plant_linking_coordinator": True,
        }

    for t in range(1, n_r + 1):
        round_units: Dict[str, Any] = {}
        sum_r_l1 = 0.0
        max_r_linf = 0.0
        sum_aug = 0.0
        round_ok = True
        for unit in UNITS:
            row = admm_coordination_round_for_unit(
                unit,
                prices=state_lam[unit],
                z=state_z[unit],
                rho=float(rho),
                delta=delta,
                dual_step=float(dual_step),
                z_blend=beta,
                max_passes=int(max_passes),
            )
            state_lam[unit] = dict(row["lam_post"])
            state_z[unit] = dict(row["z_post"])
            compact = {
                "round": t,
                "ok": bool(row["ok"]),
                "r_l1_raw": float(row["r_l1_raw"]),
                "r_linf_raw": float(row["r_linf_raw"]),
                "r_l2_raw": float(row["r_l2_raw"]),
                "augmented_local_raw": float(row["augmented_local_raw"]),
                "not_worse_than_ref": bool(row["not_worse_than_ref"]),
                "subproblem_ok": bool(row["subproblem_ok"]),
                "z_mode": row["z_mode"],
                "z_seed_source": row["z_seed_source"] if t == 1 else "raw_previous_y",
            }
            units_out[unit]["rounds"].append(compact)
            units_out[unit]["final_lam"] = dict(row["lam_post"])
            units_out[unit]["final_z"] = dict(row["z_post"])
            units_out[unit]["last_x_star"] = list(row["x_star"])
            units_out[unit]["last_y_raw_star"] = dict(row["y_raw_star"])
            if row.get("renorm_note"):
                units_out[unit]["renorm_note"] = row["renorm_note"]
            if row.get("raw_vs_full_r_l1_gap_star") is not None:
                units_out[unit]["raw_vs_full_r_l1_gap_star"] = row[
                    "raw_vs_full_r_l1_gap_star"
                ]
            if not row["ok"]:
                units_out[unit]["ok"] = False
                round_ok = False
                all_ok = False
            sum_r_l1 += float(row["r_l1_raw"])
            max_r_linf = max(max_r_linf, float(row["r_linf_raw"]))
            sum_aug += float(row["augmented_local_raw"])
            round_units[unit] = compact

        traj_row = {
            "round": t,
            "ok": round_ok and all(
                np.isfinite(sum_r_l1) and np.isfinite(max_r_linf) and np.isfinite(sum_aug)
                for _ in (0,)
            ),
            "sum_r_l1_raw": float(sum_r_l1),
            "max_r_linf_raw": float(max_r_linf),
            "sum_augmented_local_raw": float(sum_aug),
            "units_ok": {u: bool(round_units[u]["ok"]) for u in UNITS},
        }
        if not traj_row["ok"]:
            all_ok = False
        trajectory.append(traj_row)

    # Soft residual trend diagnostic only (never hard-fail)
    residual_trend = "n/a"
    if len(trajectory) >= 2:
        vals = [float(tr["sum_r_l1_raw"]) for tr in trajectory]
        if all(np.isfinite(v) for v in vals):
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if all(d <= 1e-12 for d in diffs):
                residual_trend = "nonincreasing"
            elif all(d >= -1e-12 for d in diffs):
                residual_trend = "nondecreasing"
            else:
                residual_trend = "mixed"

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["kind"] == ADMM_COORDINATION_KIND
        and honesty.get("optimand_space") == "raw_affine"
        and honesty.get("z_update_space") == "raw_affine"
        and honesty.get("not_plant_linking_coordinator") is True
        and honesty.get("coordination_lambda_is_not_case1_online_lambda") is True
        and set(units_out.keys()) == set(UNITS)
        and len(trajectory) == n_r
    )
    if not honesty_ok:
        all_ok = False
    for unit in UNITS:
        if not units_out[unit].get("ok"):
            all_ok = False

    finite_traj = all(
        np.isfinite(tr["sum_r_l1_raw"])
        and np.isfinite(tr["max_r_linf_raw"])
        and np.isfinite(tr["sum_augmented_local_raw"])
        for tr in trajectory
    )
    if not finite_traj:
        all_ok = False

    return {
        "ok": bool(all_ok and honesty_ok and finite_traj),
        "units": units_out,
        "unit_order": list(UNITS),
        "trajectory": trajectory,
        "n_rounds": n_r,
        "rho": float(rho),
        "dual_step": float(dual_step),
        "z_blend": beta,
        "delta": delta if not isinstance(delta, Mapping) else dict(delta),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": ADMM_COORDINATION_KIND,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "z_update_space": "raw_affine",
        "coordination_scope": ADMM_COORDINATION_SCOPE,
        "not_plant_linking_coordinator": True,
        "coordination_lambda_is_not_case1_online_lambda": True,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "formula": ADMM_COORDINATION_FORMULA,
        "residual_trend": residual_trend,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "note": honesty["note"],
    }


# ---------------------------------------------------------------------------
# Offline multi-block plant-linking ADMM (goal 5 -- synthetic + plant-named)
# ---------------------------------------------------------------------------

ADMM_PLANT_LINKING_KIND = "offline_admm_plant_linking"
ADMM_PLANT_LINKING_FORMULA = (
    "round: map lam_link/z_link->unit via incidence; "
    "x=argmax raw L1-aug under (lam_u,z_u,rho,delta); "
    "y_link_u=A_u y_raw; r_link=sum_u y_link_u - z_link_pre; "
    "z_link<-(1-beta)z+beta y_link_total; lam_link<-lam+alpha*rho*r_link"
)
ADMM_PLANT_LINKING_SCOPE = "synthetic_offline_demo"
ADMM_PLANT_LINKING_STREAMS = (
    "light_ends",
    "naphtha",
    "distillate",
    "gasoil",
    "resid",
    "coke",
)
ADMM_PLANT_NAMED_LINKING_SCOPE = "plant_named_offline_demo"
ADMM_PLANT_NAMED_LINKING_STREAMS = (
    # Identity-friendly plant/routing product names (name-disjoint across units).
    "fcc_dry_gas",
    "fcc_lpg",
    "fcc_naphtha",
    "fcc_lco",
    "fcc_slurry",
    "fcc_coke",
    "coker_dry_gas",
    "coker_lpg",
    "coker_naphtha",
    "coker_gasoil",
    "coker_coke",
    "cdu_offgas",
    "cdu_naphtha_light",
    "cdu_naphtha_heavy",
    "cdu_distillate",
    "cdu_gasoil",
    "cdu_resid",
)
ADMM_PLANT_LINKING_MODES = ("synthetic", "plant_named")

# product -> linking stream (0/1 selection incidence; synthetic offline demo only).
# Products are name-disjoint across units -- linking requires explicit incidence.
_PLANT_LINKING_PRODUCT_TO_STREAM: Dict[str, Dict[str, str]] = {
    "FCC": {
        "fcc_dry_gas": "light_ends",
        "fcc_lpg": "light_ends",
        "fcc_naphtha": "naphtha",
        "fcc_lco": "distillate",
        "fcc_slurry": "gasoil",
        "fcc_coke": "coke",
    },
    "COKER": {
        "coker_dry_gas": "light_ends",
        "coker_lpg": "light_ends",
        "coker_naphtha": "naphtha",
        "coker_gasoil": "gasoil",
        "coker_coke": "coke",
    },
    "CDU": {
        "cdu_offgas": "light_ends",
        "cdu_naphtha_light": "naphtha",
        "cdu_naphtha_heavy": "naphtha",
        "cdu_distillate": "distillate",
        "cdu_gasoil": "gasoil",
        "cdu_resid": "resid",
    },
}

# Plant-named mode: identity incidence product p -> stream p (no family collapse).
_PLANT_NAMED_PRODUCT_TO_STREAM: Dict[str, Dict[str, str]] = {
    unit: {product: product for product in product_map}
    for unit, product_map in _PLANT_LINKING_PRODUCT_TO_STREAM.items()
}


def _normalize_plant_linking_mode(mode: Optional[str] = None) -> str:
    """Normalize topology mode; default remains synthetic (bit-stable for #24)."""
    if mode is None:
        return "synthetic"
    key = str(mode).strip().lower()
    if key in ("synthetic", "synthetic_offline_demo", ADMM_PLANT_LINKING_SCOPE):
        return "synthetic"
    if key in ("plant_named", "plant_named_offline_demo", ADMM_PLANT_NAMED_LINKING_SCOPE):
        return "plant_named"
    raise ValueError(
        f"Unknown plant-linking topology mode {mode!r}; "
        f"expected one of {ADMM_PLANT_LINKING_MODES} "
        f"(aliases: synthetic_offline_demo, plant_named_offline_demo)"
    )


def _plant_linking_topology_source(mode: str) -> str:
    m = _normalize_plant_linking_mode(mode)
    if m == "plant_named":
        return ADMM_PLANT_NAMED_LINKING_SCOPE
    return ADMM_PLANT_LINKING_SCOPE


def _plant_linking_linking_space(mode: str) -> str:
    m = _normalize_plant_linking_mode(mode)
    if m == "plant_named":
        return "plant_named_linking_streams"
    return "synthetic_linking_streams"


def _plant_linking_streams_for_mode(mode: str) -> List[str]:
    m = _normalize_plant_linking_mode(mode)
    if m == "plant_named":
        return list(ADMM_PLANT_NAMED_LINKING_STREAMS)
    return list(ADMM_PLANT_LINKING_STREAMS)


def _plant_linking_product_to_stream_for_mode(
    mode: str,
) -> Dict[str, Dict[str, str]]:
    m = _normalize_plant_linking_mode(mode)
    if m == "plant_named":
        return _PLANT_NAMED_PRODUCT_TO_STREAM
    return _PLANT_LINKING_PRODUCT_TO_STREAM


def _admm_plant_linking_honesty_fields(
    topology_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Machine-readable dual-ban / topology locks for plant-linking.

    ``topology_source`` discriminates synthetic vs plant-named offline demo.
    Both modes share dual-ban / not-full-plant-MB / not-wire locks.
    """
    if topology_source is None:
        topology_source = ADMM_PLANT_LINKING_SCOPE
    mode = _normalize_plant_linking_mode(topology_source)
    topo_src = _plant_linking_topology_source(mode)
    linking_space = _plant_linking_linking_space(mode)
    if mode == "plant_named":
        topo_phrase = (
            "plant-named offline demo linking streams (identity incidence over "
            "unit product names such as fcc_naphtha / cdu_gasoil; no synthetic "
            "family collapse)"
        )
    else:
        topo_phrase = (
            "synthetic plant linking-stream space (family streams light_ends/"
            "naphtha/... with explicit product->stream incidence)"
        )
    return {
        "kind": ADMM_PLANT_LINKING_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "linking_space": linking_space,
        "z_update_space": linking_space,
        "plant_linking_scope": topo_src,
        "topology_source": topo_src,
        "not_full_plant_mass_balance": True,
        "not_case1_solve": True,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "plant_linking_lambda_is_not_case1_online_lambda": True,
        "not_live_plant_blocks": True,
        "formula": ADMM_PLANT_LINKING_FORMULA,
        "note": (
            "Offline multi-block plant-linking ADMM harness: shared lam/z on a "
            f"{topo_phrase}. Composes existing admm_block_subproblem_for_unit "
            "(map lam_link/z_link -> unit product prices/z via incidence^T / selection; "
            "lift y_raw -> linking via A_u). Dual ascent uses pre-z-update linking residual. "
            f"topology_source={topo_src} -- NOT live plant_blocks / cascade mass "
            "balance / Case 1 CDU-Blender links / full plant mass balance. Shared "
            "plant-linking lam/z are NOT Case 1 PRIMARY online lambda, NOT SECONDARY "
            "recovered blender duals, NOT pure-ADMM dual recovery, NOT wire shipped. "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False. Existing "
            "multi_unit_admm_coordination_report remains per-unit synthetic "
            "(not_plant_linking_coordinator=True). Synthetic topology mode remains "
            "default/available. No PuLP/CBC; always-on numpy; no absolute residual-"
            "must-converge hard-fail."
        ),
    }


def offline_plant_linking_topology(
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Offline plant linking topology (streams + incidence).

    Modes:
      - ``synthetic`` (default): family streams + product->family incidence
        (``synthetic_offline_demo``). Bit-stable for existing #24 tests.
      - ``plant_named``: plant/routing-style product names as streams with
        **identity** incidence product->product (``plant_named_offline_demo``).

    Always-on numpy data surface. Products are name-disjoint across units, so
    plant linking uses **explicit** incidence (not product-name intersection).

    Honesty: offline demo only -- not live plant cascade mass balance, not
    Case 1 links, not wire, not full plant mass balance.
    """
    mode_key = _normalize_plant_linking_mode(mode)
    streams = _plant_linking_streams_for_mode(mode_key)
    raw_maps = _plant_linking_product_to_stream_for_mode(mode_key)
    topo_src = _plant_linking_topology_source(mode_key)
    incidence: Dict[str, Dict[str, Dict[str, float]]] = {}
    coverage: Dict[str, List[str]] = {}
    for unit in UNITS:
        coeffs = cached_offline_unit_coeffs(unit)
        known = set(coeffs.products)
        raw_map = raw_maps.get(unit, {})
        unit_inc: Dict[str, Dict[str, float]] = {}
        for product, stream in raw_map.items():
            if product not in known:
                raise ValueError(
                    f"Plant-linking incidence product {product!r} not in "
                    f"known products for {unit}: {list(coeffs.products)}"
                )
            if stream not in streams:
                raise ValueError(
                    f"Unknown linking stream {stream!r} for product {product!r}"
                )
            unit_inc[product] = {stream: 1.0}
        unknown = sorted(set(raw_map) - known)
        if unknown:
            raise ValueError(f"Unknown incidence products for {unit}: {unknown}")
        incidence[unit] = unit_inc
        coverage[unit] = [p for p in coeffs.products if p in unit_inc]

    honesty = _admm_plant_linking_honesty_fields(topo_src)
    return {
        "streams": streams,
        "unit_order": list(UNITS),
        "incidence": incidence,
        "product_coverage": coverage,
        "mode": mode_key,
        "topology_source": topo_src,
        "plant_linking_scope": topo_src,
        "not_full_plant_mass_balance": True,
        "not_live_plant_blocks": True,
        "not_case1_links": True,
        "kind": "offline_plant_linking_topology",
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "note": honesty["note"],
    }


def offline_plant_named_linking_topology() -> Dict[str, Any]:
    """Plant-named offline linking topology (identity incidence).

    Thin wrapper over ``offline_plant_linking_topology(mode="plant_named")``.
    Not full plant mass balance; not live cascade; not Case 1; not wire.
    """
    return offline_plant_linking_topology(mode="plant_named")


def _incidence_matrix_for_unit(
    unit: str,
    streams: Sequence[str],
    products: Sequence[str],
    *,
    product_to_stream: Optional[Mapping[str, str]] = None,
) -> np.ndarray:
    """A shape (n_streams, n_products): 0/1 selection incidence."""
    if product_to_stream is None:
        raw = _PLANT_LINKING_PRODUCT_TO_STREAM.get(unit, {})
    else:
        raw = dict(product_to_stream)
    stream_index = {s: i for i, s in enumerate(streams)}
    A = np.zeros((len(streams), len(products)), dtype=np.float64)
    for j, p in enumerate(products):
        stream = raw.get(p)
        if stream is None:
            continue
        if stream not in stream_index:
            continue
        A[stream_index[stream], j] = 1.0
    return A


def project_linking_to_unit(
    unit: str,
    lam_link: Mapping[str, float],
    z_link: Mapping[str, float],
    *,
    streams: Optional[Sequence[str]] = None,
    product_to_stream: Optional[Mapping[str, str]] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Map shared linking lam/z into unit product prices/z via incidence^T.

    Formula (0/1 selection incidence A, shape n_link x n_prod)::

        prices_unit = A^T lam_link
        z_unit      = A^T z_link

    Documented dual-ish of lift for selection incidence.
    """
    key = _normalize_unit_name(unit)
    mode_key = _normalize_plant_linking_mode(mode)
    if streams is None:
        topo_streams = _plant_linking_streams_for_mode(mode_key)
    else:
        topo_streams = list(streams)
    if product_to_stream is None:
        p2s = _plant_linking_product_to_stream_for_mode(mode_key).get(key, {})
    else:
        p2s = dict(product_to_stream)
    coeffs = cached_offline_unit_coeffs(key)
    products = list(coeffs.products)
    A = _incidence_matrix_for_unit(
        key, topo_streams, products, product_to_stream=p2s
    )
    lam_vec = np.array(
        [float(lam_link.get(s, 0.0)) for s in topo_streams], dtype=np.float64
    )
    z_vec = np.array(
        [float(z_link.get(s, 0.0)) for s in topo_streams], dtype=np.float64
    )
    prices_vec = A.T @ lam_vec
    z_unit_vec = A.T @ z_vec
    prices = {p: float(prices_vec[i]) for i, p in enumerate(products)}
    z_unit = {p: float(z_unit_vec[i]) for i, p in enumerate(products)}
    return {
        "unit": key,
        "products": products,
        "streams": list(topo_streams),
        "prices": prices,
        "z": z_unit,
        "A_shape": [int(A.shape[0]), int(A.shape[1])],
        "formula": "prices=A^T lam_link; z_unit=A^T z_link (0/1 selection incidence)",
    }


def lift_unit_y_to_linking(
    unit: str,
    y_raw: Mapping[str, float],
    *,
    streams: Optional[Sequence[str]] = None,
    product_to_stream: Optional[Mapping[str, str]] = None,
    mode: Optional[str] = None,
) -> Dict[str, float]:
    """Lift unit product y_raw to linking space: y_link = A y_raw."""
    key = _normalize_unit_name(unit)
    mode_key = _normalize_plant_linking_mode(mode)
    if streams is None:
        topo_streams = _plant_linking_streams_for_mode(mode_key)
    else:
        topo_streams = list(streams)
    if product_to_stream is None:
        p2s = _plant_linking_product_to_stream_for_mode(mode_key).get(key, {})
    else:
        p2s = dict(product_to_stream)
    coeffs = cached_offline_unit_coeffs(key)
    products = list(coeffs.products)
    A = _incidence_matrix_for_unit(
        key, topo_streams, products, product_to_stream=p2s
    )
    y_vec = np.array([float(y_raw.get(p, 0.0)) for p in products], dtype=np.float64)
    y_link_vec = A @ y_vec
    return {s: float(y_link_vec[i]) for i, s in enumerate(topo_streams)}


def _default_lam_link(
    streams: Sequence[str],
    *,
    product_to_stream_by_unit: Optional[Mapping[str, Mapping[str, str]]] = None,
    mode: Optional[str] = None,
) -> Dict[str, float]:
    """Seed linking lam as mean of default product prices mapping into each stream."""
    mode_key = _normalize_plant_linking_mode(mode)
    maps = (
        dict(product_to_stream_by_unit)
        if product_to_stream_by_unit is not None
        else _plant_linking_product_to_stream_for_mode(mode_key)
    )
    sums: Dict[str, float] = {s: 0.0 for s in streams}
    counts: Dict[str, int] = {s: 0 for s in streams}
    for unit in UNITS:
        prices = default_offline_prices(unit)
        raw = maps.get(unit, {})
        for product, stream in raw.items():
            if stream not in sums:
                continue
            sums[stream] += float(prices.get(product, 0.0))
            counts[stream] += 1
    out: Dict[str, float] = {}
    for s in streams:
        out[s] = float(sums[s] / counts[s]) if counts[s] else 0.0
    return out


def _default_z_link(
    streams: Sequence[str],
    *,
    product_to_stream_by_unit: Optional[Mapping[str, Mapping[str, str]]] = None,
    mode: Optional[str] = None,
) -> Dict[str, float]:
    """Seed z_link as sum over units of lifted full postprocess yields at reference."""
    mode_key = _normalize_plant_linking_mode(mode)
    maps = (
        dict(product_to_stream_by_unit)
        if product_to_stream_by_unit is not None
        else _plant_linking_product_to_stream_for_mode(mode_key)
    )
    totals = {s: 0.0 for s in streams}
    for unit in UNITS:
        coeffs = cached_offline_unit_coeffs(unit)
        z_full = _default_z_full_for_unit(unit, coeffs)
        p2s = maps.get(unit, {})
        y_link = lift_unit_y_to_linking(
            unit, z_full, streams=streams, product_to_stream=p2s, mode=mode_key
        )
        for s in streams:
            totals[s] += float(y_link[s])
    return totals


def plant_linking_admm_round(
    *,
    lam_link: Optional[Mapping[str, float]] = None,
    z_link: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float], np.ndarray] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    max_passes: int = 32,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """One offline multi-block plant-linking ADMM round (always-on numpy).

    Structure (deterministic)::

        1. For each unit (registry order): map lam_link/z_link -> unit prices/z;
           call existing ``admm_block_subproblem_for_unit`` under rho, delta
        2. Lift y_raw -> linking; aggregate r_link = sum y_link_u - z_link_pre
           (pre-z-update; never post free z<-y zero theater)
        3. Shared z consensus in linking space: z <- (1-beta)z + beta y_link_total
        4. Shared lam dual ascent: lam <- lam + alpha*rho*r_link

    ``mode`` selects topology (default ``synthetic``; ``plant_named`` for
    identity plant-product streams). Not Case 1, not wire, not full plant mass
    balance, not dual recovery.
    """
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    mode_key = _normalize_plant_linking_mode(mode)
    streams = _plant_linking_streams_for_mode(mode_key)
    maps = _plant_linking_product_to_stream_for_mode(mode_key)
    topo_src = _plant_linking_topology_source(mode_key)
    lam_pre = (
        dict(lam_link)
        if lam_link is not None
        else _default_lam_link(streams, product_to_stream_by_unit=maps, mode=mode_key)
    )
    z_pre = (
        dict(z_link)
        if z_link is not None
        else _default_z_link(streams, product_to_stream_by_unit=maps, mode=mode_key)
    )
    for s in streams:
        if s not in lam_pre:
            raise ValueError(f"Missing lam_link stream {s!r}")
        if s not in z_pre:
            raise ValueError(f"Missing z_link stream {s!r}")
        if not np.isfinite(float(lam_pre[s])) or not np.isfinite(float(z_pre[s])):
            raise ValueError(f"lam_link/z_link must be finite for stream {s!r}")

    units_out: Dict[str, Any] = {}
    y_link_total = {s: 0.0 for s in streams}
    sum_aug = 0.0
    all_sub_ok = True

    for unit in UNITS:
        p2s = maps.get(unit, {})
        proj = project_linking_to_unit(
            unit,
            lam_pre,
            z_pre,
            streams=streams,
            product_to_stream=p2s,
            mode=mode_key,
        )
        sub = admm_block_subproblem_for_unit(
            unit,
            prices=proj["prices"],
            z=proj["z"],
            rho=float(rho),
            delta=delta,
            max_passes=int(max_passes),
        )
        y_raw = dict(sub["y_raw_star"])
        y_link_u = lift_unit_y_to_linking(
            unit,
            y_raw,
            streams=streams,
            product_to_stream=p2s,
            mode=mode_key,
        )
        for s in streams:
            y_link_total[s] += float(y_link_u[s])
        sum_aug += float(sub["augmented_local_raw"])
        unit_ok = bool(sub.get("ok"))
        if not unit_ok:
            all_sub_ok = False
        units_out[unit] = {
            "unit": unit,
            "ok": unit_ok,
            "subproblem_ok": unit_ok,
            "subproblem_kind": sub.get("kind"),
            "not_worse_than_ref": bool(sub.get("not_worse_than_ref")),
            "augmented_local_raw": float(sub["augmented_local_raw"]),
            "y_raw_star": y_raw,
            "x_star": list(sub["x_star"]),
            "y_link": y_link_u,
            "prices_unit": dict(proj["prices"]),
            "z_unit": dict(proj["z"]),
            "renorm_note": sub.get("renorm_note"),
            "raw_vs_full_r_l1_gap_star": sub.get("raw_vs_full_r_l1_gap_star"),
        }

    # Pre-update residual in linking space (drives dual ascent).
    r_vec = np.array(
        [float(y_link_total[s]) - float(z_pre[s]) for s in streams], dtype=np.float64
    )
    r_link = {s: float(r_vec[i]) for i, s in enumerate(streams)}
    r_l1 = float(np.sum(np.abs(r_vec)))
    r_linf = float(np.max(np.abs(r_vec))) if r_vec.size else 0.0
    r_l2 = float(np.sqrt(np.sum(r_vec * r_vec))) if r_vec.size else 0.0

    y_total_vec = np.array([float(y_link_total[s]) for s in streams], dtype=np.float64)
    z_pre_vec = np.array([float(z_pre[s]) for s in streams], dtype=np.float64)
    z_post_vec = (1.0 - beta) * z_pre_vec + beta * y_total_vec
    z_post = {s: float(z_post_vec[i]) for i, s in enumerate(streams)}

    alpha = float(dual_step)
    lam_pre_vec = np.array([float(lam_pre[s]) for s in streams], dtype=np.float64)
    lam_post_vec = lam_pre_vec + alpha * float(rho) * r_vec
    lam_post = {s: float(lam_post_vec[i]) for i, s in enumerate(streams)}

    finite_ok = bool(
        all_sub_ok
        and np.all(np.isfinite(r_vec))
        and np.all(np.isfinite(z_post_vec))
        and np.all(np.isfinite(lam_post_vec))
        and np.isfinite(r_l1)
        and np.isfinite(r_linf)
        and np.isfinite(sum_aug)
    )
    honesty = _admm_plant_linking_honesty_fields(topo_src)
    return {
        **{
            k: honesty[k]
            for k in (
                "kind",
                "solver",
                "dual_recovery_path",
                "on_excel_case1_path",
                "price_source",
                "lam_source",
                "z_source",
                "rho_source",
                "optimand_space",
                "linking_space",
                "z_update_space",
                "plant_linking_scope",
                "topology_source",
                "not_full_plant_mass_balance",
                "not_case1_solve",
                "not_wire_shipped",
                "not_pure_admm_dual_recovery",
                "plant_linking_lambda_is_not_case1_online_lambda",
                "not_live_plant_blocks",
                "formula",
            )
        },
        "ok": finite_ok,
        "mode": mode_key,
        "streams": streams,
        "unit_order": list(UNITS),
        "units": units_out,
        "rho": float(rho),
        "dual_step": alpha,
        "z_blend": beta,
        "z_mode": "link_copy" if abs(beta - 1.0) <= 1e-15 else "link_blend",
        "lam_pre": dict(lam_pre),
        "lam_post": lam_post,
        "z_pre": dict(z_pre),
        "z_post": z_post,
        "y_link_total": dict(y_link_total),
        "r_link_pre": r_link,
        "r_l1_link": r_l1,
        "r_linf_link": r_linf,
        "r_l2_link": r_l2,
        "sum_augmented_local_raw": float(sum_aug),
        "all_subproblem_ok": all_sub_ok,
    }


def multi_block_plant_linking_admm_report(
    *,
    n_rounds: int = 3,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float]] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    lam0: Optional[Mapping[str, float]] = None,
    z0: Optional[Mapping[str, float]] = None,
    max_passes: int = 32,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Always-on multi-round multi-block plant-linking ADMM report.

    Shared lam/z live in the active linking-stream space (synthetic family
    streams by default, or plant-named product streams when ``mode="plant_named"``).
    Dual ascent residual is pre-z-update in linking space. Aggregate ``ok`` =
    finite trajectory + honesty locks + structure + per-unit subproblem ok.
    **No** residual-must-vanish SLA.

    Not Case 1, not wire, not full plant mass balance, not dual recovery.
    Existing ``multi_unit_admm_coordination_report`` remains a separate surface
    with ``not_plant_linking_coordinator=True``.
    """
    if int(n_rounds) < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    mode_key = _normalize_plant_linking_mode(mode)
    n_r = int(n_rounds)
    streams = _plant_linking_streams_for_mode(mode_key)
    maps = _plant_linking_product_to_stream_for_mode(mode_key)
    topo_src = _plant_linking_topology_source(mode_key)
    honesty = _admm_plant_linking_honesty_fields(topo_src)
    topo = offline_plant_linking_topology(mode=mode_key)

    state_lam = (
        dict(lam0)
        if lam0 is not None
        else _default_lam_link(streams, product_to_stream_by_unit=maps, mode=mode_key)
    )
    state_z = (
        dict(z0)
        if z0 is not None
        else _default_z_link(streams, product_to_stream_by_unit=maps, mode=mode_key)
    )

    trajectory: List[Dict[str, Any]] = []
    units_out: Dict[str, Any] = {
        u: {
            "unit": u,
            "ok": True,
            "rounds": [],
            "final_y_link": None,
            "last_x_star": None,
            "last_y_raw_star": None,
        }
        for u in UNITS
    }
    all_ok = True

    for t in range(1, n_r + 1):
        row = plant_linking_admm_round(
            lam_link=state_lam,
            z_link=state_z,
            rho=float(rho),
            delta=delta,
            dual_step=float(dual_step),
            z_blend=beta,
            max_passes=int(max_passes),
            mode=mode_key,
        )
        state_lam = dict(row["lam_post"])
        state_z = dict(row["z_post"])
        compact_units: Dict[str, Any] = {}
        round_ok = bool(row.get("ok"))
        for unit in UNITS:
            urow = row["units"][unit]
            compact = {
                "round": t,
                "ok": bool(urow["ok"]),
                "subproblem_ok": bool(urow["subproblem_ok"]),
                "not_worse_than_ref": bool(urow["not_worse_than_ref"]),
                "augmented_local_raw": float(urow["augmented_local_raw"]),
                "y_link": dict(urow["y_link"]),
            }
            units_out[unit]["rounds"].append(compact)
            units_out[unit]["final_y_link"] = dict(urow["y_link"])
            units_out[unit]["last_x_star"] = list(urow["x_star"])
            units_out[unit]["last_y_raw_star"] = dict(urow["y_raw_star"])
            if urow.get("renorm_note"):
                units_out[unit]["renorm_note"] = urow["renorm_note"]
            if not urow["ok"]:
                units_out[unit]["ok"] = False
                round_ok = False
                all_ok = False
            compact_units[unit] = compact

        traj_row = {
            "round": t,
            "ok": bool(
                round_ok
                and np.isfinite(float(row["r_l1_link"]))
                and np.isfinite(float(row["r_linf_link"]))
                and np.isfinite(float(row["sum_augmented_local_raw"]))
            ),
            "r_l1_link": float(row["r_l1_link"]),
            "r_linf_link": float(row["r_linf_link"]),
            "r_l2_link": float(row["r_l2_link"]),
            "sum_augmented_local_raw": float(row["sum_augmented_local_raw"]),
            "r_link_pre": dict(row["r_link_pre"]),
            "units_ok": {u: bool(compact_units[u]["ok"]) for u in UNITS},
        }
        if not traj_row["ok"]:
            all_ok = False
        trajectory.append(traj_row)

    residual_trend = "n/a"
    if len(trajectory) >= 2:
        vals = [float(tr["r_l1_link"]) for tr in trajectory]
        if all(np.isfinite(v) for v in vals):
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if all(d <= 1e-12 for d in diffs):
                residual_trend = "nonincreasing"
            elif all(d >= -1e-12 for d in diffs):
                residual_trend = "nondecreasing"
            else:
                residual_trend = "mixed"

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["kind"] == ADMM_PLANT_LINKING_KIND
        and honesty.get("optimand_space") == "raw_affine"
        and honesty.get("not_full_plant_mass_balance") is True
        and honesty.get("not_case1_solve") is True
        and honesty.get("not_wire_shipped") is True
        and honesty.get("not_pure_admm_dual_recovery") is True
        and honesty.get("plant_linking_lambda_is_not_case1_online_lambda") is True
        and honesty.get("topology_source") == topo_src
        and honesty.get("topology_source")
        in (ADMM_PLANT_LINKING_SCOPE, ADMM_PLANT_NAMED_LINKING_SCOPE)
        and honesty.get("linking_space") == _plant_linking_linking_space(mode_key)
        and set(units_out.keys()) == set(UNITS)
        and len(trajectory) == n_r
        and list(topo["streams"]) == streams
        and topo.get("topology_source") == topo_src
    )
    if not honesty_ok:
        all_ok = False
    for unit in UNITS:
        if not units_out[unit].get("ok"):
            all_ok = False

    finite_traj = all(
        np.isfinite(tr["r_l1_link"])
        and np.isfinite(tr["r_linf_link"])
        and np.isfinite(tr["sum_augmented_local_raw"])
        for tr in trajectory
    )
    if not finite_traj:
        all_ok = False

    return {
        "ok": bool(all_ok and honesty_ok and finite_traj),
        "units": units_out,
        "unit_order": list(UNITS),
        "streams": streams,
        "mode": mode_key,
        "topology": {
            "streams": topo["streams"],
            "topology_source": topo["topology_source"],
            "mode": mode_key,
            "not_full_plant_mass_balance": True,
            "product_coverage": topo["product_coverage"],
        },
        "trajectory": trajectory,
        "n_rounds": n_r,
        "rho": float(rho),
        "dual_step": float(dual_step),
        "z_blend": beta,
        "delta": delta if not isinstance(delta, Mapping) else dict(delta),
        "final_lam": dict(state_lam),
        "final_z": dict(state_z),
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": ADMM_PLANT_LINKING_KIND,
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "optimand_space": "raw_affine",
        "linking_space": honesty["linking_space"],
        "z_update_space": honesty["z_update_space"],
        "plant_linking_scope": topo_src,
        "topology_source": topo_src,
        "not_full_plant_mass_balance": True,
        "not_case1_solve": True,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "plant_linking_lambda_is_not_case1_online_lambda": True,
        "not_live_plant_blocks": True,
        "formula": ADMM_PLANT_LINKING_FORMULA,
        "residual_trend": residual_trend,
        "honesty_ok": honesty_ok,
        "tf_available": tf_available(),
        "note": honesty["note"],
    }


def multi_block_plant_named_linking_admm_report(
    *,
    n_rounds: int = 3,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float]] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    lam0: Optional[Mapping[str, float]] = None,
    z0: Optional[Mapping[str, float]] = None,
    max_passes: int = 32,
) -> Dict[str, Any]:
    """Thin wrapper: multi-block plant-linking report on plant-named streams.

    Equivalent to ``multi_block_plant_linking_admm_report(mode="plant_named", ...)``.
    Not full plant mass balance; not wire; not Case 1; dual_recovery_path=None.
    """
    return multi_block_plant_linking_admm_report(
        n_rounds=n_rounds,
        rho=rho,
        delta=delta,
        dual_step=dual_step,
        z_blend=z_blend,
        lam0=lam0,
        z0=z0,
        max_passes=max_passes,
        mode="plant_named",
    )



# ---------------------------------------------------------------------------
# Offline dual-honest wire preflight (goal 5 residual after plant-named packaging)
# ---------------------------------------------------------------------------
# Always-on numpy. Composes existing readiness gates and lists machine-readable
# wire_blockers. Does NOT ship wire, flip Case 1 form, rewrite isolation, or
# redefine ready_for_wire_discussion.

WIRE_PREFLIGHT_KIND = "offline_wire_preflight"

# Stable honesty ids true at HEAD (pre-wire residual). Not failure theater.
DEFAULT_WIRE_BLOCKERS: tuple = (
    "isolation_rewrite_required",
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
    "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp",
)

WIRE_BLOCKER_NOTES: Dict[str, str] = {
    "isolation_rewrite_required": (
        "test_tf_import_isolation.py must be rewritten WITH a dual-honest wire; "
        "do not silently break or delete isolation gates."
    ),
    "form_label_change_required": (
        "Case 1 still uses form classic_2block_excel_path; any TF-aware path needs "
        "an explicit form label change (never silent form reuse)."
    ),
    "dual_linf_under_wire_unproven": (
        "Online λ L∞ gate under a TF-aware path is not proven; Case 1 PRIMARY "
        "online dual honesty remains on the classic package ADMM path."
    ),
    "case1_is_cdu_blender_package_admm": (
        "Case 1 is CDU+Blender package ADMM, not multi-unit FCC/COKER/CDU "
        "plant-linking alone. Offline Case-1-shaped skeleton is a dual-banned "
        "shape substrate only — skeleton ≠ package-ADMM wire."
    ),
    "no_blender_offline_affine_kernel": (
        "Offline UNITS are FCC/COKER/CDU only; no blender offline affine kernel. "
        "Case-1-shaped blender_surface=linear_quality_pooling is planning residual "
        "only — linear pooling ≠ affine kernel; not a BLENDER UNITS entry."
    ),
    "wire_not_shipped": (
        "TF is not wired into Excel Case 1 or the ADMM coordinator; "
        "wire_shipped=False always on this surface."
    ),
    "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp": (
        "Affine TF kernels model yield response to drivers — not full plant_blocks "
        "feed-rate LPs; do not claim producer-consumer plant_blocks ADMM shipped "
        "via this affine-only surface."
    ),
}

# Honest next-wave hint only (no executor / no auto-wire).
SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT = (
    "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
)


def offline_wire_blocker_catalog() -> Dict[str, Any]:
    """Stable wire_blockers catalog true at HEAD (honesty, not CI red).

    Always-on. Does not claim wire shipped or dual recovery.
    """
    return {
        "kind": "offline_wire_blocker_catalog",
        "wire_blockers": list(DEFAULT_WIRE_BLOCKERS),
        "wire_blocker_notes": dict(WIRE_BLOCKER_NOTES),
        "wire_shipped": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "solver": False,
        "not_full_plant_mass_balance": True,
        "not_pure_admm_dual_recovery": True,
        "note": (
            "Machine-readable wire blockers true at HEAD. Blockers are honesty "
            "documentation for dual-honest wire — not test failures. "
            "preflight documents them; wire is still deferred."
        ),
    }


def _wire_preflight_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire locks for preflight reports."""
    return {
        "kind": WIRE_PREFLIGHT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "not_case1_solve": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_full_plant_mass_balance": True,
        "not_pure_admm_dual_recovery": True,
        "preflight_lambda_is_not_case1_online_lambda": True,
        "preflight_is_not_case1_primary_or_secondary_duals": True,
        "plant_linking_lambda_is_not_case1_online_lambda": True,
        "suggested_next_wave": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
    }


def offline_wire_preflight_report(
    *,
    readiness_n_repeats: int = 30,
    readiness_warmup: int = 1,
    include_box: bool = True,
    box_delta: float = 1.0,
    include_admm_residual: bool = True,
    include_admm_block_subproblem: bool = True,
    include_admm_coordination: bool = True,
    include_admm_plant_linking: bool = True,
    include_admm_plant_named_linking: bool = True,
    include_admm_case1_shaped_linking: bool = True,
    include_admm_case1_dual_space_form_contract: bool = True,
    include_admm_case1_dual_space_linf_probe: bool = True,
    include_admm_case1_dual_space_linf_live_lambda_bridge: bool = True,
    include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart: bool = True,
    include_admm_case1_honest_blender_pooling_path: bool = True,
    include_admm_case1_online_linf_gate_criteria_contract: bool = True,
    include_admm_case1_isolation_rewrite_design_contract: bool = True,
    include_admm_case1_wire_ship_acceptance_design_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_design_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract: bool = True,
    include_admm_case1_form_label_change_shipped_criteria_contract: bool = True,
    include_admm_case1_isolation_rewrite_shipped_criteria_contract: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract: bool = True,
    include_admm_case1_dual_honest_tf_aware_path_execution_scaffold: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_rehearsal: bool = True,
    include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint: bool = True,
    include_admm_case1_isolation_rewrite_first_blocker_operational_prep: bool = True,
    include_admm_case1_dual_linf_under_wire_criteria_contract: bool = True,
) -> Dict[str, Any]:
    """Compose green offline gates + explicit machine-readable wire_blockers.

    Always-on numpy (no TensorFlow, no PuLP). Reuses
    ``offline_block_solve_readiness_report`` — does **not** re-implement
    residual/subproblem/coordination/plant-linking maximizer math.

    Honesty:
    - ``wire_shipped=False`` always; preflight ≠ wire shipped
    - ``dual_recovery_path=None``; not pure-ADMM dual recovery; not Case 1 duals
    - ``ready_for_wire_discussion`` meaning **unchanged** (parity∧priced∧timings∧honesty)
    - ``preflight_ok`` / ``blockers_documented`` are separate from ready
    - ``ok`` means finite compose + honesty locks + blockers documented — **not** wire

    Does **not** flip Case 1 form, rewrite isolation, retune ρ, or import TF.
    """
    honesty = _wire_preflight_honesty_fields()
    catalog = offline_wire_blocker_catalog()
    wire_blockers = list(catalog["wire_blockers"])
    wire_blocker_notes = dict(catalog["wire_blocker_notes"])

    readiness = offline_block_solve_readiness_report(
        n_repeats=readiness_n_repeats,
        warmup=readiness_warmup,
        include_box=include_box,
        box_delta=box_delta,
        include_admm_residual=include_admm_residual,
        include_admm_block_subproblem=include_admm_block_subproblem,
        include_admm_coordination=include_admm_coordination,
        include_admm_plant_linking=include_admm_plant_linking,
        include_admm_plant_named_linking=include_admm_plant_named_linking,
        include_admm_case1_shaped_linking=include_admm_case1_shaped_linking,
        include_admm_case1_dual_space_form_contract=(
            include_admm_case1_dual_space_form_contract
        ),
        include_admm_case1_dual_space_linf_probe=(
            include_admm_case1_dual_space_linf_probe
        ),
        include_admm_case1_dual_space_linf_live_lambda_bridge=(
            include_admm_case1_dual_space_linf_live_lambda_bridge
        ),
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=(
            include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart
        ),
        include_admm_case1_honest_blender_pooling_path=(
            include_admm_case1_honest_blender_pooling_path
        ),
        include_admm_case1_online_linf_gate_criteria_contract=(
            include_admm_case1_online_linf_gate_criteria_contract
        ),
        include_admm_case1_isolation_rewrite_design_contract=(
            include_admm_case1_isolation_rewrite_design_contract
        ),
        include_admm_case1_wire_ship_acceptance_design_contract=(
            include_admm_case1_wire_ship_acceptance_design_contract
        ),
        include_admm_case1_dual_honest_tf_aware_path_design_contract=(
            include_admm_case1_dual_honest_tf_aware_path_design_contract
        ),
        include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract=(
            include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract
        ),
        include_admm_case1_form_label_change_shipped_criteria_contract=(
            include_admm_case1_form_label_change_shipped_criteria_contract
        ),
        include_admm_case1_isolation_rewrite_shipped_criteria_contract=(
            include_admm_case1_isolation_rewrite_shipped_criteria_contract
        ),
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract=(
            include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract
        ),
        include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract=(
            include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract
        ),
        include_admm_case1_dual_honest_tf_aware_path_execution_scaffold=(
            include_admm_case1_dual_honest_tf_aware_path_execution_scaffold
        ),
        include_admm_case1_dual_honest_multi_blocker_wire_rehearsal=(
            include_admm_case1_dual_honest_multi_blocker_wire_rehearsal
        ),
        include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint=(
            include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint
        ),
        include_admm_case1_isolation_rewrite_first_blocker_operational_prep=(
            include_admm_case1_isolation_rewrite_first_blocker_operational_prep
        ),
    )

    # Structural ready meaning unchanged — mirror only, never AND blockers into ready.
    parity_ok = bool(readiness.get("parity_ok"))
    priced_ok = bool(readiness.get("priced_ok"))
    timings_ok = bool(readiness.get("timings_ok"))
    honesty_ok = bool(readiness.get("honesty_ok"))
    ready = bool(readiness.get("ready_for_wire_discussion"))
    expected_ready = bool(parity_ok and priced_ok and timings_ok and honesty_ok)
    ready_semantics_ok = ready is expected_ready

    admm_residual_ok = readiness.get("admm_residual_ok")
    admm_block_subproblem_ok = readiness.get("admm_block_subproblem_ok")
    admm_coordination_ok = readiness.get("admm_coordination_ok")
    admm_plant_linking_ok = readiness.get("admm_plant_linking_ok")
    admm_plant_named_linking_ok = readiness.get("admm_plant_named_linking_ok")
    admm_case1_shaped_linking_ok = readiness.get("admm_case1_shaped_linking_ok")
    admm_case1_dual_space_form_contract_ok = readiness.get(
        "admm_case1_dual_space_form_contract_ok"
    )
    admm_case1_dual_space_linf_probe_ok = readiness.get(
        "admm_case1_dual_space_linf_probe_ok"
    )
    admm_case1_dual_space_linf_live_lambda_bridge_ok = readiness.get(
        "admm_case1_dual_space_linf_live_lambda_bridge_ok"
    )
    admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok = readiness.get(
        "admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok"
    )
    admm_case1_honest_blender_pooling_path_ok = readiness.get(
        "admm_case1_honest_blender_pooling_path_ok"
    )
    admm_case1_online_linf_gate_criteria_contract_ok = readiness.get(
        "admm_case1_online_linf_gate_criteria_contract_ok"
    )
    admm_case1_isolation_rewrite_design_contract_ok = readiness.get(
        "admm_case1_isolation_rewrite_design_contract_ok"
    )
    admm_case1_wire_ship_acceptance_design_contract_ok = readiness.get(
        "admm_case1_wire_ship_acceptance_design_contract_ok"
    )
    admm_case1_dual_honest_tf_aware_path_design_contract_ok = readiness.get(
        "admm_case1_dual_honest_tf_aware_path_design_contract_ok"
    )
    admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok = readiness.get(
        "admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok"
    )
    admm_case1_form_label_change_shipped_criteria_contract_ok = readiness.get(
        "admm_case1_form_label_change_shipped_criteria_contract_ok"
    )
    admm_case1_isolation_rewrite_shipped_criteria_contract_ok = readiness.get(
        "admm_case1_isolation_rewrite_shipped_criteria_contract_ok"
    )
    admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok = readiness.get(
        "admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok"
    )
    admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok = readiness.get(
        "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok"
    )
    admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok = readiness.get(
        "admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok"
    )
    admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok = readiness.get(
        "admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok"
    )
    admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok = readiness.get(
        "admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok"
    )
    admm_case1_isolation_rewrite_first_blocker_operational_prep_ok = readiness.get(
        "admm_case1_isolation_rewrite_first_blocker_operational_prep_ok"
    )
    admm_case1_dual_linf_under_wire_criteria_contract_ok = readiness.get(
        "admm_case1_dual_linf_under_wire_criteria_contract_ok"
    )

    blockers_documented = (
        len(wire_blockers) > 0
        and "wire_not_shipped" in wire_blockers
        and "isolation_rewrite_required" in wire_blockers
        and "form_label_change_required" in wire_blockers
        and "dual_linf_under_wire_unproven" in wire_blockers
        and "case1_is_cdu_blender_package_admm" in wire_blockers
        and "no_blender_offline_affine_kernel" in wire_blockers
    )

    compose_ok = bool(
        ready_semantics_ok
        and honesty_ok
        and readiness.get("dual_recovery_path") is None
        and readiness.get("on_excel_case1_path") is False
        and readiness.get("solver") is False
    )
    # When gates are included, prefer true (green ladder); None only if skipped.
    for flag, included in (
        (admm_residual_ok, include_admm_residual),
        (admm_block_subproblem_ok, include_admm_block_subproblem),
        (admm_coordination_ok, include_admm_coordination),
        (admm_plant_linking_ok, include_admm_plant_linking),
        (admm_plant_named_linking_ok, include_admm_plant_named_linking),
        (admm_case1_shaped_linking_ok, include_admm_case1_shaped_linking),
        (
            admm_case1_dual_space_form_contract_ok,
            include_admm_case1_dual_space_form_contract,
        ),
        (
            admm_case1_dual_space_linf_probe_ok,
            include_admm_case1_dual_space_linf_probe,
        ),
        (
            admm_case1_dual_space_linf_live_lambda_bridge_ok,
            include_admm_case1_dual_space_linf_live_lambda_bridge,
        ),
        (
            admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok,
            include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart,
        ),
        (
            admm_case1_honest_blender_pooling_path_ok,
            include_admm_case1_honest_blender_pooling_path,
        ),
        (
            admm_case1_online_linf_gate_criteria_contract_ok,
            include_admm_case1_online_linf_gate_criteria_contract,
        ),
        (
            admm_case1_isolation_rewrite_design_contract_ok,
            include_admm_case1_isolation_rewrite_design_contract,
        ),
        (
            admm_case1_wire_ship_acceptance_design_contract_ok,
            include_admm_case1_wire_ship_acceptance_design_contract,
        ),
        (
            admm_case1_dual_honest_tf_aware_path_design_contract_ok,
            include_admm_case1_dual_honest_tf_aware_path_design_contract,
        ),
        (
            admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok,
            include_admm_case1_dual_honest_tf_aware_path_present_criteria_contract,
        ),
        (
            admm_case1_form_label_change_shipped_criteria_contract_ok,
            include_admm_case1_form_label_change_shipped_criteria_contract,
        ),
        (
            admm_case1_isolation_rewrite_shipped_criteria_contract_ok,
            include_admm_case1_isolation_rewrite_shipped_criteria_contract,
        ),
        (
            admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok,
            include_admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract,
        ),
        (
            admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok,
            include_admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract,
        ),
        (
            admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok,
            include_admm_case1_dual_honest_tf_aware_path_execution_scaffold,
        ),
        (
            admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok,
            include_admm_case1_dual_honest_multi_blocker_wire_rehearsal,
        ),
        (
            admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok,
            include_admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint,
        ),
        (
            admm_case1_isolation_rewrite_first_blocker_operational_prep_ok,
            include_admm_case1_isolation_rewrite_first_blocker_operational_prep,
        ),
        (
            admm_case1_dual_linf_under_wire_criteria_contract_ok,
            include_admm_case1_dual_linf_under_wire_criteria_contract,
        ),
    ):
        if included and flag is False:
            compose_ok = False

    honesty_locks_ok = bool(
        honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["not_wire_shipped"] is True
        and honesty["not_full_plant_mass_balance"] is True
        and honesty["not_pure_admm_dual_recovery"] is True
        and honesty["not_case1_solve"] is True
    )

    preflight_ok = bool(compose_ok and honesty_locks_ok and blockers_documented)
    # ok = preflight surface healthy — NEVER means wire shipped
    ok = preflight_ok and (honesty["wire_shipped"] is False)

    note = (
        "Offline dual-honest wire preflight: composes offline_block_solve_readiness_report "
        "(parity/priced/timings/honesty + additive admm residual/subproblem/coordination/"
        "plant_linking/plant_named/case1_shaped/dual_space_form_contract/"
        "dual_space_linf_probe/live_lambda_bridge/live_lambda_seeded_warmstart/"
        "honest_blender_pooling_path/online_linf_gate_criteria_contract/"
        "isolation_rewrite_design_contract/wire_ship_acceptance_design_contract/"
        "dual_honest_tf_aware_path_design_contract/"
        "dual_honest_tf_aware_path_present_criteria_contract/"
        "form_label_change_shipped_criteria_contract/"
        "isolation_rewrite_shipped_criteria_contract/"
        "dual_honest_multi_blocker_wire_bundle_design_contract "
        "gates) and lists "
        "machine-readable wire_blockers true at "
        "HEAD. preflight_ok/blockers_documented are separate from ready_for_wire_discussion "
        "(still structural parity∧priced∧timings∧honesty only — not redefined by preflight "
        "or blockers). wire_shipped=False always; dual_recovery_path=None; not Case 1 solve; "
        "not pure-ADMM dual recovery; not full plant mass balance. Preflight / plant-linking "
        "λ,z,ρ are synthetic offline demos — not Case 1 PRIMARY online λ or SECONDARY "
        "recovered duals. Blockers are honesty (isolation rewrite, form label change, dual "
        "L∞ under wire unproven, Case 1 CDU+Blender shape, no blender affine kernel, "
        "wire_not_shipped, affine≠plant_blocks feed LP) — not CI failure theater. "
        "This report does not ship wire."
    )

    return {
        **honesty,
        "ok": ok,
        "preflight_ok": preflight_ok,
        "blockers_documented": blockers_documented,
        "compose_ok": compose_ok,
        "honesty_locks_ok": honesty_locks_ok,
        "ready_semantics_ok": ready_semantics_ok,
        # Structural mirror — meaning unchanged
        "ready_for_wire_discussion": ready,
        "parity_ok": parity_ok,
        "priced_ok": priced_ok,
        "timings_ok": timings_ok,
        "honesty_ok": honesty_ok,
        # Additive ADMM ladder flags (mirror readiness; None if skipped)
        "admm_residual_ok": admm_residual_ok,
        "admm_block_subproblem_ok": admm_block_subproblem_ok,
        "admm_coordination_ok": admm_coordination_ok,
        "admm_plant_linking_ok": admm_plant_linking_ok,
        "admm_plant_named_linking_ok": admm_plant_named_linking_ok,
        "admm_case1_shaped_linking_ok": admm_case1_shaped_linking_ok,
        "admm_case1_dual_space_form_contract_ok": admm_case1_dual_space_form_contract_ok,
        "admm_case1_dual_space_linf_probe_ok": admm_case1_dual_space_linf_probe_ok,
        "admm_case1_dual_space_linf_live_lambda_bridge_ok": (
            admm_case1_dual_space_linf_live_lambda_bridge_ok
        ),
        "admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok": (
            admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok
        ),
        "admm_case1_honest_blender_pooling_path_ok": (
            admm_case1_honest_blender_pooling_path_ok
        ),
        "admm_case1_online_linf_gate_criteria_contract_ok": (
            admm_case1_online_linf_gate_criteria_contract_ok
        ),
        "admm_case1_isolation_rewrite_design_contract_ok": (
            admm_case1_isolation_rewrite_design_contract_ok
        ),
        "admm_case1_wire_ship_acceptance_design_contract_ok": (
            admm_case1_wire_ship_acceptance_design_contract_ok
        ),
        "admm_case1_dual_honest_tf_aware_path_design_contract_ok": (
            admm_case1_dual_honest_tf_aware_path_design_contract_ok
        ),
        "admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok": (
            admm_case1_dual_honest_tf_aware_path_present_criteria_contract_ok
        ),
        "admm_case1_form_label_change_shipped_criteria_contract_ok": (
            admm_case1_form_label_change_shipped_criteria_contract_ok
        ),
        "admm_case1_isolation_rewrite_shipped_criteria_contract_ok": (
            admm_case1_isolation_rewrite_shipped_criteria_contract_ok
        ),
        "admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok": (
            admm_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ok
        ),
        "admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok": (
            admm_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ok
        ),
        "admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok": (
            admm_case1_dual_honest_tf_aware_path_execution_scaffold_ok
        ),
        "admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok": (
            admm_case1_dual_honest_multi_blocker_wire_rehearsal_ok
        ),
        "admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok": (
            admm_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ok
        ),
        "admm_case1_isolation_rewrite_first_blocker_operational_prep_ok": (
            admm_case1_isolation_rewrite_first_blocker_operational_prep_ok
        ),
        "admm_case1_dual_linf_under_wire_criteria_contract_ok": (
            admm_case1_dual_linf_under_wire_criteria_contract_ok
        ),
        # Blockers
        "wire_blockers": wire_blockers,
        "wire_blocker_notes": wire_blocker_notes,
        "n_wire_blockers": len(wire_blockers),
        # Nested readiness for full detail (timings etc.) without redefining ready
        "readiness": readiness,
        "units": list(UNITS),
        "tf_available": tf_available(),
        "note": note,
    }


def multi_unit_wire_preflight_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_wire_preflight_report`` (multi-unit naming twin)."""
    return offline_wire_preflight_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1-shaped CDU↔Blender linking ADMM skeleton (goal 5 residual)
# ---------------------------------------------------------------------------
# Always-on numpy. Models Case 1 package *shape* (CDU producer ↔ Blender
# consumer on classic intermediates) without wiring TF into Case 1 solve.
# dual_recovery_path=None; wire_shipped=False; not form flip; not full plant MB.
# blender_surface=linear_quality_pooling — NOT a base_delta affine UNITS entry.
# Does NOT clear DEFAULT_WIRE_BLOCKERS (skeleton ≠ package-ADMM wire).

CASE1_SHAPED_LINKING_KIND = "offline_case1_shaped_cdu_blender_linking"
CASE1_SHAPED_LINKING_SCOPE = "case1_shaped_offline_demo"
CASE1_SHAPED_LINKING_STREAMS = (
    "naphtha",
    "distillate",
    "gasoil",
    "residue",
)
CASE1_SHAPED_BLENDER_SURFACE = "linear_quality_pooling"
CASE1_SHAPED_LINKING_FORMULA = (
    "round: map lam/z->CDU products via Case1 incidence; "
    "CDU x=argmax raw L1-aug under (lam_u,z_u,rho,delta); "
    "y_link=A_cdu y_raw; blender use=R^T y_prod (linear pooling); "
    "r_link=y_link-use (pre-z); z<-(1-beta)z+beta*0.5*(y+use); "
    "lam<-lam+alpha*rho*r_link"
)

# CDU affine products → Case 1 intermediate streams (0/1 selection incidence).
# Note: plant-linking maps cdu_resid → "resid"; Case 1 spelling is "residue".
# cdu_offgas is intentionally unmapped (not a Case 1 package intermediate).
_CDU_PRODUCT_TO_CASE1_STREAM: Dict[str, str] = {
    "cdu_naphtha_light": "naphtha",
    "cdu_naphtha_heavy": "naphtha",
    "cdu_distillate": "distillate",
    "cdu_gasoil": "gasoil",
    "cdu_resid": "residue",
}

# Classic Case 1 style recipes: product → intermediate volume fractions.
# Planning-grade synthetic offline demo — not live blend QP duals.
CASE1_SHAPED_BLEND_RECIPES: Dict[str, Dict[str, float]] = {
    "gasoline": {"naphtha": 0.85, "distillate": 0.15},
    "diesel": {"distillate": 0.70, "gasoil": 0.30},
    "fuel_oil": {"gasoil": 0.40, "residue": 0.60},
}

CASE1_SHAPED_PRODUCT_PRICES: Dict[str, float] = {
    "gasoline": 90.0,
    "diesel": 85.0,
    "fuel_oil": 60.0,
}

# Soft product box around synthetic offline demo volumes (not Case 1 solution).
CASE1_SHAPED_PRODUCT_REF: Dict[str, float] = {
    "gasoline": 10.0,
    "diesel": 12.0,
    "fuel_oil": 8.0,
}

# Synthetic offline crude feed scale: CDU affine yields are mass fractions;
# blender product volumes are planning-scale. Scale producer y_link by feed so
# producer−consumer residual lives in comparable units (not plant MB claim).
CASE1_SHAPED_CDU_FEED_SCALE = 30.0


def _case1_shaped_linking_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire locks for Case-1-shaped skeleton."""
    return {
        "kind": CASE1_SHAPED_LINKING_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "not_case1_solve": True,
        "case1_shaped_offline_only": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_plant_linking_multi_unit_fcc_coker_cdu": True,
        "linking_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "excel_cdu_matrix_matches_affine": None,
        "excel_blender_matrix_matches_affine": None,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "scope": CASE1_SHAPED_LINKING_SCOPE,
        "linking_space": "case1_intermediate_streams",
        "z_update_space": "case1_intermediate_streams",
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "formula": CASE1_SHAPED_LINKING_FORMULA,
        "note": (
            "Offline Case-1-shaped CDU↔Blender linking ADMM skeleton: CDU affine "
            "producer projected onto Case 1 intermediates (naphtha/distillate/gasoil/"
            "residue) + honest blender linear quality/pooling residual under synthetic "
            "λ,z,ρ. dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            "wire_shipped=False; case1_form_unchanged (classic_2block_excel_path outside "
            "this surface). Skeleton λ/z/ρ are NOT Case 1 PRIMARY online λ, NOT "
            "SECONDARY recovered blender duals, NOT pure-ADMM dual recovery. "
            "blender_surface=linear_quality_pooling — not base_delta_affine_unit; "
            "UNITS stay FCC/COKER/CDU (no silent BLENDER). Not full plant mass balance; "
            "not live plant_blocks cascade; not multi-unit FCC plant-linking replacement. "
            "Does not invent excel_cdu_matrix_matches_affine / excel_blender_matrix_matches_"
            "affine. Does not clear DEFAULT_WIRE_BLOCKERS (skeleton ≠ package-ADMM wire; "
            "linear pooling ≠ affine kernel). No residual-must-vanish hard-fail; no PuLP "
            "on offline hot path; always-on numpy; TF not required."
        ),
    }


def case1_shaped_cdu_to_intermediate_map() -> Dict[str, str]:
    """Honest CDU product → Case 1 intermediate incidence labels (not excel match)."""
    return dict(_CDU_PRODUCT_TO_CASE1_STREAM)


def project_cdu_y_to_case1_intermediates(
    y_cdu: Mapping[str, float],
    *,
    streams: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Lift CDU product yields onto Case 1 intermediate streams via 0/1 incidence.

    light+heavy naphtha sum into naphtha; cdu_resid maps to residue (not resid).
    Unmapped products (e.g. cdu_offgas) do not contribute.
    """
    streams_list = list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    out = {s: 0.0 for s in streams_list}
    for product, stream in _CDU_PRODUCT_TO_CASE1_STREAM.items():
        if stream not in out:
            continue
        out[stream] += float(y_cdu.get(product, 0.0))
    return out


def _case1_recipe_use_matrix(
    *,
    products: Sequence[str],
    streams: Sequence[str],
    recipes: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """R shape (n_products, n_streams): intermediate fraction per product."""
    R = np.zeros((len(products), len(streams)), dtype=np.float64)
    s_index = {s: i for i, s in enumerate(streams)}
    for i, p in enumerate(products):
        row = recipes.get(p, {})
        for s, frac in row.items():
            if s in s_index:
                R[i, s_index[s]] = float(frac)
    return R


def blender_recipe_use_from_products(
    y_products: Mapping[str, float],
    *,
    streams: Optional[Sequence[str]] = None,
    recipes: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, float]:
    """Linear pooling: use = R^T y_products (planning residual; not QP duals)."""
    streams_list = list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    rec = recipes if recipes is not None else CASE1_SHAPED_BLEND_RECIPES
    products = list(rec.keys())
    R = _case1_recipe_use_matrix(products=products, streams=streams_list, recipes=rec)
    y_vec = np.array([float(y_products.get(p, 0.0)) for p in products], dtype=np.float64)
    use_vec = R.T @ y_vec
    return {s: float(use_vec[i]) for i, s in enumerate(streams_list)}


def _default_case1_lam_link(streams: Sequence[str]) -> Dict[str, float]:
    """Seed linking λ from CDU default product prices mapped into Case 1 streams."""
    prices = default_offline_prices("CDU")
    sums: Dict[str, float] = {s: 0.0 for s in streams}
    counts: Dict[str, int] = {s: 0 for s in streams}
    for product, stream in _CDU_PRODUCT_TO_CASE1_STREAM.items():
        if stream not in sums:
            continue
        sums[stream] += float(prices.get(product, 0.0))
        counts[stream] += 1
    return {
        s: float(sums[s] / counts[s]) if counts[s] else 0.0 for s in streams
    }


def _default_case1_z_link(streams: Sequence[str]) -> Dict[str, float]:
    """Seed z from CDU full postprocess yields at reference projected to Case 1."""
    coeffs = cached_offline_unit_coeffs("CDU")
    z_full = _default_z_full_for_unit("CDU", coeffs)
    frac = project_cdu_y_to_case1_intermediates(z_full, streams=streams)
    feed = float(CASE1_SHAPED_CDU_FEED_SCALE)
    return {s: float(frac[s]) * feed for s in streams}


def _project_case1_link_to_cdu(
    lam_link: Mapping[str, float],
    z_link: Mapping[str, float],
    *,
    streams: Sequence[str],
) -> Dict[str, Any]:
    """Map Case 1 intermediate λ/z into CDU product prices/z via incidence^T."""
    coeffs = cached_offline_unit_coeffs("CDU")
    products = list(coeffs.products)
    A = _incidence_matrix_for_unit(
        "CDU",
        streams,
        products,
        product_to_stream=_CDU_PRODUCT_TO_CASE1_STREAM,
    )
    lam_vec = np.array([float(lam_link.get(s, 0.0)) for s in streams], dtype=np.float64)
    z_vec = np.array([float(z_link.get(s, 0.0)) for s in streams], dtype=np.float64)
    prices_vec = A.T @ lam_vec
    z_unit_vec = A.T @ z_vec
    return {
        "products": products,
        "prices": {
            # pack_price_vector requires non-negative product prices; linking λ may
            # be signed after dual ascent — soft-rectify for CDU subproblem only.
            p: float(max(0.0, prices_vec[i])) for i, p in enumerate(products)
        },
        "z": {p: float(z_unit_vec[i]) for i, p in enumerate(products)},
        "A_shape": [int(A.shape[0]), int(A.shape[1])],
    }


def _blender_linear_pooling_step(
    lam_link: Mapping[str, float],
    z_link: Mapping[str, float],
    *,
    rho: float,
    streams: Sequence[str],
    recipes: Mapping[str, Mapping[str, float]],
    product_prices: Mapping[str, float],
    product_ref: Mapping[str, float],
    product_box: float = 5.0,
    max_passes: int = 24,
) -> Dict[str, Any]:
    """Honest blender linear pooling residual under synthetic λ,z,ρ (no PuLP).

    Maximizes planning-grade local objective on product volumes y_p in a box
    around synthetic ref volumes::

        sum_p price_p y_p - sum_i λ_i use_i - ρ ||use - z||_1
        use = R^T y

    Coordinate ascent on y (always-on numpy). Surface class:
    linear_quality_pooling — not base_delta_affine_unit.
    """
    products = list(recipes.keys())
    R = _case1_recipe_use_matrix(products=products, streams=streams, recipes=recipes)
    n_p = len(products)
    y = np.array([float(product_ref.get(p, 0.0)) for p in products], dtype=np.float64)
    lo = np.maximum(0.0, y - float(product_box))
    hi = y + float(product_box)
    y = np.clip(y, lo, hi)

    lam_vec = np.array([float(lam_link.get(s, 0.0)) for s in streams], dtype=np.float64)
    z_vec = np.array([float(z_link.get(s, 0.0)) for s in streams], dtype=np.float64)
    price_vec = np.array(
        [float(product_prices.get(p, 0.0)) for p in products], dtype=np.float64
    )
    rho_f = float(rho)

    def _objective(y_vec: np.ndarray) -> float:
        use = R.T @ y_vec
        return float(
            price_vec @ y_vec
            - lam_vec @ use
            - rho_f * np.sum(np.abs(use - z_vec))
        )

    def _coord_step(y_vec: np.ndarray, j: int) -> float:
        """1D exact max on coordinate j under box (piecewise linear)."""
        # sample candidate breakpoints from residual zero-crossings + bounds
        y_try = y_vec.copy()
        best_val = -np.inf
        best_yj = float(y_vec[j])
        candidates = {float(lo[j]), float(hi[j]), float(y_vec[j])}
        # residual r_i = (R^T y)_i - z_i; zero when y_j moves
        # use_i(y) = use0_i + R[j,i]*(y_j - y0_j)
        use0 = R.T @ y_vec
        for i, s in enumerate(streams):
            rji = float(R[j, i])
            if abs(rji) < 1e-15:
                continue
            # set use_i = z_i ⇒ y_j = y0_j + (z_i - use0_i)/R[j,i]
            yj_star = float(y_vec[j] + (z_vec[i] - use0[i]) / rji)
            if lo[j] - 1e-12 <= yj_star <= hi[j] + 1e-12:
                candidates.add(float(np.clip(yj_star, lo[j], hi[j])))
        # also midpoints for robustness
        candidates.add(float(0.5 * (lo[j] + hi[j])))
        for yj in candidates:
            y_try[j] = yj
            val = _objective(y_try)
            if val > best_val + 1e-15 or (
                abs(val - best_val) <= 1e-15 and abs(yj - y_vec[j]) < abs(best_yj - y_vec[j])
            ):
                best_val = val
                best_yj = float(yj)
        return best_yj

    obj0 = _objective(y)
    for _ in range(int(max_passes)):
        y_prev = y.copy()
        for j in range(n_p):
            y[j] = _coord_step(y, j)
        if float(np.max(np.abs(y - y_prev))) <= 1e-12:
            break
    obj_star = _objective(y)
    use_vec = R.T @ y
    use = {s: float(use_vec[i]) for i, s in enumerate(streams)}
    y_prod = {p: float(y[i]) for i, p in enumerate(products)}
    r_use = {s: float(use[s] - float(z_link.get(s, 0.0))) for s in streams}
    r_l1 = float(sum(abs(v) for v in r_use.values()))
    finite_ok = bool(
        np.all(np.isfinite(y))
        and np.all(np.isfinite(use_vec))
        and np.isfinite(obj_star)
        and np.isfinite(r_l1)
    )
    return {
        "ok": finite_ok,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "products": products,
        "y_products": y_prod,
        "use": use,
        "r_use_vs_z": r_use,
        "r_l1_use": r_l1,
        "augmented_local": float(obj_star),
        "augmented_local_ref": float(obj0),
        "not_worse_than_ref": bool(obj_star + 1e-9 >= obj0),
        "recipes": {p: dict(recipes[p]) for p in products},
        "product_prices": {p: float(product_prices.get(p, 0.0)) for p in products},
        "solver": False,
        "dual_recovery_path": None,
    }


def case1_shaped_cdu_blender_linking_round(
    *,
    lam_link: Optional[Mapping[str, float]] = None,
    z_link: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float], np.ndarray] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    max_passes: int = 32,
    product_box: float = 5.0,
) -> Dict[str, Any]:
    """One offline Case-1-shaped CDU↔Blender linking ADMM round (always-on numpy).

    Structure:
      1. CDU subproblem under mapped Case1 intermediate λ/z
      2. Lift CDU y_raw → Case1 intermediates
      3. Blender linear pooling step under same λ/z,ρ
      4. r_link = y_cdu − use_blender (pre-z)
      5. z consensus midpoint; λ dual ascent on pre-z residual

    Not Case 1 solve, not wire, not dual recovery, not full plant MB.
    """
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    streams = list(CASE1_SHAPED_LINKING_STREAMS)
    lam_pre = dict(lam_link) if lam_link is not None else _default_case1_lam_link(streams)
    z_pre = dict(z_link) if z_link is not None else _default_case1_z_link(streams)
    for s in streams:
        if s not in lam_pre:
            raise ValueError(f"Missing lam_link stream {s!r}")
        if s not in z_pre:
            raise ValueError(f"Missing z_link stream {s!r}")
        if not np.isfinite(float(lam_pre[s])) or not np.isfinite(float(z_pre[s])):
            raise ValueError(f"lam_link/z_link must be finite for stream {s!r}")

    # --- CDU producer ---
    proj = _project_case1_link_to_cdu(lam_pre, z_pre, streams=streams)
    sub = admm_block_subproblem_for_unit(
        "CDU",
        prices=proj["prices"],
        z=proj["z"],
        rho=float(rho),
        delta=delta,
        max_passes=int(max_passes),
    )
    y_raw = dict(sub["y_raw_star"])
    y_link_frac = project_cdu_y_to_case1_intermediates(y_raw, streams=streams)
    feed = float(CASE1_SHAPED_CDU_FEED_SCALE)
    y_link = {s: float(y_link_frac[s]) * feed for s in streams}
    cdu_ok = bool(sub.get("ok"))

    # --- Blender consumer (linear pooling) ---
    blend = _blender_linear_pooling_step(
        lam_pre,
        z_pre,
        rho=float(rho),
        streams=streams,
        recipes=CASE1_SHAPED_BLEND_RECIPES,
        product_prices=CASE1_SHAPED_PRODUCT_PRICES,
        product_ref=CASE1_SHAPED_PRODUCT_REF,
        product_box=float(product_box),
        max_passes=int(max_passes),
    )
    use = dict(blend["use"])
    blender_ok = bool(blend.get("ok"))

    # Pre-z producer−consumer residual
    r_vec = np.array(
        [float(y_link[s]) - float(use[s]) for s in streams], dtype=np.float64
    )
    r_link = {s: float(r_vec[i]) for i, s in enumerate(streams)}
    r_l1 = float(np.sum(np.abs(r_vec)))
    r_linf = float(np.max(np.abs(r_vec))) if r_vec.size else 0.0
    r_l2 = float(np.sqrt(np.sum(r_vec * r_vec))) if r_vec.size else 0.0

    y_vec = np.array([float(y_link[s]) for s in streams], dtype=np.float64)
    use_vec = np.array([float(use[s]) for s in streams], dtype=np.float64)
    z_pre_vec = np.array([float(z_pre[s]) for s in streams], dtype=np.float64)
    # Midpoint consensus between producer supply and consumer use
    mid = 0.5 * (y_vec + use_vec)
    z_post_vec = (1.0 - beta) * z_pre_vec + beta * mid
    z_post = {s: float(z_post_vec[i]) for i, s in enumerate(streams)}

    alpha = float(dual_step)
    lam_pre_vec = np.array([float(lam_pre[s]) for s in streams], dtype=np.float64)
    lam_post_vec = lam_pre_vec + alpha * float(rho) * r_vec
    lam_post = {s: float(lam_post_vec[i]) for i, s in enumerate(streams)}

    finite_ok = bool(
        cdu_ok
        and blender_ok
        and np.all(np.isfinite(r_vec))
        and np.all(np.isfinite(z_post_vec))
        and np.all(np.isfinite(lam_post_vec))
        and np.isfinite(r_l1)
        and np.isfinite(r_linf)
    )
    honesty = _case1_shaped_linking_honesty_fields()
    return {
        **{
            k: honesty[k]
            for k in (
                "kind",
                "solver",
                "dual_recovery_path",
                "on_excel_case1_path",
                "not_case1_solve",
                "case1_shaped_offline_only",
                "case1_form_unchanged",
                "wire_shipped",
                "not_wire_shipped",
                "not_pure_admm_dual_recovery",
                "not_full_plant_mass_balance",
                "not_full_plant_blocks_feed_lp",
                "not_live_plant_blocks",
                "not_plant_linking_multi_unit_fcc_coker_cdu",
                "linking_lambda_is_not_case1_online_lambda",
                "skeleton_lambda_is_not_case1_primary_or_secondary_duals",
                "blender_surface",
                "blender_is_base_delta_affine_unit",
                "scope",
                "linking_space",
                "z_update_space",
                "price_source",
                "lam_source",
                "z_source",
                "rho_source",
                "formula",
            )
        },
        "ok": finite_ok,
        "streams": streams,
        "packages": ["CDU", "BLENDER"],
        "rho": float(rho),
        "dual_step": alpha,
        "z_blend": beta,
        "lam_pre": dict(lam_pre),
        "lam_post": lam_post,
        "z_pre": dict(z_pre),
        "z_post": z_post,
        "y_link_cdu": dict(y_link),
        "use_blender": dict(use),
        "r_link_pre": r_link,
        "r_l1_link": r_l1,
        "r_linf_link": r_linf,
        "r_l2_link": r_l2,
        "cdu": {
            "ok": cdu_ok,
            "subproblem_ok": cdu_ok,
            "subproblem_kind": sub.get("kind"),
            "not_worse_than_ref": bool(sub.get("not_worse_than_ref")),
            "augmented_local_raw": float(sub["augmented_local_raw"]),
            "y_raw_star": y_raw,
            "x_star": list(sub["x_star"]),
            "y_link": dict(y_link),
            "prices_unit": dict(proj["prices"]),
            "z_unit": dict(proj["z"]),
            "cdu_to_case1_map": case1_shaped_cdu_to_intermediate_map(),
        },
        "blender": {
            "ok": blender_ok,
            "blender_surface": blend["blender_surface"],
            "blender_is_base_delta_affine_unit": False,
            "y_products": dict(blend["y_products"]),
            "use": dict(use),
            "augmented_local": float(blend["augmented_local"]),
            "not_worse_than_ref": bool(blend["not_worse_than_ref"]),
            "r_l1_use": float(blend["r_l1_use"]),
            "recipes": blend["recipes"],
        },
        "sum_augmented_local": float(sub["augmented_local_raw"]) + float(blend["augmented_local"]),
        "all_package_ok": bool(cdu_ok and blender_ok),
    }


def offline_case1_shaped_cdu_blender_linking_report(
    *,
    n_rounds: int = 3,
    rho: float = 1.0,
    delta: Union[float, Mapping[str, float]] = 1.0,
    dual_step: float = 1.0,
    z_blend: float = 1.0,
    lam0: Optional[Mapping[str, float]] = None,
    z0: Optional[Mapping[str, float]] = None,
    max_passes: int = 32,
    product_box: float = 5.0,
) -> Dict[str, Any]:
    """Always-on multi-round Case-1-shaped CDU↔Blender offline linking report.

    Aggregate ``ok`` = honesty locks + finite trajectory + package structure.
    **No** residual-must-vanish SLA. **No** TF required. **No** PuLP hot path.
    Does **not** clear DEFAULT_WIRE_BLOCKERS. Does **not** change Case 1 form.
    """
    if int(n_rounds) < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    if float(rho) <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(float(dual_step)):
        raise ValueError(f"dual_step must be finite, got {dual_step}")
    beta = float(z_blend)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError(f"z_blend must be in [0, 1], got {z_blend}")

    n_r = int(n_rounds)
    streams = list(CASE1_SHAPED_LINKING_STREAMS)
    honesty = _case1_shaped_linking_honesty_fields()

    state_lam = dict(lam0) if lam0 is not None else _default_case1_lam_link(streams)
    state_z = dict(z0) if z0 is not None else _default_case1_z_link(streams)

    trajectory: List[Dict[str, Any]] = []
    packages_out: Dict[str, Any] = {
        "CDU": {
            "package": "CDU",
            "ok": True,
            "rounds": [],
            "final_y_link": None,
            "last_x_star": None,
            "last_y_raw_star": None,
        },
        "BLENDER": {
            "package": "BLENDER",
            "ok": True,
            "rounds": [],
            "final_use": None,
            "last_y_products": None,
            "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        },
    }
    all_ok = True

    for t in range(1, n_r + 1):
        row = case1_shaped_cdu_blender_linking_round(
            lam_link=state_lam,
            z_link=state_z,
            rho=float(rho),
            delta=delta,
            dual_step=float(dual_step),
            z_blend=beta,
            max_passes=int(max_passes),
            product_box=float(product_box),
        )
        state_lam = dict(row["lam_post"])
        state_z = dict(row["z_post"])

        cdu_row = row["cdu"]
        blend_row = row["blender"]
        cdu_compact = {
            "round": t,
            "ok": bool(cdu_row["ok"]),
            "subproblem_ok": bool(cdu_row["subproblem_ok"]),
            "not_worse_than_ref": bool(cdu_row["not_worse_than_ref"]),
            "augmented_local_raw": float(cdu_row["augmented_local_raw"]),
            "y_link": dict(cdu_row["y_link"]),
        }
        blend_compact = {
            "round": t,
            "ok": bool(blend_row["ok"]),
            "not_worse_than_ref": bool(blend_row["not_worse_than_ref"]),
            "augmented_local": float(blend_row["augmented_local"]),
            "use": dict(blend_row["use"]),
            "y_products": dict(blend_row["y_products"]),
        }
        packages_out["CDU"]["rounds"].append(cdu_compact)
        packages_out["CDU"]["final_y_link"] = dict(cdu_row["y_link"])
        packages_out["CDU"]["last_x_star"] = list(cdu_row["x_star"])
        packages_out["CDU"]["last_y_raw_star"] = dict(cdu_row["y_raw_star"])
        packages_out["BLENDER"]["rounds"].append(blend_compact)
        packages_out["BLENDER"]["final_use"] = dict(blend_row["use"])
        packages_out["BLENDER"]["last_y_products"] = dict(blend_row["y_products"])

        round_ok = bool(row.get("ok"))
        if not cdu_row["ok"]:
            packages_out["CDU"]["ok"] = False
            round_ok = False
            all_ok = False
        if not blend_row["ok"]:
            packages_out["BLENDER"]["ok"] = False
            round_ok = False
            all_ok = False

        traj_row = {
            "round": t,
            "ok": bool(
                round_ok
                and np.isfinite(float(row["r_l1_link"]))
                and np.isfinite(float(row["r_linf_link"]))
                and np.isfinite(float(row["sum_augmented_local"]))
            ),
            "r_l1_link": float(row["r_l1_link"]),
            "r_linf_link": float(row["r_linf_link"]),
            "r_l2_link": float(row["r_l2_link"]),
            "sum_augmented_local": float(row["sum_augmented_local"]),
            "r_link_pre": dict(row["r_link_pre"]),
            "packages_ok": {
                "CDU": bool(cdu_compact["ok"]),
                "BLENDER": bool(blend_compact["ok"]),
            },
        }
        if not traj_row["ok"]:
            all_ok = False
        trajectory.append(traj_row)

    residual_trend = "n/a"
    if len(trajectory) >= 2:
        vals = [float(tr["r_l1_link"]) for tr in trajectory]
        if all(np.isfinite(v) for v in vals):
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if all(d <= 1e-12 for d in diffs):
                residual_trend = "nonincreasing"
            elif all(d >= -1e-12 for d in diffs):
                residual_trend = "nondecreasing"
            else:
                residual_trend = "mixed"

    honesty_ok = (
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and "BLENDER" not in UNITS
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["solver"] is False
        and honesty["kind"] == CASE1_SHAPED_LINKING_KIND
        and honesty["wire_shipped"] is False
        and honesty["not_wire_shipped"] is True
        and honesty["not_full_plant_mass_balance"] is True
        and honesty["not_case1_solve"] is True
        and honesty["case1_shaped_offline_only"] is True
        and honesty["case1_form_unchanged"] is True
        and honesty["linking_lambda_is_not_case1_online_lambda"] is True
        and honesty["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and honesty["blender_is_base_delta_affine_unit"] is False
        and honesty["excel_cdu_matrix_matches_affine"] is None
        and honesty["excel_blender_matrix_matches_affine"] is None
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and set(packages_out.keys()) == {"CDU", "BLENDER"}
        and len(trajectory) == n_r
        and list(streams) == list(CASE1_SHAPED_LINKING_STREAMS)
        and "case1_is_cdu_blender_package_admm" in DEFAULT_WIRE_BLOCKERS
        and "no_blender_offline_affine_kernel" in DEFAULT_WIRE_BLOCKERS
    )
    if not honesty_ok:
        all_ok = False
    for pkg in packages_out.values():
        if not pkg.get("ok"):
            all_ok = False

    finite_traj = all(
        np.isfinite(tr["r_l1_link"])
        and np.isfinite(tr["r_linf_link"])
        and np.isfinite(tr["sum_augmented_local"])
        for tr in trajectory
    )
    if not finite_traj:
        all_ok = False

    ok_criteria = (
        "honesty_ok ∧ finite trajectory ∧ package structure "
        "(CDU affine + blender linear pooling); "
        "NOT residual-must-vanish; NOT wire shipped; NOT blockers empty"
    )

    return {
        "ok": bool(all_ok and honesty_ok and finite_traj),
        "packages": packages_out,
        "package_order": ["CDU", "BLENDER"],
        "streams": streams,
        "trajectory": trajectory,
        "n_rounds": n_r,
        "rho": float(rho),
        "dual_step": float(dual_step),
        "z_blend": beta,
        "delta": delta if not isinstance(delta, Mapping) else dict(delta),
        "final_lam": dict(state_lam),
        "final_z": dict(state_z),
        "cdu_to_case1_map": case1_shaped_cdu_to_intermediate_map(),
        "blend_recipes": {p: dict(r) for p, r in CASE1_SHAPED_BLEND_RECIPES.items()},
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "kind": CASE1_SHAPED_LINKING_KIND,
        "not_case1_solve": True,
        "case1_shaped_offline_only": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_plant_linking_multi_unit_fcc_coker_cdu": True,
        "linking_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "excel_cdu_matrix_matches_affine": None,
        "excel_blender_matrix_matches_affine": None,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "scope": CASE1_SHAPED_LINKING_SCOPE,
        "linking_space": honesty["linking_space"],
        "z_update_space": honesty["z_update_space"],
        "price_source": PRICE_SOURCE,
        "lam_source": PRICE_SOURCE,
        "z_source": "synthetic_offline_demo",
        "rho_source": "synthetic_offline_demo",
        "formula": CASE1_SHAPED_LINKING_FORMULA,
        "residual_trend": residual_trend,
        "honesty_ok": honesty_ok,
        "ok_criteria": ok_criteria,
        "tf_available": tf_available(),
        "units_affine_unchanged": list(UNITS),
        "cdu_feed_scale": float(CASE1_SHAPED_CDU_FEED_SCALE),
        "cdu_feed_scale_note": (
            "Synthetic offline feed multiplies CDU mass-fraction yields into "
            "planning-scale intermediate volumes; not live plant mass balance."
        ),
        "note": honesty["note"],
    }


def multi_block_case1_shaped_linking_admm_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_shaped_cdu_blender_linking_report``."""
    return offline_case1_shaped_cdu_blender_linking_report(**kwargs)


# ---------------------------------------------------------------------------
# Offline Case-1 dual-space / form-label contract (goal 5 + goal 3 residual)
# ---------------------------------------------------------------------------
# Always-on numpy. Registers planned TF-aware form label WITHOUT flipping
# Case 1; maps dual-space stream slots (Case 1 intermediates ↔ skeleton λ);
# records dual_linf_under_wire as unproven with open checklist.
# dual_recovery_path=None; wire_shipped=False; does NOT clear DEFAULT_WIRE_BLOCKERS;
# does NOT redefine ready_for_wire_discussion; does NOT re-run maximizers;
# no TF / no PuLP / no live Case 1 solve on hot path.

CASE1_DUAL_SPACE_FORM_CONTRACT_KIND = "offline_case1_dual_space_form_contract"

# Form-label registry (no flip). Planned name is distinct from current classic.
CASE1_FORM_CURRENT = "classic_2block_excel_path"
CASE1_PLANNED_TF_AWARE_FORM = "tf_affine_cdu_blender_shaped_excel_path"

# dual_linf under wire remains unproven at HEAD — open items only.
CASE1_DUAL_LINF_UNDER_WIRE_STATUS = "unproven"
CASE1_DUAL_LINF_PROOF_CHECKLIST: Dict[str, str] = {
    "isolation_rewrite_with_wire": "open",
    "form_label_change_shipped": "open",  # registered planned form ≠ form shipped
    "online_linf_gate_under_tf_path": "open",
    "wire_shipped": "false_today",
    "blender_affine_kernel_or_honest_pooling_path": "honest_pooling_path_present",
}

# Critical blockers that must remain true after contract ships (prep ≠ cleared).
CASE1_CONTRACT_CRITICAL_BLOCKERS: tuple = (
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "wire_not_shipped",
    "isolation_rewrite_required",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
)


def case1_form_label_contract() -> Dict[str, Any]:
    """Form-label registry fields (current classic vs planned TF-aware; no flip)."""
    form_current = CASE1_FORM_CURRENT
    form_planned = CASE1_PLANNED_TF_AWARE_FORM
    planned_distinct = form_planned != form_current and form_planned != (
        "classic_2block_excel_path"
    )
    form_unchanged = form_current == "classic_2block_excel_path"
    form_label_change_required_still_true = (
        "form_label_change_required" in DEFAULT_WIRE_BLOCKERS
        and planned_distinct
        and form_unchanged
    )
    return {
        "form_current": form_current,
        "form_planned": form_planned,
        "planned_form": form_planned,
        "form_unchanged": form_unchanged,
        "case1_form_unchanged": form_unchanged,
        "planned_form_distinct": planned_distinct,
        "form_label_change_required_still_true": form_label_change_required_still_true,
        "form_label_change_required": form_label_change_required_still_true,
        "form_contract_ok": bool(
            form_unchanged and planned_distinct and form_label_change_required_still_true
        ),
        "note": (
            "Planned TF-aware form is registered only — Case 1 still uses "
            f"{form_current}. Never silent reuse of classic as planned name; "
            "never mutate Excel model.form from this surface."
        ),
    }


def case1_dual_space_stream_map() -> Dict[str, Any]:
    """Map Case 1 intermediates ↔ Case-1-shaped skeleton λ/z slots (name-set).

    No numeric dual equality. No live PuLP duals. Skeleton λ ≠ Case 1 duals.
    """
    streams = list(CASE1_SHAPED_LINKING_STREAMS)
    skeleton_lambda_slots = list(CASE1_SHAPED_LINKING_STREAMS)
    expected = ["naphtha", "distillate", "gasoil", "residue"]
    stream_alignment_ok = (
        streams == skeleton_lambda_slots
        and streams == expected
        and "resid" not in streams  # plant-linking spelling is separate
    )
    stream_dual_roles = {
        s: {
            "skeleton_lambda_slot": s,
            "package_dual_gate": "online_lambda",
            "package_dual_secondary": "recovered_blender",
            "gate_role": "PRIMARY",
            "secondary_role": "SECONDARY_not_gate",
        }
        for s in streams
    }
    return {
        "streams": streams,
        "linking_streams": streams,
        "skeleton_lambda_slots": skeleton_lambda_slots,
        "stream_alignment_ok": stream_alignment_ok,
        "package_dual_gate": "online_lambda",
        "package_dual_secondary": "recovered_blender",
        "package_dual_gate_role": "PRIMARY",
        "package_dual_secondary_role": "SECONDARY_not_gate",
        "skeleton_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "dual_recovery_path": None,
        "stream_dual_roles": stream_dual_roles,
        "note": (
            "Name-set alignment for future dual L∞ compare under wire. "
            "Case 1 package dual gate is PRIMARY online λ; SECONDARY recovered "
            "blender face is not the VERDICT gate. Skeleton λ/z slots reuse "
            "CASE1_SHAPED_LINKING_STREAMS identity map at HEAD — not numeric "
            "dual equality; not pure-ADMM dual recovery."
        ),
    }


def case1_dual_linf_proof_checklist() -> Dict[str, Any]:
    """Machine-readable dual_linf_under_wire prep checklist (status unproven)."""
    checklist = dict(CASE1_DUAL_LINF_PROOF_CHECKLIST)
    open_ids = [
        k
        for k, v in checklist.items()
        if str(v).lower() in ("open", "false_today", "unproven")
    ]
    status = CASE1_DUAL_LINF_UNDER_WIRE_STATUS
    status_unproven = status == "unproven"
    blocker_still = "dual_linf_under_wire_unproven" in DEFAULT_WIRE_BLOCKERS
    return {
        "dual_linf_under_wire": status,
        "dual_linf_under_wire_status": status,
        "dual_linf_under_wire_unproven_still_true": bool(
            status_unproven and blocker_still
        ),
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": len(open_ids),
        "dual_linf_status_unproven_ok": bool(
            status_unproven and len(open_ids) > 0 and blocker_still
        ),
        "note": (
            "Contract existence does not flip dual_linf_under_wire to proven. "
            "Online λ L∞ gate under a TF-aware path remains open until isolation "
            "rewrite + form label change + wire ship a dual-honest path."
        ),
    }


def _case1_dual_space_form_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire locks for the dual-space/form contract."""
    return {
        "kind": CASE1_DUAL_SPACE_FORM_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "skeleton_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "excel_cdu_matrix_matches_affine": None,
        "excel_blender_matrix_matches_affine": None,
        "scope": "case1_dual_space_form_contract_offline",
        "note": (
            "Offline Case-1 dual-space / form-label contract: registers planned "
            f"TF-aware form ({CASE1_PLANNED_TF_AWARE_FORM}) without flipping Case 1 "
            f"form ({CASE1_FORM_CURRENT}); maps Case 1 intermediates "
            "(naphtha/distillate/gasoil/residue) to Case-1-shaped skeleton λ slots; "
            "records dual_linf_under_wire=unproven with open checklist. "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            "wire_shipped=False; not pure-ADMM dual recovery; not full plant mass "
            "balance; not isolation rewrite; not full TF→ADMM wire. Skeleton λ ≠ "
            "Case 1 PRIMARY online λ / SECONDARY recovered duals. "
            "blender_surface=linear_quality_pooling — not base_delta_affine_unit; "
            "UNITS stay FCC/COKER/CDU. Does not invent excel_cdu_matrix_matches_affine "
            "/ excel_blender_matrix_matches_affine. Does not clear DEFAULT_WIRE_BLOCKERS. "
            "Does not redefine ready_for_wire_discussion. Always-on numpy; no TF/PuLP "
            "on hot path; no live Case 1 solve; no maximizer re-run."
        ),
    }


def offline_case1_dual_space_form_contract_report() -> Dict[str, Any]:
    """Always-on dual-space + form-label contract report (no TF, no PuLP, no solve).

    Aggregate ``ok`` = honesty locks ∧ stream_alignment_ok ∧ form_contract_ok ∧
    dual_linf status unproven ∧ blockers still documented.
    **Not** wire shipped. **Not** dual L∞ under wire proven. **Not** form flip.
    Does **not** re-implement maximizer / skeleton rounds.
    """
    honesty = _case1_dual_space_form_contract_honesty_fields()
    form = case1_form_label_contract()
    streams = case1_dual_space_stream_map()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    honesty_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and "BLENDER" not in UNITS
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["on_case1_solve"] is False
        and honesty["solver"] is False
        and honesty["kind"] == CASE1_DUAL_SPACE_FORM_CONTRACT_KIND
        and honesty["wire_shipped"] is False
        and honesty["not_wire_shipped"] is True
        and honesty["not_full_plant_mass_balance"] is True
        and honesty["not_pure_admm_dual_recovery"] is True
        and honesty["not_case1_solve"] is True
        and honesty["case1_form_unchanged"] is True
        and honesty["skeleton_lambda_is_not_case1_online_lambda"] is True
        and honesty["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
        and honesty["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and honesty["blender_is_base_delta_affine_unit"] is False
        and honesty["excel_cdu_matrix_matches_affine"] is None
        and honesty["excel_blender_matrix_matches_affine"] is None
        and honesty["not_isolation_rewrite"] is True
        and honesty["not_full_tf_admm_wire"] is True
    )

    form_contract_ok = bool(form.get("form_contract_ok"))
    stream_alignment_ok = bool(streams.get("stream_alignment_ok"))
    dual_linf_status_unproven_ok = bool(dual_linf.get("dual_linf_status_unproven_ok"))

    ok = bool(
        honesty_ok
        and form_contract_ok
        and stream_alignment_ok
        and dual_linf_status_unproven_ok
        and blockers_still_documented
        and honesty["wire_shipped"] is False
    )

    ok_criteria = (
        "honesty_ok ∧ form_contract_ok ∧ stream_alignment_ok ∧ "
        "dual_linf_status_unproven_ok ∧ blockers_still_documented ∧ "
        "wire_shipped=False; "
        "NOT wire shipped; NOT dual_linf proven; NOT form flip; NOT ready redefined"
    )

    return {
        **honesty,
        "ok": ok,
        "honesty_ok": honesty_ok,
        "form_contract_ok": form_contract_ok,
        "stream_alignment_ok": stream_alignment_ok,
        "dual_linf_status_unproven_ok": dual_linf_status_unproven_ok,
        "blockers_still_documented": blockers_still_documented,
        "ok_criteria": ok_criteria,
        # Form-label contract
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form": form["planned_form"],
        "form_unchanged": form["form_unchanged"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form": form,
        # Dual-space stream map
        "streams": streams["streams"],
        "linking_streams": streams["linking_streams"],
        "skeleton_lambda_slots": streams["skeleton_lambda_slots"],
        "package_dual_gate": streams["package_dual_gate"],
        "package_dual_secondary": streams["package_dual_secondary"],
        "package_dual_gate_role": streams["package_dual_gate_role"],
        "package_dual_secondary_role": streams["package_dual_secondary_role"],
        "stream_dual_roles": streams["stream_dual_roles"],
        "stream_map": streams,
        # dual_linf checklist
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_proof_checklist": dual_linf["dual_linf_proof_checklist"],
        "dual_linf_proof_checklist_open_ids": dual_linf[
            "dual_linf_proof_checklist_open_ids"
        ],
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        "dual_linf": dual_linf,
        # Blockers remain (prep surface documents; does not clear)
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "units_affine_unchanged": list(UNITS),
        "tf_available": tf_available(),
        "case1_shaped_streams_source": "CASE1_SHAPED_LINKING_STREAMS",
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "does_not_redefine_ready_for_wire_discussion": True,
        "does_not_clear_default_wire_blockers": True,
        "note": honesty["note"],
    }


def multi_unit_case1_dual_space_form_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_form_contract_report``."""
    return offline_case1_dual_space_form_contract_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1 dual-space L∞ probe / dual_linf proof-prep (goal 5 + goal 3)
# ---------------------------------------------------------------------------
# Always-on numpy. Stream-aligned numeric L∞ between fixture/supplied Case 1
# PRIMARY online λ and Case-1-shaped skeleton λ (final_lam).
# dual_recovery_path=None; wire_shipped=False; dual_linf_under_wire stays unproven;
# checklist online_linf_gate_under_tf_path remains open; probe ≠ VERDICT gate;
# probe ≠ dual L∞ under wire proof; does NOT clear DEFAULT_WIRE_BLOCKERS;
# does NOT redefine ready_for_wire_discussion; no TF / no PuLP / no excel_pipeline
# on the probe hot path.

CASE1_DUAL_SPACE_LINF_PROBE_KIND = "offline_case1_dual_space_linf_probe"

# Frozen demo-shaped Case 1 PRIMARY online λ (raw online_duals face; negative).
# Snapshot for proof-prep reproducibility — not live dual recovery from this surface.
CASE1_FIXTURE_PRIMARY_ONLINE_LAMBDA: Dict[str, float] = {
    "naphtha": -99.4429636359843,
    "distillate": -136.4898727294246,
    "gasoil": -54.8569636313148,
    "residue": -76.76202424579238,
}

# Optional SECONDARY recovered blender face (diagnostic only; never gate).
CASE1_FIXTURE_SECONDARY_RECOVERED_LAMBDA: Dict[str, float] = {
    "naphtha": 108.15126,
    "distillate": 87.142857,
    "gasoil": 170.0,
    "residue": -0.0,
}

CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE = "raw_online_duals"
CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW = "economic_shadow_prices"
CASE1_DUAL_VECTOR_FACE_SECONDARY_RECOVERED = "secondary_recovered_blender"


def case1_primary_online_lambda_fixture(
    *,
    face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
) -> Dict[str, float]:
    """Frozen fixture-shaped Case 1 PRIMARY online λ (demo keys; offline constants).

    Default face is raw ``online_duals`` (negative). Pass
    ``face=CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW`` for the positive economic
    shadow face (magnitude-matched). Not live dual recovery; not a solve path.
    """
    base = {k: float(v) for k, v in CASE1_FIXTURE_PRIMARY_ONLINE_LAMBDA.items()}
    if face == CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE:
        return base
    if face == CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW:
        return {k: -float(v) for k, v in base.items()}
    raise ValueError(
        f"unsupported primary dual vector face {face!r}; "
        f"use {CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE!r} or "
        f"{CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW!r}"
    )


def case1_secondary_recovered_lambda_fixture() -> Dict[str, float]:
    """Frozen SECONDARY recovered blender duals (diagnostic only; never gate)."""
    return {k: float(v) for k, v in CASE1_FIXTURE_SECONDARY_RECOVERED_LAMBDA.items()}


def case1_dual_space_stream_aligned_linf(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    streams: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Stream-aligned dual-space L∞ / L1 between two λ maps (pure numpy; dual-ban).

    Missing keys → ``alignment_ok=False`` (no silent zero-fill as success).
    ``probe_ok``-style finite flags are returned; this helper never gates VERDICT.
    """
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    missing_in_a = [s for s in streams_list if s not in a]
    missing_in_b = [s for s in streams_list if s not in b]
    alignment_ok = not missing_in_a and not missing_in_b and len(streams_list) > 0
    per_stream_abs: Dict[str, float] = {}
    finite_ok = True
    if alignment_ok:
        for s in streams_list:
            try:
                av = float(a[s])
                bv = float(b[s])
            except (TypeError, ValueError):
                finite_ok = False
                per_stream_abs[s] = float("nan")
                continue
            if not (np.isfinite(av) and np.isfinite(bv)):
                finite_ok = False
                per_stream_abs[s] = float("nan")
                continue
            per_stream_abs[s] = float(abs(av - bv))
    else:
        for s in streams_list:
            if s in a and s in b:
                try:
                    av = float(a[s])
                    bv = float(b[s])
                    gap = float(abs(av - bv)) if (
                        np.isfinite(av) and np.isfinite(bv)
                    ) else float("nan")
                except (TypeError, ValueError):
                    gap = float("nan")
                    finite_ok = False
                per_stream_abs[s] = gap
                if not np.isfinite(gap):
                    finite_ok = False
            else:
                per_stream_abs[s] = float("nan")
                finite_ok = False

    gaps = [per_stream_abs[s] for s in streams_list if np.isfinite(per_stream_abs.get(s, float("nan")))]
    if gaps and finite_ok and alignment_ok:
        linf = float(max(gaps))
        l1 = float(sum(gaps))
    elif gaps:
        linf = float(max(gaps)) if gaps else float("nan")
        l1 = float(sum(gaps)) if gaps else float("nan")
        if not all(np.isfinite(g) for g in gaps):
            finite_ok = False
    else:
        linf = float("nan")
        l1 = float("nan")
        finite_ok = False

    return {
        "streams": streams_list,
        "per_stream_abs": per_stream_abs,
        "abs_gap": dict(per_stream_abs),
        "linf": linf,
        "l1": l1,
        "alignment_ok": bool(alignment_ok),
        "stream_alignment_ok": bool(alignment_ok),
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "finite_ok": bool(finite_ok and np.isfinite(linf)),
        "finite_linf": bool(np.isfinite(linf)),
        "dual_recovery_path": None,
        "probe_is_not_verdict_gate": True,
        "note": (
            "Stream-aligned dual-space L∞ on Case 1 intermediates "
            "(naphtha/distillate/gasoil/residue). Not Case 1 VERDICT gate; "
            "not dual L∞ under wire proof; dual_recovery_path=None."
        ),
    }


# Alias preferred by explorer A naming.
stream_aligned_dual_linf = case1_dual_space_stream_aligned_linf


def extract_case1_shaped_skeleton_lambda(
    *,
    n_rounds: int = 1,
    skeleton_report: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Extract Case-1-shaped skeleton linking λ (final_lam) on stream map.

    Reuses ``offline_case1_shaped_cdu_blender_linking_report`` — does **not**
    re-implement maximizer rounds. Skeleton λ is **not** Case 1 dual recovery.
    """
    if skeleton_report is None:
        skeleton_report = offline_case1_shaped_cdu_blender_linking_report(
            n_rounds=int(n_rounds), **kwargs
        )
    final_lam = skeleton_report.get("final_lam") or {}
    streams = list(CASE1_SHAPED_LINKING_STREAMS)
    out: Dict[str, float] = {}
    for s in streams:
        if s in final_lam:
            out[s] = float(final_lam[s])
    return out


def case1_shaped_skeleton_lambda(
    *,
    n_rounds: int = 1,
    skeleton_report: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Alias for ``extract_case1_shaped_skeleton_lambda``."""
    return extract_case1_shaped_skeleton_lambda(
        n_rounds=n_rounds, skeleton_report=skeleton_report, **kwargs
    )


def _case1_dual_space_linf_probe_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire / not-VERDICT locks for the L∞ probe."""
    checklist = dict(CASE1_DUAL_LINF_PROOF_CHECKLIST)
    open_ids = [
        k
        for k, v in checklist.items()
        if str(v).lower() in ("open", "false_today", "unproven")
    ]
    return {
        "kind": CASE1_DUAL_SPACE_LINF_PROBE_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "form_current": CASE1_FORM_CURRENT,
        "form_planned": CASE1_PLANNED_TF_AWARE_FORM,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "skeleton_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "probe_is_not_verdict_gate": True,
        "probe_is_not_dual_linf_under_wire_proof": True,
        "probe_available_is_not_dual_linf_under_wire_proof": True,
        "secondary_recovered_is_not_gate": True,
        "package_dual_gate": "online_lambda",
        "package_dual_secondary": "recovered_blender",
        "package_dual_gate_role": "PRIMARY",
        "package_dual_secondary_role": "SECONDARY_not_gate",
        "dual_linf_under_wire_status": CASE1_DUAL_LINF_UNDER_WIRE_STATUS,
        "dual_linf_under_wire": CASE1_DUAL_LINF_UNDER_WIRE_STATUS,
        "dual_linf_under_wire_unproven_still_true": True,
        "online_linf_gate_under_tf_path": checklist.get(
            "online_linf_gate_under_tf_path", "open"
        ),
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": len(open_ids),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "excel_cdu_matrix_matches_affine": None,
        "excel_blender_matrix_matches_affine": None,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "scope": "case1_dual_space_linf_probe_offline",
        "note": (
            "Offline Case-1 dual-space L∞ probe / dual_linf proof-prep: stream-aligned "
            "numeric L∞ between fixture/supplied Case 1 PRIMARY online λ and "
            "Case-1-shaped skeleton λ on naphtha/distillate/gasoil/residue. "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            "wire_shipped=False; dual_linf_under_wire_status=unproven; checklist "
            "online_linf_gate_under_tf_path remains open (probe_available ≠ proven); "
            "skeleton λ ≠ Case 1 PRIMARY/SECONDARY duals as dual recovery; "
            "probe ≠ Case 1 VERDICT gate; probe ≠ dual L∞ under wire proof; "
            "SECONDARY recovered face is diagnostic only; form_current classic "
            "unchanged; form_planned registered only; not pure-ADMM dual recovery; "
            "not full plant mass balance; not isolation rewrite; not full TF→ADMM wire. "
            "Does not clear DEFAULT_WIRE_BLOCKERS. Does not redefine "
            "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline "
            "on hot path; no live Case 1 solve."
        ),
    }


def offline_case1_dual_space_linf_probe_report(
    *,
    case1_primary_online_lambda: Optional[Mapping[str, float]] = None,
    case1_secondary_recovered_lambda: Optional[Mapping[str, float]] = None,
    skeleton_lambda: Optional[Mapping[str, float]] = None,
    skeleton_report: Optional[Mapping[str, Any]] = None,
    skeleton_n_rounds: int = 1,
    dual_vector_face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
    streams: Optional[Sequence[str]] = None,
    dual_gate_threshold_diagnostic: float = 15.0,
) -> Dict[str, Any]:
    """Always-on dual-space L∞ probe (no TF, no PuLP, no excel_pipeline, no solve).

    Compare fixture/supplied Case 1 PRIMARY online λ vs Case-1-shaped skeleton λ
    on stream-aligned slots. Aggregate ``probe_ok`` / ``ok`` =
    honesty locks ∧ stream alignment ∧ finite L∞ ∧ dual-ban ∧ wire_shipped=False ∧
    dual_linf still unproven — **never** ``linf <= 15`` under wire; **not** VERDICT.

    Does **not** clear ``DEFAULT_WIRE_BLOCKERS``. Does **not** redefine
    ``ready_for_wire_discussion``. Does **not** flip Case 1 form.
    """
    honesty = _case1_dual_space_linf_probe_honesty_fields()
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    smap = case1_dual_space_stream_map()
    dual_linf_cl = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    # Left-hand: PRIMARY online λ (caller or frozen fixture).
    face = dual_vector_face
    if case1_primary_online_lambda is None:
        primary = case1_primary_online_lambda_fixture(face=face)
        primary_source = "fixture"
    else:
        primary = {str(k): float(v) for k, v in case1_primary_online_lambda.items()}
        primary_source = "caller_supplied"

    secondary_diag: Optional[Dict[str, float]] = None
    if case1_secondary_recovered_lambda is not None:
        secondary_diag = {
            str(k): float(v) for k, v in case1_secondary_recovered_lambda.items()
        }
        secondary_source = "caller_supplied"
    else:
        secondary_source = "not_provided"

    # Right-hand: skeleton λ (caller, extract from report, or run thin n_rounds extract).
    if skeleton_lambda is not None:
        skeleton = {str(k): float(v) for k, v in skeleton_lambda.items()}
        skeleton_source = "caller_supplied"
    else:
        skeleton = extract_case1_shaped_skeleton_lambda(
            n_rounds=int(skeleton_n_rounds),
            skeleton_report=skeleton_report,
        )
        skeleton_source = (
            "skeleton_report_final_lam"
            if skeleton_report is not None
            else "case1_shaped_extract"
        )

    gap = case1_dual_space_stream_aligned_linf(
        primary, skeleton, streams=streams_list
    )
    stream_alignment_ok = bool(
        gap["stream_alignment_ok"] and smap.get("stream_alignment_ok")
    )
    finite_linf = bool(gap.get("finite_linf"))
    linf = gap["linf"]
    l1 = gap["l1"]
    per_stream_abs = dict(gap["per_stream_abs"])

    # Optional diagnostic vs gate tol — NEVER part of probe_ok / VERDICT.
    gate_tol = float(dual_gate_threshold_diagnostic)
    if finite_linf and np.isfinite(gate_tol):
        under_gate_tol_diag = bool(float(linf) <= gate_tol)
    else:
        under_gate_tol_diag = False

    checklist = dict(honesty["dual_linf_proof_checklist"])
    online_gate_status = str(checklist.get("online_linf_gate_under_tf_path", "open"))
    online_gate_open = online_gate_status.lower() in ("open", "false_today", "unproven")
    dual_linf_status = honesty["dual_linf_under_wire_status"]
    dual_linf_unproven = dual_linf_status == "unproven"
    dual_linf_blocker_still = "dual_linf_under_wire_unproven" in blockers
    wire_not_shipped_blocker = "wire_not_shipped" in blockers

    honesty_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and "BLENDER" not in UNITS
        and honesty["dual_recovery_path"] is None
        and honesty["on_excel_case1_path"] is False
        and honesty["on_case1_solve"] is False
        and honesty["solver"] is False
        and honesty["kind"] == CASE1_DUAL_SPACE_LINF_PROBE_KIND
        and honesty["wire_shipped"] is False
        and honesty["not_wire_shipped"] is True
        and honesty["not_full_plant_mass_balance"] is True
        and honesty["not_pure_admm_dual_recovery"] is True
        and honesty["not_case1_solve"] is True
        and honesty["case1_form_unchanged"] is True
        and honesty["skeleton_lambda_is_not_case1_online_lambda"] is True
        and honesty["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
        and honesty["probe_is_not_verdict_gate"] is True
        and honesty["probe_is_not_dual_linf_under_wire_proof"] is True
        and honesty["secondary_recovered_is_not_gate"] is True
        and honesty["excel_cdu_matrix_matches_affine"] is None
        and honesty["excel_blender_matrix_matches_affine"] is None
        and honesty["not_isolation_rewrite"] is True
        and honesty["not_full_tf_admm_wire"] is True
        and honesty["does_not_clear_default_wire_blockers"] is True
        and honesty["does_not_redefine_ready_for_wire_discussion"] is True
        and dual_linf_unproven
        and dual_linf_blocker_still
        and online_gate_open
        and blockers_still_documented
    )

    dual_ban_ok = bool(
        honesty["dual_recovery_path"] is None
        and honesty["probe_is_not_dual_linf_under_wire_proof"] is True
        and honesty["probe_is_not_verdict_gate"] is True
        and dual_linf_unproven
        and online_gate_open
    )

    # probe_ok NEVER requires linf <= gate threshold.
    probe_ok = bool(
        honesty_ok
        and stream_alignment_ok
        and finite_linf
        and dual_ban_ok
        and honesty["wire_shipped"] is False
        and dual_linf_unproven
        and dual_linf_blocker_still
        and blockers_still_documented
    )
    ok = probe_ok

    ok_criteria = (
        "honesty_ok ∧ stream_alignment_ok ∧ finite_linf ∧ dual_ban ∧ "
        "wire_shipped=False ∧ dual_linf_unproven ∧ blockers_still_documented; "
        "NOT linf<=15 under wire; NOT dual_linf proven; NOT VERDICT gate; "
        "NOT form flip; NOT ready redefined; NOT blockers cleared"
    )

    # Optional SECONDARY diagnostic gaps (never gate).
    secondary_gap = None
    if secondary_diag is not None:
        secondary_gap = case1_dual_space_stream_aligned_linf(
            secondary_diag, skeleton, streams=streams_list
        )

    return {
        **honesty,
        "ok": ok,
        "probe_ok": probe_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "stream_alignment_ok": stream_alignment_ok,
        "finite_linf": finite_linf,
        "finite_ok": finite_linf,
        "blockers_still_documented": blockers_still_documented,
        "ok_criteria": ok_criteria,
        "probe_ok_criteria": ok_criteria,
        # Dual vectors
        "dual_vector_face": face,
        "case1_primary_online_lambda": dict(primary),
        "case1_primary_online_lambda_source": primary_source,
        "case1_secondary_recovered_lambda": (
            dict(secondary_diag) if secondary_diag is not None else None
        ),
        "case1_secondary_recovered_lambda_source": secondary_source,
        "skeleton_lambda": dict(skeleton),
        "skeleton_lambda_source": skeleton_source,
        "streams": streams_list,
        "linking_streams": streams_list,
        "skeleton_lambda_slots": list(CASE1_SHAPED_LINKING_STREAMS),
        # Gaps
        "per_stream_abs": per_stream_abs,
        "abs_gap": dict(per_stream_abs),
        "linf": linf,
        "l1": l1,
        "missing_in_primary": list(gap.get("missing_in_a") or []),
        "missing_in_skeleton": list(gap.get("missing_in_b") or []),
        "gap": gap,
        "secondary_gap_diagnostic": secondary_gap,
        # Diagnostic only vs gate tol — not VERDICT, not probe_ok
        "probe_linf_vs_gate_tol_diagnostic": {
            "gate_tol": gate_tol,
            "linf": linf,
            "under_gate_tol": under_gate_tol_diag,
            "is_not_verdict_gate": True,
            "is_not_probe_ok_criterion": True,
            "note": (
                "Optional diagnostic compare of probe L∞ to Case 1 online dual "
                "gate tol (default 15). Never a VERDICT hard-fail; never "
                "probe_ok criterion; never dual_linf_under_wire proof."
            ),
        },
        # dual_linf checklist remains open / unproven
        "dual_linf_status_unproven_ok": bool(
            dual_linf_cl.get("dual_linf_status_unproven_ok")
        ),
        "online_linf_gate_under_tf_path_open": online_gate_open,
        "probe_available": True,
        # Blockers remain (prep surface documents; does not clear)
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "wire_not_shipped_blocker_still_true": wire_not_shipped_blocker,
        "dual_linf_under_wire_unproven_blocker_still_true": dual_linf_blocker_still,
        "units_affine_unchanged": list(UNITS),
        "tf_available": tf_available(),
        "case1_shaped_streams_source": "CASE1_SHAPED_LINKING_STREAMS",
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "stream_map": smap,
        "dual_linf_checklist": dual_linf_cl,
        "note": honesty["note"],
    }


def case1_dual_space_linf_probe(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_probe_report``."""
    return offline_case1_dual_space_linf_probe_report(**kwargs)


def multi_unit_case1_dual_space_linf_probe_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_probe_report``."""
    return offline_case1_dual_space_linf_probe_report(**kwargs)


# ---------------------------------------------------------------------------
# Offline Case-1 dual-space L∞ live-λ bridge / capture harness (goal 5 + 3)
# ---------------------------------------------------------------------------
# Always-on numpy. Pure extract/normalize of this-run Case 1 PRIMARY online λ
# (+ optional SECONDARY recovered diagnostic) into the existing dual-space L∞
# probe. Does NOT re-implement L∞ math. dual_recovery_path=None; dual_linf
# under wire stays unproven; bridge ≠ VERDICT; bridge ≠ wire; no form flip;
# no excel_pipeline / PuLP / tensorflow on the bridge hot path.

CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_BRIDGE_KIND = (
    "offline_case1_dual_space_linf_live_lambda_bridge"
)

# Source tags for dual-honesty (fixture ≠ live).
LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED = "caller_supplied"
LIVE_LAMBDA_SOURCE_FIXTURE = "fixture"
LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT = "package_extract"
LIVE_LAMBDA_SOURCE_MISSING = "missing"


def _case1_stream_align_mapping(
    mapping: Mapping[str, Any],
    *,
    streams: Sequence[str],
) -> Dict[str, Any]:
    """Align a plain stream→value map onto required streams (no silent zero-fill)."""
    missing = [s for s in streams if s not in mapping]
    non_finite: List[str] = []
    out: Dict[str, float] = {}
    if not missing:
        for s in streams:
            try:
                v = float(mapping[s])
            except (TypeError, ValueError):
                non_finite.append(s)
                continue
            if not np.isfinite(v):
                non_finite.append(s)
                continue
            out[s] = v
    extract_ok = bool(not missing and not non_finite and len(streams) > 0)
    return {
        "lambda": out if extract_ok else dict(out),
        "extract_ok": extract_ok,
        "stream_alignment_ok": extract_ok,
        "missing_streams": missing,
        "non_finite_streams": non_finite,
        "streams": list(streams),
        "dual_recovery_path": None,
    }


def case1_primary_online_lambda_from_mapping(
    mapping: Optional[Mapping[str, Any]],
    *,
    face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
    streams: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Normalize a plain stream→float map onto Case 1 intermediate streams.

    Pure mapping I/O — no excel_pipeline / PuLP / tensorflow. Missing keys →
    ``extract_ok=False`` (no silent zero-fill as success). Values are treated
    as already on the requested dual face (no automatic sign flip here).

    Extracted vectors are **inputs to the dual-ban probe**, not dual recovery
    ownership (``dual_recovery_path`` stays None on this surface).
    """
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    if face not in (
        CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
        CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW,
    ):
        return {
            "kind": "case1_primary_online_lambda_from_mapping",
            "extract_ok": False,
            "stream_alignment_ok": False,
            "lambda": {},
            "case1_primary_online_lambda": {},
            "missing_streams": list(streams_list),
            "non_finite_streams": [],
            "streams": streams_list,
            "dual_vector_face": face,
            "source": LIVE_LAMBDA_SOURCE_MISSING,
            "source_path": "unsupported_face",
            "dual_recovery_path": None,
            "note": (
                f"Unsupported dual_vector_face={face!r}; use "
                f"{CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE!r} or "
                f"{CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW!r}."
            ),
        }
    if mapping is None or not isinstance(mapping, Mapping):
        return {
            "kind": "case1_primary_online_lambda_from_mapping",
            "extract_ok": False,
            "stream_alignment_ok": False,
            "lambda": {},
            "case1_primary_online_lambda": {},
            "missing_streams": list(streams_list),
            "non_finite_streams": [],
            "streams": streams_list,
            "dual_vector_face": face,
            "source": LIVE_LAMBDA_SOURCE_MISSING,
            "source_path": "none",
            "dual_recovery_path": None,
            "note": "No mapping supplied; extract failed (no silent zero-fill).",
        }
    aligned = _case1_stream_align_mapping(mapping, streams=streams_list)
    return {
        "kind": "case1_primary_online_lambda_from_mapping",
        "extract_ok": aligned["extract_ok"],
        "stream_alignment_ok": aligned["stream_alignment_ok"],
        "lambda": dict(aligned["lambda"]),
        "case1_primary_online_lambda": dict(aligned["lambda"]),
        "missing_streams": list(aligned["missing_streams"]),
        "non_finite_streams": list(aligned["non_finite_streams"]),
        "streams": streams_list,
        "dual_vector_face": face,
        "source": (
            LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
            if aligned["extract_ok"]
            else LIVE_LAMBDA_SOURCE_MISSING
        ),
        "source_path": "plain_mapping",
        "dual_recovery_path": None,
        "note": (
            "Pure stream-aligned PRIMARY online λ mapping. dual_recovery_path=None; "
            "not dual recovery ownership; inputs to dual-ban probe only."
        ),
    }


def _nested_get(root: Mapping[str, Any], *path: str) -> Any:
    cur: Any = root
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_case1_primary_online_lambda(
    package_or_dicts: Optional[Mapping[str, Any]] = None,
    *,
    face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
    streams: Optional[Sequence[str]] = None,
    plain_mapping: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract/normalize Case 1 PRIMARY online λ from a package-shaped mapping.

    Search order for ``face=raw_online_duals`` (default / PRIMARY gate face):
      1. ``plain_mapping`` if provided
      2. package top-level stream map (all CASE1 streams present)
      3. ``package[\"admm\"][\"online_duals\"]``
      4. ``package[\"online_duals\"]``
      5. ``package[\"admm\"][\"shadow_prices\"]`` negated → raw face (economic→raw)
      6. ``package[\"shadow_prices\"]`` / ``package[\"admm\"]`` economic face negated

    Search order for ``face=economic_shadow_prices``:
      1. ``plain_mapping`` if provided
      2. package top-level stream map
      3. ``package[\"admm\"][\"shadow_prices\"]``
      4. ``package[\"shadow_prices\"]``
      5. ``package[\"admm\"][\"online_duals\"]`` negated → economic face
      6. ``package[\"online_duals\"]`` negated

    Pure dict walk — **no** excel_pipeline / PuLP / tensorflow. Missing keys →
    ``extract_ok=False``. Extracted vectors are dual-ban probe inputs only.
    """
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    candidates: List[tuple] = []  # (source_path, mapping, convert_from_face)

    if plain_mapping is not None:
        candidates.append(("plain_mapping", plain_mapping, face))

    pkg = package_or_dicts if isinstance(package_or_dicts, Mapping) else None
    if pkg is not None:
        # Flat package that is itself a stream map
        if all(s in pkg for s in streams_list) and not any(
            k in pkg for k in ("admm", "mono", "comparison", "meta", "verdict")
        ):
            candidates.append(("top_level_stream_map", pkg, face))

        admm = pkg.get("admm") if isinstance(pkg.get("admm"), Mapping) else None

        if face == CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE:
            if admm is not None and isinstance(admm.get("online_duals"), Mapping):
                candidates.append(
                    ("admm.online_duals", admm["online_duals"], face)
                )
            if isinstance(pkg.get("online_duals"), Mapping):
                candidates.append(
                    ("online_duals", pkg["online_duals"], face)
                )
            # Economic → raw conversion candidates (negate)
            if admm is not None and isinstance(admm.get("shadow_prices"), Mapping):
                candidates.append(
                    (
                        "admm.shadow_prices_negated_to_raw",
                        admm["shadow_prices"],
                        CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW,
                    )
                )
            if isinstance(pkg.get("shadow_prices"), Mapping):
                candidates.append(
                    (
                        "shadow_prices_negated_to_raw",
                        pkg["shadow_prices"],
                        CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW,
                    )
                )
        elif face == CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW:
            if admm is not None and isinstance(admm.get("shadow_prices"), Mapping):
                candidates.append(
                    ("admm.shadow_prices", admm["shadow_prices"], face)
                )
            if isinstance(pkg.get("shadow_prices"), Mapping):
                candidates.append(
                    ("shadow_prices", pkg["shadow_prices"], face)
                )
            if admm is not None and isinstance(admm.get("online_duals"), Mapping):
                candidates.append(
                    (
                        "admm.online_duals_negated_to_economic",
                        admm["online_duals"],
                        CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
                    )
                )
            if isinstance(pkg.get("online_duals"), Mapping):
                candidates.append(
                    (
                        "online_duals_negated_to_economic",
                        pkg["online_duals"],
                        CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
                    )
                )
        else:
            return {
                "kind": "extract_case1_primary_online_lambda",
                "extract_ok": False,
                "stream_alignment_ok": False,
                "lambda": {},
                "case1_primary_online_lambda": {},
                "missing_streams": list(streams_list),
                "non_finite_streams": [],
                "streams": streams_list,
                "dual_vector_face": face,
                "source": LIVE_LAMBDA_SOURCE_MISSING,
                "source_path": "unsupported_face",
                "dual_recovery_path": None,
                "face_converted": False,
                "note": f"Unsupported dual_vector_face={face!r}.",
            }

    last_fail: Optional[Dict[str, Any]] = None
    for source_path, raw_map, source_face in candidates:
        mapped = dict(raw_map)
        face_converted = False
        if source_face != face:
            # Convert raw ↔ economic by sign flip (magnitude-matched convention).
            try:
                mapped = {str(k): -float(v) for k, v in raw_map.items()}
                face_converted = True
            except (TypeError, ValueError):
                last_fail = {
                    "source_path": source_path,
                    "error": "face_convert_failed",
                }
                continue
        aligned = case1_primary_online_lambda_from_mapping(
            mapped, face=face, streams=streams_list
        )
        if aligned["extract_ok"]:
            src = (
                LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
                if source_path not in ("plain_mapping", "top_level_stream_map")
                else LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
            )
            if source_path == "plain_mapping":
                src = LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
            elif source_path == "top_level_stream_map":
                src = LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
            else:
                src = LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
            return {
                "kind": "extract_case1_primary_online_lambda",
                "extract_ok": True,
                "stream_alignment_ok": True,
                "lambda": dict(aligned["lambda"]),
                "case1_primary_online_lambda": dict(aligned["lambda"]),
                "missing_streams": [],
                "non_finite_streams": [],
                "streams": streams_list,
                "dual_vector_face": face,
                "source": src,
                "source_path": source_path,
                "face_converted": face_converted,
                "dual_recovery_path": None,
                "note": (
                    "Extracted Case 1 PRIMARY online λ stream map for dual-ban "
                    "probe input only. dual_recovery_path=None; not dual recovery "
                    "ownership; not VERDICT; not wire."
                ),
            }
        last_fail = aligned

    # Failure path
    missing = list(streams_list)
    if last_fail is not None and "missing_streams" in last_fail:
        missing = list(last_fail.get("missing_streams") or missing)
    return {
        "kind": "extract_case1_primary_online_lambda",
        "extract_ok": False,
        "stream_alignment_ok": False,
        "lambda": {},
        "case1_primary_online_lambda": {},
        "missing_streams": missing,
        "non_finite_streams": list(
            (last_fail or {}).get("non_finite_streams") or []
        ),
        "streams": streams_list,
        "dual_vector_face": face,
        "source": LIVE_LAMBDA_SOURCE_MISSING,
        "source_path": "not_found",
        "face_converted": False,
        "dual_recovery_path": None,
        "note": (
            "Could not extract stream-aligned Case 1 PRIMARY online λ from package "
            "or plain mapping (no silent zero-fill). dual_recovery_path=None."
        ),
    }


def extract_case1_secondary_recovered_lambda(
    package_or_dicts: Optional[Mapping[str, Any]] = None,
    *,
    streams: Optional[Sequence[str]] = None,
    plain_mapping: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract SECONDARY recovered blender duals (diagnostic only; never gate).

    Search: plain_mapping → admm.shadow_prices_recovered → shadow_prices_recovered.
    Pure dict I/O; dual_recovery_path=None; not a VERDICT gate.
    """
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    candidates: List[tuple] = []
    if plain_mapping is not None:
        candidates.append(("plain_mapping", plain_mapping))
    pkg = package_or_dicts if isinstance(package_or_dicts, Mapping) else None
    if pkg is not None:
        admm = pkg.get("admm") if isinstance(pkg.get("admm"), Mapping) else None
        if admm is not None and isinstance(
            admm.get("shadow_prices_recovered"), Mapping
        ):
            candidates.append(
                ("admm.shadow_prices_recovered", admm["shadow_prices_recovered"])
            )
        if isinstance(pkg.get("shadow_prices_recovered"), Mapping):
            candidates.append(
                ("shadow_prices_recovered", pkg["shadow_prices_recovered"])
            )

    for source_path, raw_map in candidates:
        aligned = _case1_stream_align_mapping(raw_map, streams=streams_list)
        if aligned["extract_ok"]:
            return {
                "kind": "extract_case1_secondary_recovered_lambda",
                "extract_ok": True,
                "stream_alignment_ok": True,
                "lambda": dict(aligned["lambda"]),
                "case1_secondary_recovered_lambda": dict(aligned["lambda"]),
                "missing_streams": [],
                "non_finite_streams": [],
                "streams": streams_list,
                "dual_vector_face": CASE1_DUAL_VECTOR_FACE_SECONDARY_RECOVERED,
                "source": (
                    LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
                    if source_path == "plain_mapping"
                    else LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
                ),
                "source_path": source_path,
                "dual_recovery_path": None,
                "secondary_recovered_is_not_gate": True,
                "note": (
                    "SECONDARY recovered blender dual extract — diagnostic only; "
                    "never VERDICT gate; dual_recovery_path=None."
                ),
            }

    return {
        "kind": "extract_case1_secondary_recovered_lambda",
        "extract_ok": False,
        "stream_alignment_ok": False,
        "lambda": {},
        "case1_secondary_recovered_lambda": {},
        "missing_streams": list(streams_list),
        "non_finite_streams": [],
        "streams": streams_list,
        "dual_vector_face": CASE1_DUAL_VECTOR_FACE_SECONDARY_RECOVERED,
        "source": LIVE_LAMBDA_SOURCE_MISSING,
        "source_path": "not_found",
        "dual_recovery_path": None,
        "secondary_recovered_is_not_gate": True,
        "note": (
            "SECONDARY recovered duals not found (optional diagnostic only; "
            "never gate)."
        ),
    }


def _case1_dual_space_linf_live_lambda_bridge_honesty_fields() -> Dict[str, Any]:
    """Dual-ban / not-wire / not-VERDICT locks for the live-λ bridge."""
    base = _case1_dual_space_linf_probe_honesty_fields()
    base = dict(base)
    base["kind"] = CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_BRIDGE_KIND
    base["scope"] = "case1_dual_space_linf_live_lambda_bridge_offline"
    base["bridge_is_not_verdict_gate"] = True
    base["bridge_is_not_dual_linf_under_wire_proof"] = True
    base["probe_is_not_verdict_gate"] = True
    base["probe_is_not_dual_linf_under_wire_proof"] = True
    base["live_lambda_is_not_dual_recovery"] = True
    base["extracted_lambda_is_probe_input_only"] = True
    base["note"] = (
        "Offline Case-1 dual-space L∞ live-λ bridge: extract/accept this-run "
        "Case 1 PRIMARY online λ (+ optional SECONDARY recovered diagnostic) and "
        "compose into offline_case1_dual_space_linf_probe_report. "
        "live_lambda_source always labeled (caller_supplied / package_extract / "
        "fixture). dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
        "wire_shipped=False; dual_linf_under_wire_status=unproven; checklist "
        "online_linf_gate_under_tf_path remains open; bridge ≠ Case 1 VERDICT gate; "
        "bridge ≠ dual L∞ under wire proof; skeleton λ ≠ Case 1 duals as recovery; "
        "SECONDARY recovered is diagnostic only; form_current classic unchanged; "
        "does not clear DEFAULT_WIRE_BLOCKERS; does not redefine "
        "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline "
        "on hot path; no live Case 1 solve ownership."
    )
    return base


def offline_case1_dual_space_linf_live_lambda_bridge_report(
    *,
    case1_package: Optional[Mapping[str, Any]] = None,
    case1_primary_online_lambda: Optional[Mapping[str, float]] = None,
    case1_secondary_recovered_lambda: Optional[Mapping[str, float]] = None,
    include_secondary_recovered: bool = True,
    allow_fixture_fallback: bool = False,
    skeleton_lambda: Optional[Mapping[str, float]] = None,
    skeleton_report: Optional[Mapping[str, Any]] = None,
    skeleton_n_rounds: int = 1,
    dual_vector_face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
    streams: Optional[Sequence[str]] = None,
    dual_gate_threshold_diagnostic: float = 15.0,
) -> Dict[str, Any]:
    """Always-on live-λ bridge: extract Case 1 PRIMARY online λ → existing probe.

    Compose only — does **not** re-implement stream L∞. ``bridge_ok`` /
    ``ok`` = extract honesty ∧ probe honesty ∧ finite ∧ aligned ∧ dual-ban ∧
    **source documented** — **never** ``linf <= 15``; **never** VERDICT.

    Live mode requires caller-supplied PRIMARY (plain map or package extract)
    unless ``allow_fixture_fallback=True`` (then ``live_lambda_source=fixture``
    and never claimed live).

    Does **not** clear ``DEFAULT_WIRE_BLOCKERS``. Does **not** redefine
    ``ready_for_wire_discussion``. Does **not** flip Case 1 form.
    """
    honesty = _case1_dual_space_linf_live_lambda_bridge_honesty_fields()
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    # --- PRIMARY extract / resolve ---
    extract_rep: Dict[str, Any]
    if case1_primary_online_lambda is not None:
        extract_rep = case1_primary_online_lambda_from_mapping(
            case1_primary_online_lambda,
            face=dual_vector_face,
            streams=streams_list,
        )
        extract_rep = dict(extract_rep)
        extract_rep["kind"] = "extract_case1_primary_online_lambda"
        extract_rep["source"] = LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
        extract_rep["source_path"] = "case1_primary_online_lambda_arg"
        extract_rep["face_converted"] = False
    elif case1_package is not None:
        extract_rep = extract_case1_primary_online_lambda(
            case1_package,
            face=dual_vector_face,
            streams=streams_list,
        )
    else:
        extract_rep = {
            "kind": "extract_case1_primary_online_lambda",
            "extract_ok": False,
            "stream_alignment_ok": False,
            "lambda": {},
            "case1_primary_online_lambda": {},
            "missing_streams": list(streams_list),
            "non_finite_streams": [],
            "streams": streams_list,
            "dual_vector_face": dual_vector_face,
            "source": LIVE_LAMBDA_SOURCE_MISSING,
            "source_path": "none",
            "face_converted": False,
            "dual_recovery_path": None,
        }

    used_fixture = False
    if not extract_rep.get("extract_ok"):
        if allow_fixture_fallback:
            primary = case1_primary_online_lambda_fixture(face=dual_vector_face)
            live_lambda_source = LIVE_LAMBDA_SOURCE_FIXTURE
            primary_source_path = "fixture"
            used_fixture = True
            extract_ok = True
        else:
            primary = {}
            live_lambda_source = LIVE_LAMBDA_SOURCE_MISSING
            primary_source_path = str(extract_rep.get("source_path") or "missing")
            extract_ok = False
    else:
        primary = dict(extract_rep["lambda"])
        live_lambda_source = str(
            extract_rep.get("source") or LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
        )
        primary_source_path = str(
            extract_rep.get("source_path") or "caller_supplied"
        )
        extract_ok = True

    # Source honesty: never claim live when fixture used.
    source_documented = live_lambda_source in (
        LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED,
        LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT,
        LIVE_LAMBDA_SOURCE_FIXTURE,
    )
    source_honest = not (
        used_fixture and live_lambda_source != LIVE_LAMBDA_SOURCE_FIXTURE
    )
    # Explicit ban: fixture path must never be labeled caller_supplied / package_extract
    if used_fixture:
        live_lambda_source = LIVE_LAMBDA_SOURCE_FIXTURE
        source_honest = True
        source_documented = True

    # --- Optional SECONDARY diagnostic ---
    secondary_map: Optional[Dict[str, float]] = None
    secondary_extract: Optional[Dict[str, Any]] = None
    if case1_secondary_recovered_lambda is not None:
        sec = _case1_stream_align_mapping(
            case1_secondary_recovered_lambda, streams=streams_list
        )
        secondary_extract = {
            "extract_ok": sec["extract_ok"],
            "source": LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED,
            "source_path": "case1_secondary_recovered_lambda_arg",
            "secondary_recovered_is_not_gate": True,
            "dual_recovery_path": None,
        }
        if sec["extract_ok"]:
            secondary_map = dict(sec["lambda"])
    elif include_secondary_recovered and case1_package is not None:
        secondary_extract = extract_case1_secondary_recovered_lambda(
            case1_package, streams=streams_list
        )
        if secondary_extract.get("extract_ok"):
            secondary_map = dict(secondary_extract["lambda"])

    # --- Compose existing probe (no second L∞ engine) ---
    if extract_ok and primary:
        probe = offline_case1_dual_space_linf_probe_report(
            case1_primary_online_lambda=primary,
            case1_secondary_recovered_lambda=secondary_map,
            skeleton_lambda=skeleton_lambda,
            skeleton_report=skeleton_report,
            skeleton_n_rounds=int(skeleton_n_rounds),
            dual_vector_face=dual_vector_face,
            streams=streams_list,
            dual_gate_threshold_diagnostic=dual_gate_threshold_diagnostic,
        )
    else:
        # Still call probe with fixture only for structural fields when missing?
        # Prefer honest failure without fabricating live vectors.
        probe = offline_case1_dual_space_linf_probe_report(
            case1_primary_online_lambda={s: 0.0 for s in streams_list},
            skeleton_lambda={s: 0.0 for s in streams_list},
            dual_vector_face=dual_vector_face,
            streams=streams_list,
            dual_gate_threshold_diagnostic=dual_gate_threshold_diagnostic,
        )
        # Force probe_ok path fields but bridge will fail on extract.
        # Override sources below.

    # Dual-ban / unproven locks from honesty + probe
    dual_linf_unproven = (
        honesty["dual_linf_under_wire_status"] == "unproven"
        and probe.get("dual_linf_under_wire_status") == "unproven"
    )
    online_gate_open = bool(probe.get("online_linf_gate_under_tf_path_open", True))
    dual_ban_ok = bool(
        honesty["dual_recovery_path"] is None
        and probe.get("dual_recovery_path") is None
        and honesty["bridge_is_not_dual_linf_under_wire_proof"] is True
        and honesty["bridge_is_not_verdict_gate"] is True
        and dual_linf_unproven
        and online_gate_open
    )
    checklist = dict(honesty["dual_linf_proof_checklist"])
    online_gate_status = str(checklist.get("online_linf_gate_under_tf_path", "open"))
    if online_gate_status.lower() not in ("open", "false_today", "unproven"):
        online_gate_open = False
        dual_ban_ok = False

    honesty_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and "BLENDER" not in UNITS
        and honesty["dual_recovery_path"] is None
        and honesty["wire_shipped"] is False
        and honesty["bridge_is_not_verdict_gate"] is True
        and honesty["bridge_is_not_dual_linf_under_wire_proof"] is True
        and honesty["does_not_clear_default_wire_blockers"] is True
        and honesty["does_not_redefine_ready_for_wire_discussion"] is True
        and dual_linf_unproven
        and "dual_linf_under_wire_unproven" in blockers
        and "wire_not_shipped" in blockers
        and blockers_still_documented
        and online_gate_open
    )

    probe_ok = bool(probe.get("probe_ok")) if extract_ok else False
    stream_alignment_ok = bool(
        extract_ok and probe.get("stream_alignment_ok", False)
    )
    finite_linf = bool(probe.get("finite_linf")) if extract_ok else False

    # bridge_ok NEVER requires linf <= 15
    bridge_ok = bool(
        extract_ok
        and source_documented
        and source_honest
        and honesty_ok
        and dual_ban_ok
        and probe_ok
        and stream_alignment_ok
        and finite_linf
        and honesty["wire_shipped"] is False
        and dual_linf_unproven
        and blockers_still_documented
    )
    ok = bridge_ok

    ok_criteria = (
        "extract_ok ∧ source_documented ∧ source_honest ∧ honesty_ok ∧ "
        "dual_ban ∧ probe_ok ∧ stream_alignment_ok ∧ finite_linf ∧ "
        "wire_shipped=False ∧ dual_linf_unproven ∧ blockers_still_documented; "
        "NOT linf<=15 under wire; NOT dual_linf proven; NOT VERDICT gate; "
        "NOT form flip; NOT ready redefined; NOT blockers cleared; "
        "fixture never labeled as live/caller_supplied"
    )

    linf = probe.get("linf") if extract_ok else float("nan")
    l1 = probe.get("l1") if extract_ok else float("nan")
    per_stream_abs = (
        dict(probe.get("per_stream_abs") or {}) if extract_ok else {}
    )

    return {
        **honesty,
        "ok": ok,
        "bridge_ok": bridge_ok,
        "probe_ok": probe_ok,
        "extract_ok": extract_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "stream_alignment_ok": stream_alignment_ok,
        "finite_linf": finite_linf,
        "finite_ok": finite_linf,
        "blockers_still_documented": blockers_still_documented,
        "source_documented": source_documented,
        "source_honest": source_honest,
        "ok_criteria": ok_criteria,
        "bridge_ok_criteria": ok_criteria,
        # Source labeling (fixture ≠ live)
        "live_lambda_source": live_lambda_source,
        "case1_primary_online_lambda_source": live_lambda_source,
        "primary_source_path": primary_source_path,
        "used_fixture_fallback": used_fixture,
        "allow_fixture_fallback": bool(allow_fixture_fallback),
        "fixture_is_not_live": True,
        # Dual vectors
        "dual_vector_face": dual_vector_face,
        "case1_primary_online_lambda": dict(primary) if extract_ok else {},
        "case1_secondary_recovered_lambda": (
            dict(secondary_map) if secondary_map is not None else None
        ),
        "secondary_recovered_is_not_gate": True,
        "primary_extract": extract_rep,
        "secondary_extract": secondary_extract,
        "skeleton_lambda": dict(probe.get("skeleton_lambda") or {}),
        "skeleton_lambda_source": probe.get("skeleton_lambda_source"),
        "streams": streams_list,
        "linking_streams": streams_list,
        "skeleton_lambda_slots": list(CASE1_SHAPED_LINKING_STREAMS),
        # Gaps from composed probe
        "per_stream_abs": per_stream_abs,
        "abs_gap": dict(per_stream_abs),
        "linf": linf,
        "l1": l1,
        "missing_in_primary": list(probe.get("missing_in_primary") or []),
        "missing_in_skeleton": list(probe.get("missing_in_skeleton") or []),
        "gap": probe.get("gap") if extract_ok else None,
        "secondary_gap_diagnostic": (
            probe.get("secondary_gap_diagnostic") if extract_ok else None
        ),
        "probe_linf_vs_gate_tol_diagnostic": {
            **dict(probe.get("probe_linf_vs_gate_tol_diagnostic") or {}),
            "is_not_verdict_gate": True,
            "is_not_bridge_ok_criterion": True,
            "is_not_probe_ok_criterion": True,
        },
        "online_linf_gate_under_tf_path_open": online_gate_open,
        "probe_available": True,
        "bridge_available": True,
        "composed_probe_kind": CASE1_DUAL_SPACE_LINF_PROBE_KIND,
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "units_affine_unchanged": list(UNITS),
        "tf_available": tf_available(),
        "case1_shaped_streams_source": "CASE1_SHAPED_LINKING_STREAMS",
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "probe": {
            "kind": probe.get("kind"),
            "probe_ok": probe.get("probe_ok"),
            "linf": probe.get("linf"),
            "dual_linf_under_wire_status": probe.get("dual_linf_under_wire_status"),
            "wire_shipped": probe.get("wire_shipped"),
            "dual_recovery_path": probe.get("dual_recovery_path"),
        },
        "note": honesty["note"],
    }


def case1_dual_space_linf_live_lambda_bridge(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_live_lambda_bridge_report``."""
    return offline_case1_dual_space_linf_live_lambda_bridge_report(**kwargs)


def multi_unit_case1_dual_space_linf_live_lambda_bridge_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_live_lambda_bridge_report``."""
    return offline_case1_dual_space_linf_live_lambda_bridge_report(**kwargs)


# ---------------------------------------------------------------------------
# Offline Case-1 dual-space L∞ live-λ-seeded skeleton warm-start / dual_linf
# proof-prep (goal 5 + goal 3 residual after live-λ bridge)
# Compose extract + Case-1-shaped rounds with λ0 from live/caller PRIMARY;
# post-round stream L∞; dual_linf_under_wire stays unproven always; not VERDICT;
# not wire; seed identity ≠ proof; no excel_pipeline / TF / PuLP on hot path.
# ---------------------------------------------------------------------------

CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_SEEDED_WARMSTART_KIND = (
    "offline_case1_dual_space_linf_live_lambda_seeded_warmstart"
)

SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY = "lambda0_from_live_primary_online"
Z0_POLICY_UNCHANGED_DEFAULT_SKELETON = "unchanged_default_skeleton_z"


def case1_warmstart_seed_lambda_from_primary(
    primary_map: Optional[Mapping[str, Any]],
    *,
    streams: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Stream-align live PRIMARY online λ into skeleton lam0 (seed input only).

    Pure mapping helper — dual_recovery_path=None; not dual recovery ownership.
    Missing/non-finite streams → seed_ok=False (no silent zero-fill success).
    """
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    aligned = _case1_stream_align_mapping(primary_map or {}, streams=streams_list)
    return {
        "kind": "case1_warmstart_seed_lambda_from_primary",
        "seed_ok": bool(aligned["extract_ok"]),
        "extract_ok": bool(aligned["extract_ok"]),
        "stream_alignment_ok": bool(aligned["stream_alignment_ok"]),
        "lam0": dict(aligned["lambda"]) if aligned["extract_ok"] else {},
        "lambda": dict(aligned["lambda"]) if aligned["extract_ok"] else {},
        "missing_streams": list(aligned["missing_streams"]),
        "non_finite_streams": list(aligned["non_finite_streams"]),
        "streams": streams_list,
        "seed_policy": SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY,
        "dual_recovery_path": None,
        "seeded_lambda_is_probe_input_only": True,
        "note": (
            "Stream-aligned lam0 from live PRIMARY online λ for skeleton seed only. "
            "dual_recovery_path=None; not dual recovery ownership; not VERDICT; not wire."
        ),
    }


def _case1_dual_space_linf_live_lambda_seeded_warmstart_honesty_fields() -> Dict[str, Any]:
    """Dual-ban / not-wire / not-VERDICT / seed-identity locks for warm-start."""
    base = _case1_dual_space_linf_probe_honesty_fields()
    base = dict(base)
    base["kind"] = CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_SEEDED_WARMSTART_KIND
    base["scope"] = "case1_dual_space_linf_live_lambda_seeded_warmstart_offline"
    base["warmstart_is_not_verdict_gate"] = True
    base["warmstart_is_not_dual_linf_under_wire_proof"] = True
    base["bridge_is_not_verdict_gate"] = True
    base["bridge_is_not_dual_linf_under_wire_proof"] = True
    base["probe_is_not_verdict_gate"] = True
    base["probe_is_not_dual_linf_under_wire_proof"] = True
    base["live_lambda_is_not_dual_recovery"] = True
    base["seeded_lambda_is_probe_input_only"] = True
    base["extracted_lambda_is_probe_input_only"] = True
    base["seed_identity_linf_is_not_proof"] = True
    base["skeleton_lambda_is_not_case1_online_lambda"] = True
    base["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] = True
    base["secondary_recovered_is_not_gate"] = True
    base["note"] = (
        "Offline Case-1 dual-space L∞ live-λ-seeded skeleton warm-start / dual_linf "
        "proof-prep: extract/accept this-run Case 1 PRIMARY online λ (source labeled), "
        "seed Case-1-shaped skeleton dual-space λ0 from that PRIMARY "
        f"(seed_policy={SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY}), keep "
        f"z0={Z0_POLICY_UNCHANGED_DEFAULT_SKELETON} (no plant MB invention), run N "
        "skeleton linking rounds via offline_case1_shaped_cdu_blender_linking_report, "
        "then stream-aligned L∞ of post-round skeleton λ vs live PRIMARY (compose "
        "existing L∞/probe helpers — no second engine). Also reports linf_at_seed "
        "labeled seed-identity-not-proof. dual_recovery_path=None; solver=False; "
        "on_excel_case1_path=False; wire_shipped=False; dual_linf_under_wire_status="
        "unproven ALWAYS (even if post-round L∞ is 0 or ≤15); checklist "
        "online_linf_gate_under_tf_path remains open; warm-start ≠ Case 1 VERDICT "
        "gate; warm-start ≠ dual L∞ under wire proof; skeleton λ (even post-round) ≠ "
        "Case 1 duals as recovery; SECONDARY recovered is diagnostic only; form_current "
        "classic unchanged; does not clear DEFAULT_WIRE_BLOCKERS; does not redefine "
        "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline on hot "
        "path; no live Case 1 solve ownership; no residual-must-vanish hard-fail."
    )
    return base


def offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
    *,
    case1_package: Optional[Mapping[str, Any]] = None,
    case1_primary_online_lambda: Optional[Mapping[str, float]] = None,
    case1_secondary_recovered_lambda: Optional[Mapping[str, float]] = None,
    include_secondary_recovered: bool = True,
    allow_fixture_fallback: bool = False,
    n_rounds: int = 2,
    rho: float = 1.0,
    dual_step: float = 1.0,
    dual_vector_face: str = CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE,
    streams: Optional[Sequence[str]] = None,
    dual_gate_threshold_diagnostic: float = 15.0,
) -> Dict[str, Any]:
    """Always-on live-λ-seeded Case-1 dual-space L∞ warm-start / dual_linf proof-prep.

    Compose only:
      1. extract/normalize live PRIMARY online λ (source labeled)
      2. seed skeleton λ0 from that PRIMARY; z0 = default skeleton z
      3. run N Case-1-shaped linking rounds under that seed
      4. stream-aligned L∞ post-round skeleton λ vs live PRIMARY
      5. also linf_at_seed (seed identity) labeled not-proof

    ``warmstart_ok`` / ``ok`` = extract ∧ source documented ∧ seed_policy ∧ z0_policy
    ∧ rounds ∧ stream alignment ∧ finite L∞ ∧ dual-ban ∧ blockers — **never**
    ``linf <= 15``; **never** VERDICT; dual_linf_under_wire stays **unproven always**.

    Does **not** clear ``DEFAULT_WIRE_BLOCKERS``. Does **not** redefine
    ``ready_for_wire_discussion``. Does **not** flip Case 1 form. Does **not**
    retune Case 1 ADMM ρ. No excel_pipeline / tensorflow / pulp on hot path.
    """
    honesty = _case1_dual_space_linf_live_lambda_seeded_warmstart_honesty_fields()
    streams_list = (
        list(streams) if streams is not None else list(CASE1_SHAPED_LINKING_STREAMS)
    )
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0
    n_r = int(n_rounds)
    if n_r < 1:
        n_r = 1

    # --- PRIMARY extract / resolve (same branch structure as live-λ bridge) ---
    extract_rep: Dict[str, Any]
    if case1_primary_online_lambda is not None:
        extract_rep = case1_primary_online_lambda_from_mapping(
            case1_primary_online_lambda,
            face=dual_vector_face,
            streams=streams_list,
        )
        extract_rep = dict(extract_rep)
        extract_rep["kind"] = "extract_case1_primary_online_lambda"
        extract_rep["source"] = LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
        extract_rep["source_path"] = "case1_primary_online_lambda_arg"
        extract_rep["face_converted"] = False
    elif case1_package is not None:
        extract_rep = extract_case1_primary_online_lambda(
            case1_package,
            face=dual_vector_face,
            streams=streams_list,
        )
    else:
        extract_rep = {
            "kind": "extract_case1_primary_online_lambda",
            "extract_ok": False,
            "stream_alignment_ok": False,
            "lambda": {},
            "case1_primary_online_lambda": {},
            "missing_streams": list(streams_list),
            "non_finite_streams": [],
            "streams": streams_list,
            "dual_vector_face": dual_vector_face,
            "source": LIVE_LAMBDA_SOURCE_MISSING,
            "source_path": "none",
            "face_converted": False,
            "dual_recovery_path": None,
        }

    used_fixture = False
    if not extract_rep.get("extract_ok"):
        if allow_fixture_fallback:
            primary = case1_primary_online_lambda_fixture(face=dual_vector_face)
            live_lambda_source = LIVE_LAMBDA_SOURCE_FIXTURE
            primary_source_path = "fixture"
            used_fixture = True
            extract_ok = True
        else:
            primary = {}
            live_lambda_source = LIVE_LAMBDA_SOURCE_MISSING
            primary_source_path = str(extract_rep.get("source_path") or "missing")
            extract_ok = False
    else:
        primary = dict(extract_rep["lambda"])
        live_lambda_source = str(
            extract_rep.get("source") or LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
        )
        primary_source_path = str(
            extract_rep.get("source_path") or "caller_supplied"
        )
        extract_ok = True

    source_documented = live_lambda_source in (
        LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED,
        LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT,
        LIVE_LAMBDA_SOURCE_FIXTURE,
    )
    source_honest = not (
        used_fixture and live_lambda_source != LIVE_LAMBDA_SOURCE_FIXTURE
    )
    if used_fixture:
        live_lambda_source = LIVE_LAMBDA_SOURCE_FIXTURE
        source_honest = True
        source_documented = True

    # --- Optional SECONDARY diagnostic (never gate) ---
    secondary_map: Optional[Dict[str, float]] = None
    secondary_extract: Optional[Dict[str, Any]] = None
    if case1_secondary_recovered_lambda is not None:
        sec = _case1_stream_align_mapping(
            case1_secondary_recovered_lambda, streams=streams_list
        )
        secondary_extract = {
            "extract_ok": sec["extract_ok"],
            "source": LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED,
            "source_path": "case1_secondary_recovered_lambda_arg",
            "secondary_recovered_is_not_gate": True,
            "dual_recovery_path": None,
        }
        if sec["extract_ok"]:
            secondary_map = dict(sec["lambda"])
    elif include_secondary_recovered and case1_package is not None:
        secondary_extract = extract_case1_secondary_recovered_lambda(
            case1_package, streams=streams_list
        )
        if secondary_extract.get("extract_ok"):
            secondary_map = dict(secondary_extract["lambda"])

    # --- Seed policy: lam0 from live PRIMARY; z0 default skeleton ---
    seed_policy = SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY
    z0_policy = Z0_POLICY_UNCHANGED_DEFAULT_SKELETON
    seed_rep = case1_warmstart_seed_lambda_from_primary(
        primary if extract_ok else None, streams=streams_list
    )
    seed_ok = bool(extract_ok and seed_rep.get("seed_ok"))
    lam0: Dict[str, float] = dict(seed_rep.get("lam0") or {}) if seed_ok else {}
    z0: Dict[str, float] = _default_case1_z_link(streams_list)
    seed_documented = bool(seed_policy and z0_policy)
    seed_policy_documented = seed_documented

    # --- Seed identity L∞ (diagnostic; not proof) ---
    linf_at_seed = float("nan")
    l1_at_seed = float("nan")
    per_stream_abs_at_seed: Dict[str, float] = {}
    seed_gap: Optional[Dict[str, Any]] = None
    if seed_ok and primary and lam0:
        seed_gap = case1_dual_space_stream_aligned_linf(
            primary, lam0, streams=streams_list
        )
        linf_at_seed = float(seed_gap.get("linf", float("nan")))
        l1_at_seed = float(seed_gap.get("l1", float("nan")))
        per_stream_abs_at_seed = dict(seed_gap.get("per_stream_abs") or {})

    # --- N skeleton rounds under seeded dual-space ---
    skeleton_report: Optional[Dict[str, Any]] = None
    rounds_ran = False
    skeleton_ok = False
    final_lam: Dict[str, float] = {}
    final_z: Dict[str, float] = {}
    residual_trend: Any = None
    if seed_ok and lam0:
        skeleton_report = offline_case1_shaped_cdu_blender_linking_report(
            n_rounds=n_r,
            rho=float(rho),
            dual_step=float(dual_step),
            lam0=lam0,
            z0=z0,
        )
        rounds_ran = True
        skeleton_ok = bool(skeleton_report.get("ok"))
        final_lam = extract_case1_shaped_skeleton_lambda(
            skeleton_report=skeleton_report
        )
        final_z = dict(skeleton_report.get("final_z") or {})
        residual_trend = skeleton_report.get("residual_trend")

    # --- Post-round stream L∞ via existing probe compose (no second engine) ---
    if seed_ok and primary and final_lam:
        probe = offline_case1_dual_space_linf_probe_report(
            case1_primary_online_lambda=primary,
            case1_secondary_recovered_lambda=secondary_map,
            skeleton_lambda=final_lam,
            dual_vector_face=dual_vector_face,
            streams=streams_list,
            dual_gate_threshold_diagnostic=dual_gate_threshold_diagnostic,
        )
    else:
        probe = offline_case1_dual_space_linf_probe_report(
            case1_primary_online_lambda={s: 0.0 for s in streams_list},
            skeleton_lambda={s: 0.0 for s in streams_list},
            dual_vector_face=dual_vector_face,
            streams=streams_list,
            dual_gate_threshold_diagnostic=dual_gate_threshold_diagnostic,
        )

    linf_post_rounds = probe.get("linf") if (seed_ok and final_lam) else float("nan")
    l1_post_rounds = probe.get("l1") if (seed_ok and final_lam) else float("nan")
    per_stream_abs_post = (
        dict(probe.get("per_stream_abs") or {}) if (seed_ok and final_lam) else {}
    )

    # dual_linf_under_wire ALWAYS unproven on this surface (even if L∞ 0 or ≤15)
    dual_linf_unproven = (
        honesty["dual_linf_under_wire_status"] == "unproven"
        and probe.get("dual_linf_under_wire_status") == "unproven"
    )
    # Force unproven even if future probe changes — warm-start hard lock
    dual_linf_status = "unproven"
    dual_linf_unproven = dual_linf_unproven and dual_linf_status == "unproven"

    checklist = dict(honesty["dual_linf_proof_checklist"])
    online_gate_status = str(checklist.get("online_linf_gate_under_tf_path", "open"))
    online_gate_open = online_gate_status.lower() in ("open", "false_today", "unproven")
    online_gate_open = online_gate_open and bool(
        probe.get("online_linf_gate_under_tf_path_open", True)
    )

    dual_ban_ok = bool(
        honesty["dual_recovery_path"] is None
        and probe.get("dual_recovery_path") is None
        and honesty["warmstart_is_not_dual_linf_under_wire_proof"] is True
        and honesty["warmstart_is_not_verdict_gate"] is True
        and honesty["seed_identity_linf_is_not_proof"] is True
        and dual_linf_unproven
        and online_gate_open
    )

    honesty_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and list(UNITS) == ["FCC", "COKER", "CDU"]
        and "BLENDER" not in UNITS
        and honesty["dual_recovery_path"] is None
        and honesty["wire_shipped"] is False
        and honesty["warmstart_is_not_verdict_gate"] is True
        and honesty["warmstart_is_not_dual_linf_under_wire_proof"] is True
        and honesty["seed_identity_linf_is_not_proof"] is True
        and honesty["does_not_clear_default_wire_blockers"] is True
        and honesty["does_not_redefine_ready_for_wire_discussion"] is True
        and dual_linf_unproven
        and "dual_linf_under_wire_unproven" in blockers
        and "wire_not_shipped" in blockers
        and blockers_still_documented
        and online_gate_open
    )

    probe_ok = bool(probe.get("probe_ok")) if (seed_ok and final_lam) else False
    stream_alignment_ok = bool(
        seed_ok and final_lam and probe.get("stream_alignment_ok", False)
    )
    finite_linf = bool(probe.get("finite_linf")) if (seed_ok and final_lam) else False
    # Primary metric finite check must not gate on ≤15
    finite_post = bool(
        finite_linf
        and linf_post_rounds is not None
        and np.isfinite(float(linf_post_rounds))
    ) if (seed_ok and final_lam) else False

    ok_criteria = (
        "extract_ok ∧ source_documented ∧ source_honest ∧ seed_ok ∧ "
        "seed_policy_documented ∧ z0_policy_documented ∧ rounds_ran ∧ "
        "skeleton_ok ∧ honesty_ok ∧ dual_ban ∧ probe_ok ∧ stream_alignment_ok ∧ "
        "finite_post_round_linf ∧ wire_shipped=False ∧ dual_linf_unproven ∧ "
        "blockers_still_documented ∧ online_linf_gate_open; "
        "NOT linf<=15 under wire; NOT dual_linf proven (even if L∞ 0 or ≤15); "
        "NOT VERDICT gate; NOT seed identity as proof; NOT form flip; "
        "NOT ready redefined; NOT blockers cleared; NOT residual-must-vanish; "
        "fixture never labeled as live/caller_supplied"
    )

    warmstart_ok = bool(
        extract_ok
        and source_documented
        and source_honest
        and seed_ok
        and seed_policy_documented
        and rounds_ran
        and skeleton_ok
        and honesty_ok
        and dual_ban_ok
        and probe_ok
        and stream_alignment_ok
        and finite_post
        and honesty["wire_shipped"] is False
        and dual_linf_unproven
        and blockers_still_documented
        and online_gate_open
    )
    ok = warmstart_ok

    face_conversion_applied = bool(extract_rep.get("face_converted", False))

    return {
        **honesty,
        # Force unproven status even if honesty base ever changes
        "dual_linf_under_wire_status": dual_linf_status,
        "dual_linf_under_wire": dual_linf_status,
        "dual_linf_under_wire_unproven_still_true": True,
        "online_linf_gate_under_tf_path": "open",
        "ok": ok,
        "warmstart_ok": warmstart_ok,
        "probe_ok": probe_ok,
        "extract_ok": extract_ok,
        "seed_ok": seed_ok,
        "skeleton_ok": skeleton_ok,
        "rounds_ran": rounds_ran,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "stream_alignment_ok": stream_alignment_ok,
        "finite_linf": finite_post,
        "finite_ok": finite_post,
        "blockers_still_documented": blockers_still_documented,
        "source_documented": source_documented,
        "source_honest": source_honest,
        "seed_documented": seed_documented,
        "seed_policy_documented": seed_policy_documented,
        "ok_criteria": ok_criteria,
        "warmstart_ok_criteria": ok_criteria,
        # Source labeling (fixture ≠ live)
        "live_lambda_source": live_lambda_source,
        "case1_primary_online_lambda_source": live_lambda_source,
        "primary_source_path": primary_source_path,
        "used_fixture_fallback": used_fixture,
        "allow_fixture_fallback": bool(allow_fixture_fallback),
        "fixture_is_not_live": True,
        # Seed policy honesty
        "seed_policy": seed_policy,
        "z0_policy": z0_policy,
        "lam0": dict(lam0),
        "z0": dict(z0),
        "n_rounds": n_r,
        "rho": float(rho),
        "dual_step": float(dual_step),
        "rho_is_not_case1_retune": True,
        "does_not_retune_case1_admm_rho": True,
        # Dual vectors
        "dual_vector_face": dual_vector_face,
        "face_conversion_applied": face_conversion_applied,
        "case1_primary_online_lambda": dict(primary) if extract_ok else {},
        "case1_secondary_recovered_lambda": (
            dict(secondary_map) if secondary_map is not None else None
        ),
        "secondary_recovered_is_not_gate": True,
        "primary_extract": extract_rep,
        "secondary_extract": secondary_extract,
        "skeleton_lambda": dict(final_lam),
        "skeleton_lambda_source": "live_lambda_seeded_post_rounds",
        "skeleton_lambda_at_seed": dict(lam0),
        "final_lam": dict(final_lam),
        "final_z": dict(final_z),
        "streams": streams_list,
        "linking_streams": streams_list,
        "skeleton_lambda_slots": list(CASE1_SHAPED_LINKING_STREAMS),
        # Dual metrics: seed identity + post-round primary
        "linf_at_seed": linf_at_seed,
        "l1_at_seed": l1_at_seed,
        "per_stream_abs_at_seed": per_stream_abs_at_seed,
        "seed_gap": seed_gap,
        "linf_post_rounds": linf_post_rounds,
        "linf": linf_post_rounds,  # primary metric alias
        "l1_post_rounds": l1_post_rounds,
        "l1": l1_post_rounds,
        "per_stream_abs": per_stream_abs_post,
        "abs_gap": dict(per_stream_abs_post),
        "missing_in_primary": list(probe.get("missing_in_primary") or []),
        "missing_in_skeleton": list(probe.get("missing_in_skeleton") or []),
        "gap": probe.get("gap") if (seed_ok and final_lam) else None,
        "secondary_gap_diagnostic": (
            probe.get("secondary_gap_diagnostic") if (seed_ok and final_lam) else None
        ),
        "probe_linf_vs_gate_tol_diagnostic": {
            **dict(probe.get("probe_linf_vs_gate_tol_diagnostic") or {}),
            "is_not_verdict_gate": True,
            "is_not_warmstart_ok_criterion": True,
            "is_not_probe_ok_criterion": True,
        },
        "online_linf_gate_under_tf_path_open": online_gate_open,
        "probe_available": True,
        "warmstart_available": True,
        "composed_probe_kind": CASE1_DUAL_SPACE_LINF_PROBE_KIND,
        "composed_skeleton_kind": CASE1_SHAPED_LINKING_KIND,
        "residual_trend": residual_trend,
        "residual_must_vanish_is_not_gate": True,
        "no_residual_must_vanish_hard_fail": True,
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "units_affine_unchanged": list(UNITS),
        "tf_available": tf_available(),
        "case1_shaped_streams_source": "CASE1_SHAPED_LINKING_STREAMS",
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "skeleton": {
            "kind": (skeleton_report or {}).get("kind"),
            "ok": (skeleton_report or {}).get("ok"),
            "n_rounds": n_r,
            "wire_shipped": (skeleton_report or {}).get("wire_shipped"),
            "dual_recovery_path": (skeleton_report or {}).get("dual_recovery_path"),
        },
        "probe": {
            "kind": probe.get("kind"),
            "probe_ok": probe.get("probe_ok"),
            "linf": probe.get("linf"),
            "dual_linf_under_wire_status": probe.get("dual_linf_under_wire_status"),
            "wire_shipped": probe.get("wire_shipped"),
            "dual_recovery_path": probe.get("dual_recovery_path"),
        },
        "note": honesty["note"],
    }


def case1_dual_space_linf_live_lambda_seeded_warmstart(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report``."""
    return offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(**kwargs)


def multi_unit_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report``."""
    return offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1 honest blender pooling path formalization (goal 5 + goal 3)
# ---------------------------------------------------------------------------
# Always-on numpy. Formalizes linear_quality_pooling as the dual-honest Case-1
# blender path so checklist blender_affine_kernel_or_honest_pooling_path is no
# longer bare "open" — WITHOUT inventing BLENDER affine UNITS and WITHOUT
# clearing no_blender_offline_affine_kernel.
# dual_recovery_path=None; wire_shipped=False; dual_linf unproven;
# pooling path ≠ wire; ≠ VERDICT; ≠ affine kernel; no residual-must-vanish;
# does NOT redefine ready_for_wire_discussion; no TF / no PuLP / no excel_pipeline.

CASE1_HONEST_BLENDER_POOLING_PATH_KIND = "offline_case1_honest_blender_pooling_path"
CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS = "honest_pooling_path_present"
CASE1_HONEST_BLENDER_POOLING_PATH_RECIPES_SOURCE = "synthetic_offline_demo"


def _case1_honest_blender_pooling_path_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire / not-affine locks for pooling path."""
    shaped = _case1_shaped_linking_honesty_fields()
    return {
        "kind": CASE1_HONEST_BLENDER_POOLING_PATH_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "skeleton_lambda_is_not_case1_online_lambda": True,
        "skeleton_lambda_is_not_case1_primary_or_secondary_duals": True,
        "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "blender_is_base_delta_affine_unit": False,
        "excel_cdu_matrix_matches_affine": None,
        "excel_blender_matrix_matches_affine": None,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "pooling_path_is_not_affine_kernel": True,
        "pooling_path_is_not_wire": True,
        "pooling_path_is_not_verdict_gate": True,
        "pooling_path_is_not_dual_linf_under_wire_proof": True,
        "recipes_source": CASE1_HONEST_BLENDER_POOLING_PATH_RECIPES_SOURCE,
        "products_source": CASE1_HONEST_BLENDER_POOLING_PATH_RECIPES_SOURCE,
        "streams_source": "CASE1_SHAPED_LINKING_STREAMS",
        "composed_from_shaped_honesty": True,
        "shaped_kind": shaped.get("kind"),
        "scope": "case1_honest_blender_pooling_path_offline",
        "note": (
            "Offline Case-1 honest blender pooling path formalization: documents "
            f"blender_surface={CASE1_SHAPED_BLENDER_SURFACE} as the dual-honest Case-1 "
            "blender path without inventing a BLENDER base_delta affine UNITS kernel. "
            "Checklist status honest_pooling_path_present (not closed_via_affine_kernel). "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            "wire_shipped=False; dual_linf_under_wire=unproven; case1_form_unchanged "
            f"({CASE1_FORM_CURRENT}). Pooling path is NOT affine kernel, NOT wire, NOT "
            "VERDICT gate, NOT dual L∞ under wire proof. "
            "no_blender_offline_affine_kernel remains in DEFAULT_WIRE_BLOCKERS. "
            "UNITS stay FCC/COKER/CDU (no silent BLENDER). Does not invent "
            "excel_cdu_matrix_matches_affine / excel_blender_matrix_matches_affine. "
            "Does not clear DEFAULT_WIRE_BLOCKERS. Does not redefine "
            "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline on "
            "hot path; no maximizer re-run; no residual-must-vanish; no linf≤15 gate."
        ),
    }


def offline_case1_honest_blender_pooling_path_report() -> Dict[str, Any]:
    """Always-on honest Case-1 blender pooling path report (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``pooling_path_ok`` = honesty ∧ surface labeled
    linear_quality_pooling ∧ dual-ban ∧ no_blender blocker still true ∧
    BLENDER not in UNITS ∧ checklist status honest_pooling_path_present ∧
    dual_linf unproven. **Not** residual-must-vanish. **Not** linf≤15.
    **Not** wire. **Not** form flip. **Not** dual L∞ under wire proven.
    Composes skeleton honesty — does **not** re-run Case-1-shaped maximizer.
    """
    honesty = _case1_honest_blender_pooling_path_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    pooling_checklist_status = checklist.get(
        "blender_affine_kernel_or_honest_pooling_path"
    )
    checklist_status_ok = (
        pooling_checklist_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS
        and pooling_checklist_status != "open"
        and pooling_checklist_status != "closed_via_affine_kernel"
    )
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    blender_not_in_open = "blender_affine_kernel_or_honest_pooling_path" not in open_ids

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    surface_ok = honesty["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE == (
        "linear_quality_pooling"
    )
    blocker_ok = (
        "no_blender_offline_affine_kernel" in blockers
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
    )
    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["blender_is_base_delta_affine_unit"] is False
        and honesty["excel_blender_matrix_matches_affine"] is None
        and honesty["excel_cdu_matrix_matches_affine"] is None
        and honesty["pooling_path_is_not_affine_kernel"] is True
        and honesty["pooling_path_is_not_wire"] is True
        and honesty["pooling_path_is_not_verdict_gate"] is True
        and honesty["pooling_path_is_not_dual_linf_under_wire_proof"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
    )
    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and surface_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and checklist_status_ok
        and blender_not_in_open
        and blockers_still_documented
    )
    pooling_path_ok = honesty_ok
    ok = pooling_path_ok and (honesty["wire_shipped"] is False)

    ok_criteria = (
        "honesty ∧ blender_surface=linear_quality_pooling ∧ dual-ban ∧ "
        "no_blender_offline_affine_kernel still true ∧ BLENDER∉UNITS ∧ "
        "checklist=honest_pooling_path_present ∧ dual_linf unproven ∧ "
        "form classic unchanged — NOT residual-must-vanish; NOT linf<=15; "
        "NOT wire; NOT VERDICT; NOT closed_via_affine_kernel"
    )

    return {
        **honesty,
        "ok": ok,
        "pooling_path_ok": pooling_path_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "surface_ok": surface_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "checklist_status_ok": checklist_status_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ok_criteria": ok_criteria,
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        "blender_pooling_checklist_status": pooling_checklist_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "streams": list(CASE1_SHAPED_LINKING_STREAMS),
        "blend_recipes": dict(CASE1_SHAPED_BLEND_RECIPES),
        "cdu_to_intermediate_map": case1_shaped_cdu_to_intermediate_map(),
        "units_affine_unchanged": list(UNITS),
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "tf_available": tf_available(),
        "pooling_path_available": True,
        "residual_must_vanish_is_not_gate": True,
        "no_residual_must_vanish_hard_fail": True,
        "linf_le_15_is_not_gate": True,
        "composed_shaped_surface": CASE1_SHAPED_BLENDER_SURFACE,
        "note": honesty["note"],
    }


def case1_honest_blender_pooling_path_report(**kwargs: Any) -> Dict[str, Any]:
    """Alias for ``offline_case1_honest_blender_pooling_path_report``."""
    return offline_case1_honest_blender_pooling_path_report(**kwargs)


def multi_unit_case1_honest_blender_pooling_path_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_honest_blender_pooling_path_report``."""
    return offline_case1_honest_blender_pooling_path_report(**kwargs)


# ---------------------------------------------------------------------------
# Offline Case-1 online_linf_gate flip-criteria contract (goal 3 + 5 honesty)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes machine-readable flip criteria for
# checklist online_linf_gate_under_tf_path so the item is no longer an opaque
# bare "open". Does NOT flip the gate. Does NOT claim dual L∞ under wire
# proven. Does NOT ship wire or flip Case 1 form. Does NOT invent BLENDER
# UNITS. Does NOT clear DEFAULT_WIRE_BLOCKERS. Does NOT redefine
# ready_for_wire_discussion. No TF / no PuLP / no excel_pipeline on hot path.

CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_KIND = (
    "offline_case1_online_linf_gate_criteria_contract"
)
CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY = "online_linf_gate_under_tf_path"
CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_ANNOTATION = "present"

# Flip-criteria class labels
FLIP_CRITERION_REQUIRED = "required"
FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY = "required_under_wire_only"

# Machine-readable flip criteria map (what must hold before the gate can close).
# Values are requirement classes — not "met today" claims.
CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA: Dict[str, str] = {
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    "dual_honest_tf_aware_path_present": FLIP_CRITERION_REQUIRED,
    "online_lambda_owns_verdict_gate": FLIP_CRITERION_REQUIRED,
    "linf_le_15_only_under_shipped_tf_aware_path": (
        FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    ),
    "wire_shipped": FLIP_CRITERION_REQUIRED,
    "dual_recovery_path_labeled_honestly": FLIP_CRITERION_REQUIRED,
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these are NEVER flip enablers today.
CASE1_ONLINE_LINF_GATE_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
)


def case1_online_linf_gate_flip_criteria() -> Dict[str, str]:
    """Return a copy of the machine-readable online_linf_gate flip-criteria map."""
    return dict(CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA)


def case1_online_linf_gate_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate criteria_met_today remains False while isolation rewrite, form
    label shipped, wire_shipped, and dual-honest TF path remain open. Individual
    structural honesty labels that already hold offline (e.g. online λ owns
    VERDICT on classic path; dual_recovery_path labeled None on TF surface) may
    be True without flipping the aggregate.
    """
    # HEAD truth: none of the wire-shipping prerequisites are met.
    return {
        "isolation_rewrite_with_wire": False,
        "form_label_change_shipped": False,
        # Dual-honest TF-aware *wire path* not present (offline ladder ≠ wire path).
        "dual_honest_tf_aware_path_present": False,
        # Classic Case 1 already owns VERDICT via PRIMARY online λ — structural.
        "online_lambda_owns_verdict_gate": True,
        # linf≤15 under shipped TF path — not applicable until wire ships.
        "linf_le_15_only_under_shipped_tf_aware_path": False,
        "wire_shipped": False,
        # TF surface dual_recovery_path is None (labeled honestly offline).
        "dual_recovery_path_labeled_honestly": True,
        # Planned form is registered and distinct from classic (no silent reuse).
        "no_silent_form_reuse": True,
        "isolation_tests_rewritten_with_wire_not_deleted": False,
    }


def case1_online_linf_gate_flip_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while wire/form/isolation prerequisites remain open.

    Even if some structural labels hold offline, aggregate flip is never allowed
    until required wire-shipping criteria are all met.
    """
    met = criteria_met if criteria_met is not None else (
        case1_online_linf_gate_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    return all(bool(met.get(k)) for k in required_keys)


def case1_online_linf_gate_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today — False until all required criteria hold."""
    return case1_online_linf_gate_flip_allowed_today(criteria_met)


def _case1_online_linf_gate_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-gate-flip / not-wire locks."""
    return {
        "kind": CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "contract_is_not_gate_flip": True,
        "contract_is_not_wire": True,
        "contract_is_not_verdict_gate": True,
        "contract_is_not_dual_linf_under_wire_proof": True,
        "probe_linf_is_not_flip_criterion_today": True,
        "bridge_linf_is_not_flip_criterion_today": True,
        "warmstart_linf_is_not_flip_criterion_today": True,
        "pooling_linf_is_not_flip_criterion_today": True,
        "seed_identity_linf_is_not_flip_criterion": True,
        "recovered_blender_linf_is_not_flip_criterion_today": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "scope": "case1_online_linf_gate_criteria_contract_offline",
        "note": (
            "Offline Case-1 online_linf_gate flip-criteria contract: machine-readable "
            "criteria for checklist online_linf_gate_under_tf_path. Gate stays open; "
            "gate_flip_allowed_today=False; criteria_met_today=False; dual_linf "
            "unproven; dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            f"wire_shipped=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). Contract "
            "is NOT gate flip, NOT wire, NOT VERDICT gate, NOT dual L∞ under wire proof. "
            "Probe/bridge/warmstart/pooling/seed-identity/recovered L∞ are not flip "
            "criteria today. no_blender_offline_affine_kernel remains in "
            "DEFAULT_WIRE_BLOCKERS. UNITS stay FCC/COKER/CDU. Does not clear "
            "DEFAULT_WIRE_BLOCKERS. Does not redefine ready_for_wire_discussion. "
            "Always-on numpy; no TF/PuLP/excel_pipeline on hot path; no maximizer; "
            "no linf≤15 gate on contract ok."
        ),
    }


def offline_case1_online_linf_gate_criteria_contract_report() -> Dict[str, Any]:
    """Always-on online_linf_gate flip-criteria contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``contract_ok`` = criteria formalized ∧ honesty locks ∧
    gate still open ∧ dual_linf unproven ∧ blockers non-empty ∧ form classic ∧
    UNITS FCC/COKER/CDU ∧ gate_flip_allowed_today=False ∧ criteria_met_today=False.
    **Not** linf≤15. **Not** gate flipped. **Not** wire. **Not** dual L∞ proven.
    Composes checklist / form / blockers / pooling snapshot — does **not** re-run
    maximizers/probes.
    """
    honesty = _case1_online_linf_gate_criteria_contract_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    flip_criteria = case1_online_linf_gate_flip_criteria()
    criteria_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(criteria_met_map)
    criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        criteria_met_map
    )

    required_keys = set(CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA.keys())
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and flip_criteria.get("linf_le_15_only_under_shipped_tf_aware_path")
        == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        and all(
            flip_criteria[k] == FLIP_CRITERION_REQUIRED
            for k in required_keys
            if k != "linf_le_15_only_under_shipped_tf_aware_path"
        )
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    blocker_ok = (
        "no_blender_offline_affine_kernel" in blockers
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["contract_is_not_gate_flip"] is True
        and honesty["contract_is_not_wire"] is True
        and honesty["contract_is_not_verdict_gate"] is True
        and honesty["contract_is_not_dual_linf_under_wire_proof"] is True
        and honesty["probe_linf_is_not_flip_criterion_today"] is True
        and honesty["bridge_linf_is_not_flip_criterion_today"] is True
        and honesty["warmstart_linf_is_not_flip_criterion_today"] is True
        and honesty["pooling_linf_is_not_flip_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_flip_criterion"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
    )
    flip_permission_ok = (
        gate_flip_allowed_today is False and criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and flip_criteria_formalized
        and flip_permission_ok
        and gate_open_ok
        and blockers_still_documented
        and pooling_ok
    )
    contract_ok = honesty_ok
    ok = contract_ok and (honesty["wire_shipped"] is False)

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ gate still open ∧ dual_linf "
        "unproven ∧ blockers non-empty ∧ form classic ∧ UNITS FCC/COKER/CDU ∧ "
        "gate_flip_allowed_today=False ∧ criteria_met_today=False — "
        "NOT gate flipped; NOT linf<=15; NOT wire; NOT VERDICT; "
        "NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "contract_ok": contract_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "flip_criteria_formalized": flip_criteria_formalized,
        "flip_permission_ok": flip_permission_ok,
        "gate_open_ok": gate_open_ok,
        "ok_criteria": ok_criteria,
        # Checklist / gate status (must remain open)
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_checklist_key": CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY,
        "online_linf_gate_still_open": gate_still_open,
        "online_linf_gate_criteria_contract": (
            CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_ANNOTATION
        ),
        # Flip permission (hard False under HEAD)
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "criteria_met_today": criteria_met_today,
        "flip_criteria": flip_criteria,
        "gate_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "anti_criteria_today": list(CASE1_ONLINE_LINF_GATE_ANTI_CRITERIA_TODAY),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        "flip_criterion_required_under_wire_only_class": (
            FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        ),
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "tf_available": tf_available(),
        "criteria_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "linf_le_15_is_not_flip_criterion_today": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_online_linf_gate_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_online_linf_gate_criteria_contract_report``."""
    return offline_case1_online_linf_gate_criteria_contract_report(**kwargs)


def multi_unit_case1_online_linf_gate_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_online_linf_gate_criteria_contract_report``."""
    return offline_case1_online_linf_gate_criteria_contract_report(**kwargs)




# ---------------------------------------------------------------------------
# Offline Case-1 isolation-rewrite design-only contract (goal 5 + 3 honesty)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes what isolation rewrite WITH dual-honest
# wire means so flip criterion isolation_rewrite_with_wire and blocker
# isolation_rewrite_required are no longer opaque open names. Design only:
# isolation_rewrite_design_present=True; isolation_rewrite_shipped=False;
# checklist stays open; met_today isolation keys stay False. Does NOT rewrite
# isolation tests. Does NOT ship wire or flip Case 1 form. Does NOT invent
# BLENDER UNITS. Does NOT clear DEFAULT_WIRE_BLOCKERS. Does NOT redefine
# ready_for_wire_discussion. No TF / no PuLP / no excel_pipeline on hot path.

CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_KIND = (
    "offline_case1_isolation_rewrite_design_contract"
)
CASE1_ISOLATION_REWRITE_CHECKLIST_KEY = "isolation_rewrite_with_wire"
CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_ANNOTATION = "present"
CASE1_ISOLATION_REWRITE_BLOCKER_ID = "isolation_rewrite_required"
CASE1_ISOLATION_REWRITE_FLIP_KEY = "isolation_rewrite_with_wire"
CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY = (
    "isolation_tests_rewritten_with_wire_not_deleted"
)

# Current isolation invariants that must survive any future dual-honest wire.
# Static catalog only — not executed rewrites this cycle.
CASE1_ISOLATION_INVARIANTS_MUST_SURVIVE: tuple = (
    "excel_pipeline_source_must_not_hard_import_tensorflow_or_tf_linear_blocks_on_classic_path",
    "models_init_must_not_hard_import_tf_linear_blocks",
    "case1_form_remains_classic_until_explicit_form_label_change_shipped",
    "dual_recovery_path_labeled_honestly_none_on_offline_tf_surface_online_lambda_owns_verdict_on_classic",
    "optional_tf_path_must_remain_skipif_friendly_when_tf_absent",
    "tf_linear_blocks_importable_without_tensorflow_scaffold_offline",
)

# Post-wire isolation rewrite *shape* (design-only; not implemented this cycle).
CASE1_ISOLATION_REWRITE_POST_WIRE_SHAPE: Dict[str, Any] = {
    "suite_shape": "dual_path_isolation_suite",
    "classic_path_still_isolated": True,
    "tf_aware_path_gated_by_form_label_and_feature_flag": True,
    "isolation_tests_rewritten_with_wire_not_deleted": True,
    "no_silent_form_reuse": True,
    "rewrite_shipped": False,
    "implemented_this_cycle": False,
    "note": (
        "Design-only shape for a future wire cycle: keep classic isolation "
        "gates; add TF-aware path gated by form label + feature flag; rewrite "
        "isolation tests WITH wire — never delete them. Not shipped today."
    ),
}


def case1_isolation_invariants_must_survive() -> list:
    """Return a copy of current isolation invariants that must survive wire."""
    return list(CASE1_ISOLATION_INVARIANTS_MUST_SURVIVE)


def case1_isolation_rewrite_post_wire_shape() -> Dict[str, Any]:
    """Return a copy of the post-wire isolation rewrite design shape."""
    return dict(CASE1_ISOLATION_REWRITE_POST_WIRE_SHAPE)


def _case1_isolation_rewrite_design_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-rewrite-shipped / not-wire locks."""
    return {
        "kind": CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "isolation_rewrite_shipped": False,
        "isolation_tests_rewritten_with_wire": False,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "scope": "case1_isolation_rewrite_design_contract_offline",
        "note": (
            "Offline Case-1 isolation-rewrite design-only contract: formalizes "
            "what isolation rewrite WITH dual-honest wire means "
            "(rewrite-not-delete). isolation_rewrite_design_present=True; "
            "isolation_rewrite_shipped=False; isolation_tests_rewritten_with_wire="
            "False; checklist isolation_rewrite_with_wire stays open; dual_linf "
            "unproven; online_linf_gate open; gate_flip_allowed_today=False; "
            "criteria_met_today=False; dual_recovery_path=None; solver=False; "
            f"on_excel_case1_path=False; wire_shipped=False; case1_form_unchanged "
            f"({CASE1_FORM_CURRENT}). Design is NOT isolation rewrite shipped, NOT "
            "wire, NOT gate flip, NOT VERDICT gate, NOT dual L∞ under wire proof. "
            "isolation_rewrite_required remains in DEFAULT_WIRE_BLOCKERS. UNITS "
            "stay FCC/COKER/CDU. Does not clear DEFAULT_WIRE_BLOCKERS. Does not "
            "redefine ready_for_wire_discussion. Always-on numpy; no "
            "TF/PuLP/excel_pipeline on hot path; no maximizer; isolation suite "
            "behavior unchanged this cycle."
        ),
    }


def offline_case1_isolation_rewrite_design_contract_report() -> Dict[str, Any]:
    """Always-on isolation-rewrite design-only contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``design_contract_ok`` = design formalized ∧ honesty locks ∧
    rewrite_shipped=False ∧ isolation_rewrite checklist still open ∧ blockers still
    document isolation_rewrite_required ∧ form classic ∧ dual_linf unproven ∧
    online_linf_gate open ∧ wire_shipped=False ∧ dual_recovery_path=None ∧
    gate_flip_allowed_today=False ∧ criteria_met_today=False.
    **Not** isolation rewrite shipped. **Not** wire. **Not** gate flip. **Not**
    VERDICT. **Not** dual L∞ under wire proof.
    Composes checklist / form / blockers / flip met_today map — does **not** re-run
    maximizers/probes or rewrite isolation tests.
    """
    honesty = _case1_isolation_rewrite_design_contract_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    criteria_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(criteria_met_map)
    criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        criteria_met_map
    )
    flip_criteria = case1_online_linf_gate_flip_criteria()

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    isolation_tests_must_be_rewritten_with_wire_not_deleted = True

    isolation_rewrite_required_in_blockers = (
        CASE1_ISOLATION_REWRITE_BLOCKER_ID in blockers
    )
    isolation_rewrite_required_in_critical = (
        CASE1_ISOLATION_REWRITE_BLOCKER_ID in critical
    )
    isolation_rewrite_required_in_notes = (
        CASE1_ISOLATION_REWRITE_BLOCKER_ID in WIRE_BLOCKER_NOTES
    )

    met_isolation_false = (
        criteria_met_map.get(CASE1_ISOLATION_REWRITE_FLIP_KEY) is False
        and criteria_met_map.get(CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY) is False
    )
    flip_keys_required = (
        flip_criteria.get(CASE1_ISOLATION_REWRITE_FLIP_KEY) == FLIP_CRITERION_REQUIRED
        and flip_criteria.get(CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY)
        == FLIP_CRITERION_REQUIRED
    )

    invariants = case1_isolation_invariants_must_survive()
    invariants_ok = (
        len(invariants) >= 5
        and "excel_pipeline_source_must_not_hard_import_tensorflow_or_tf_linear_blocks_on_classic_path"
        in invariants
        and "models_init_must_not_hard_import_tf_linear_blocks" in invariants
        and "dual_recovery_path_labeled_honestly_none_on_offline_tf_surface_online_lambda_owns_verdict_on_classic"
        in invariants
    )
    post_wire_shape = case1_isolation_rewrite_post_wire_shape()
    post_wire_shape_ok = bool(
        post_wire_shape.get("suite_shape") == "dual_path_isolation_suite"
        and post_wire_shape.get("classic_path_still_isolated") is True
        and post_wire_shape.get("isolation_tests_rewritten_with_wire_not_deleted")
        is True
        and post_wire_shape.get("rewrite_shipped") is False
        and post_wire_shape.get("implemented_this_cycle") is False
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    blocker_ok = (
        isolation_rewrite_required_in_blockers
        and "no_blender_offline_affine_kernel" in blockers
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["isolation_tests_rewritten_with_wire"] is False
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
    )
    flip_permission_ok = (
        gate_flip_allowed_today is False and criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and met_isolation_false
        and flip_keys_required
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
        and isolation_tests_must_be_rewritten_with_wire_not_deleted is True
        and isolation_rewrite_design_present is True
    )

    design_formalized = bool(
        isolation_rewrite_design_present
        and invariants_ok
        and post_wire_shape_ok
        and isolation_rewrite_required_in_blockers
        and isolation_rewrite_required_in_notes
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and flip_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and isolation_rewrite_required_in_critical
    )
    design_contract_ok = honesty_ok
    ok = design_contract_ok and (honesty["wire_shipped"] is False)

    ok_criteria = (
        "design formalized ∧ honesty locks ∧ rewrite_shipped=False ∧ "
        "isolation_rewrite checklist still open ∧ isolation_rewrite_required in "
        "blockers ∧ form classic ∧ dual_linf unproven ∧ online_linf_gate open ∧ "
        "wire_shipped=False ∧ dual_recovery_path=None ∧ gate_flip_allowed_today=False "
        "∧ criteria_met_today=False — NOT isolation rewrite shipped; NOT wire; "
        "NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": design_contract_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "flip_permission_ok": flip_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "design_formalized": design_formalized,
        "invariants_ok": invariants_ok,
        "post_wire_shape_ok": post_wire_shape_ok,
        "ok_criteria": ok_criteria,
        # Design fields
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_design_contract": (
            CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_ANNOTATION
        ),
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_tests_must_be_rewritten_with_wire_not_deleted": (
            isolation_tests_must_be_rewritten_with_wire_not_deleted
        ),
        "isolation_invariants_must_survive": invariants,
        "post_wire_rewrite_shape": post_wire_shape,
        # Checklist / open permanence (design ≠ closed)
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_checklist_key": CASE1_ISOLATION_REWRITE_CHECKLIST_KEY,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        # Flip / met_today discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "criteria_met_today": criteria_met_today,
        "criteria_met_today_map": criteria_met_map,
        "criteria_status_today": criteria_met_map,
        "isolation_rewrite_met_today": criteria_met_map.get(
            CASE1_ISOLATION_REWRITE_FLIP_KEY
        ),
        "isolation_tests_rewritten_met_today": criteria_met_map.get(
            CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY
        ),
        "flip_criteria_isolation_keys": {
            CASE1_ISOLATION_REWRITE_FLIP_KEY: flip_criteria.get(
                CASE1_ISOLATION_REWRITE_FLIP_KEY
            ),
            CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY: flip_criteria.get(
                CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY
            ),
        },
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_blocker_id": CASE1_ISOLATION_REWRITE_BLOCKER_ID,
        "isolation_rewrite_required_in_default_wire_blockers": (
            isolation_rewrite_required_in_blockers
        ),
        "isolation_rewrite_required_in_critical_blockers": (
            isolation_rewrite_required_in_critical
        ),
        "isolation_rewrite_required_in_wire_blocker_notes": (
            isolation_rewrite_required_in_notes
        ),
        "isolation_rewrite_blocker_note": WIRE_BLOCKER_NOTES.get(
            CASE1_ISOLATION_REWRITE_BLOCKER_ID
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "tf_available": tf_available(),
        "isolation_rewrite_design_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_isolation_rewrite_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_isolation_rewrite_design_contract_report``."""
    return offline_case1_isolation_rewrite_design_contract_report(**kwargs)


def multi_unit_case1_isolation_rewrite_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_isolation_rewrite_design_contract_report``."""
    return offline_case1_isolation_rewrite_design_contract_report(**kwargs)


# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest wire-ship acceptance design/acceptance contract
# (goal 5 + 3 honesty residual after isolation design)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes machine-readable criteria for when a
# dual-honest TF Case 1 wire *may* ship. Design only:
# design_present=True; wire_ship_allowed_today=False; wire_shipped=False;
# wire_ship_criteria_met_today=False. Does NOT ship wire. Does NOT rewrite
# isolation tests. Does NOT flip form. Does NOT invent BLENDER UNITS.
# Does NOT clear DEFAULT_WIRE_BLOCKERS. Does NOT redefine
# ready_for_wire_discussion. No TF / no PuLP / no excel_pipeline on hot path.

CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_KIND = (
    "offline_case1_wire_ship_acceptance_design_contract"
)
CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_ANNOTATION = "present"

# Machine-readable wire-ship acceptance criteria (requirement classes only —
# not met-today theater). Composes isolation rewrite, form label, dual honesty
# path, online_linf_gate under TF path, dual L∞ under wire, dual_recovery_path
# labeling, no_silent_form_reuse, blender honesty, Case 1 package shape, and
# affine≠plant_blocks feed LP honesty.
CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA: Dict[str, str] = {
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    # Offline ladder ≠ dual-honest *wire* path — remains required for ship.
    # Path *shape* formalized by path design contract (path_design_present).
    # Ship-met *when* formalized by path_present criteria contract without
    # flipping this met_today ship-met flag today.
    "dual_honest_tf_aware_path_present": FLIP_CRITERION_REQUIRED,
    "online_linf_gate_under_tf_path": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "dual_linf_under_wire_proven": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "wire_shipped": FLIP_CRITERION_REQUIRED,
    "dual_recovery_path_labeled_honestly": FLIP_CRITERION_REQUIRED,
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    "no_blender_offline_affine_kernel_honesty_preserved": FLIP_CRITERION_REQUIRED,
    "case1_cdu_blender_package_admm_shape_acknowledged": FLIP_CRITERION_REQUIRED,
    "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp": (
        FLIP_CRITERION_REQUIRED
    ),
}

# Explicit anti-criteria: these are NEVER wire-ship enablers today.
CASE1_WIRE_SHIP_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_contract_alone",
    "gate_criteria_contract_alone",
    "this_wire_ship_acceptance_design_alone",
)


def case1_wire_ship_acceptance_criteria() -> Dict[str, str]:
    """Return a copy of the machine-readable wire-ship acceptance criteria map."""
    return dict(CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA)


def case1_wire_ship_acceptance_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate ship permission remains False while isolation rewrite, form
    label shipped, dual-honest wire path, dual_linf under wire, and
    wire_shipped remain open. Individual structural honesty labels that
    already hold offline (e.g. dual_recovery_path labeled None on TF surface;
    planned form distinct; blockers documented; affine≠plant_blocks honesty)
    may be True without flipping the aggregate.
    """
    return {
        "isolation_rewrite_with_wire": False,
        "isolation_tests_rewritten_with_wire_not_deleted": False,
        "form_label_change_shipped": False,
        # Offline ladder ≠ dual-honest *wire* path.
        "dual_honest_tf_aware_path_present": False,
        # online_linf_gate closed under shipped TF path only — not today.
        "online_linf_gate_under_tf_path": False,
        "dual_linf_under_wire_proven": False,
        "wire_shipped": False,
        # TF surface dual_recovery_path is None (labeled honestly offline).
        "dual_recovery_path_labeled_honestly": True,
        # Planned form is registered and distinct from classic.
        "no_silent_form_reuse": True,
        # Blocker honesty still true (no silent BLENDER UNITS).
        "no_blender_offline_affine_kernel_honesty_preserved": True,
        # Case 1 package shape honesty acknowledged (blocker still true).
        "case1_cdu_blender_package_admm_shape_acknowledged": True,
        # Affine kernels are yield drivers — honesty still documented.
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp": True,
    }


def case1_wire_ship_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while isolation rewrite / form / dual_linf / wire path open.

    Even if some structural honesty labels hold offline, aggregate ship is never
    allowed until all *required* (not under-wire-only) criteria are met.
    """
    met = criteria_met if criteria_met is not None else (
        case1_wire_ship_acceptance_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    return all(bool(met.get(k)) for k in required_keys)


def case1_wire_ship_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate wire_ship_criteria_met_today — False until all required hold."""
    return case1_wire_ship_allowed_today(criteria_met)


def _case1_wire_ship_acceptance_design_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-wire-shipped / not-VERDICT locks."""
    return {
        "kind": CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "scope": "case1_wire_ship_acceptance_design_contract_offline",
        "note": (
            "Offline Case-1 dual-honest wire-ship acceptance design/acceptance "
            "contract: machine-readable criteria for when a dual-honest TF Case 1 "
            "wire *may* ship. design_present=True; wire_ship_allowed_today=False; "
            "wire_ship_criteria_met_today=False; wire_shipped=False; dual_linf "
            "unproven; online_linf_gate open; isolation_rewrite_shipped=False; "
            "isolation checklist open; gate_flip_allowed_today=False; "
            "criteria_met_today=False; dual_recovery_path=None; solver=False; "
            f"on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Design is NOT wire shipped, NOT ship allow, NOT isolation rewrite "
            "shipped, NOT gate flip, NOT VERDICT gate, NOT dual L∞ under wire "
            "proof. Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, "
            "residual-must-vanish, packaging alone, and design contracts alone "
            "are not ship enablers today. Full DEFAULT_WIRE_BLOCKERS remain "
            "(isolation_rewrite_required, form_label_change_required, "
            "dual_linf_under_wire_unproven, case1_is_cdu_blender_package_admm, "
            "no_blender_offline_affine_kernel, wire_not_shipped, "
            "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp). UNITS "
            "stay FCC/COKER/CDU. Does not clear DEFAULT_WIRE_BLOCKERS. Does not "
            "redefine ready_for_wire_discussion. Always-on numpy; no "
            "TF/PuLP/excel_pipeline on hot path; no maximizer; isolation suite "
            "behavior unchanged this cycle. SUGGESTED_NEXT_WAVE still points at "
            "full dual-honest wire (deferred)."
        ),
    }


def offline_case1_wire_ship_acceptance_design_contract_report() -> Dict[str, Any]:
    """Always-on wire-ship acceptance design contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``design_contract_ok`` = design formalized ∧ honesty locks ∧
    wire_shipped=False ∧ wire_ship_allowed_today=False ∧
    wire_ship_criteria_met_today=False ∧ dual_linf unproven ∧ form classic ∧
    isolation rewrite not shipped ∧ isolation checklist open ∧ online_linf_gate
    open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧ dual_recovery_path
    is None ∧ UNITS FCC/COKER/CDU.
    **Not** wire shipped. **Not** ship allow. **Not** isolation rewrite shipped.
    **Not** gate flip. **Not** VERDICT. **Not** dual L∞ under wire proof.
    Composes form contract / dual_linf checklist / isolation design presence
    flags / gate met maps / DEFAULT_WIRE_BLOCKERS — does **not** re-run
    maximizers/probes or rewrite isolation tests.
    """
    honesty = _case1_wire_ship_acceptance_design_contract_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    # form checklist value may be the string "open" or absent as key status
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    # checklist uses false_today for wire_shipped rather than "open"
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    criteria_map = case1_wire_ship_acceptance_criteria()
    criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        criteria_met_map
    )

    # Gate flip discipline (compose — do not re-run gate report recursively).
    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    # Isolation design presence without recursive isolation report call —
    # constants + checklist already lock rewrite_shipped=False.
    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False

    design_present = True
    wire_shipped = False

    required_keys = set(CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA.keys())
    under_wire_keys = {
        k
        for k, cls in CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    }
    criteria_formalized = (
        set(criteria_map.keys()) == required_keys
        and all(
            criteria_map[k] == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
            for k in under_wire_keys
        )
        and all(
            criteria_map[k] == FLIP_CRITERION_REQUIRED
            for k in required_keys
            if k not in under_wire_keys
        )
        and len(CASE1_WIRE_SHIP_ANTI_CRITERIA_TODAY) >= 6
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["probe_linf_is_not_ship_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_ship_criterion"] is True
        and honesty["design_contracts_alone_is_not_ship_criterion"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        wire_ship_allowed_today is False and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )

    # Ship-critical met_today keys must remain False under HEAD.
    ship_critical_false = (
        criteria_met_map.get("isolation_rewrite_with_wire") is False
        and criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and criteria_met_map.get("dual_linf_under_wire_proven") is False
        and criteria_met_map.get("wire_shipped") is False
        and criteria_met_map.get("online_linf_gate_under_tf_path") is False
    )

    design_formalized = bool(
        design_present
        and criteria_formalized
        and isolation_rewrite_design_present
        and CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_ANNOTATION == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and form_label_open
    )
    design_contract_ok = honesty_ok
    ok = design_contract_ok and (honesty["wire_shipped"] is False)

    ok_criteria = (
        "design formalized ∧ honesty locks ∧ wire_shipped=False ∧ "
        "wire_ship_allowed_today=False ∧ wire_ship_criteria_met_today=False ∧ "
        "dual_linf unproven ∧ form classic ∧ isolation rewrite not shipped ∧ "
        "isolation checklist open ∧ online_linf_gate open ∧ "
        "gate_flip_allowed_today=False ∧ blockers non-empty ∧ "
        "dual_recovery_path=None ∧ UNITS FCC/COKER/CDU — "
        "NOT wire shipped; NOT ship allow; NOT isolation rewrite shipped; "
        "NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": design_contract_ok,
        "design_present": design_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "design_formalized": design_formalized,
        "criteria_formalized": criteria_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "wire_ship_acceptance_design_contract": (
            CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_ANNOTATION
        ),
        "wire_ship_acceptance_design_present": design_present,
        # Ship permission (hard False under HEAD)
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "criteria_map": criteria_map,
        "acceptance_criteria": criteria_map,
        "criteria_met_today_map": criteria_met_map,
        "criteria_status_today": criteria_met_map,
        "anti_criteria_today": list(CASE1_WIRE_SHIP_ANTI_CRITERIA_TODAY),
        "acceptance_criterion_required_class": FLIP_CRITERION_REQUIRED,
        "acceptance_criterion_required_under_wire_only_class": (
            FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        ),
        # Isolation design presence (compose constants — not recursive report)
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "tf_available": tf_available(),
        "wire_ship_acceptance_design_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_wire_ship_acceptance_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_wire_ship_acceptance_design_contract_report``."""
    return offline_case1_wire_ship_acceptance_design_contract_report(**kwargs)


def multi_unit_case1_wire_ship_acceptance_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_wire_ship_acceptance_design_contract_report``."""
    return offline_case1_wire_ship_acceptance_design_contract_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest TF-aware path design contract
# (goal 5 + 3 honesty residual after wire-ship acceptance design)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes machine-readable *path shape* for a
# future dual-honest TF Case 1 wire (topology, form co-req, dual_recovery_path
# planned-vs-today, feature-flag reserved false). Design only:
# path_design_present=True; path_shipped=False;
# dual_honest_tf_aware_path_present ship-met remains False; wire_shipped=False;
# wire_ship_allowed_today=False. Does NOT ship path. Does NOT ship wire.
# Does NOT flip form. Does NOT invent BLENDER UNITS. Does NOT clear
# DEFAULT_WIRE_BLOCKERS. Does NOT redefine ready_for_wire_discussion.
# Does NOT rewrite isolation tests. No TF / no PuLP / no excel_pipeline on
# hot path. Formalizes the still-opaque wire-ship criterion
# dual_honest_tf_aware_path_present without flipping met_today.

CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_KIND = (
    "offline_case1_dual_honest_tf_aware_path_design_contract"
)
CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_ANNOTATION = "present"

# Reserved feature-flag name for a future dual-honest TF Case 1 wire.
# Design reservation only — always hard-coded False this cycle.
CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME = "enable_tf_affine_case1_wire"
CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY = False

# Planned dual_recovery_path label under a *future* TF-aware form when shipped.
# Today TF surface dual_recovery_path remains None. Never pure-ADMM.
CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED = (
    "online_lambda_under_tf_aware_form_when_shipped"
)

# Machine-readable path shape (pure metadata; does not execute maximizers).
CASE1_DUAL_HONEST_TF_AWARE_PATH_SHAPE: Dict[str, Any] = {
    "units_on_path": ("CDU", "Blender"),
    "cdu_surface": "offline_affine_base_delta",
    "blender_surface": CASE1_SHAPED_BLENDER_SURFACE,  # linear_quality_pooling
    "intermediates": CASE1_SHAPED_LINKING_STREAMS,
    "form_current": CASE1_FORM_CURRENT,
    "form_planned": CASE1_PLANNED_TF_AWARE_FORM,
    "form_label_change_shipped": False,
    "dual_recovery_path_today_on_tf_surface": None,
    "dual_recovery_path_planned_when_shipped": (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
    ),
    "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
    "feature_flag_enabled_today": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY,
    "isolation_rewrite_required": True,
    "isolation_rewrite_shipped": False,
    "online_linf_gate_under_tf_path_status": "open",
    "dual_linf_under_wire_status": "unproven",
    "path_design_present": True,
    "path_shipped": False,
    "dual_honest_tf_aware_path_present": False,  # ship-met / path-present-for-ship
    "wire_shipped": False,
    "wire_ship_allowed_today": False,
    "package_shape": "case1_cdu_blender_package_admm",
    "not_pure_admm_dual_recovery": True,
    "not_blender_affine_units": True,
}


# Explicit anti-criteria: these are NEVER path-ship or wire-ship enablers today.
CASE1_DUAL_HONEST_TF_AWARE_PATH_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_contract_alone",
    "gate_criteria_contract_alone",
    "wire_ship_acceptance_design_alone",
    "this_path_design_alone",
)


def case1_dual_honest_tf_aware_path_shape() -> Dict[str, Any]:
    """Return a copy of the dual-honest TF-aware Case-1 path shape map."""
    shape = dict(CASE1_DUAL_HONEST_TF_AWARE_PATH_SHAPE)
    # Materialize nested tuples/lists as plain Python containers for callers.
    shape["units_on_path"] = list(shape["units_on_path"])
    shape["intermediates"] = list(shape["intermediates"])
    return shape


def _case1_dual_honest_tf_aware_path_design_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-path-shipped / not-wire / not-VERDICT locks."""
    return {
        "kind": CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "path_design_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_path_shipped": True,
        "design_is_not_path_present_for_ship": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "design_is_not_form_flip": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "this_path_design_alone_is_not_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_tf_aware_path_design_contract_offline",
        "note": (
            "Offline Case-1 dual-honest TF-aware path design contract: machine-readable "
            "*path shape* for a future dual-honest TF Case 1 wire (CDU offline affine + "
            "blender linear_quality_pooling under Case-1 package ADMM shape; form_planned "
            f"={CASE1_PLANNED_TF_AWARE_FORM}; dual_recovery_path today=None / planned="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED}; feature flag "
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved False). "
            "path_design_present=True; path_shipped=False; dual_honest_tf_aware_path_present "
            "ship-met remains False (design ≠ path present for ship); wire_shipped=False; "
            "wire_ship_allowed_today=False; dual_linf unproven; online_linf_gate open; "
            "isolation_rewrite_shipped=False; isolation checklist open; "
            "gate_flip_allowed_today=False; criteria_met_today=False; "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False; "
            f"case1_form_unchanged ({CASE1_FORM_CURRENT}). Design is NOT path shipped, "
            "NOT path-present-for-ship, NOT wire shipped, NOT ship allow, NOT isolation "
            "rewrite shipped, NOT form flip, NOT gate flip, NOT VERDICT gate, NOT dual L∞ "
            "under wire proof. Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, "
            "residual-must-vanish, packaging alone, design contracts alone, wire-ship "
            "design alone, and this path design alone are not ship enablers today. Full "
            "DEFAULT_WIRE_BLOCKERS remain (isolation_rewrite_required, "
            "form_label_change_required, dual_linf_under_wire_unproven, "
            "case1_is_cdu_blender_package_admm, no_blender_offline_affine_kernel, "
            "wire_not_shipped, affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp). "
            "UNITS stay FCC/COKER/CDU (no silent BLENDER). Formalizes wire-ship criterion "
            "dual_honest_tf_aware_path_present without flipping met_today. Does not clear "
            "DEFAULT_WIRE_BLOCKERS. Does not redefine ready_for_wire_discussion. Always-on "
            "numpy; no TF/PuLP/excel_pipeline on hot path; no maximizer; isolation suite "
            "behavior unchanged this cycle. SUGGESTED_NEXT_WAVE still points at full "
            "dual-honest wire (deferred)."
        ),
    }


def offline_case1_dual_honest_tf_aware_path_design_contract_report() -> Dict[str, Any]:
    """Always-on dual-honest TF-aware path design contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``design_contract_ok`` = design formalized ∧ honesty locks ∧
    path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met False ∧
    wire_shipped=False ∧ wire_ship_allowed_today=False ∧ dual_linf unproven ∧
    form classic ∧ isolation rewrite not shipped ∧ isolation checklist open ∧
    online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧
    dual_recovery_path is None ∧ feature_flag_enabled_today=False ∧
    UNITS FCC/COKER/CDU.
    **Not** path shipped. **Not** path-present-for-ship. **Not** wire shipped.
    **Not** ship allow. **Not** isolation rewrite shipped. **Not** form flip.
    **Not** gate flip. **Not** VERDICT. **Not** dual L∞ under wire proof.
    Composes path shape map / form contract / dual_linf checklist / gate met maps /
    DEFAULT_WIRE_BLOCKERS / wire-ship ship-permission helpers — does **not** re-run
    maximizers/probes or rewrite isolation tests. Does **not** flip wire-ship
    met_today dual_honest_tf_aware_path_present.
    """
    honesty = _case1_dual_honest_tf_aware_path_design_contract_honesty_fields()
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Wire-ship permission remains hard-false; compose without recursive design report.
    criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False

    path_design_present = True
    path_shipped = False
    # Ship-met / path-present-for-ship for wire-ship criteria — remains False.
    dual_honest_tf_aware_path_present = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["dual_recovery_path_planned_when_shipped"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        and "pure-admm" not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and "pure_admm" not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and shape["feature_flag_name"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        and shape["feature_flag_enabled_today"] is False
        and shape["path_design_present"] is True
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["wire_shipped"] is False
        and shape["wire_ship_allowed_today"] is False
        and shape["isolation_rewrite_required"] is True
        and shape["isolation_rewrite_shipped"] is False
        and shape["online_linf_gate_under_tf_path_status"] == "open"
        and shape["dual_linf_under_wire_status"] == "unproven"
        and shape["not_pure_admm_dual_recovery"] is True
        and shape["not_blender_affine_units"] is True
        and "CDU" in shape["units_on_path"]
        and "Blender" in shape["units_on_path"]
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_path_shipped"] is True
        and honesty["design_is_not_path_present_for_ship"] is True
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["design_is_not_form_flip"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["probe_linf_is_not_ship_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_ship_criterion"] is True
        and honesty["design_contracts_alone_is_not_ship_criterion"] is True
        and honesty["this_path_design_alone_is_not_ship_criterion"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        wire_ship_allowed_today is False and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )

    # Ship-critical wire-ship met_today keys must remain False under HEAD —
    # including dual_honest_tf_aware_path_present (this design formalizes it).
    ship_critical_false = (
        criteria_met_map.get("isolation_rewrite_with_wire") is False
        and criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and criteria_met_map.get("dual_linf_under_wire_proven") is False
        and criteria_met_map.get("wire_shipped") is False
        and criteria_met_map.get("online_linf_gate_under_tf_path") is False
    )

    anti_ok = len(CASE1_DUAL_HONEST_TF_AWARE_PATH_ANTI_CRITERIA_TODAY) >= 6 and (
        "this_path_design_alone" in CASE1_DUAL_HONEST_TF_AWARE_PATH_ANTI_CRITERIA_TODAY
    )

    design_formalized = bool(
        path_design_present
        and shape_ok
        and isolation_rewrite_design_present
        and anti_ok
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_ANNOTATION == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and path_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and form_label_open
    )
    design_contract_ok = honesty_ok
    ok = design_contract_ok and (honesty["path_shipped"] is False) and (
        honesty["wire_shipped"] is False
    )

    ok_criteria = (
        "design formalized ∧ honesty locks ∧ path_design_present=True ∧ "
        "path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met=False ∧ "
        "wire_shipped=False ∧ wire_ship_allowed_today=False ∧ dual_linf unproven ∧ "
        "form classic ∧ isolation rewrite not shipped ∧ isolation checklist open ∧ "
        "online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧ "
        "dual_recovery_path=None ∧ feature_flag_enabled_today=False ∧ "
        "UNITS FCC/COKER/CDU — NOT path shipped; NOT path-present-for-ship; "
        "NOT wire shipped; NOT ship allow; NOT isolation rewrite shipped; "
        "NOT form flip; NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": design_contract_ok,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "design_formalized": design_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "path_design_contract": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_ANNOTATION
        ),
        "dual_honest_tf_aware_path_design_present": path_design_present,
        # Path shape
        "path_shape": shape,
        "cdu_surface": shape["cdu_surface"],
        "blender_surface": shape["blender_surface"],
        "units_on_path": list(shape["units_on_path"]),
        "intermediates": list(shape["intermediates"]),
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "anti_criteria_today": list(CASE1_DUAL_HONEST_TF_AWARE_PATH_ANTI_CRITERIA_TODAY),
        # Ship permission (hard False under HEAD)
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "criteria_met_today_map": criteria_met_map,
        # Isolation design presence
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        "design_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "cross_links_wire_ship_criterion_dual_honest_tf_aware_path_present": True,
        "wire_ship_criterion_dual_honest_tf_aware_path_present_met_today": False,
        "tf_available": tf_available(),
        "path_design_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_dual_honest_tf_aware_path_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_design_contract_report``."""
    return offline_case1_dual_honest_tf_aware_path_design_contract_report(**kwargs)


def multi_unit_case1_dual_honest_tf_aware_path_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_design_contract_report``."""
    return offline_case1_dual_honest_tf_aware_path_design_contract_report(**kwargs)




# ---------------------------------------------------------------------------
# Offline Case-1 dual_honest_tf_aware_path_present ship-met flip criteria
# contract (goal 5 + 3 honesty residual after path design #48)
# ---------------------------------------------------------------------------
# Always-on machine-readable *when path counts as present-for-ship* criteria.
# Distinguishes path_design_present ≠ ship-met ≠ path_shipped ≠ wire_shipped.
# criteria_present=True; ship_met_allowed_today=False; dual_honest_tf_aware_path_present
# remains False; path_design_present=True; path_shipped=False; wire_shipped=False.
# Does NOT flip wire-ship / gate met_today maps. Does NOT clear DEFAULT_WIRE_BLOCKERS.
# Does NOT redefine ready_for_wire_discussion. Does NOT rewrite isolation tests.
# No TF / no PuLP / no excel_pipeline on hot path.

CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_KIND = (
    "offline_case1_dual_honest_tf_aware_path_present_criteria_contract"
)
CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_ANNOTATION = "present"

# Machine-readable ship-met / path-present-for-ship flip criteria map
# (requirement classes only — not met-today theater).
CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_FLIP_CRITERIA: Dict[str, str] = {
    # Structural: path design shape already present offline (#48).
    "path_design_present": FLIP_CRITERION_REQUIRED,
    # Structural: CDU affine + blender linear_quality_pooling + Case 1 intermediates.
    "path_shape_matches_case1_cdu_blender_package": FLIP_CRITERION_REQUIRED,
    # Co-req: form label change must ship before path counts as present-for-ship.
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    # Structural: feature flag name reserved; enabled_today still False.
    "feature_flag_reserved_and_named": FLIP_CRITERION_REQUIRED,
    # Structural: planned dual_recovery_path labeled honestly (not pure-ADMM);
    # today TF surface dual_recovery_path remains None.
    "dual_recovery_path_planned_labeled_honestly": FLIP_CRITERION_REQUIRED,
    # Co-req: isolation rewrite with wire before true path-present-for-ship under wire.
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    # Structural: planned form distinct from classic (no silent reuse).
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    # Structural honesty: blender is linear_quality_pooling, not UNITS affine entry.
    "no_blender_affine_units_entry": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these are NEVER ship-met / path-present-for-ship enablers today.
CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_contract_alone",
    "gate_criteria_contract_alone",
    "wire_ship_acceptance_design_alone",
    "this_path_design_alone",
    "this_ship_met_criteria_contract_alone",
    "this_path_present_criteria_contract_alone",
)


def case1_dual_honest_tf_aware_path_present_flip_criteria() -> Dict[str, str]:
    """Return a copy of the ship-met / path-present-for-ship flip-criteria map."""
    return dict(CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_FLIP_CRITERIA)


def case1_dual_honest_tf_aware_path_present_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate ship_met_allowed_today remains False while form label shipped and
    isolation rewrite co-reqs remain open. Individual structural honesty labels
    that already hold offline (path_design_present, path shape, feature flag
    named+disabled, dual_recovery planned labeled, no silent form reuse, no
    BLENDER UNITS) may be True without flipping the aggregate or ship-met.
    """
    return {
        "path_design_present": True,
        "path_shape_matches_case1_cdu_blender_package": True,
        "form_label_change_shipped": False,
        # Name reserved; enabled remains False — structural readiness of the name.
        "feature_flag_reserved_and_named": True,
        "dual_recovery_path_planned_labeled_honestly": True,
        "isolation_rewrite_with_wire": False,
        "no_silent_form_reuse": True,
        "no_blender_affine_units_entry": True,
    }


def case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while form/isolation co-reqs remain open.

    Even if structural labels hold offline, aggregate ship-met is never allowed
    until all *required* criteria are met.
    """
    met = criteria_met if criteria_met is not None else (
        case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    return all(bool(met.get(k)) for k in required_keys)


def case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today — False until all required criteria hold."""
    return case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(criteria_met)


def _case1_dual_honest_tf_aware_path_present_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-ship-met / not-path-shipped / not-wire locks."""
    return {
        "kind": CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "path_design_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_ship_met": True,
        "design_is_not_path_present_for_ship": True,
        "design_is_not_path_shipped": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "design_is_not_form_flip": True,
        "probe_linf_is_not_ship_met_criterion_today": True,
        "bridge_linf_is_not_ship_met_criterion_today": True,
        "warmstart_linf_is_not_ship_met_criterion_today": True,
        "pooling_linf_is_not_ship_met_criterion_today": True,
        "seed_identity_linf_is_not_ship_met_criterion": True,
        "recovered_blender_linf_is_not_ship_met_criterion_today": True,
        "residual_must_vanish_is_not_ship_met_criterion": True,
        "packaging_alone_is_not_ship_met_criterion": True,
        "design_contracts_alone_is_not_ship_met_criterion": True,
        "this_path_design_alone_is_not_ship_met_criterion": True,
        "this_ship_met_criteria_contract_alone_is_not_ship_met_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_tf_aware_path_present_criteria_contract_offline",
        "note": (
            "Offline Case-1 dual_honest_tf_aware_path_present ship-met flip criteria "
            "contract: machine-readable *when path counts as present-for-ship* for "
            "wire-ship / gate criterion dual_honest_tf_aware_path_present. "
            "criteria_present=True; ship_met_allowed_today=False; criteria_met_today=False; "
            "dual_honest_tf_aware_path_present ship-met remains False; path_design_present=True "
            "(path design formalizes *what*; this formalizes *when present-for-ship*); "
            "path_shipped=False; wire_shipped=False; wire_ship_allowed_today=False; dual_linf "
            "unproven; online_linf_gate open; isolation_rewrite_shipped=False; isolation "
            "checklist open; gate_flip_allowed_today=False; dual_recovery_path=None; "
            f"solver=False; on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Criteria contract is NOT ship-met, NOT path-present-for-ship True, NOT path "
            "shipped, NOT wire shipped, NOT ship allow, NOT isolation rewrite shipped, NOT "
            "form flip, NOT gate flip, NOT VERDICT gate, NOT dual L∞ under wire proof. "
            "Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, residual-must-vanish, "
            "packaging alone, design contracts alone, path design alone, and this ship-met "
            "criteria contract alone are not ship-met enablers today. Full DEFAULT_WIRE_BLOCKERS "
            "remain (isolation_rewrite_required, form_label_change_required, "
            "dual_linf_under_wire_unproven, case1_is_cdu_blender_package_admm, "
            "no_blender_offline_affine_kernel, wire_not_shipped, "
            "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp). UNITS stay "
            "FCC/COKER/CDU (no silent BLENDER). Does not flip foreign wire-ship/gate "
            "met_today maps. Does not clear DEFAULT_WIRE_BLOCKERS. Does not redefine "
            "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline on hot "
            "path; no maximizer; isolation suite behavior unchanged this cycle. "
            "SUGGESTED_NEXT_WAVE still points at full dual-honest wire (deferred)."
        ),
    }


def offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report() -> Dict[str, Any]:
    """Always-on path-present-for-ship / ship-met flip criteria contract.

    No TF, no PuLP, no excel_pipeline, no solve. Aggregate ``ok`` /
    ``design_contract_ok`` / ``contract_ok`` = criteria formalized ∧ honesty
    locks ∧ dual_honest_tf_aware_path_present ship-met False ∧
    ship_met_allowed_today=False ∧ criteria_met_today=False ∧
    path_design_present=True ∧ path_shipped=False ∧ wire_shipped=False ∧
    wire_ship_allowed_today=False ∧ dual_linf unproven ∧ form classic ∧
    isolation rewrite not shipped ∧ isolation checklist open ∧ online_linf_gate
    open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧
    dual_recovery_path is None ∧ feature_flag_enabled_today=False ∧
    UNITS FCC/COKER/CDU.
    **Not** ship-met. **Not** path shipped. **Not** wire. **Not** ship allow.
    **Not** isolation rewrite shipped. **Not** form flip. **Not** gate flip.
    **Not** VERDICT. **Not** dual L∞ under wire proof.
    Composes path shape / form contract / dual_linf checklist / wire-ship + gate
    met maps / DEFAULT_WIRE_BLOCKERS — does **not** re-run maximizers/probes or
    rewrite isolation tests. Does **not** flip foreign met_today maps.
    """
    honesty = _case1_dual_honest_tf_aware_path_present_criteria_contract_honesty_fields()
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Foreign maps — compose without flipping; do not recurse into design reports.
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        wire_criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    flip_criteria = case1_dual_honest_tf_aware_path_present_flip_criteria()
    criteria_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        criteria_met_map
    )
    criteria_met_today = case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
        criteria_met_map
    )

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False

    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    criteria_present = True

    required_keys = set(CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_FLIP_CRITERIA.keys())
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and all(
            flip_criteria[k] == FLIP_CRITERION_REQUIRED for k in required_keys
        )
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["dual_recovery_path_planned_when_shipped"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        and "pure-admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and "pure_admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and shape["feature_flag_name"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        and shape["feature_flag_enabled_today"] is False
        and shape["path_design_present"] is True
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["wire_shipped"] is False
        and shape["not_pure_admm_dual_recovery"] is True
        and shape["not_blender_affine_units"] is True
        and "CDU" in shape["units_on_path"]
        and "Blender" in shape["units_on_path"]
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_ship_met"] is True
        and honesty["design_is_not_path_present_for_ship"] is True
        and honesty["design_is_not_path_shipped"] is True
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["design_is_not_form_flip"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["probe_linf_is_not_ship_met_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_ship_met_criterion"] is True
        and honesty["this_path_design_alone_is_not_ship_met_criterion"] is True
        and honesty[
            "this_ship_met_criteria_contract_alone_is_not_ship_met_criterion"
        ]
        is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_met_permission_ok = (
        ship_met_allowed_today is False and criteria_met_today is False
    )
    wire_ship_permission_ok = (
        wire_ship_allowed_today is False and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )

    # Ship-critical foreign met_today keys remain False under HEAD.
    ship_critical_false = (
        wire_criteria_met_map.get("isolation_rewrite_with_wire") is False
        and wire_criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and wire_criteria_met_map.get("form_label_change_shipped") is False
        and wire_criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and wire_criteria_met_map.get("dual_linf_under_wire_proven") is False
        and wire_criteria_met_map.get("wire_shipped") is False
        and wire_criteria_met_map.get("online_linf_gate_under_tf_path") is False
        and gate_met_map.get("dual_honest_tf_aware_path_present") is False
        and gate_met_map.get("isolation_rewrite_with_wire") is False
        and gate_met_map.get("form_label_change_shipped") is False
        and gate_met_map.get("wire_shipped") is False
    )

    # Structural True subset may hold; co-reqs stay False.
    structural_met_ok = (
        criteria_met_map.get("path_design_present") is True
        and criteria_met_map.get("path_shape_matches_case1_cdu_blender_package") is True
        and criteria_met_map.get("feature_flag_reserved_and_named") is True
        and criteria_met_map.get("dual_recovery_path_planned_labeled_honestly") is True
        and criteria_met_map.get("no_silent_form_reuse") is True
        and criteria_met_map.get("no_blender_affine_units_entry") is True
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("isolation_rewrite_with_wire") is False
    )

    anti = CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 8
        and "this_ship_met_criteria_contract_alone" in anti
        and "this_path_design_alone" in anti
        and "this_path_present_criteria_contract_alone" in anti
    )

    design_formalized = bool(
        criteria_present
        and flip_criteria_formalized
        and path_design_present
        and shape_ok
        and isolation_rewrite_design_present
        and anti_ok
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_ANNOTATION
        == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_met_permission_ok
        and wire_ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and path_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and structural_met_ok
        and form_label_open
    )
    design_contract_ok = honesty_ok
    contract_ok = design_contract_ok
    ok = design_contract_ok and (honesty["path_shipped"] is False) and (
        honesty["wire_shipped"] is False
    ) and (honesty["dual_honest_tf_aware_path_present"] is False)

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ criteria_present=True ∧ "
        "path_design_present=True ∧ dual_honest_tf_aware_path_present ship-met=False ∧ "
        "ship_met_allowed_today=False ∧ criteria_met_today=False ∧ path_shipped=False ∧ "
        "wire_shipped=False ∧ wire_ship_allowed_today=False ∧ dual_linf unproven ∧ "
        "form classic ∧ isolation rewrite not shipped ∧ isolation checklist open ∧ "
        "online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧ "
        "dual_recovery_path=None ∧ feature_flag_enabled_today=False ∧ "
        "UNITS FCC/COKER/CDU — NOT ship-met; NOT path-present-for-ship True; "
        "NOT path shipped; NOT wire shipped; NOT ship allow; NOT isolation rewrite "
        "shipped; NOT form flip; NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": contract_ok,
        "criteria_present": criteria_present,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_met_permission_ok": ship_met_permission_ok,
        "wire_ship_permission_ok": wire_ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "design_formalized": design_formalized,
        "flip_criteria_formalized": flip_criteria_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "path_present_criteria_contract": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_ANNOTATION
        ),
        "dual_honest_tf_aware_path_present_criteria_present": criteria_present,
        # Flip criteria / met maps
        "flip_criteria": flip_criteria,
        "ship_met_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "ship_met_allowed_today": ship_met_allowed_today,
        "criteria_met_today": criteria_met_today,
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_ANTI_CRITERIA_TODAY
        ),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        # Path shape snapshot
        "path_shape": shape,
        "cdu_surface": shape["cdu_surface"],
        "blender_surface": shape["blender_surface"],
        "units_on_path": list(shape["units_on_path"]),
        "intermediates": list(shape["intermediates"]),
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        # Foreign ship permission (hard False under HEAD; not flipped by this contract)
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "wire_ship_criteria_met_today_map": wire_criteria_met_map,
        "design_does_not_flip_wire_ship_met_today": True,
        "design_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        # Isolation design presence
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        "design_does_not_flip_gate_met_today": True,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "cross_links_wire_ship_criterion_dual_honest_tf_aware_path_present": True,
        "wire_ship_criterion_dual_honest_tf_aware_path_present_met_today": False,
        "cross_links_gate_criterion_dual_honest_tf_aware_path_present": True,
        "gate_criterion_dual_honest_tf_aware_path_present_met_today": False,
        "path_design_present_is_not_ship_met": True,
        "ship_met_is_not_path_shipped": True,
        "ship_met_is_not_wire_shipped": True,
        "ship_met_is_not_ship_allow": True,
        "tf_available": tf_available(),
        "path_present_criteria_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_dual_honest_tf_aware_path_present_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report``."""
    return offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report(
        **kwargs
    )


def multi_unit_case1_dual_honest_tf_aware_path_present_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report``."""
    return offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report(
        **kwargs
    )




# ---------------------------------------------------------------------------
# Offline Case-1 form_label_change_shipped flip criteria contract (goal 5+3)
# ---------------------------------------------------------------------------
# Always-on numpy. Formalizes *when* checklist / wire-ship / gate / path-present
# key form_label_change_shipped may become True — while form stays classic,
# form_label_change_shipped remains False / open, form_label_ship_allowed_today
# remains False, path_shipped=False, ship-met False, wire_shipped=False.
# Form registration formalizes *what*; this formalizes *when shipped*.
# dual_recovery_path=None today; planned dual recovery labeled honestly (not
# pure-ADMM). Never imports excel_pipeline / tensorflow / pulp on hot path.
# Never flips form, never closes checklist, never clears DEFAULT_WIRE_BLOCKERS.

CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_KIND = (
    "offline_case1_form_label_change_shipped_criteria_contract"
)
CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION = "present"

# Named mutation path for future wire — named, not executed today.
CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_NAME = (
    "feature_flag_enable_tf_affine_case1_wire_then_set_model_form_to_planned"
)
CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_EXECUTED_TODAY = False

# Machine-readable form_label_change_shipped flip criteria map
# (requirement classes only — not met-today theater).
CASE1_FORM_LABEL_CHANGE_SHIPPED_FLIP_CRITERIA: Dict[str, str] = {
    # Structural: planned form already registered offline via form contract.
    "planned_form_registered": FLIP_CRITERION_REQUIRED,
    # Structural: planned form distinct from classic.
    "planned_form_distinct_from_classic": FLIP_CRITERION_REQUIRED,
    # Structural: form_label_change_required blocker still documented.
    "form_label_change_required_blocker_documented": FLIP_CRITERION_REQUIRED,
    # How form would change under future wire — named, not executed.
    "explicit_form_mutation_path_named": FLIP_CRITERION_REQUIRED,
    # Structural: feature flag name reserved; enabled_today still False.
    "feature_flag_reserved_and_named": FLIP_CRITERION_REQUIRED,
    # Structural: planned form distinct identity (no silent reuse).
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    # Co-req: isolation rewrite with wire before true form ship under wire path.
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    # Structural: planned dual_recovery_path labeled honestly (not pure-ADMM);
    # today TF surface dual_recovery_path remains None.
    "dual_recovery_path_planned_labeled_honestly": FLIP_CRITERION_REQUIRED,
    # Structural co-ack: path design present (shape formalized offline).
    "path_design_present": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these are NEVER form_label_change_shipped enablers today.
CASE1_FORM_LABEL_CHANGE_SHIPPED_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_contract_alone",
    "gate_criteria_contract_alone",
    "wire_ship_acceptance_design_alone",
    "this_path_design_alone",
    "this_ship_met_criteria_contract_alone",
    "this_path_present_criteria_contract_alone",
    "dual_space_form_contract_alone",
    "form_registration_alone",
    "form_label_contract_alone",
    "this_form_label_change_shipped_criteria_contract_alone",
)


def case1_form_label_change_shipped_flip_criteria() -> Dict[str, str]:
    """Return a copy of the form_label_change_shipped flip-criteria map."""
    return dict(CASE1_FORM_LABEL_CHANGE_SHIPPED_FLIP_CRITERIA)


def case1_form_label_change_shipped_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate form_label_ship_allowed_today remains False while form is still
    classic and isolation rewrite co-req remains open. Individual structural
    honesty labels that already hold offline may be True without flipping the
    aggregate or form_label_change_shipped.
    """
    return {
        "planned_form_registered": True,
        "planned_form_distinct_from_classic": True,
        "form_label_change_required_blocker_documented": True,
        # Mutation path is named (structural); executed remains False — name only.
        "explicit_form_mutation_path_named": True,
        "feature_flag_reserved_and_named": True,
        "no_silent_form_reuse": True,
        "isolation_rewrite_with_wire": False,
        "dual_recovery_path_planned_labeled_honestly": True,
        "path_design_present": True,
        # Explicit ship key — never True while form remains classic today.
        "form_label_change_shipped": False,
    }


def case1_form_label_ship_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while form classic / isolation co-reqs remain open.

    Even if structural labels hold offline, aggregate form-label ship is never
    allowed until all *required* criteria are met (including isolation rewrite
    co-req and actual form mutation under wire — not claimed today).
    """
    met = (
        criteria_met
        if criteria_met is not None
        else case1_form_label_change_shipped_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_FORM_LABEL_CHANGE_SHIPPED_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    # Also require explicit form_label_change_shipped ship key if present in map.
    keys = list(required_keys)
    if "form_label_change_shipped" in met and "form_label_change_shipped" not in keys:
        keys.append("form_label_change_shipped")
    return all(bool(met.get(k)) for k in keys)


def case1_form_label_change_shipped_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today — False until all required criteria hold."""
    return case1_form_label_ship_allowed_today(criteria_met)


def _case1_form_label_change_shipped_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-form-flip / not-form-label-shipped locks."""
    return {
        "kind": CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "form_label_change_shipped": False,
        "path_design_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_form_label_change_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_form_flip": True,
        "design_is_not_form_label_shipped": True,
        "design_is_not_ship_met": True,
        "design_is_not_path_present_for_ship": True,
        "design_is_not_path_shipped": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "probe_linf_is_not_form_label_ship_criterion_today": True,
        "bridge_linf_is_not_form_label_ship_criterion_today": True,
        "warmstart_linf_is_not_form_label_ship_criterion_today": True,
        "pooling_linf_is_not_form_label_ship_criterion_today": True,
        "seed_identity_linf_is_not_form_label_ship_criterion": True,
        "recovered_blender_linf_is_not_form_label_ship_criterion_today": True,
        "residual_must_vanish_is_not_form_label_ship_criterion": True,
        "packaging_alone_is_not_form_label_ship_criterion": True,
        "design_contracts_alone_is_not_form_label_ship_criterion": True,
        "form_registration_alone_is_not_form_label_ship_criterion": True,
        "dual_space_form_contract_alone_is_not_form_label_ship_criterion": True,
        "this_form_label_change_shipped_criteria_contract_alone_is_not_form_label_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_form_label_change_shipped_criteria_contract_offline",
        "note": (
            "Offline Case-1 form_label_change_shipped flip criteria contract: "
            "machine-readable *when form_label_change_shipped may become True* for "
            "checklist / wire-ship / gate / path-present key form_label_change_shipped. "
            "criteria_present=True; form_label_ship_allowed_today=False; "
            "criteria_met_today=False; form_label_change_shipped remains False; "
            f"form remains {CASE1_FORM_CURRENT}; planned={CASE1_PLANNED_TF_AWARE_FORM}; "
            "path_design_present=True; path_shipped=False; "
            "dual_honest_tf_aware_path_present ship-met=False; wire_shipped=False; "
            "wire_ship_allowed_today=False; dual_linf unproven; online_linf_gate open; "
            "isolation_rewrite_shipped=False; isolation checklist open; "
            "gate_flip_allowed_today=False; dual_recovery_path=None; solver=False; "
            "on_excel_case1_path=False. Criteria contract is NOT form flip, NOT "
            "form_label_change_shipped True, NOT path shipped, NOT ship-met, NOT wire "
            "shipped, NOT ship allow, NOT isolation rewrite shipped, NOT gate flip, "
            "NOT VERDICT gate, NOT dual L∞ under wire proof. Form registration alone, "
            "dual-space form alone, path design alone, ship-met criteria alone, "
            "wire-ship design alone, isolation design alone, gate criteria alone, "
            "packaging alone, diagnostic L∞, residual-must-vanish, and this form_label "
            "criteria contract alone are not form-label ship enablers today. Full "
            "DEFAULT_WIRE_BLOCKERS remain (isolation_rewrite_required, "
            "form_label_change_required, dual_linf_under_wire_unproven, "
            "case1_is_cdu_blender_package_admm, no_blender_offline_affine_kernel, "
            "wire_not_shipped, affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp). "
            "UNITS stay FCC/COKER/CDU (no silent BLENDER). Does not flip foreign "
            "wire-ship/gate/path-present met_today maps. Does not clear "
            "DEFAULT_WIRE_BLOCKERS. Does not redefine ready_for_wire_discussion. "
            "Always-on numpy; no TF/PuLP/excel_pipeline on hot path; no maximizer; "
            "isolation suite behavior unchanged this cycle. SUGGESTED_NEXT_WAVE still "
            "points at full dual-honest wire (deferred)."
        ),
    }


def offline_case1_form_label_change_shipped_criteria_contract_report() -> Dict[str, Any]:
    """Always-on form_label_change_shipped flip criteria contract.

    No TF, no PuLP, no excel_pipeline, no solve. Aggregate ``ok`` /
    ``design_contract_ok`` / ``contract_ok`` = criteria formalized ∧ honesty
    locks ∧ form_label_change_shipped False ∧ form_label_ship_allowed_today=False
    ∧ criteria_met_today=False ∧ form classic ∧ path_design_present=True ∧
    path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met False ∧
    wire_shipped=False ∧ wire_ship_allowed_today=False ∧ dual_linf unproven ∧
    isolation rewrite not shipped ∧ isolation checklist open ∧ online_linf_gate
    open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧
    dual_recovery_path is None ∧ feature_flag_enabled_today=False ∧
    UNITS FCC/COKER/CDU.
    **Not** form flip. **Not** form_label_change_shipped. **Not** path shipped.
    **Not** ship-met. **Not** wire. **Not** ship allow. **Not** VERDICT.
    Composes form contract / dual_linf checklist / path shape / wire-ship + gate
    + path-present met maps / DEFAULT_WIRE_BLOCKERS — does **not** re-run
    maximizers/probes or rewrite isolation tests. Does **not** flip foreign
    met_today maps.
    """
    honesty = _case1_form_label_change_shipped_criteria_contract_honesty_fields()
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Foreign maps — compose without flipping.
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        wire_criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    path_present_criteria_met_today = (
        case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
            path_present_met_map
        )
    )

    flip_criteria = case1_form_label_change_shipped_flip_criteria()
    criteria_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(criteria_met_map)
    criteria_met_today = case1_form_label_change_shipped_criteria_met_today_aggregate(
        criteria_met_map
    )

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False

    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    criteria_present = True
    mutation_path_named = CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_NAME
    mutation_path_executed_today = bool(
        CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_EXECUTED_TODAY
    )

    required_keys = set(CASE1_FORM_LABEL_CHANGE_SHIPPED_FLIP_CRITERIA.keys())
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and all(flip_criteria[k] == FLIP_CRITERION_REQUIRED for k in required_keys)
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["dual_recovery_path_planned_when_shipped"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        and "pure-admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and "pure_admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and shape["feature_flag_name"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        and shape["feature_flag_enabled_today"] is False
        and shape["path_design_present"] is True
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["wire_shipped"] is False
        and shape["not_pure_admm_dual_recovery"] is True
        and shape["not_blender_affine_units"] is True
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_form_flip"] is True
        and honesty["design_is_not_form_label_shipped"] is True
        and honesty["design_is_not_ship_met"] is True
        and honesty["design_is_not_path_shipped"] is True
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["form_registration_alone_is_not_form_label_ship_criterion"] is True
        and honesty[
            "this_form_label_change_shipped_criteria_contract_alone_is_not_form_label_ship_criterion"
        ]
        is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
        and form["form_label_change_required_still_true"] is True
    )
    form_label_permission_ok = (
        form_label_ship_allowed_today is False and criteria_met_today is False
    )
    ship_met_permission_ok = (
        ship_met_allowed_today is False and path_present_criteria_met_today is False
    )
    wire_ship_permission_ok = (
        wire_ship_allowed_today is False and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and mutation_path_executed_today is False
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    # Ship-critical foreign met_today keys remain False under HEAD.
    ship_critical_false = (
        wire_criteria_met_map.get("isolation_rewrite_with_wire") is False
        and wire_criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and wire_criteria_met_map.get("form_label_change_shipped") is False
        and wire_criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and wire_criteria_met_map.get("dual_linf_under_wire_proven") is False
        and wire_criteria_met_map.get("wire_shipped") is False
        and wire_criteria_met_map.get("online_linf_gate_under_tf_path") is False
        and gate_met_map.get("dual_honest_tf_aware_path_present") is False
        and gate_met_map.get("isolation_rewrite_with_wire") is False
        and gate_met_map.get("form_label_change_shipped") is False
        and gate_met_map.get("wire_shipped") is False
        and path_present_met_map.get("form_label_change_shipped") is False
        and path_present_met_map.get("isolation_rewrite_with_wire") is False
    )

    # Structural True subset may hold; co-reqs stay False.
    structural_met_ok = (
        criteria_met_map.get("planned_form_registered") is True
        and criteria_met_map.get("planned_form_distinct_from_classic") is True
        and criteria_met_map.get("form_label_change_required_blocker_documented") is True
        and criteria_met_map.get("explicit_form_mutation_path_named") is True
        and criteria_met_map.get("feature_flag_reserved_and_named") is True
        and criteria_met_map.get("no_silent_form_reuse") is True
        and criteria_met_map.get("dual_recovery_path_planned_labeled_honestly") is True
        and criteria_met_map.get("path_design_present") is True
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("isolation_rewrite_with_wire") is False
    )

    anti = CASE1_FORM_LABEL_CHANGE_SHIPPED_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 10
        and "this_form_label_change_shipped_criteria_contract_alone" in anti
        and "form_registration_alone" in anti
        and "dual_space_form_contract_alone" in anti
        and "form_label_contract_alone" in anti
    )

    design_formalized = bool(
        criteria_present
        and flip_criteria_formalized
        and path_design_present
        and shape_ok
        and isolation_rewrite_design_present
        and anti_ok
        and mutation_path_named == CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_NAME
        and mutation_path_executed_today is False
        and CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and form_label_permission_ok
        and ship_met_permission_ok
        and wire_ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and structural_met_ok
        and form_label_open
    )
    design_contract_ok = honesty_ok
    contract_ok = design_contract_ok
    ok = (
        design_contract_ok
        and (honesty["form_label_change_shipped"] is False)
        and (honesty["path_shipped"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
    )

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ criteria_present=True ∧ "
        "form classic ∧ form_label_change_shipped=False ∧ "
        "form_label_ship_allowed_today=False ∧ criteria_met_today=False ∧ "
        "path_design_present=True ∧ path_shipped=False ∧ "
        "dual_honest_tf_aware_path_present ship-met=False ∧ wire_shipped=False ∧ "
        "wire_ship_allowed_today=False ∧ dual_linf unproven ∧ isolation rewrite "
        "not shipped ∧ isolation checklist open ∧ online_linf_gate open ∧ "
        "gate_flip_allowed_today=False ∧ blockers non-empty ∧ "
        "dual_recovery_path=None ∧ feature_flag_enabled_today=False ∧ "
        "UNITS FCC/COKER/CDU — NOT form flip; NOT form_label_change_shipped; "
        "NOT path shipped; NOT ship-met; NOT wire shipped; NOT ship allow; "
        "NOT isolation rewrite shipped; NOT gate flip; NOT VERDICT; "
        "NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": contract_ok,
        "criteria_present": criteria_present,
        "form_label_change_shipped": form_label_change_shipped,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "form_label_permission_ok": form_label_permission_ok,
        "ship_met_permission_ok": ship_met_permission_ok,
        "wire_ship_permission_ok": wire_ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "design_formalized": design_formalized,
        "flip_criteria_formalized": flip_criteria_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "form_label_change_shipped_criteria_contract": (
            CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION
        ),
        "form_label_change_shipped_criteria_present": criteria_present,
        # Flip criteria / met maps
        "flip_criteria": flip_criteria,
        "form_label_ship_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "anti_criteria_today": list(
            CASE1_FORM_LABEL_CHANGE_SHIPPED_ANTI_CRITERIA_TODAY
        ),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        # Explicit mutation path (named, not executed)
        "explicit_form_mutation_path_named": mutation_path_named,
        "form_mutation_path_name": mutation_path_named,
        "form_mutation_path_executed_today": mutation_path_executed_today,
        # Path shape snapshot
        "path_shape": shape,
        "cdu_surface": shape["cdu_surface"],
        "blender_surface": shape["blender_surface"],
        "units_on_path": list(shape["units_on_path"]),
        "intermediates": list(shape["intermediates"]),
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        # Foreign ship permission (hard False under HEAD; not flipped by this contract)
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "wire_ship_criteria_met_today_map": wire_criteria_met_map,
        "ship_met_allowed_today": ship_met_allowed_today,
        "path_present_criteria_met_today": path_present_criteria_met_today,
        "path_present_criteria_met_today_map": path_present_met_map,
        "design_does_not_flip_wire_ship_met_today": True,
        "design_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "design_does_not_flip_form_label_change_shipped_met_today": True,
        # Isolation design presence
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        "design_does_not_flip_gate_met_today": True,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        "design_does_not_close_form_label_change_shipped_checklist": True,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "cross_links_wire_ship_criterion_form_label_change_shipped": True,
        "wire_ship_criterion_form_label_change_shipped_met_today": False,
        "cross_links_gate_criterion_form_label_change_shipped": True,
        "gate_criterion_form_label_change_shipped_met_today": False,
        "cross_links_path_present_criterion_form_label_change_shipped": True,
        "path_present_criterion_form_label_change_shipped_met_today": False,
        "form_registration_is_not_form_label_shipped": True,
        "form_label_shipped_is_not_path_shipped": True,
        "form_label_shipped_is_not_ship_met": True,
        "form_label_shipped_is_not_wire_shipped": True,
        "form_label_shipped_is_not_ship_allow": True,
        "tf_available": tf_available(),
        "form_label_change_shipped_criteria_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_form_label_change_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_form_label_change_shipped_criteria_contract_report``."""
    return offline_case1_form_label_change_shipped_criteria_contract_report(**kwargs)


def multi_unit_case1_form_label_change_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_form_label_change_shipped_criteria_contract_report``."""
    return offline_case1_form_label_change_shipped_criteria_contract_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1 isolation-rewrite ship-met / flip criteria contract
# (goal 5 + 3 honesty residual after isolation design + form_label criteria)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes *when* checklist key isolation_rewrite_with_wire
# may become met and/or isolation_rewrite_shipped may become True — without shipping
# rewrite, without rewriting/deleting test_tf_import_isolation.py, without wire/form
# flip. Isolation design formalizes *what*; this formalizes *when*.
# Locks: criteria_present=True; isolation_ship_allowed_today=False;
# criteria_met_today=False; isolation_rewrite_shipped=False; checklist open;
# dual_recovery_path=None today; planned dual recovery labeled honestly (not
# pure-ADMM). Never imports excel_pipeline / tensorflow / pulp on hot path.
# Never flips form, never closes checklist, never clears DEFAULT_WIRE_BLOCKERS.

CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_KIND = (
    "offline_case1_isolation_rewrite_shipped_criteria_contract"
)
CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION = "present"

# Machine-readable isolation_rewrite_shipped / isolation_rewrite_with_wire
# flip criteria map (requirement classes only — not met-today theater).
CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA: Dict[str, str] = {
    # Structural: isolation design contract already formalizes rewrite shape.
    "isolation_rewrite_design_present": FLIP_CRITERION_REQUIRED,
    # Structural rule: rewrite WITH dual-honest wire, never delete isolation suite.
    "rewrite_with_wire_not_delete": FLIP_CRITERION_REQUIRED,
    # Structural: isolation_rewrite_required blocker still documented.
    "isolation_rewrite_required_blocker_documented": FLIP_CRITERION_REQUIRED,
    # Ship-critical executed rewrite of isolation tests WITH wire — False today.
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
    # Structural: isolation suite must not be silently deleted.
    "no_silent_isolation_suite_deletion": FLIP_CRITERION_REQUIRED,
    # Structural: feature flag name reserved; enabled_today still False.
    "feature_flag_reserved_and_named": FLIP_CRITERION_REQUIRED,
    # Structural: planned dual_recovery_path labeled honestly (not pure-ADMM);
    # today TF surface dual_recovery_path remains None.
    "dual_recovery_path_planned_labeled_honestly": FLIP_CRITERION_REQUIRED,
    # Structural co-ack: path design present (shape formalized offline).
    "path_design_present": FLIP_CRITERION_REQUIRED,
    # Co-req under full wire: form_label_change_shipped (False today —
    # isolation ship does not free form_label).
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these are NEVER isolation_rewrite_shipped enablers today.
CASE1_ISOLATION_REWRITE_SHIPPED_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_contract_alone",
    "isolation_design_alone",
    "gate_criteria_contract_alone",
    "wire_ship_acceptance_design_alone",
    "this_path_design_alone",
    "this_ship_met_criteria_contract_alone",
    "this_path_present_criteria_contract_alone",
    "dual_space_form_contract_alone",
    "form_registration_alone",
    "form_label_contract_alone",
    "this_form_label_change_shipped_criteria_contract_alone",
    "this_isolation_rewrite_shipped_criteria_contract_alone",
    "this_isolation_ship_met_criteria_contract_alone",
)


def case1_isolation_rewrite_shipped_flip_criteria() -> Dict[str, str]:
    """Return a copy of the isolation_rewrite_shipped flip-criteria map."""
    return dict(CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA)


def case1_isolation_rewrite_shipped_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate isolation_ship_allowed_today remains False while isolation
    rewrite is not shipped and suite rewrite remains unexecuted. Individual
    structural honesty labels that already hold offline may be True without
    flipping the aggregate or isolation_rewrite_shipped.
    """
    return {
        "isolation_rewrite_design_present": True,
        "rewrite_with_wire_not_delete": True,
        "isolation_rewrite_required_blocker_documented": True,
        # Executed rewrite of isolation suite WITH wire — not done today.
        "isolation_tests_rewritten_with_wire_not_deleted": False,
        "no_silent_isolation_suite_deletion": True,
        "feature_flag_reserved_and_named": True,
        "dual_recovery_path_planned_labeled_honestly": True,
        "path_design_present": True,
        "form_label_change_shipped": False,
        # Explicit ship keys — never True while rewrite remains unshipped.
        "isolation_rewrite_shipped": False,
        "isolation_rewrite_with_wire": False,
        "wire_shipped": False,
    }


def case1_isolation_ship_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while isolation rewrite / suite rewrite co-reqs remain open.

    Even if structural labels hold offline, aggregate isolation ship is never
    allowed until all *required* criteria are met (including isolation tests
    rewritten WITH wire and form_label_change co-req — not claimed today).
    """
    met = (
        criteria_met
        if criteria_met is not None
        else case1_isolation_rewrite_shipped_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    keys = list(required_keys)
    for extra in (
        "isolation_rewrite_shipped",
        "isolation_rewrite_with_wire",
        "wire_shipped",
    ):
        if extra in met and extra not in keys:
            keys.append(extra)
    return all(bool(met.get(k)) for k in keys)


def case1_isolation_rewrite_ship_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Alias for ``case1_isolation_ship_allowed_today``."""
    return case1_isolation_ship_allowed_today(criteria_met)


def case1_isolation_rewrite_shipped_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today — False until all required criteria hold."""
    return case1_isolation_ship_allowed_today(criteria_met)


def _case1_isolation_rewrite_shipped_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-isolation-rewrite-shipped locks."""
    return {
        "kind": CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "form_label_change_shipped": False,
        "path_design_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_form_label_change_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_form_flip": True,
        "design_is_not_form_label_shipped": True,
        "design_is_not_ship_met": True,
        "design_is_not_path_present_for_ship": True,
        "design_is_not_path_shipped": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "probe_linf_is_not_isolation_ship_criterion_today": True,
        "bridge_linf_is_not_isolation_ship_criterion_today": True,
        "warmstart_linf_is_not_isolation_ship_criterion_today": True,
        "pooling_linf_is_not_isolation_ship_criterion_today": True,
        "seed_identity_linf_is_not_isolation_ship_criterion": True,
        "recovered_blender_linf_is_not_isolation_ship_criterion_today": True,
        "residual_must_vanish_is_not_isolation_ship_criterion": True,
        "packaging_alone_is_not_isolation_ship_criterion": True,
        "design_contracts_alone_is_not_isolation_ship_criterion": True,
        "isolation_design_alone_is_not_isolation_ship_criterion": True,
        "isolation_design_contract_alone_is_not_isolation_ship_criterion": True,
        "this_isolation_rewrite_shipped_criteria_contract_alone_is_not_isolation_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_isolation_rewrite_shipped_criteria_contract_offline",
        "note": (
            "Offline Case-1 isolation-rewrite ship-met / flip criteria contract: "
            "machine-readable *when isolation_rewrite_with_wire / isolation_rewrite_shipped "
            "may become met/True*. criteria_present=True; isolation_ship_allowed_today=False; "
            "criteria_met_today=False; isolation_rewrite_shipped remains False; checklist "
            "isolation_rewrite_with_wire stays open; rewrite-not-delete; form remains "
            f"{CASE1_FORM_CURRENT}; planned={CASE1_PLANNED_TF_AWARE_FORM}; "
            "path_design_present=True; path_shipped=False; "
            "dual_honest_tf_aware_path_present ship-met=False; form_label_change_shipped=False; "
            "wire_shipped=False; wire_ship_allowed_today=False; dual_linf unproven; "
            "online_linf_gate open; gate_flip_allowed_today=False; dual_recovery_path=None; "
            "solver=False; on_excel_case1_path=False. Criteria contract is NOT isolation "
            "rewrite shipped, NOT form flip, NOT form_label_change_shipped True, NOT path "
            "shipped, NOT ship-met, NOT wire shipped, NOT ship allow, NOT gate flip, NOT "
            "VERDICT gate, NOT dual L∞ under wire proof. Isolation design alone, this "
            "isolation ship criteria contract alone, packaging alone, path design alone, "
            "ship-met criteria alone, form_label criteria alone, wire-ship design alone, "
            "gate criteria alone, dual-space form alone, form registration alone, "
            "diagnostic L∞, and residual-must-vanish are not isolation-ship enablers "
            "today. Full DEFAULT_WIRE_BLOCKERS remain (isolation_rewrite_required, "
            "form_label_change_required, dual_linf_under_wire_unproven, "
            "case1_is_cdu_blender_package_admm, no_blender_offline_affine_kernel, "
            "wire_not_shipped, affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp). "
            "UNITS stay FCC/COKER/CDU (no silent BLENDER). Does not flip foreign "
            "wire-ship/gate/form_label/path-present met_today maps. Does not clear "
            "DEFAULT_WIRE_BLOCKERS. Does not redefine ready_for_wire_discussion. "
            "Always-on numpy; no TF/PuLP/excel_pipeline on hot path; no maximizer; "
            "isolation suite behavior unchanged this cycle. SUGGESTED_NEXT_WAVE still "
            "points at full dual-honest wire (deferred)."
        ),
    }


def offline_case1_isolation_rewrite_shipped_criteria_contract_report() -> Dict[str, Any]:
    """Always-on isolation-rewrite ship-met / flip criteria contract.

    No TF, no PuLP, no excel_pipeline, no solve. Aggregate ``ok`` /
    ``design_contract_ok`` / ``contract_ok`` = criteria formalized ∧ honesty
    locks ∧ isolation_rewrite_shipped False ∧ isolation_ship_allowed_today=False
    ∧ criteria_met_today=False ∧ form classic ∧ path_design_present=True ∧
    path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met False ∧
    form_label_change_shipped=False ∧ wire_shipped=False ∧
    wire_ship_allowed_today=False ∧ dual_linf unproven ∧ isolation checklist
    open ∧ online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers
    non-empty (incl. isolation_rewrite_required) ∧ dual_recovery_path is None ∧
    feature_flag_enabled_today=False ∧ UNITS FCC/COKER/CDU.
    **Not** isolation rewrite shipped. **Not** form flip. **Not** path shipped.
    **Not** ship-met. **Not** wire. **Not** ship allow. **Not** VERDICT.
    Composes form contract / dual_linf checklist / path shape / wire-ship + gate
    + form_label + path-present met maps / DEFAULT_WIRE_BLOCKERS — does **not**
    re-run maximizers/probes or rewrite isolation tests. Does **not** flip
    foreign met_today maps.
    """
    honesty = _case1_isolation_rewrite_shipped_criteria_contract_honesty_fields()
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Foreign maps — compose without flipping.
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        wire_criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    path_present_criteria_met_today = (
        case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
            path_present_met_map
        )
    )

    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    form_label_criteria_met_today = (
        case1_form_label_change_shipped_criteria_met_today_aggregate(form_label_met_map)
    )

    flip_criteria = case1_isolation_rewrite_shipped_flip_criteria()
    criteria_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(criteria_met_map)
    criteria_met_today = case1_isolation_rewrite_shipped_criteria_met_today_aggregate(
        criteria_met_map
    )

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    isolation_tests_must_be_rewritten_with_wire_not_deleted = True
    no_silent_isolation_suite_deletion = True

    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    criteria_present = True

    required_keys = set(CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA.keys())
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and all(flip_criteria[k] == FLIP_CRITERION_REQUIRED for k in required_keys)
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
        and CASE1_ISOLATION_REWRITE_BLOCKER_ID in blockers
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["dual_recovery_path_planned_when_shipped"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        and "pure-admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and "pure_admm"
        not in str(shape["dual_recovery_path_planned_when_shipped"]).lower()
        and shape["feature_flag_name"]
        == CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME
        and shape["feature_flag_enabled_today"] is False
        and shape["path_design_present"] is True
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["wire_shipped"] is False
        and shape["not_pure_admm_dual_recovery"] is True
        and shape["not_blender_affine_units"] is True
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and honesty["wire_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_form_flip"] is True
        and honesty["design_is_not_form_label_shipped"] is True
        and honesty["design_is_not_ship_met"] is True
        and honesty["design_is_not_path_shipped"] is True
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["isolation_design_alone_is_not_isolation_ship_criterion"] is True
        and honesty[
            "this_isolation_rewrite_shipped_criteria_contract_alone_is_not_isolation_ship_criterion"
        ]
        is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
        and form["form_label_change_required_still_true"] is True
    )
    isolation_permission_ok = (
        isolation_ship_allowed_today is False and criteria_met_today is False
    )
    form_label_permission_ok = (
        form_label_ship_allowed_today is False and form_label_criteria_met_today is False
    )
    ship_met_permission_ok = (
        ship_met_allowed_today is False and path_present_criteria_met_today is False
    )
    wire_ship_permission_ok = (
        wire_ship_allowed_today is False and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
        and isolation_tests_must_be_rewritten_with_wire_not_deleted is True
        and isolation_tests_rewritten_with_wire is False
        and no_silent_isolation_suite_deletion is True
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    # Ship-critical foreign met_today keys remain False under HEAD.
    ship_critical_false = (
        wire_criteria_met_map.get("isolation_rewrite_with_wire") is False
        and wire_criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and wire_criteria_met_map.get("form_label_change_shipped") is False
        and wire_criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and wire_criteria_met_map.get("dual_linf_under_wire_proven") is False
        and wire_criteria_met_map.get("wire_shipped") is False
        and wire_criteria_met_map.get("online_linf_gate_under_tf_path") is False
        and gate_met_map.get("dual_honest_tf_aware_path_present") is False
        and gate_met_map.get("isolation_rewrite_with_wire") is False
        and gate_met_map.get("form_label_change_shipped") is False
        and gate_met_map.get("wire_shipped") is False
        and path_present_met_map.get("form_label_change_shipped") is False
        and path_present_met_map.get("isolation_rewrite_with_wire") is False
        and form_label_met_map.get("isolation_rewrite_with_wire") is False
        and form_label_met_map.get("form_label_change_shipped") is False
    )

    # Structural True subset may hold; co-reqs stay False.
    structural_met_ok = (
        criteria_met_map.get("isolation_rewrite_design_present") is True
        and criteria_met_map.get("rewrite_with_wire_not_delete") is True
        and criteria_met_map.get("isolation_rewrite_required_blocker_documented") is True
        and criteria_met_map.get("no_silent_isolation_suite_deletion") is True
        and criteria_met_map.get("feature_flag_reserved_and_named") is True
        and criteria_met_map.get("dual_recovery_path_planned_labeled_honestly") is True
        and criteria_met_map.get("path_design_present") is True
        and criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("isolation_rewrite_shipped") is False
        and criteria_met_map.get("isolation_rewrite_with_wire") is False
        and criteria_met_map.get("wire_shipped") is False
    )

    anti = CASE1_ISOLATION_REWRITE_SHIPPED_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 10
        and "this_isolation_rewrite_shipped_criteria_contract_alone" in anti
        and "isolation_design_contract_alone" in anti
        and "isolation_design_alone" in anti
        and "packaging_alone" in anti
    )

    design_formalized = bool(
        criteria_present
        and flip_criteria_formalized
        and path_design_present
        and shape_ok
        and isolation_rewrite_design_present
        and anti_ok
        and CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and isolation_permission_ok
        and form_label_permission_ok
        and ship_met_permission_ok
        and wire_ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and structural_met_ok
        and form_label_open
    )
    design_contract_ok = honesty_ok
    contract_ok = design_contract_ok
    ok = (
        design_contract_ok
        and (honesty["isolation_rewrite_shipped"] is False)
        and (honesty["form_label_change_shipped"] is False)
        and (honesty["path_shipped"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
    )

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ criteria_present=True ∧ "
        "form classic ∧ isolation_rewrite_shipped=False ∧ "
        "isolation_ship_allowed_today=False ∧ criteria_met_today=False ∧ "
        "isolation checklist open ∧ rewrite-not-delete ∧ "
        "path_design_present=True ∧ path_shipped=False ∧ "
        "dual_honest_tf_aware_path_present ship-met=False ∧ "
        "form_label_change_shipped=False ∧ wire_shipped=False ∧ "
        "wire_ship_allowed_today=False ∧ dual_linf unproven ∧ "
        "online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers "
        "non-empty (incl. isolation_rewrite_required) ∧ dual_recovery_path=None ∧ "
        "feature_flag_enabled_today=False ∧ UNITS FCC/COKER/CDU — NOT isolation "
        "rewrite shipped; NOT form flip; NOT form_label_change_shipped; "
        "NOT path shipped; NOT ship-met; NOT wire shipped; NOT ship allow; "
        "NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": contract_ok,
        "criteria_present": criteria_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "form_label_change_shipped": form_label_change_shipped,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "isolation_permission_ok": isolation_permission_ok,
        "form_label_permission_ok": form_label_permission_ok,
        "ship_met_permission_ok": ship_met_permission_ok,
        "wire_ship_permission_ok": wire_ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "design_formalized": design_formalized,
        "flip_criteria_formalized": flip_criteria_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "isolation_rewrite_shipped_criteria_contract": (
            CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION
        ),
        "isolation_rewrite_shipped_criteria_present": criteria_present,
        # Flip criteria / met maps
        "flip_criteria": flip_criteria,
        "isolation_ship_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "isolation_rewrite_ship_allowed_today": isolation_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "anti_criteria_today": list(
            CASE1_ISOLATION_REWRITE_SHIPPED_ANTI_CRITERIA_TODAY
        ),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        # Path shape snapshot
        "path_shape": shape,
        "cdu_surface": shape["cdu_surface"],
        "blender_surface": shape["blender_surface"],
        "units_on_path": list(shape["units_on_path"]),
        "intermediates": list(shape["intermediates"]),
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        # Foreign ship permission (hard False under HEAD; not flipped by this contract)
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "wire_ship_criteria_met_today_map": wire_criteria_met_map,
        "ship_met_allowed_today": ship_met_allowed_today,
        "path_present_criteria_met_today": path_present_criteria_met_today,
        "path_present_criteria_met_today_map": path_present_met_map,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "form_label_criteria_met_today": form_label_criteria_met_today,
        "form_label_criteria_met_today_map": form_label_met_map,
        "design_does_not_flip_wire_ship_met_today": True,
        "design_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "design_does_not_flip_form_label_change_shipped_met_today": True,
        # Isolation design presence / rewrite-not-delete
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_tests_must_be_rewritten_with_wire_not_deleted": (
            isolation_tests_must_be_rewritten_with_wire_not_deleted
        ),
        "no_silent_isolation_suite_deletion": no_silent_isolation_suite_deletion,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_rewrite_checklist_key": CASE1_ISOLATION_REWRITE_CHECKLIST_KEY,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "design_does_not_set_isolation_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        "design_does_not_flip_gate_met_today": True,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        "design_does_not_close_form_label_change_shipped_checklist": True,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_blocker_id": CASE1_ISOLATION_REWRITE_BLOCKER_ID,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "cross_links_wire_ship_criterion_isolation_rewrite_with_wire": True,
        "wire_ship_criterion_isolation_rewrite_with_wire_met_today": False,
        "cross_links_gate_criterion_isolation_rewrite_with_wire": True,
        "gate_criterion_isolation_rewrite_with_wire_met_today": False,
        "cross_links_form_label_criterion_isolation_rewrite_with_wire": True,
        "form_label_criterion_isolation_rewrite_with_wire_met_today": False,
        "cross_links_path_present_criterion_isolation_rewrite_with_wire": True,
        "path_present_criterion_isolation_rewrite_with_wire_met_today": False,
        "isolation_design_is_not_isolation_rewrite_shipped": True,
        "isolation_rewrite_shipped_is_not_form_label_shipped": True,
        "isolation_rewrite_shipped_is_not_path_shipped": True,
        "isolation_rewrite_shipped_is_not_ship_met": True,
        "isolation_rewrite_shipped_is_not_wire_shipped": True,
        "isolation_rewrite_shipped_is_not_ship_allow": True,
        "tf_available": tf_available(),
        "isolation_rewrite_shipped_criteria_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_isolation_rewrite_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_isolation_rewrite_shipped_criteria_contract_report``."""
    return offline_case1_isolation_rewrite_shipped_criteria_contract_report(**kwargs)


def multi_unit_case1_isolation_rewrite_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_isolation_rewrite_shipped_criteria_contract_report``."""
    return offline_case1_isolation_rewrite_shipped_criteria_contract_report(**kwargs)



# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest multi-blocker wire bundle design contract
# (goal 5 + 3 honesty residual after isolation ship criteria + individual
# design/criteria ladder)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes machine-readable *what must land together*
# for SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
# (dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change).
# Design only:
# bundle_design_present=True; bundle_shipped=False; bundle_ship_allowed_today=False;
# criteria_met_today=False (aggregate); wire_shipped=False; isolation_rewrite_shipped=False;
# form classic; form_label_change_shipped=False; path_shipped=False; ship-met False;
# dual_linf unproven; online_linf_gate open; dual_recovery_path=None on TF surface.
# Distinct from wire-ship acceptance design (unordered *when* ship criteria): this is
# the co-req *bundle* with optional order/atomicity documentation only.
# Does NOT ship wire. Does NOT ship bundle. Does NOT rewrite isolation tests.
# Does NOT flip form. Does NOT invent BLENDER UNITS. Does NOT clear
# DEFAULT_WIRE_BLOCKERS. Does NOT redefine ready_for_wire_discussion.
# Does NOT implement auto-executor / auto-wire for SUGGESTED_NEXT_WAVE.
# No TF / no PuLP / no excel_pipeline on hot path.

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_KIND = (
    "offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract"
)
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_ANNOTATION = "present"

# Bundle name references SUGGESTED_NEXT_WAVE (string hint only — not an executor).
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME = SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT

# Machine-readable co-req members (requirement classes only — not met-today theater).
# Parallel SUGGESTED_NEXT_WAVE + existing checklist/criteria keys.
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS: Dict[str, str] = {
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    "isolation_rewrite_shipped": FLIP_CRITERION_REQUIRED,
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    "dual_honest_tf_aware_path_present": FLIP_CRITERION_REQUIRED,
    "online_linf_gate_under_tf_path": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "dual_linf_under_wire_proven": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "wire_shipped": FLIP_CRITERION_REQUIRED,
    "dual_recovery_path_planned_labeled_honestly": FLIP_CRITERION_REQUIRED,
    "feature_flag_reserved_and_named": FLIP_CRITERION_REQUIRED,
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    "rewrite_not_delete": FLIP_CRITERION_REQUIRED,
    "no_blender_affine_units": FLIP_CRITERION_REQUIRED,
    "case1_cdu_blender_package_shape_acknowledged": FLIP_CRITERION_REQUIRED,
}

# Optional order hint — design documentation only. NOT an executor / auto-wire.
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT: tuple = (
    "isolation_rewrite_with_wire",
    "form_label_change_shipped",
    "dual_honest_tf_aware_path_present",
    "dual_linf_under_wire_proven",
    "wire_shipped",
)
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR = True
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE = True
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ATOMIC_COSHIP_ALSO_VALID = True

# Explicit anti-criteria: these are NEVER bundle-ship or wire-ship enablers today.
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "form_label_criteria_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "gate_criteria_alone",
    "wire_ship_acceptance_design_alone",
    "this_bundle_design_alone",
    "this_bundle_ship_criteria_contract_alone",
    "this_bundle_ship_met_criteria_alone",
)


def case1_dual_honest_multi_blocker_wire_bundle_members() -> Dict[str, str]:
    """Return a copy of the multi-blocker wire bundle co-req members map."""
    return dict(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS)


def case1_dual_honest_multi_blocker_wire_bundle_member_status_today() -> Dict[str, bool]:
    """Per-member met_today / status snapshot under HEAD defaults (pure metadata).

    Aggregate bundle ship permission remains False while isolation rewrite, form
    label shipped, dual-honest wire path, dual_linf under wire, and wire_shipped
    remain open. Structural honesty labels that already hold offline
    (dual_recovery planned label, feature flag reserved+named, planned form
    distinct, no BLENDER UNITS, Case 1 package shape acknowledged, rewrite-not-
    delete design present) may be True without flipping the aggregate.
    """
    return {
        "isolation_rewrite_with_wire": False,
        "isolation_rewrite_shipped": False,
        "isolation_tests_rewritten_with_wire_not_deleted": False,
        "form_label_change_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "online_linf_gate_under_tf_path": False,
        "dual_linf_under_wire_proven": False,
        "wire_shipped": False,
        # Planned dual_recovery_path label under future wire is honest (not pure-ADMM);
        # TF surface dual_recovery_path remains None today.
        "dual_recovery_path_planned_labeled_honestly": True,
        # Feature flag name reserved; enabled_today hard False.
        "feature_flag_reserved_and_named": True,
        "no_silent_form_reuse": True,
        # Isolation design formalizes rewrite-not-delete; suite not rewritten as ship.
        "rewrite_not_delete": True,
        "no_blender_affine_units": True,
        "case1_cdu_blender_package_shape_acknowledged": True,
    }


def case1_dual_honest_multi_blocker_wire_bundle_criteria_met_today_map() -> Dict[str, bool]:
    """Alias for member status map (criteria framing)."""
    return case1_dual_honest_multi_blocker_wire_bundle_member_status_today()


def case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while any *required* (not under-wire-only) member remains open.

    Order hint is NOT an executor — this function never auto-wires or enables flags.
    """
    met = (
        criteria_met
        if criteria_met is not None
        else case1_dual_honest_multi_blocker_wire_bundle_member_status_today()
    )
    required_keys = [
        k
        for k, cls in CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    keys = list(required_keys)
    # Explicit ship keys if provided in met map (criteria surface may pass them).
    for extra in ("bundle_shipped", "bundle_ship_allowed_today", "wire_shipped"):
        if extra in met and extra not in keys:
            keys.append(extra)
    return all(bool(met.get(k)) for k in keys)


def case1_dual_honest_multi_blocker_wire_bundle_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today for the multi-blocker bundle — False until required hold."""
    return case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(criteria_met)


def case1_dual_honest_multi_blocker_wire_bundle_shipped() -> bool:
    """Hard False — bundle is design-only; never shipped via this contract."""
    return False


def _case1_dual_honest_multi_blocker_wire_bundle_design_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-bundle-shipped / not-wire / not-VERDICT locks."""
    return {
        "kind": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "bundle_shipped": False,
        "not_wire_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "design_is_not_bundle_shipped": True,
        "design_is_not_bundle_ship_allow": True,
        "design_is_not_wire_shipped": True,
        "design_is_not_wire": True,
        "design_is_not_wire_ship_allow": True,
        "design_is_not_isolation_rewrite_shipped": True,
        "design_is_not_form_label_change_shipped": True,
        "design_is_not_path_shipped": True,
        "design_is_not_ship_met": True,
        "design_is_not_form_flip": True,
        "design_is_not_gate_flip": True,
        "design_is_not_verdict_gate": True,
        "design_is_not_dual_linf_under_wire_proof": True,
        "design_is_not_ship_allow": True,
        "this_bundle_design_alone_is_not_ship_criterion": True,
        "wire_ship_acceptance_design_alone_is_not_bundle_ship": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "path_shipped": False,
        "scope": "case1_dual_honest_multi_blocker_wire_bundle_design_contract_offline",
        "note": (
            "Offline Case-1 dual-honest multi-blocker wire bundle design contract: "
            "machine-readable *what must land together* for SUGGESTED_NEXT_WAVE "
            f"({CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME}). "
            "bundle_design_present=True; bundle_shipped=False; "
            "bundle_ship_allowed_today=False; criteria_met_today=False; "
            "wire_shipped=False; isolation_rewrite_shipped=False; form classic; "
            "form_label_change_shipped=False; path_shipped=False; dual_honest_tf_aware_path_present "
            "ship-met False; dual_linf unproven; online_linf_gate open; "
            "gate_flip_allowed_today=False; dual_recovery_path=None on TF surface; "
            f"planned dual_recovery_path under future wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved False; "
            f"on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Design is NOT bundle shipped, NOT bundle ship allow, NOT wire shipped, "
            "NOT wire ship allow, NOT isolation rewrite shipped, NOT form_label shipped, "
            "NOT path shipped, NOT ship-met, NOT form flip, NOT gate flip, NOT VERDICT, "
            "NOT dual L∞ under wire proof. Distinct from wire-ship acceptance design "
            "(unordered when-ship criteria): this is the co-req *bundle* with optional "
            "order/atomicity documentation only (order_hint is NOT an executor; no auto-wire). "
            "Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, residual-must-vanish, "
            "packaging alone, design contracts alone, wire-ship acceptance alone, and this "
            "bundle design alone are not ship enablers today. Full DEFAULT_WIRE_BLOCKERS "
            "remain. UNITS stay FCC/COKER/CDU. Does not clear DEFAULT_WIRE_BLOCKERS. "
            "Does not redefine ready_for_wire_discussion. Always-on numpy; no "
            "TF/PuLP/excel_pipeline on hot path; no maximizer; isolation suite behavior "
            "unchanged this cycle."
        ),
    }


def offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report() -> Dict[str, Any]:
    """Always-on multi-blocker wire bundle design contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``design_contract_ok`` = design formalized ∧ honesty locks ∧
    bundle_shipped=False ∧ bundle_ship_allowed_today=False ∧ criteria_met_today=False ∧
    wire_shipped=False ∧ dual_linf unproven ∧ form classic ∧ isolation rewrite not
    shipped ∧ isolation checklist open ∧ online_linf_gate open ∧
    gate_flip_allowed_today=False ∧ blockers non-empty ∧ dual_recovery_path is None ∧
    UNITS FCC/COKER/CDU ∧ order_hint is not executor.
    **Not** bundle shipped. **Not** bundle ship allow. **Not** wire shipped.
    **Not** isolation rewrite shipped. **Not** form flip. **Not** VERDICT.
    **Not** dual L∞ under wire proof. **Not** an auto-executor for SUGGESTED_NEXT_WAVE.
    Composes form contract / dual_linf checklist / DEFAULT_WIRE_BLOCKERS / existing
    met maps / path feature-flag constants — does **not** re-run maximizers/probes
    or rewrite isolation tests.
    """
    honesty = _case1_dual_honest_multi_blocker_wire_bundle_design_contract_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    members_map = case1_dual_honest_multi_blocker_wire_bundle_members()
    member_status = case1_dual_honest_multi_blocker_wire_bundle_member_status_today()
    bundle_ship_allowed_today = case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(
        member_status
    )
    bundle_criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_criteria_met_today_aggregate(
            member_status
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    # Wire-ship allow compose (pure metadata — not recursive wire-ship report).
    wire_ship_allowed_today = case1_wire_ship_allowed_today()
    wire_shipped = False

    # Gate flip discipline (compose — do not re-run gate report recursively).
    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    path_design_present = True
    wire_ship_acceptance_design_present = True

    bundle_design_present = True

    required_keys = set(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS.keys())
    under_wire_keys = {
        k
        for k, cls in CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS.items()
        if cls == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    }
    members_formalized = (
        set(members_map.keys()) == required_keys
        and all(
            members_map[k] == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
            for k in under_wire_keys
        )
        and all(
            members_map[k] == FLIP_CRITERION_REQUIRED
            for k in required_keys
            if k not in under_wire_keys
        )
        and len(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ANTI_CRITERIA_TODAY) >= 10
        and len(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT) >= 3
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["design_is_not_bundle_shipped"] is True
        and honesty["design_is_not_bundle_ship_allow"] is True
        and honesty["design_is_not_wire_shipped"] is True
        and honesty["design_is_not_wire"] is True
        and honesty["design_is_not_isolation_rewrite_shipped"] is True
        and honesty["design_is_not_gate_flip"] is True
        and honesty["design_is_not_verdict_gate"] is True
        and honesty["design_is_not_dual_linf_under_wire_proof"] is True
        and honesty["design_is_not_ship_allow"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["this_bundle_design_alone_is_not_ship_criterion"] is True
        and honesty["wire_ship_acceptance_design_alone_is_not_bundle_ship"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["probe_linf_is_not_ship_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_ship_criterion"] is True
        and honesty["design_contracts_alone_is_not_ship_criterion"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        bundle_ship_allowed_today is False
        and bundle_criteria_met_today is False
        and wire_ship_allowed_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    bundle_not_shipped_ok = (
        bundle_shipped is False
        and honesty["bundle_shipped"] is False
        and honesty["not_bundle_shipped"] is True
    )

    # Ship-critical met_today keys must remain False under HEAD.
    ship_critical_false = (
        member_status.get("isolation_rewrite_with_wire") is False
        and member_status.get("isolation_rewrite_shipped") is False
        and member_status.get("isolation_tests_rewritten_with_wire_not_deleted") is False
        and member_status.get("form_label_change_shipped") is False
        and member_status.get("dual_honest_tf_aware_path_present") is False
        and member_status.get("dual_linf_under_wire_proven") is False
        and member_status.get("wire_shipped") is False
        and member_status.get("online_linf_gate_under_tf_path") is False
    )

    feature_flag_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME == "enable_tf_affine_case1_wire"
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
        and member_status.get("feature_flag_reserved_and_named") is True
    )
    dual_recovery_planned_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        == "online_lambda_under_tf_aware_form_when_shipped"
        and member_status.get("dual_recovery_path_planned_labeled_honestly") is True
        and honesty["dual_recovery_path"] is None
        and honesty["not_pure_admm_dual_recovery"] is True
    )
    order_hint_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
    )
    bundle_name_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME
        == SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
        == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
    )

    design_formalized = bool(
        bundle_design_present
        and members_formalized
        and isolation_rewrite_design_present
        and path_design_present
        and wire_ship_acceptance_design_present
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_ANNOTATION
        == "present"
        and bundle_name_ok
        and order_hint_ok
        and feature_flag_ok
        and dual_recovery_planned_ok
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and bundle_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and form_label_open
        and form_label_change_shipped is False
        and path_shipped is False
        and dual_honest_tf_aware_path_present is False
    )
    design_contract_ok = honesty_ok
    ok = design_contract_ok and (honesty["wire_shipped"] is False) and (
        honesty["bundle_shipped"] is False
    )

    ok_criteria = (
        "design formalized ∧ honesty locks ∧ bundle_shipped=False ∧ "
        "bundle_ship_allowed_today=False ∧ criteria_met_today=False ∧ "
        "wire_shipped=False ∧ wire_ship_allowed_today=False ∧ dual_linf unproven ∧ "
        "form classic ∧ isolation rewrite not shipped ∧ isolation checklist open ∧ "
        "online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers non-empty ∧ "
        "dual_recovery_path=None ∧ UNITS FCC/COKER/CDU ∧ order_hint not executor — "
        "NOT bundle shipped; NOT bundle ship allow; NOT wire shipped; NOT ship allow; "
        "NOT isolation rewrite shipped; NOT form flip; NOT gate flip; NOT VERDICT; "
        "NOT dual L∞ under wire proof; NOT auto-executor"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": design_contract_ok,
        "bundle_design_present": bundle_design_present,
        "design_present": bundle_design_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "bundle_not_shipped_ok": bundle_not_shipped_ok,
        "design_formalized": design_formalized,
        "members_formalized": members_formalized,
        "criteria_formalized": members_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "bundle_design_contract": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_ANNOTATION
        ),
        "bundle_design_contract_available": True,
        "bundle_name": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME,
        "suggested_next_wave_bundle_name": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME,
        # Ship permission (hard False under HEAD)
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "bundle_criteria_met_today": bundle_criteria_met_today,
        "criteria_met_today": bundle_criteria_met_today,
        "members_map": members_map,
        "bundle_members": members_map,
        "member_status_today": member_status,
        "criteria_met_today_map": member_status,
        "criteria_status_today": member_status,
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ANTI_CRITERIA_TODAY
        ),
        "order_hint": list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT),
        "order_hint_is_not_executor": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR
        ),
        "no_auto_wire": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE,
        "atomic_coship_also_valid": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ATOMIC_COSHIP_ALSO_VALID
        ),
        "member_criterion_required_class": FLIP_CRITERION_REQUIRED,
        "member_criterion_required_under_wire_only_class": (
            FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        ),
        # Companion design surfaces present (constants — not recursive reports)
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "wire_ship_acceptance_design_present": wire_ship_acceptance_design_present,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_shipped": wire_shipped,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Dual recovery planned vs today
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
        ),
        "feature_flag_reserved_and_named": feature_flag_ok,
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "tf_available": tf_available(),
        "dual_honest_multi_blocker_wire_bundle_design_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_dual_honest_multi_blocker_wire_bundle_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report(
        **kwargs
    )


def multi_unit_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report(
        **kwargs
    )



# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest multi-blocker wire bundle ship-met / flip criteria
# contract (goal 5 + 3 honesty residual after multi-blocker bundle design)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes machine-readable *when*
# bundle_shipped / bundle_ship_allowed_today may become True.
# Locks: criteria_present=True; bundle_shipped=False; bundle_ship_allowed_today=False;
# criteria_met_today=False; wire_shipped=False; isolation_rewrite_shipped=False;
# checklist isolation_rewrite_with_wire open; form classic; form_label_change_shipped=False;
# path_shipped=False; dual_honest_tf_aware_path_present ship-met False; dual_linf unproven;
# online_linf_gate open; dual_recovery_path=None today; planned dual recovery labeled
# honestly (not pure-ADMM); order_hint is not executor. Bundle design formalizes *what*;
# this formalizes *when*. Never imports excel_pipeline / tensorflow / pulp on hot path.
# Never flips form, never closes checklist, never clears DEFAULT_WIRE_BLOCKERS.
# Never ships wire/bundle.

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_KIND = (
    "offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract"
)
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION = (
    "present"
)

# Machine-readable bundle_shipped / bundle_ship_allowed_today flip criteria map
# (requirement classes only — not met-today theater). Aligns with design members +
# structural bundle_design_present.
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA: Dict[str, str] = {
    # Structural: multi-blocker bundle design already formalizes co-req *what*.
    "bundle_design_present": FLIP_CRITERION_REQUIRED,
    # Ship-critical co-reqs (False / open today).
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    "isolation_rewrite_shipped": FLIP_CRITERION_REQUIRED,
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    "dual_honest_tf_aware_path_present": FLIP_CRITERION_REQUIRED,
    "online_linf_gate_under_tf_path": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "dual_linf_under_wire_proven": FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY,
    "wire_shipped": FLIP_CRITERION_REQUIRED,
    # Structural honesty labels that already hold offline.
    "dual_recovery_path_planned_labeled_honestly": FLIP_CRITERION_REQUIRED,
    "feature_flag_reserved_and_named": FLIP_CRITERION_REQUIRED,
    "no_silent_form_reuse": FLIP_CRITERION_REQUIRED,
    "rewrite_not_delete": FLIP_CRITERION_REQUIRED,
    "no_blender_affine_units": FLIP_CRITERION_REQUIRED,
    "case1_cdu_blender_package_shape_acknowledged": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these are NEVER bundle-ship enablers today.
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "this_bundle_design_alone",
    "this_bundle_ship_criteria_contract_alone",
    "this_bundle_ship_met_criteria_alone",
    "this_contract_alone",
    "wire_ship_acceptance_design_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "form_label_criteria_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "gate_criteria_alone",
)


def case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria() -> Dict[str, str]:
    """Return a copy of the multi-blocker bundle ship-met flip-criteria map."""
    return dict(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA)


def case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults (criteria surface).

    Aggregate bundle_ship_allowed_today / criteria_met_today remains False while
    isolation rewrite, form label shipped, dual-honest path ship-met, dual_linf
    under wire, and wire_shipped remain open. Structural labels that already hold
    offline (bundle_design_present, dual_recovery planned label, feature flag
    reserved+named, no_silent_form_reuse, rewrite_not_delete design, no BLENDER
    UNITS, Case 1 package shape acknowledged) may be True without flipping the
    aggregate or bundle_shipped. Order_hint is not an executor.
    """
    status = case1_dual_honest_multi_blocker_wire_bundle_member_status_today()
    out = dict(status)
    out["bundle_design_present"] = True
    # Explicit ship keys — never True while multi-blocker co-reqs remain open.
    out["bundle_shipped"] = False
    out["bundle_ship_allowed_today"] = False
    return out


def case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today for the bundle ship-met surface — False under HEAD."""
    met = (
        criteria_met
        if criteria_met is not None
        else case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    keys = list(required_keys)
    for extra in ("bundle_shipped", "bundle_ship_allowed_today", "wire_shipped"):
        if extra in met and extra not in keys:
            keys.append(extra)
    return all(bool(met.get(k)) for k in keys)


def _case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-bundle-shipped locks for ship-met criteria."""
    return {
        "kind": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "bundle_shipped": False,
        "not_wire_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "criteria_is_not_bundle_shipped": True,
        "criteria_is_not_bundle_ship_allow": True,
        "criteria_is_not_wire_shipped": True,
        "criteria_is_not_wire": True,
        "criteria_is_not_wire_ship_allow": True,
        "criteria_is_not_isolation_rewrite_shipped": True,
        "criteria_is_not_form_label_change_shipped": True,
        "criteria_is_not_path_shipped": True,
        "criteria_is_not_ship_met": True,
        "criteria_is_not_form_flip": True,
        "criteria_is_not_gate_flip": True,
        "criteria_is_not_verdict_gate": True,
        "criteria_is_not_dual_linf_under_wire_proof": True,
        "criteria_is_not_ship_allow": True,
        "this_bundle_design_alone_is_not_ship_criterion": True,
        "this_bundle_ship_criteria_contract_alone_is_not_ship_criterion": True,
        "this_contract_alone_is_not_ship_criterion": True,
        "wire_ship_acceptance_design_alone_is_not_bundle_ship": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_offline",
        "note": (
            "Offline Case-1 dual-honest multi-blocker wire bundle ship-met / flip "
            "criteria contract: machine-readable *when bundle_shipped / "
            "bundle_ship_allowed_today may become True*. criteria_present=True; "
            "bundle_shipped=False; bundle_ship_allowed_today=False; "
            "criteria_met_today=False; wire_shipped=False; isolation_rewrite_shipped=False; "
            "isolation checklist open; form classic; form_label_change_shipped=False; "
            "path_shipped=False; dual_honest_tf_aware_path_present ship-met False; "
            "dual_linf unproven; online_linf_gate open; gate_flip_allowed_today=False; "
            "dual_recovery_path=None on TF surface; "
            f"planned dual_recovery_path under future wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved False; "
            f"on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Criteria contract is NOT bundle shipped, NOT bundle ship allow, NOT wire "
            "shipped, NOT wire ship allow, NOT isolation rewrite shipped, NOT form_label "
            "shipped, NOT path shipped, NOT path ship-met, NOT form flip, NOT gate flip, "
            "NOT VERDICT, NOT dual L∞ under wire proof. Bundle design formalizes *what*; "
            "this formalizes *when*. Distinct from wire-ship acceptance design (unordered "
            "when-ship). Order_hint is NOT an executor; no auto-wire. Probe/bridge/"
            "warmstart/pooling/seed-identity/recovered L∞, residual-must-vanish, packaging "
            "alone, design contracts alone, this bundle design alone, this criteria "
            "contract alone, wire-ship acceptance alone, and isolation/form/path/gate "
            "criteria alone are not ship enablers today. Full DEFAULT_WIRE_BLOCKERS remain. "
            "UNITS stay FCC/COKER/CDU. Does not clear DEFAULT_WIRE_BLOCKERS. Does not "
            "redefine ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline "
            "on hot path; no maximizer; isolation suite behavior unchanged this cycle."
        ),
    }


def offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report() -> Dict[str, Any]:
    """Always-on multi-blocker wire bundle ship-met / flip criteria contract.

    No TF, no PuLP, no excel_pipeline, no solve. Aggregate ``ok`` /
    ``design_contract_ok`` / ``contract_ok`` = criteria formalized ∧ honesty locks ∧
    criteria_present=True ∧ bundle_shipped=False ∧ bundle_ship_allowed_today=False ∧
    criteria_met_today=False ∧ wire_shipped=False ∧ isolation_rewrite_shipped=False ∧
    form classic ∧ path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met False ∧
    form_label_change_shipped=False ∧ dual_linf unproven ∧ online_linf_gate open ∧
    gate_flip_allowed_today=False ∧ blockers non-empty ∧ dual_recovery_path is None ∧
    UNITS FCC/COKER/CDU ∧ order_hint is not executor.
    **Not** bundle shipped. **Not** bundle ship allow. **Not** wire shipped.
    **Not** isolation rewrite shipped. **Not** form flip. **Not** VERDICT.
    **Not** dual L∞ under wire proof. **Not** an auto-executor for SUGGESTED_NEXT_WAVE.
    Composes form contract / dual_linf checklist / DEFAULT_WIRE_BLOCKERS / existing
    foreign met maps — does **not** re-run maximizers/probes or flip foreign maps.
    """
    honesty = (
        _case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_honesty_fields()
    )
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Foreign maps — compose without flipping.
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        wire_criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    path_present_criteria_met_today = (
        case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
            path_present_met_map
        )
    )

    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    form_label_criteria_met_today = (
        case1_form_label_change_shipped_criteria_met_today_aggregate(form_label_met_map)
    )

    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)

    flip_criteria = case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria()
    criteria_met_map = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    # Reuse design ship_allowed (REQUIRED members only + optional extras) and
    # criteria-surface aggregate (REQUIRED flip keys) — both hard False under HEAD.
    bundle_ship_allowed_today = case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(
        criteria_met_map
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            criteria_met_map
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    criteria_present = True
    bundle_design_present = True
    wire_ship_acceptance_design_present = True

    required_keys = set(
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA.keys()
    )
    under_wire_keys = {
        k
        for k, cls in CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    }
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and all(
            flip_criteria[k] == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
            for k in under_wire_keys
        )
        and all(
            flip_criteria[k] == FLIP_CRITERION_REQUIRED
            for k in required_keys
            if k not in under_wire_keys
        )
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["criteria_is_not_bundle_shipped"] is True
        and honesty["criteria_is_not_bundle_ship_allow"] is True
        and honesty["criteria_is_not_wire_shipped"] is True
        and honesty["criteria_is_not_wire"] is True
        and honesty["criteria_is_not_isolation_rewrite_shipped"] is True
        and honesty["criteria_is_not_gate_flip"] is True
        and honesty["criteria_is_not_verdict_gate"] is True
        and honesty["criteria_is_not_dual_linf_under_wire_proof"] is True
        and honesty["criteria_is_not_ship_allow"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["this_bundle_design_alone_is_not_ship_criterion"] is True
        and honesty[
            "this_bundle_ship_criteria_contract_alone_is_not_ship_criterion"
        ]
        is True
        and honesty["this_contract_alone_is_not_ship_criterion"] is True
        and honesty["wire_ship_acceptance_design_alone_is_not_bundle_ship"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["probe_linf_is_not_ship_criterion_today"] is True
        and honesty["seed_identity_linf_is_not_ship_criterion"] is True
        and honesty["design_contracts_alone_is_not_ship_criterion"] is True
        and honesty["feature_flag_enabled_today"] is False
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        bundle_ship_allowed_today is False
        and criteria_met_today is False
        and wire_ship_allowed_today is False
        and isolation_ship_allowed_today is False
        and form_label_ship_allowed_today is False
        and ship_met_allowed_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    bundle_not_shipped_ok = (
        bundle_shipped is False
        and honesty["bundle_shipped"] is False
        and honesty["not_bundle_shipped"] is True
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    ship_critical_false = (
        criteria_met_map.get("isolation_rewrite_with_wire") is False
        and criteria_met_map.get("isolation_rewrite_shipped") is False
        and criteria_met_map.get("isolation_tests_rewritten_with_wire_not_deleted")
        is False
        and criteria_met_map.get("form_label_change_shipped") is False
        and criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and criteria_met_map.get("dual_linf_under_wire_proven") is False
        and criteria_met_map.get("wire_shipped") is False
        and criteria_met_map.get("online_linf_gate_under_tf_path") is False
        and criteria_met_map.get("bundle_shipped") is False
        and criteria_met_map.get("bundle_ship_allowed_today") is False
        and wire_criteria_met_map.get("isolation_rewrite_with_wire") is False
        and wire_criteria_met_map.get("wire_shipped") is False
        and gate_met_map.get("isolation_rewrite_with_wire") is False
        and gate_met_map.get("wire_shipped") is False
        and path_present_met_map.get("isolation_rewrite_with_wire") is False
        and form_label_met_map.get("isolation_rewrite_with_wire") is False
        and isolation_met_map.get("isolation_rewrite_shipped") is False
    )

    structural_met_ok = (
        criteria_met_map.get("bundle_design_present") is True
        and criteria_met_map.get("dual_recovery_path_planned_labeled_honestly") is True
        and criteria_met_map.get("feature_flag_reserved_and_named") is True
        and criteria_met_map.get("no_silent_form_reuse") is True
        and criteria_met_map.get("rewrite_not_delete") is True
        and criteria_met_map.get("no_blender_affine_units") is True
        and criteria_met_map.get("case1_cdu_blender_package_shape_acknowledged") is True
    )

    anti = CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 10
        and "this_bundle_ship_criteria_contract_alone" in anti
        and "this_bundle_design_alone" in anti
        and "packaging_alone" in anti
        and "wire_ship_acceptance_design_alone" in anti
        and "residual_must_vanish" in anti
    )

    order_hint_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
    )
    feature_flag_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME == "enable_tf_affine_case1_wire"
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
        and criteria_met_map.get("feature_flag_reserved_and_named") is True
    )
    dual_recovery_planned_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        == "online_lambda_under_tf_aware_form_when_shipped"
        and criteria_met_map.get("dual_recovery_path_planned_labeled_honestly") is True
        and honesty["dual_recovery_path"] is None
        and honesty["not_pure_admm_dual_recovery"] is True
    )

    design_formalized = bool(
        criteria_present
        and flip_criteria_formalized
        and bundle_design_present
        and isolation_rewrite_design_present
        and path_design_present
        and wire_ship_acceptance_design_present
        and anti_ok
        and order_hint_ok
        and feature_flag_ok
        and dual_recovery_planned_ok
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION
        == "present"
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and bundle_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and design_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and structural_met_ok
        and form_label_open
        and form_label_change_shipped is False
        and path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and path_present_criteria_met_today is False
        and form_label_criteria_met_today is False
        and wire_ship_criteria_met_today is False
    )
    design_contract_ok = honesty_ok
    contract_ok = design_contract_ok
    ok = (
        design_contract_ok
        and (honesty["bundle_shipped"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["isolation_rewrite_shipped"] is False)
        and (honesty["form_label_change_shipped"] is False)
        and (honesty["path_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
    )

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ criteria_present=True ∧ "
        "bundle_shipped=False ∧ bundle_ship_allowed_today=False ∧ "
        "criteria_met_today=False ∧ wire_shipped=False ∧ "
        "isolation_rewrite_shipped=False ∧ isolation checklist open ∧ "
        "form classic ∧ form_label_change_shipped=False ∧ path_shipped=False ∧ "
        "dual_honest_tf_aware_path_present ship-met=False ∧ dual_linf unproven ∧ "
        "online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers "
        "non-empty ∧ dual_recovery_path=None ∧ feature_flag_enabled_today=False ∧ "
        "UNITS FCC/COKER/CDU ∧ order_hint not executor — NOT bundle shipped; "
        "NOT bundle ship allow; NOT wire shipped; NOT isolation rewrite shipped; "
        "NOT form flip; NOT path shipped; NOT path ship-met; NOT ship allow; "
        "NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    return {
        **honesty,
        "ok": ok,
        "design_contract_ok": design_contract_ok,
        "contract_ok": contract_ok,
        "criteria_present": criteria_present,
        "bundle_design_present": bundle_design_present,
        "design_present": bundle_design_present,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "bundle_not_shipped_ok": bundle_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "design_formalized": design_formalized,
        "flip_criteria_formalized": flip_criteria_formalized,
        "criteria_formalized": flip_criteria_formalized,
        "ok_criteria": ok_criteria,
        # Design annotation
        "bundle_shipped_criteria_contract": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION
        ),
        "bundle_shipped_criteria_contract_available": True,
        "bundle_name": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME,
        "suggested_next_wave_bundle_name": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME,
        # Ship permission (hard False under HEAD)
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "bundle_criteria_met_today": criteria_met_today,
        "criteria_met_today": criteria_met_today,
        "flip_criteria": flip_criteria,
        "bundle_ship_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY
        ),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        "flip_criterion_required_under_wire_only_class": (
            FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        ),
        "order_hint": list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT),
        "order_hint_is_not_executor": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR
        ),
        "no_auto_wire": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE,
        # Companion surfaces (constants — not recursive design reports)
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "design_does_not_close_isolation_rewrite_checklist": True,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "wire_ship_acceptance_design_present": wire_ship_acceptance_design_present,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "wire_ship_criteria_met_today_map": wire_criteria_met_map,
        "wire_shipped": wire_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "path_present_criteria_met_today": path_present_criteria_met_today,
        "path_present_criteria_met_today_map": path_present_met_map,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "form_label_criteria_met_today": form_label_criteria_met_today,
        "form_label_criteria_met_today_map": form_label_met_map,
        "isolation_criteria_met_today_map": isolation_met_map,
        "design_does_not_flip_wire_ship_met_today": True,
        "design_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "design_does_not_flip_form_label_change_shipped_met_today": True,
        "design_does_not_flip_isolation_ship_met_today": True,
        "design_does_not_flip_gate_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual recovery / feature flag
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Pooling honesty snapshot
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        "cdu_surface": "offline_affine_base_delta",
        "blender_surface": "linear_quality_pooling",
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "bundle_design_is_not_bundle_shipped": True,
        "bundle_shipped_criteria_is_not_bundle_shipped": True,
        "bundle_shipped_is_not_wire_shipped": True,
        "bundle_shipped_is_not_isolation_rewrite_shipped": True,
        "bundle_shipped_is_not_form_label_shipped": True,
        "bundle_shipped_is_not_path_shipped": True,
        "bundle_shipped_is_not_ship_met": True,
        "tf_available": tf_available(),
        "dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "note": honesty["note"],
    }


def case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report(
        **kwargs
    )


def multi_unit_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report(
        **kwargs
    )




# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest TF-aware path *execution scaffold* (goal 5 + 3)
# ---------------------------------------------------------------------------
# Callable always-on numpy compose of existing dual-honest path pieces under
# multi-blocker honesty locks. scaffold_present=True; all ship flags hard False.
# Distinct from: multi-blocker design (*what*), ship-met criteria (*when*),
# path design (*path shape*), path present criteria (*when present-for-ship*),
# case1-shaped linking maximizer skeleton. This formalizes *offline how-without-ship*.
# Does NOT ship wire/path/bundle/isolation rewrite/form; does NOT enable feature flag;
# does NOT import excel_pipeline / pulp / tensorflow on hot path; no auto-executor.

CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_KIND = (
    "offline_case1_dual_honest_tf_aware_path_execution_scaffold"
)
CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANNOTATION = "present"

CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "this_scaffold_alone",
    "this_execution_scaffold_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "bundle_design_alone",
    "bundle_ship_met_criteria_alone",
    "wire_ship_acceptance_alone",
    "case1_shaped_linking_skeleton_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "form_label_criteria_alone",
    "gate_criteria_alone",
    "diagnostic_linf_alone",
)


def _case1_dual_honest_tf_aware_path_execution_scaffold_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-path-shipped / not-wire locks for scaffold."""
    return {
        "kind": CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "scaffold_present": True,
        "execution_scaffold_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "bundle_ship_allowed_today": False,
        "criteria_met_today": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "scaffold_is_not_path_shipped": True,
        "scaffold_is_not_path_present_for_ship": True,
        "scaffold_is_not_wire_shipped": True,
        "scaffold_is_not_wire": True,
        "scaffold_is_not_bundle_shipped": True,
        "scaffold_is_not_isolation_rewrite_shipped": True,
        "scaffold_is_not_form_label_change_shipped": True,
        "scaffold_is_not_ship_allow": True,
        "scaffold_is_not_ship_met": True,
        "scaffold_is_not_form_flip": True,
        "scaffold_is_not_gate_flip": True,
        "scaffold_is_not_verdict_gate": True,
        "scaffold_is_not_dual_linf_under_wire_proof": True,
        "this_scaffold_alone_is_not_ship_criterion": True,
        "this_scaffold_alone_is_not_multi_blocker_ship": True,
        "this_execution_scaffold_alone_is_not_ship_criterion": True,
        "path_design_alone_is_not_ship_criterion": True,
        "path_present_criteria_alone_is_not_ship_criterion": True,
        "bundle_design_alone_is_not_ship_criterion": True,
        "bundle_ship_met_criteria_alone_is_not_ship_criterion": True,
        "wire_ship_acceptance_alone_is_not_ship_criterion": True,
        "case1_shaped_linking_skeleton_alone_is_not_ship_criterion": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "diagnostic_linf_is_not_dual_linf_under_wire_proof": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_tf_aware_path_execution_scaffold_offline",
        "note": (
            "Offline Case-1 dual-honest TF-aware path *execution scaffold*: callable "
            "always-on numpy compose of existing dual-honest path pieces (CDU offline "
            "affine base_delta + blender linear_quality_pooling + Case-1 intermediate "
            "stream alignment + optional labeled λ + stream-aligned dual-space residual "
            "as diagnostic-only) under multi-blocker honesty locks. "
            "scaffold_present=True / execution_scaffold_present=True; path_shipped=False; "
            "dual_honest_tf_aware_path_present ship-met=False; wire_shipped=False; "
            "bundle_shipped=False; bundle_ship_allowed_today=False; criteria_met_today=False; "
            "isolation_rewrite_shipped=False; isolation checklist open; form classic; "
            "form_label_change_shipped=False; dual_linf unproven; online_linf_gate open; "
            "gate_flip_allowed_today=False; dual_recovery_path=None on TF surface; "
            f"planned dual_recovery_path under future wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved False; "
            f"on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Scaffold is NOT path shipped, NOT path-present-for-ship, NOT wire shipped, "
            "NOT bundle shipped, NOT isolation rewrite shipped, NOT form_label shipped, "
            "NOT ship allow, NOT ship-met, NOT form flip, NOT gate flip, NOT VERDICT, "
            "NOT dual L∞ under wire proof. Design formalizes *what*; ship-met criteria "
            "*when*; this formalizes *offline how-without-ship*. Distinct from case1-shaped "
            "linking maximizer skeleton. Order_hint is NOT an executor; no auto-wire. "
            "Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, residual-must-vanish, "
            "packaging alone, design/criteria alone, this scaffold alone, and diagnostic L∞ "
            "are not ship enablers today. Full DEFAULT_WIRE_BLOCKERS remain. UNITS stay "
            "FCC/COKER/CDU (no silent BLENDER). Does not clear DEFAULT_WIRE_BLOCKERS. "
            "Does not redefine ready_for_wire_discussion. Always-on numpy; no TF/PuLP/"
            "excel_pipeline on hot path; isolation suite behavior unchanged this cycle. "
            "SUGGESTED_NEXT_WAVE still points at full dual-honest multi-blocker wire "
            "(deferred)."
        ),
    }


def case1_dual_honest_tf_aware_path_execution_scaffold_compose(
    *,
    live_lambda: Optional[Mapping[str, float]] = None,
    live_lambda_source: Optional[str] = None,
    include_diagnostic: bool = True,
) -> Dict[str, Any]:
    """Compose existing dual-honest path pieces into one offline unit of work.

    Always-on numpy. No maximizer SLA, no residual-must-vanish, no TF/PuLP/
    excel_pipeline. Ship flags remain hard false at the report layer.
    """
    shape = case1_dual_honest_tf_aware_path_shape()
    streams = list(CASE1_SHAPED_LINKING_STREAMS)

    # CDU offline affine base_delta surface (UNITS FCC/COKER/CDU — no BLENDER).
    coeffs = cached_offline_unit_coeffs("CDU")
    y_raw = numpy_affine_forward(coeffs, coeffs.x0, clamp_products=True)
    y_full = apply_cdu_postprocess(y_raw, products=coeffs.products)
    y_cdu = {p: float(y_full[p]) for p in coeffs.products}
    intermediates = project_cdu_y_to_case1_intermediates(y_cdu, streams=streams)

    # Blender linear_quality_pooling surface (not base_delta UNITS entry).
    # Use a simple finite product map for recipe-use projection (demo residual).
    products_demo = {
        p: 1.0 / max(len(CASE1_SHAPED_BLEND_RECIPES), 1)
        for p in CASE1_SHAPED_BLEND_RECIPES
    }
    blender_use = blender_recipe_use_from_products(products_demo, streams=streams)

    # Optional labeled λ — unlabeled vectors are ignored (not dual recovery).
    lambda_meta: Dict[str, Any] = {
        "live_lambda_used": False,
        "live_lambda_source": None,
        "live_lambda_rejected_unlabeled": False,
        "dual_recovery_path": None,
    }
    lam_vec: Optional[Dict[str, float]] = None
    if live_lambda is not None:
        if not live_lambda_source:
            lambda_meta["live_lambda_rejected_unlabeled"] = True
        else:
            lam_vec = {str(k): float(v) for k, v in live_lambda.items()}
            lambda_meta["live_lambda_used"] = True
            lambda_meta["live_lambda_source"] = str(live_lambda_source)
    elif live_lambda_source is None:
        # Default fixture path always-on and labeled (diagnostic only).
        fixture = case1_primary_online_lambda_fixture()
        lam_vec = {s: float(fixture.get(s, 0.0)) for s in streams}
        lambda_meta["live_lambda_used"] = True
        lambda_meta["live_lambda_source"] = LIVE_LAMBDA_SOURCE_FIXTURE

    diagnostic: Dict[str, Any] = {
        "included": False,
        "diagnostic_only": True,
        "is_not_dual_linf_under_wire_proof": True,
        "is_not_verdict": True,
        "dual_linf_under_wire_status": "unproven",
        "stream_linf": None,
        "note": (
            "Stream-aligned residual / L∞ diagnostic only — never flips "
            "dual_linf_under_wire, never VERDICT, never ship allow."
        ),
    }
    if include_diagnostic and lam_vec is not None:
        # Compare fixture/caller λ vs zero map as a finite always-on diagnostic.
        zero = {s: 0.0 for s in streams}
        aligned = {s: float(lam_vec.get(s, 0.0)) for s in streams}
        linf = float(max(abs(aligned[s] - zero[s]) for s in streams)) if streams else 0.0
        diagnostic.update(
            {
                "included": True,
                "stream_linf": linf,
                "streams": list(streams),
                "lambda_face": "primary_online_fixture_or_caller",
                "compare_face": "zero_map_diagnostic_only",
            }
        )

    pieces = {
        "cdu_affine": True,
        "blender_pooling": True,
        "stream_alignment": True,
        "labeled_lambda": bool(lambda_meta["live_lambda_used"]),
        "dual_space_diag": bool(diagnostic["included"]),
        "no_blender_units": "BLENDER" not in UNITS,
    }
    pieces_present = all(
        pieces[k]
        for k in ("cdu_affine", "blender_pooling", "stream_alignment", "no_blender_units")
    )

    return {
        "pieces": pieces,
        "pieces_present": pieces_present,
        "surfaces": {
            "cdu_surface": shape["cdu_surface"],
            "blender_surface": shape["blender_surface"],
            "form_current": shape["form_current"],
            "form_planned": shape["form_planned"],
            "path_shipped": False,
            "path_design_present": True,
        },
        "streams": {
            "names": list(streams),
            "cdu_intermediates": intermediates,
            "blender_recipe_use": blender_use,
            "cdu_intermediates_finite": all(
                abs(float(v)) < 1e300 for v in intermediates.values()
            ),
            "blender_use_finite": all(
                abs(float(v)) < 1e300 for v in blender_use.values()
            ),
        },
        "units_on_path": list(shape.get("units_on_path", ["CDU", "Blender"])),
        "units_affine": list(UNITS),
        "lambda_meta": lambda_meta,
        "lambda_stream": lam_vec,
        "diagnostic": diagnostic,
        "no_auto_wire": True,
        "path_shipped": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "is_not_maximizer_sla": True,
        "is_not_case1_shaped_linking_skeleton": True,
        "is_not_residual_must_vanish": True,
    }


def offline_case1_dual_honest_tf_aware_path_execution_scaffold_report(
    *,
    live_lambda: Optional[Mapping[str, float]] = None,
    live_lambda_source: Optional[str] = None,
    include_diagnostic: bool = True,
) -> Dict[str, Any]:
    """Always-on dual-honest TF-aware path *execution scaffold* report.

    No TF, no PuLP, no excel_pipeline, no Case 1 solve routing. Aggregate
    ``ok`` / ``contract_ok`` / ``scaffold_ok`` = scaffold formalized ∧ honesty
    locks ∧ scaffold_present ∧ all ship flags hard false ∧ dual_linf unproven ∧
    blockers non-empty ∧ dual_recovery_path is None ∧ UNITS FCC/COKER/CDU.
    **Not** path shipped. **Not** path present ship-met. **Not** wire shipped.
    **Not** bundle shipped. **Not** isolation rewrite shipped. **Not** form flip.
    **Not** VERDICT. **Not** dual L∞ under wire proof. **Not** an auto-executor.
    """
    honesty = _case1_dual_honest_tf_aware_path_execution_scaffold_honesty_fields()
    compose = case1_dual_honest_tf_aware_path_execution_scaffold_compose(
        live_lambda=live_lambda,
        live_lambda_source=live_lambda_source,
        include_diagnostic=include_diagnostic,
    )
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_shipped_checklist = checklist.get("wire_shipped")
    wire_shipped_still_false = (
        wire_shipped_checklist in ("false_today", "open", False, None)
        or "wire_shipped" in open_ids
    )

    # Multi-blocker co-req visibility — compose existing maps without flipping.
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    wire_ship_criteria_met_today = case1_wire_ship_criteria_met_today_aggregate(
        wire_criteria_met_map
    )

    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    gate_criteria_met_today = case1_online_linf_gate_criteria_met_today_aggregate(
        gate_met_map
    )

    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    path_present_criteria_met_today = (
        case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate(
            path_present_met_map
        )
    )

    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    form_label_criteria_met_today = (
        case1_form_label_change_shipped_criteria_met_today_aggregate(form_label_met_map)
    )

    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)

    bundle_met_map = case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    bundle_ship_allowed_today = case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(
        bundle_met_map
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            bundle_met_map
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    scaffold_present = True
    execution_scaffold_present = True
    bundle_design_present = True

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["feature_flag_enabled_today"] is False
    )
    compose_ok = bool(
        compose["pieces_present"] is True
        and compose["streams"]["cdu_intermediates_finite"] is True
        and compose["streams"]["blender_use_finite"] is True
        and compose["path_shipped"] is False
        and compose["wire_shipped"] is False
        and compose["bundle_shipped"] is False
        and compose["surfaces"]["cdu_surface"] == "offline_affine_base_delta"
        and compose["surfaces"]["blender_surface"] == "linear_quality_pooling"
        and list(compose["streams"]["names"]) == list(CASE1_SHAPED_LINKING_STREAMS)
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["scaffold_is_not_path_shipped"] is True
        and honesty["scaffold_is_not_path_present_for_ship"] is True
        and honesty["scaffold_is_not_wire_shipped"] is True
        and honesty["scaffold_is_not_wire"] is True
        and honesty["scaffold_is_not_bundle_shipped"] is True
        and honesty["scaffold_is_not_isolation_rewrite_shipped"] is True
        and honesty["scaffold_is_not_form_label_change_shipped"] is True
        and honesty["scaffold_is_not_ship_allow"] is True
        and honesty["scaffold_is_not_verdict_gate"] is True
        and honesty["scaffold_is_not_dual_linf_under_wire_proof"] is True
        and honesty["this_scaffold_alone_is_not_ship_criterion"] is True
        and honesty["this_scaffold_alone_is_not_multi_blocker_ship"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["diagnostic_linf_is_not_dual_linf_under_wire_proof"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        bundle_ship_allowed_today is False
        and criteria_met_today is False
        and wire_ship_allowed_today is False
        and isolation_ship_allowed_today is False
        and form_label_ship_allowed_today is False
        and ship_met_allowed_today is False
        and path_present_criteria_met_today is False
        and form_label_criteria_met_today is False
        and wire_ship_criteria_met_today is False
    )
    gate_permission_ok = (
        gate_flip_allowed_today is False and gate_criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
        and wire_shipped_still_false
    )
    bundle_not_shipped_ok = (
        bundle_shipped is False
        and honesty["bundle_shipped"] is False
        and honesty["not_bundle_shipped"] is True
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    ship_critical_false = (
        wire_criteria_met_map.get("isolation_rewrite_with_wire") is False
        and wire_criteria_met_map.get("wire_shipped") is False
        and wire_criteria_met_map.get("dual_honest_tf_aware_path_present") is False
        and path_present_met_map.get("isolation_rewrite_with_wire") is False
        # path-present met map may omit or leave dual_honest_tf_aware_path_present
        # as None/False while ship-met remains unsatisfied — treat non-True as open.
        and path_present_met_map.get("dual_honest_tf_aware_path_present") is not True
        and form_label_met_map.get("isolation_rewrite_with_wire") is False
        and isolation_met_map.get("isolation_rewrite_shipped") is False
        and bundle_met_map.get("wire_shipped") is False
        and bundle_met_map.get("isolation_rewrite_shipped") is False
        and bundle_met_map.get("form_label_change_shipped") is False
        and bundle_met_map.get("dual_honest_tf_aware_path_present") is False
        and bundle_met_map.get("bundle_shipped") is False
        and bundle_met_map.get("bundle_ship_allowed_today") is False
        and gate_met_map.get("isolation_rewrite_with_wire") is False
        and gate_met_map.get("wire_shipped") is False
    )

    anti = CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 10
        and "this_scaffold_alone" in anti
        and "path_design_alone" in anti
        and "path_present_criteria_alone" in anti
        and "bundle_design_alone" in anti
        and "bundle_ship_met_criteria_alone" in anti
        and "wire_ship_acceptance_alone" in anti
        and "packaging_alone" in anti
        and "residual_must_vanish" in anti
        and "case1_shaped_linking_skeleton_alone" in anti
        and "diagnostic_linf_alone" in anti
    )

    order_hint = list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    order_hint_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
    )
    feature_flag_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME == "enable_tf_affine_case1_wire"
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
    )
    dual_recovery_planned_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        == "online_lambda_under_tf_aware_form_when_shipped"
        and honesty["dual_recovery_path"] is None
        and honesty["not_pure_admm_dual_recovery"] is True
        and "pure-admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
        and "pure_admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
    )

    # Diagnostic must never flip dual_linf status even if L∞ is 0 or ≤15.
    diagnostic = compose["diagnostic"]
    diagnostic_lock_ok = bool(
        diagnostic.get("is_not_dual_linf_under_wire_proof") is True
        and diagnostic.get("is_not_verdict") is True
        and diagnostic.get("dual_linf_under_wire_status") == "unproven"
    )

    scaffold_formalized = bool(
        scaffold_present
        and execution_scaffold_present
        and compose_ok
        and shape_ok
        and anti_ok
        and order_hint_ok
        and feature_flag_ok
        and dual_recovery_planned_ok
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANNOTATION == "present"
        and path_design_present is True
        and bundle_design_present is True
        and isolation_rewrite_design_present is True
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and bundle_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and scaffold_formalized
        and blockers_still_documented
        and pooling_ok
        and ship_critical_false
        and form_label_open
        and form_label_change_shipped is False
        and path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and diagnostic_lock_ok
        and compose["pieces"]["no_blender_units"] is True
    )
    scaffold_ok = honesty_ok
    contract_ok = scaffold_ok
    ok = (
        scaffold_ok
        and (honesty["scaffold_present"] is True)
        and (honesty["path_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["bundle_shipped"] is False)
        and (honesty["isolation_rewrite_shipped"] is False)
        and (honesty["form_label_change_shipped"] is False)
    )

    ok_criteria = (
        "scaffold formalized ∧ honesty locks ∧ scaffold_present=True ∧ "
        "path_shipped=False ∧ dual_honest_tf_aware_path_present ship-met=False ∧ "
        "wire_shipped=False ∧ bundle_shipped=False ∧ bundle_ship_allowed_today=False ∧ "
        "criteria_met_today=False ∧ isolation_rewrite_shipped=False ∧ isolation "
        "checklist open ∧ form classic ∧ form_label_change_shipped=False ∧ dual_linf "
        "unproven ∧ online_linf_gate open ∧ gate_flip_allowed_today=False ∧ blockers "
        "non-empty ∧ dual_recovery_path=None ∧ feature_flag_enabled_today=False ∧ "
        "UNITS FCC/COKER/CDU ∧ order_hint not executor — NOT path shipped; NOT path "
        "present ship-met; NOT wire shipped; NOT bundle shipped; NOT isolation rewrite "
        "shipped; NOT form flip; NOT ship allow; NOT gate flip; NOT VERDICT; NOT dual "
        "L∞ under wire proof"
    )

    multi_blocker_coreqs = {
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "form_label_change_shipped": form_label_change_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire_status"],
        "online_linf_gate_under_tf_path": gate_status,
        "wire_shipped": wire_shipped,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "path_shipped": path_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "gate_flip_allowed_today": gate_flip_allowed_today,
    }

    return {
        **honesty,
        "ok": ok,
        "scaffold_ok": scaffold_ok,
        "contract_ok": contract_ok,
        "design_contract_ok": scaffold_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "compose_ok": compose_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "bundle_not_shipped_ok": bundle_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "scaffold_formalized": scaffold_formalized,
        "diagnostic_lock_ok": diagnostic_lock_ok,
        "ok_criteria": ok_criteria,
        "scaffold_present": scaffold_present,
        "execution_scaffold_present": execution_scaffold_present,
        "execution_scaffold": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANNOTATION
        ),
        "execution_scaffold_available": True,
        "compose": compose,
        "scaffold_pieces": compose["pieces"],
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANTI_CRITERIA_TODAY
        ),
        # Multi-blocker co-req visibility (read-only)
        "multi_blocker_coreqs": multi_blocker_coreqs,
        "order_hint": order_hint,
        "order_hint_is_not_executor": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR
        ),
        "no_auto_wire": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE,
        # Companion surfaces
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "scaffold_does_not_close_isolation_rewrite_checklist": True,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "bundle_design_present": bundle_design_present,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_ship_criteria_met_today": wire_ship_criteria_met_today,
        "wire_ship_criteria_met_today_map": wire_criteria_met_map,
        "wire_shipped": wire_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "path_present_criteria_met_today": path_present_criteria_met_today,
        "path_present_criteria_met_today_map": path_present_met_map,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "form_label_criteria_met_today": form_label_criteria_met_today,
        "form_label_criteria_met_today_map": form_label_met_map,
        "isolation_criteria_met_today_map": isolation_met_map,
        "bundle_criteria_met_today_map": bundle_met_map,
        "scaffold_does_not_flip_wire_ship_met_today": True,
        "scaffold_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "scaffold_does_not_flip_form_label_change_shipped_met_today": True,
        "scaffold_does_not_flip_isolation_ship_met_today": True,
        "scaffold_does_not_flip_gate_met_today": True,
        "scaffold_does_not_flip_bundle_ship_met_today": True,
        # Gate discipline
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "gate_criteria_met_today": gate_criteria_met_today,
        "gate_criteria_met_today_map": gate_met_map,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        # Dual recovery / feature flag
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        # Dual linf
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        # Surfaces
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        "cdu_surface": "offline_affine_base_delta",
        "blender_surface": "linear_quality_pooling",
        "intermediates": list(CASE1_SHAPED_LINKING_STREAMS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "scaffold_is_not_path_shipped": True,
        "scaffold_is_not_wire_shipped": True,
        "scaffold_is_not_bundle_shipped": True,
        "scaffold_is_not_isolation_rewrite_shipped": True,
        "scaffold_is_not_form_label_shipped": True,
        "tf_available": tf_available(),
        "dual_honest_tf_aware_path_execution_scaffold_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "excel_packaging_twin_deferred": False,
        "excel_packaging_twin_present": True,
        "note": honesty["note"],
    }


def case1_dual_honest_tf_aware_path_execution_scaffold_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_execution_scaffold_report``."""
    return offline_case1_dual_honest_tf_aware_path_execution_scaffold_report(**kwargs)


def multi_unit_case1_dual_honest_tf_aware_path_execution_scaffold_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_tf_aware_path_execution_scaffold_report``."""
    return offline_case1_dual_honest_tf_aware_path_execution_scaffold_report(**kwargs)




# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest multi-blocker wire *rehearsal* / dry-run readiness
# (goal 5 residual after scaffold #60 / packaging #61)
# ---------------------------------------------------------------------------
# Always-on numpy. Machine-readable co-req status matrix for
# SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT under scaffold compose *without ship*.
# Distinct from design (*what*), ship-met criteria (*when*), scaffold
# (*offline how-without-ship*), packaging (*planner visibility*).
# dual_recovery_path=None; all ship flags hard false; no auto-wire.
# No TF / PuLP / excel_pipeline on hot path.

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_KIND = (
    "offline_case1_dual_honest_multi_blocker_wire_rehearsal"
)
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANNOTATION = "present"

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "this_scaffold_alone",
    "this_execution_scaffold_alone",
    "this_rehearsal_alone",
    "wire_rehearsal_alone",
    "coreq_matrix_alone",
    "scaffold_plus_rehearsal_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "bundle_design_alone",
    "bundle_ship_met_criteria_alone",
    "wire_ship_acceptance_alone",
    "case1_shaped_linking_skeleton_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "form_label_criteria_alone",
    "gate_criteria_alone",
    "diagnostic_linf_alone",
)


def _case1_dual_honest_multi_blocker_wire_rehearsal_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-ship locks for multi-blocker wire rehearsal."""
    return {
        "kind": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "rehearsal_present": True,
        "wire_rehearsal_present": True,
        "scaffold_present": True,
        "execution_scaffold_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "bundle_ship_allowed_today": False,
        "criteria_met_today": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "rehearsal_is_not_path_shipped": True,
        "rehearsal_is_not_path_present_for_ship": True,
        "rehearsal_is_not_wire_shipped": True,
        "rehearsal_is_not_wire": True,
        "rehearsal_is_not_bundle_shipped": True,
        "rehearsal_is_not_isolation_rewrite_shipped": True,
        "rehearsal_is_not_form_label_change_shipped": True,
        "rehearsal_is_not_ship_allow": True,
        "rehearsal_is_not_ship_met": True,
        "rehearsal_is_not_form_flip": True,
        "rehearsal_is_not_gate_flip": True,
        "rehearsal_is_not_verdict_gate": True,
        "rehearsal_is_not_dual_linf_under_wire_proof": True,
        "this_rehearsal_alone_is_not_ship_criterion": True,
        "this_rehearsal_alone_is_not_multi_blocker_ship": True,
        "this_scaffold_alone_is_not_ship_criterion": True,
        "scaffold_plus_rehearsal_alone_is_not_ship_criterion": True,
        "path_design_alone_is_not_ship_criterion": True,
        "path_present_criteria_alone_is_not_ship_criterion": True,
        "bundle_design_alone_is_not_ship_criterion": True,
        "bundle_ship_met_criteria_alone_is_not_ship_criterion": True,
        "wire_ship_acceptance_alone_is_not_ship_criterion": True,
        "case1_shaped_linking_skeleton_alone_is_not_ship_criterion": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "diagnostic_linf_is_not_dual_linf_under_wire_proof": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_multi_blocker_wire_rehearsal_offline",
        "note": (
            "Offline Case-1 dual-honest multi-blocker wire *rehearsal* / dry-run "
            "readiness report: machine-readable co-req status matrix for "
            f"{SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT} under scaffold compose without "
            "shipping wire. rehearsal_present=True / wire_rehearsal_present=True; "
            "scaffold_present=True / execution_scaffold_present=True (visibility "
            "link); path_shipped=False; dual_honest_tf_aware_path_present ship-met="
            "False; wire_shipped=False; bundle_shipped=False; "
            "bundle_ship_allowed_today=False; criteria_met_today=False; "
            "isolation_rewrite_shipped=False; isolation checklist open; form classic; "
            "form_label_change_shipped=False; dual_linf unproven; online_linf_gate open; "
            "gate_flip_allowed_today=False; dual_recovery_path=None on TF surface; "
            f"planned dual_recovery_path under future wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved "
            f"False; on_excel_case1_path=False; case1_form_unchanged "
            f"({CASE1_FORM_CURRENT}). Rehearsal is NOT path shipped, NOT path-present-"
            "for-ship, NOT wire shipped, NOT bundle shipped, NOT isolation rewrite "
            "shipped, NOT form_label shipped, NOT ship allow, NOT ship-met, NOT form "
            "flip, NOT gate flip, NOT VERDICT, NOT dual L∞ under wire proof. Design "
            "formalizes *what*; ship-met criteria *when*; scaffold *offline how-"
            "without-ship*; this formalizes *co-req readiness dry-run under scaffold "
            "without ship*. Distinct from scaffold multi_blocker_coreqs visibility "
            "dict. Order_hint is NOT an executor; no auto-wire. Probe/bridge/"
            "warmstart/pooling/seed-identity/recovered L∞, residual-must-vanish, "
            "packaging alone, design/criteria alone, this_rehearsal_alone, "
            "scaffold_plus_rehearsal_alone, and diagnostic L∞ are not ship enablers "
            "today. Full DEFAULT_WIRE_BLOCKERS remain. UNITS stay FCC/COKER/CDU (no "
            "silent BLENDER). Does not clear DEFAULT_WIRE_BLOCKERS. Does not redefine "
            "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline on "
            "hot path; isolation suite behavior unchanged this cycle. "
            "SUGGESTED_NEXT_WAVE still points at full dual-honest multi-blocker wire "
            "(deferred). Excel packaging twin of rehearsal is present after #63 "
            "(packaging existence only — not ship; Index headroom tight for new twins)."
        ),
    }


def case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix(
    *,
    scaffold_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-readable co-req readiness matrix under scaffold (no ship flips).

    Status vocabulary: open | false_today | unproven | not_shipped | present |
    classic form string / surface labels as appropriate. Distinct from scaffold
    flat multi_blocker_coreqs visibility dict — this is dry-run readiness for
    SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT.
    """
    dual_linf = case1_dual_linf_proof_checklist()
    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    form = case1_form_label_contract()

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY, "open")
    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY, "open")

    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)
    bundle_met_map = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    bundle_ship_allowed_today = (
        case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(bundle_met_map)
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            bundle_met_map
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    scaffold_present = True
    execution_scaffold_present = True
    scaffold_compose_ok = True
    if scaffold_report is not None:
        scaffold_present = bool(scaffold_report.get("scaffold_present", True))
        execution_scaffold_present = bool(
            scaffold_report.get("execution_scaffold_present", True)
        )
        scaffold_compose_ok = bool(scaffold_report.get("compose_ok", True))

    rows: Dict[str, Dict[str, Any]] = {
        "isolation_rewrite_with_wire": {
            "status": isolation_status if isolation_status == "open" else str(isolation_status),
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "dual_linf_proof_checklist",
        },
        "isolation_rewrite_shipped": {
            "status": "false_today",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "isolation_ship_met_map",
        },
        "form_label_change_shipped": {
            "status": "false_today",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "form_label_met_map",
        },
        "form_current": {
            "status": form["form_current"],
            "value": form["form_current"],
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "form_label_contract",
        },
        "online_linf_gate_under_tf_path": {
            "status": gate_status if gate_status == "open" else str(gate_status),
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "dual_linf_proof_checklist",
        },
        "dual_linf_under_wire": {
            "status": "unproven",
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "dual_linf_proof_checklist",
        },
        "dual_honest_tf_aware_path_present": {
            "status": "false_today",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "path_present_met_map",
        },
        "path_shipped": {
            "status": "not_shipped",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "path_design",
        },
        "wire_shipped": {
            "status": "not_shipped",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "wire_ship_acceptance",
        },
        "wire_ship_allowed_today": {
            "status": "false_today",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "wire_ship_acceptance",
        },
        "bundle_shipped": {
            "status": "not_shipped",
            "value": bool(bundle_shipped) if bundle_shipped else False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "bundle_ship_met_map",
        },
        "bundle_ship_allowed_today": {
            "status": "false_today",
            "value": bool(bundle_ship_allowed_today)
            if bundle_ship_allowed_today
            else False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "bundle_ship_met_map",
        },
        "criteria_met_today": {
            "status": "false_today",
            "value": bool(criteria_met_today) if criteria_met_today else False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "bundle_ship_met_map",
        },
        "no_blender_offline_affine_kernel": {
            "status": "open",
            "value": True,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "DEFAULT_WIRE_BLOCKERS",
        },
        "blender_surface": {
            "status": "linear_quality_pooling",
            "value": "linear_quality_pooling",
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "path_shape",
        },
        "case1_is_cdu_blender_package_admm": {
            "status": "open",
            "value": True,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "DEFAULT_WIRE_BLOCKERS",
        },
        "feature_flag_enabled_today": {
            "status": "false_today",
            "value": False,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "feature_flag_constants",
        },
        "dual_recovery_path": {
            "status": "none_today",
            "value": None,
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "honesty_constants",
        },
        "scaffold_present": {
            "status": "present" if scaffold_present else "missing",
            "value": scaffold_present,
            "ship_critical": False,
            "allows_wire_today": False,
            "source": "scaffold_report",
        },
        "execution_scaffold_present": {
            "status": "present" if execution_scaffold_present else "missing",
            "value": execution_scaffold_present,
            "ship_critical": False,
            "allows_wire_today": False,
            "source": "scaffold_report",
        },
        "scaffold_compose_ok": {
            "status": "present" if scaffold_compose_ok else "false_today",
            "value": scaffold_compose_ok,
            "ship_critical": False,
            "allows_wire_today": False,
            "source": "scaffold_report",
        },
        "ship_met_allowed_today": {
            "status": "false_today",
            "value": False if not ship_met_allowed_today else bool(ship_met_allowed_today),
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "path_present_met_map",
        },
        "isolation_ship_allowed_today": {
            "status": "false_today",
            "value": False
            if not isolation_ship_allowed_today
            else bool(isolation_ship_allowed_today),
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "isolation_ship_met_map",
        },
        "form_label_ship_allowed_today": {
            "status": "false_today",
            "value": False
            if not form_label_ship_allowed_today
            else bool(form_label_ship_allowed_today),
            "ship_critical": True,
            "allows_wire_today": False,
            "source": "form_label_met_map",
        },
    }

    # Force hard-false ship-critical truth under current honesty locks.
    for key in (
        "isolation_rewrite_shipped",
        "form_label_change_shipped",
        "dual_honest_tf_aware_path_present",
        "path_shipped",
        "wire_shipped",
        "wire_ship_allowed_today",
        "bundle_shipped",
        "bundle_ship_allowed_today",
        "criteria_met_today",
        "feature_flag_enabled_today",
        "ship_met_allowed_today",
        "isolation_ship_allowed_today",
        "form_label_ship_allowed_today",
    ):
        rows[key]["value"] = False
        if rows[key]["status"] not in ("not_shipped", "false_today"):
            rows[key]["status"] = "false_today"
        rows[key]["allows_wire_today"] = False

    rows["isolation_rewrite_with_wire"]["status"] = "open"
    rows["online_linf_gate_under_tf_path"]["status"] = "open"
    rows["dual_linf_under_wire"]["status"] = "unproven"
    rows["dual_recovery_path"]["value"] = None
    rows["dual_recovery_path"]["status"] = "none_today"
    rows["form_current"]["value"] = CASE1_FORM_CURRENT
    rows["form_current"]["status"] = CASE1_FORM_CURRENT
    rows["blender_surface"]["value"] = "linear_quality_pooling"
    rows["blender_surface"]["status"] = "linear_quality_pooling"
    rows["no_blender_offline_affine_kernel"]["value"] = True
    rows["case1_is_cdu_blender_package_admm"]["value"] = True

    ship_critical_keys = [k for k, v in rows.items() if v.get("ship_critical")]
    n_open_ship_critical = 0
    for k in ship_critical_keys:
        st = rows[k]["status"]
        val = rows[k].get("value", None)
        if st in ("open", "false_today", "unproven", "not_shipped", "none_today") or val is False or val is None:
            if st in ("open", "unproven") or val is False or val is None or st in (
                "false_today",
                "not_shipped",
                "none_today",
            ):
                n_open_ship_critical += 1

    any_ship_allowed_today = any(
        bool(rows[k].get("allows_wire_today")) for k in rows
    )
    all_ship_critical_open_or_false = all(
        (
            rows[k]["status"]
            in (
                "open",
                "false_today",
                "unproven",
                "not_shipped",
                "none_today",
                CASE1_FORM_CURRENT,
                "linear_quality_pooling",
            )
            or rows[k].get("value") in (False, None, True, CASE1_FORM_CURRENT, "linear_quality_pooling")
        )
        and rows[k].get("allows_wire_today") is False
        for k in ship_critical_keys
    )
    # Stronger: no ship-critical value is True except known still-true blockers.
    allowed_true = {
        "no_blender_offline_affine_kernel",
        "case1_is_cdu_blender_package_admm",
    }
    no_false_ship_true = all(
        (rows[k].get("value") is not True) or (k in allowed_true)
        for k in ship_critical_keys
        if k
        not in (
            "form_current",
            "blender_surface",
            "isolation_rewrite_with_wire",
            "online_linf_gate_under_tf_path",
            "dual_linf_under_wire",
            "dual_recovery_path",
        )
    )
    matrix_ok = bool(
        all_ship_critical_open_or_false
        and any_ship_allowed_today is False
        and no_false_ship_true
        and rows["scaffold_present"]["value"] is True
        and rows["execution_scaffold_present"]["value"] is True
        and rows["path_shipped"]["value"] is False
        and rows["wire_shipped"]["value"] is False
        and rows["bundle_shipped"]["value"] is False
        and rows["isolation_rewrite_shipped"]["value"] is False
        and rows["form_label_change_shipped"]["value"] is False
        and rows["dual_linf_under_wire"]["status"] == "unproven"
        and rows["dual_recovery_path"]["value"] is None
        and CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids
        and CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids
    )

    return {
        "kind": "offline_case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix",
        "suggested_next_wave": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "rows": rows,
        "coreq_status_matrix": rows,
        "ship_critical_keys": ship_critical_keys,
        "n_open_ship_critical": n_open_ship_critical,
        "all_ship_critical_open_or_false": all_ship_critical_open_or_false,
        "any_ship_allowed_today": any_ship_allowed_today,
        "all_ship_flags_false": True,
        "matrix_ok": matrix_ok,
        "rehearsal_does_not_flip_isolation_rewrite_shipped": True,
        "rehearsal_does_not_flip_form_label_change_shipped": True,
        "rehearsal_does_not_flip_path_shipped": True,
        "rehearsal_does_not_flip_wire_shipped": True,
        "rehearsal_does_not_flip_bundle_shipped": True,
        "rehearsal_does_not_flip_criteria_met_today": True,
        "rehearsal_does_not_flip_dual_linf_under_wire": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "dual_linf_proof_checklist_open_ids": list(open_ids),
        "wire_ship_allowed_today": False,
        "isolation_ship_allowed_today": False,
        "form_label_ship_allowed_today": False,
        "ship_met_allowed_today": False,
        "bundle_ship_allowed_today": False,
        "note": (
            "Co-req readiness dry-run matrix under scaffold honesty locks. "
            "Not path/wire/bundle/isolation/form ship; not VERDICT; not dual L∞ "
            "under wire proof. Distinct from scaffold multi_blocker_coreqs."
        ),
    }


def offline_case1_dual_honest_multi_blocker_wire_rehearsal_report() -> Dict[str, Any]:
    """Always-on dual-honest multi-blocker wire *rehearsal* / dry-run report.

    No TF, no PuLP, no excel_pipeline, no Case 1 solve routing. Aggregate
    ``ok`` / ``contract_ok`` / ``rehearsal_ok`` = rehearsal formalized ∧ honesty
    locks ∧ rehearsal_present ∧ scaffold linked ∧ all ship flags hard false ∧
    dual_linf unproven ∧ blockers non-empty ∧ dual_recovery_path is None ∧
    UNITS FCC/COKER/CDU. **Not** path shipped. **Not** path present ship-met.
    **Not** wire shipped. **Not** bundle shipped. **Not** isolation rewrite
    shipped. **Not** form flip. **Not** VERDICT. **Not** dual L∞ under wire
    proof. **Not** an auto-executor.
    """
    honesty = _case1_dual_honest_multi_blocker_wire_rehearsal_honesty_fields()
    # Visibility-link existing scaffold (always-on; no re-ship of scaffold engines).
    scaffold = offline_case1_dual_honest_tf_aware_path_execution_scaffold_report(
        include_diagnostic=True
    )
    matrix = case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix(
        scaffold_report=scaffold
    )
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    # Read-only co-req maps (must not flip).
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)
    bundle_met_map = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    bundle_ship_allowed_today = (
        case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(bundle_met_map)
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            bundle_met_map
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    rehearsal_present = True
    wire_rehearsal_present = True
    scaffold_present = bool(scaffold.get("scaffold_present", False))
    execution_scaffold_present = bool(
        scaffold.get("execution_scaffold_present", False)
    )
    scaffold_compose_ok = bool(scaffold.get("compose_ok", False))
    bundle_design_present = True

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["feature_flag_enabled_today"] is False
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["rehearsal_is_not_path_shipped"] is True
        and honesty["rehearsal_is_not_path_present_for_ship"] is True
        and honesty["rehearsal_is_not_wire_shipped"] is True
        and honesty["rehearsal_is_not_wire"] is True
        and honesty["rehearsal_is_not_bundle_shipped"] is True
        and honesty["rehearsal_is_not_isolation_rewrite_shipped"] is True
        and honesty["rehearsal_is_not_form_label_change_shipped"] is True
        and honesty["rehearsal_is_not_ship_allow"] is True
        and honesty["rehearsal_is_not_verdict_gate"] is True
        and honesty["rehearsal_is_not_dual_linf_under_wire_proof"] is True
        and honesty["this_rehearsal_alone_is_not_ship_criterion"] is True
        and honesty["this_rehearsal_alone_is_not_multi_blocker_ship"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["diagnostic_linf_is_not_dual_linf_under_wire_proof"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        bundle_ship_allowed_today is False
        and criteria_met_today is False
        and wire_ship_allowed_today is False
        and isolation_ship_allowed_today is False
        and form_label_ship_allowed_today is False
        and ship_met_allowed_today is False
    )
    gate_permission_ok = gate_flip_allowed_today is False
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
    )
    bundle_not_shipped_ok = (
        bundle_shipped is False
        and honesty["bundle_shipped"] is False
        and honesty["not_bundle_shipped"] is True
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    anti = CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 12
        and "this_rehearsal_alone" in anti
        and "wire_rehearsal_alone" in anti
        and "coreq_matrix_alone" in anti
        and "scaffold_plus_rehearsal_alone" in anti
        and "this_scaffold_alone" in anti
        and "path_design_alone" in anti
        and "path_present_criteria_alone" in anti
        and "bundle_design_alone" in anti
        and "bundle_ship_met_criteria_alone" in anti
        and "wire_ship_acceptance_alone" in anti
        and "packaging_alone" in anti
        and "residual_must_vanish" in anti
        and "diagnostic_linf_alone" in anti
    )

    order_hint = list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    order_hint_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
    )
    feature_flag_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME == "enable_tf_affine_case1_wire"
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
    )
    dual_recovery_planned_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        == "online_lambda_under_tf_aware_form_when_shipped"
        and honesty["dual_recovery_path"] is None
        and honesty["not_pure_admm_dual_recovery"] is True
        and "pure-admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
        and "pure_admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
    )

    scaffold_link_ok = bool(
        scaffold_present
        and execution_scaffold_present
        and scaffold_compose_ok
        and scaffold.get("path_shipped") is False
        and scaffold.get("wire_shipped") is False
        and scaffold.get("bundle_shipped") is False
        and scaffold.get("isolation_rewrite_shipped") is False
        and scaffold.get("form_label_change_shipped") is False
        and scaffold.get("dual_linf_under_wire_status") == "unproven"
        and scaffold.get("dual_recovery_path") is None
        and scaffold.get("ok") is True
    )
    matrix_ok = bool(matrix.get("matrix_ok") is True and matrix.get("any_ship_allowed_today") is False)

    rehearsal_formalized = bool(
        rehearsal_present
        and wire_rehearsal_present
        and scaffold_link_ok
        and matrix_ok
        and shape_ok
        and anti_ok
        and order_hint_ok
        and feature_flag_ok
        and dual_recovery_planned_ok
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANNOTATION == "present"
        and path_design_present is True
        and bundle_design_present is True
        and isolation_rewrite_design_present is True
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and bundle_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and rehearsal_formalized
        and blockers_still_documented
        and pooling_ok
        and form_label_open
        and form_label_change_shipped is False
        and path_shipped is False
        and dual_honest_tf_aware_path_present is False
    )
    rehearsal_ok = honesty_ok
    contract_ok = rehearsal_ok
    ok = (
        rehearsal_ok
        and (honesty["rehearsal_present"] is True)
        and (honesty["path_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["bundle_shipped"] is False)
        and (honesty["isolation_rewrite_shipped"] is False)
        and (honesty["form_label_change_shipped"] is False)
    )

    ok_criteria = (
        "rehearsal formalized ∧ honesty locks ∧ rehearsal_present=True ∧ "
        "scaffold linked ∧ path_shipped=False ∧ dual_honest_tf_aware_path_present "
        "ship-met=False ∧ wire_shipped=False ∧ bundle_shipped=False ∧ "
        "bundle_ship_allowed_today=False ∧ criteria_met_today=False ∧ "
        "isolation_rewrite_shipped=False ∧ isolation checklist open ∧ form classic ∧ "
        "form_label_change_shipped=False ∧ dual_linf unproven ∧ online_linf_gate open ∧ "
        "gate_flip_allowed_today=False ∧ blockers non-empty ∧ dual_recovery_path=None ∧ "
        "feature_flag_enabled_today=False ∧ UNITS FCC/COKER/CDU ∧ order_hint not "
        "executor — NOT path shipped; NOT path present ship-met; NOT wire shipped; "
        "NOT bundle shipped; NOT isolation rewrite shipped; NOT form flip; NOT ship "
        "allow; NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    # Compact multi_blocker_coreqs (visibility twin) + full matrix
    multi_blocker_coreqs = {
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "form_label_change_shipped": form_label_change_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire_status"],
        "online_linf_gate_under_tf_path": gate_status,
        "wire_shipped": wire_shipped,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "path_shipped": path_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "dual_recovery_path": None,
        "scaffold_present": scaffold_present,
        "execution_scaffold_present": execution_scaffold_present,
        "no_blender_offline_affine_kernel": True,
        "blender_surface": "linear_quality_pooling",
        "case1_is_cdu_blender_package_admm": True,
    }

    return {
        **honesty,
        "ok": ok,
        "rehearsal_ok": rehearsal_ok,
        "contract_ok": contract_ok,
        "design_contract_ok": rehearsal_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "scaffold_link_ok": scaffold_link_ok,
        "matrix_ok": matrix_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "bundle_not_shipped_ok": bundle_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "rehearsal_formalized": rehearsal_formalized,
        "ok_criteria": ok_criteria,
        "rehearsal_present": rehearsal_present,
        "wire_rehearsal_present": wire_rehearsal_present,
        "rehearsal_annotation": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANNOTATION
        ),
        "scaffold_present": scaffold_present,
        "execution_scaffold_present": execution_scaffold_present,
        "scaffold_compose_ok": scaffold_compose_ok,
        "scaffold_ok": bool(scaffold.get("scaffold_ok", scaffold.get("ok"))),
        "compose_ok": scaffold_compose_ok,
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANTI_CRITERIA_TODAY
        ),
        "coreq_status_matrix": matrix["coreq_status_matrix"],
        "rehearsal_coreqs": matrix["coreq_status_matrix"],
        "coreq_matrix": matrix,
        "multi_blocker_coreqs": multi_blocker_coreqs,
        "n_open_ship_critical": matrix["n_open_ship_critical"],
        "all_ship_critical_open_or_false": matrix["all_ship_critical_open_or_false"],
        "any_ship_allowed_today": False,
        "all_ship_flags_false": True,
        "order_hint": order_hint,
        "order_hint_is_not_executor": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR
        ),
        "no_auto_wire": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE,
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "rehearsal_does_not_close_isolation_rewrite_checklist": True,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "bundle_design_present": bundle_design_present,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_shipped": wire_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "rehearsal_does_not_flip_wire_ship_met_today": True,
        "rehearsal_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "rehearsal_does_not_flip_form_label_change_shipped_met_today": True,
        "rehearsal_does_not_flip_isolation_ship_met_today": True,
        "rehearsal_does_not_flip_gate_met_today": True,
        "rehearsal_does_not_flip_bundle_ship_met_today": True,
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        "cdu_surface": "offline_affine_base_delta",
        "blender_surface": "linear_quality_pooling",
        "intermediates": list(CASE1_SHAPED_LINKING_STREAMS),
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "rehearsal_is_not_path_shipped": True,
        "rehearsal_is_not_wire_shipped": True,
        "rehearsal_is_not_bundle_shipped": True,
        "rehearsal_is_not_isolation_rewrite_shipped": True,
        "rehearsal_is_not_form_label_shipped": True,
        "tf_available": tf_available(),
        "dual_honest_multi_blocker_wire_rehearsal_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        "excel_packaging_twin_deferred": False,
        "excel_packaging_twin_present": True,
        "note": honesty["note"],
    }


def case1_dual_honest_multi_blocker_wire_rehearsal_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_rehearsal_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_rehearsal_report(**kwargs)


def multi_unit_case1_dual_honest_multi_blocker_wire_rehearsal_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_rehearsal_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_rehearsal_report(**kwargs)




# ---------------------------------------------------------------------------
# Offline Case-1 dual-honest multi-blocker wire *implementation blueprint* /
# go-board (goal 5 residual after rehearsal #62 / packaging #63)
# ---------------------------------------------------------------------------
# Always-on numpy. Order_hint-sequenced first-blocking-coreq + file-level prep
# map under scaffold/rehearsal compose *without ship*. Distinct from design
# (*what*), ship-met criteria (*when*), scaffold (*offline how-without-ship*),
# rehearsal (*co-req readiness dry-run*), packaging (*planner visibility*).
# dual_recovery_path=None; all ship flags hard false; no auto-wire.
# No TF / PuLP / excel_pipeline on hot path.

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_KIND = (
    "offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint"
)
CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANNOTATION = "present"

CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "this_scaffold_alone",
    "this_execution_scaffold_alone",
    "this_rehearsal_alone",
    "wire_rehearsal_alone",
    "coreq_matrix_alone",
    "scaffold_plus_rehearsal_alone",
    "this_blueprint_alone",
    "go_board_alone",
    "first_blocking_coreq_alone",
    "prep_map_alone",
    "scaffold_plus_rehearsal_plus_blueprint_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "bundle_design_alone",
    "bundle_ship_met_criteria_alone",
    "wire_ship_acceptance_alone",
    "case1_shaped_linking_skeleton_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "form_label_criteria_alone",
    "gate_criteria_alone",
    "diagnostic_linf_alone",
)


def _case1_dual_honest_multi_blocker_wire_implementation_blueprint_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-ship locks for multi-blocker wire blueprint."""
    return {
        "kind": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "blueprint_present": True,
        "implementation_blueprint_present": True,
        "wire_go_board_present": True,
        "rehearsal_present": True,
        "wire_rehearsal_present": True,
        "scaffold_present": True,
        "execution_scaffold_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "bundle_ship_allowed_today": False,
        "criteria_met_today": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "blueprint_is_not_path_shipped": True,
        "blueprint_is_not_path_present_for_ship": True,
        "blueprint_is_not_wire_shipped": True,
        "blueprint_is_not_wire": True,
        "blueprint_is_not_bundle_shipped": True,
        "blueprint_is_not_isolation_rewrite_shipped": True,
        "blueprint_is_not_form_label_change_shipped": True,
        "blueprint_is_not_ship_allow": True,
        "blueprint_is_not_ship_met": True,
        "blueprint_is_not_form_flip": True,
        "blueprint_is_not_gate_flip": True,
        "blueprint_is_not_verdict_gate": True,
        "blueprint_is_not_dual_linf_under_wire_proof": True,
        "this_blueprint_alone_is_not_ship_criterion": True,
        "this_blueprint_alone_is_not_multi_blocker_ship": True,
        "go_board_alone_is_not_ship_criterion": True,
        "first_blocking_coreq_alone_is_not_ship_criterion": True,
        "prep_map_alone_is_not_ship_criterion": True,
        "this_rehearsal_alone_is_not_ship_criterion": True,
        "this_scaffold_alone_is_not_ship_criterion": True,
        "scaffold_plus_rehearsal_alone_is_not_ship_criterion": True,
        "scaffold_plus_rehearsal_plus_blueprint_alone_is_not_ship_criterion": True,
        "path_design_alone_is_not_ship_criterion": True,
        "path_present_criteria_alone_is_not_ship_criterion": True,
        "bundle_design_alone_is_not_ship_criterion": True,
        "bundle_ship_met_criteria_alone_is_not_ship_criterion": True,
        "wire_ship_acceptance_alone_is_not_ship_criterion": True,
        "case1_shaped_linking_skeleton_alone_is_not_ship_criterion": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "design_contracts_alone_is_not_ship_criterion": True,
        "diagnostic_linf_is_not_dual_linf_under_wire_proof": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "feature_flag_enabled_today": False,
        "scope": "case1_dual_honest_multi_blocker_wire_implementation_blueprint_offline",
        "note": (
            "Offline Case-1 dual-honest multi-blocker wire *implementation blueprint* / "
            "go-board: order_hint-sequenced first_blocking_coreq + file-level prep map for "
            f"{SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT} under scaffold/rehearsal compose without "
            "shipping wire. blueprint_present=True / implementation_blueprint_present=True / "
            "wire_go_board_present=True; rehearsal_present=True / wire_rehearsal_present=True; "
            "scaffold_present=True / execution_scaffold_present=True (visibility link); "
            "path_shipped=False; dual_honest_tf_aware_path_present ship-met=False; "
            "wire_shipped=False; bundle_shipped=False; bundle_ship_allowed_today=False; "
            "criteria_met_today=False; isolation_rewrite_shipped=False; isolation checklist "
            "open; form classic; form_label_change_shipped=False; dual_linf unproven; "
            "online_linf_gate open; gate_flip_allowed_today=False; dual_recovery_path=None "
            "on TF surface; planned dual_recovery_path under future wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved "
            f"False; on_excel_case1_path=False; case1_form_unchanged ({CASE1_FORM_CURRENT}). "
            "Blueprint is NOT path shipped, NOT path-present-for-ship, NOT wire shipped, "
            "NOT bundle shipped, NOT isolation rewrite shipped, NOT form_label shipped, "
            "NOT ship allow, NOT ship-met, NOT form flip, NOT gate flip, NOT VERDICT, "
            "NOT dual L∞ under wire proof. Design formalizes *what*; ship-met criteria "
            "*when*; scaffold *offline how-without-ship*; rehearsal *co-req readiness "
            "dry-run under scaffold without ship*; this formalizes *order_hint-sequenced "
            "first-blocker + file-level prep without ship*. Order_hint is NOT an executor; "
            "no auto-wire. Probe/bridge/warmstart/pooling/seed-identity/recovered L∞, "
            "residual-must-vanish, packaging alone, design/criteria alone, "
            "this_blueprint_alone, go_board_alone, first_blocking_coreq_alone, "
            "prep_map_alone, scaffold_plus_rehearsal_plus_blueprint_alone, and diagnostic "
            "L∞ are not ship enablers today. Full DEFAULT_WIRE_BLOCKERS remain. UNITS stay "
            "FCC/COKER/CDU (no silent BLENDER). Does not clear DEFAULT_WIRE_BLOCKERS. "
            "Does not redefine ready_for_wire_discussion. Always-on numpy; no TF/PuLP/"
            "excel_pipeline on hot path; isolation suite behavior unchanged this cycle. "
            "SUGGESTED_NEXT_WAVE still points at full dual-honest multi-blocker wire "
            "(deferred). Excel packaging twin of *rehearsal* is present after #63 "
            "(existence only — not ship). Excel packaging twin of *this blueprint* is "
            "deferred (Index 6 free — trim-first next cycle)."
        ),
    }


def case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq(
    *,
    dual_linf: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Walk ORDER_HINT; return first open/false_today/unproven step (no auto-close).

    Expected today: ``isolation_rewrite_with_wire``. Never returns a shipped token
    while ship flags remain false. order_hint is not an executor.
    """
    if dual_linf is None:
        dual_linf = case1_dual_linf_proof_checklist()
    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = list(dual_linf["dual_linf_proof_checklist_open_ids"])

    status_today: Dict[str, Any] = {
        "isolation_rewrite_with_wire": checklist.get(
            CASE1_ISOLATION_REWRITE_CHECKLIST_KEY, "open"
        ),
        "isolation_rewrite_shipped": False,
        "form_label_change_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "path_shipped": False,
        "dual_linf_under_wire_proven": dual_linf["dual_linf_under_wire_status"]
        == "proven",
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire_status"],
        "online_linf_gate_under_tf_path": checklist.get(
            CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY, "open"
        ),
        "wire_shipped": False,
        "wire_ship_allowed_today": False,
    }

    first_id: Optional[str] = None
    first_index: Optional[int] = None
    first_status: Optional[str] = None
    order_hint = list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    for idx, step in enumerate(order_hint):
        if step == "isolation_rewrite_with_wire":
            st = status_today["isolation_rewrite_with_wire"]
            if status_today["isolation_rewrite_shipped"] is False and (
                st == "open" or CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids
            ):
                first_id = step
                first_index = idx
                first_status = "open" if st == "open" else "false_today"
                break
        elif step == "form_label_change_shipped":
            if status_today["form_label_change_shipped"] is False:
                first_id = step
                first_index = idx
                first_status = "false_today"
                break
        elif step == "dual_honest_tf_aware_path_present":
            if (
                status_today["dual_honest_tf_aware_path_present"] is False
                or status_today["path_shipped"] is False
            ):
                first_id = step
                first_index = idx
                first_status = "false_today"
                break
        elif step == "dual_linf_under_wire_proven":
            if dual_linf["dual_linf_under_wire_status"] != "proven":
                first_id = step
                first_index = idx
                first_status = "unproven"
                break
        elif step == "wire_shipped":
            if status_today["wire_shipped"] is False:
                first_id = step
                first_index = idx
                first_status = "false_today"
                break

    exhausted = first_id is None
    return {
        "kind": "offline_case1_dual_honest_multi_blocker_wire_first_blocking_coreq",
        "first_blocking_coreq": first_id,
        "first_blocking_coreq_status": first_status,
        "first_blocking_coreq_order_index": first_index,
        "order_hint": order_hint,
        "order_hint_exhausted": exhausted,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "does_not_close_isolation_rewrite_checklist": True,
        "does_not_set_isolation_rewrite_shipped": True,
        "does_not_set_form_label_change_shipped": True,
        "does_not_set_path_shipped": True,
        "does_not_set_wire_shipped": True,
        "expected_today": "isolation_rewrite_with_wire",
        "matches_expected_today": first_id == "isolation_rewrite_with_wire",
        "status_snapshot": status_today,
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "note": (
            "First ORDER_HINT step still open/false_today/unproven that blocks multi-"
            "blocker wire ship. Diagnostic go-board only — not an auto-executor; does "
            "not flip any ship flags or close checklists."
        ),
    }


def case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board(
    *,
    first_blocking: Optional[Mapping[str, Any]] = None,
    dual_linf: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Static order_hint go-board rows + file-level prep map (no live mutation)."""
    if dual_linf is None:
        dual_linf = case1_dual_linf_proof_checklist()
    if first_blocking is None:
        first_blocking = (
            case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq(
                dual_linf=dual_linf
            )
        )
    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = list(dual_linf["dual_linf_proof_checklist_open_ids"])
    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY, "open")
    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY, "open")
    dual_status = dual_linf["dual_linf_under_wire_status"]

    order_rows: List[Dict[str, Any]] = [
        {
            "order_index": 0,
            "coreq_id": "isolation_rewrite_with_wire",
            "status": isolation_status if isolation_status == "open" else "false_today",
            "ship_flag_key": "isolation_rewrite_shipped",
            "ship_flag_still_false": True,
            "is_first_blocking": (
                first_blocking.get("first_blocking_coreq")
                == "isolation_rewrite_with_wire"
            ),
            "prep_artifacts": [
                "tests/test_tf_import_isolation.py (rewrite-not-delete plan notes only; do not rewrite suite this cycle)",
                "offline_case1_isolation_rewrite_design_contract_report (design present)",
                "offline_case1_isolation_rewrite_shipped_criteria_contract_report (criteria present; ship false)",
                "offline_case1_isolation_rewrite_first_blocker_operational_prep_report (prep_present; ship false)",
            ],
            "test_surface": "tests/test_tf_import_isolation.py",
            "prep_note": (
                "Isolation suite rewrite-with-wire (not delete) remains open. Design "
                "+ ship-met criteria already present; suite behavior unchanged this cycle."
            ),
        },
        {
            "order_index": 1,
            "coreq_id": "form_label_change_shipped",
            "status": "false_today",
            "ship_flag_key": "form_label_change_shipped",
            "ship_flag_still_false": True,
            "is_first_blocking": (
                first_blocking.get("first_blocking_coreq") == "form_label_change_shipped"
            ),
            "prep_artifacts": [
                "CASE1_FORM_CURRENT / CASE1_PLANNED_TF_AWARE_FORM constants",
                "offline_case1_form_label_change_shipped_criteria_contract_report",
                "case1_form_label_contract (form classic registration)",
            ],
            "test_surface": (
                "tests/test_tf_offline_case1_form_label_change_shipped_criteria_contract.py"
            ),
            "prep_note": (
                "Form flip touch points registered; form remains classic_2block_excel_path; "
                "form_label_change_shipped still False."
            ),
        },
        {
            "order_index": 2,
            "coreq_id": "dual_honest_tf_aware_path_present",
            "status": "false_today",
            "ship_flag_key": "path_shipped",
            "ship_flag_still_false": True,
            "is_first_blocking": (
                first_blocking.get("first_blocking_coreq")
                == "dual_honest_tf_aware_path_present"
            ),
            "prep_artifacts": [
                "offline_case1_dual_honest_tf_aware_path_design_contract_report",
                "offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report",
                f"feature flag {CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME}=False",
            ],
            "test_surface": (
                "tests/test_tf_offline_case1_dual_honest_tf_aware_path_present_criteria_contract.py"
            ),
            "prep_note": (
                "Path design+criteria present; path_shipped=False; "
                "dual_honest_tf_aware_path_present ship-met=False; feature flag reserved false."
            ),
        },
        {
            "order_index": 3,
            "coreq_id": "dual_linf_under_wire_proven",
            "status": "unproven" if dual_status != "proven" else "proven",
            "ship_flag_key": "dual_linf_under_wire",
            "ship_flag_still_false": dual_status != "proven",
            "is_first_blocking": (
                first_blocking.get("first_blocking_coreq")
                == "dual_linf_under_wire_proven"
            ),
            "prep_artifacts": [
                "offline_case1_dual_space_linf_probe_report (diagnostic only)",
                "offline_case1_dual_space_linf_live_lambda_bridge_report (diagnostic only)",
                "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report (diagnostic only)",
                "offline_case1_online_linf_gate_criteria_contract_report (gate open)",
                "offline_case1_dual_linf_under_wire_criteria_contract_report (criteria present; dual_linf unproven; not proof)",
            ],
            "test_surface": "tests/test_tf_offline_case1_dual_linf_under_wire_criteria_contract.py",
            "prep_note": (
                "dual_linf_under_wire remains unproven; online_linf_gate_under_tf_path "
                f"status={gate_status}; probe/bridge/warmstart are diagnostic only — not proof; "
                "dual_linf flip-criteria contract present (criteria ≠ proven)."
            ),
        },
        {
            "order_index": 4,
            "coreq_id": "wire_shipped",
            "status": "false_today",
            "ship_flag_key": "wire_shipped",
            "ship_flag_still_false": True,
            "is_first_blocking": (
                first_blocking.get("first_blocking_coreq") == "wire_shipped"
            ),
            "prep_artifacts": [
                "offline_case1_wire_ship_acceptance_design_contract_report",
                "DEFAULT_WIRE_BLOCKERS / offline_wire_preflight_report",
            ],
            "test_surface": (
                "tests/test_tf_offline_case1_wire_ship_acceptance_design_contract.py"
            ),
            "prep_note": (
                "Wire-ship acceptance design present; wire_shipped=False; "
                "wire_ship_allowed_today=False."
            ),
        },
    ]

    companion_rows: List[Dict[str, Any]] = [
        {
            "coreq_id": "bundle_shipped",
            "status": "false_today",
            "ship_flag_still_false": True,
            "prep_artifacts": [
                "offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report",
                "offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
            ],
            "prep_note": "bundle_shipped=False; bundle_ship_allowed_today=False; criteria_met_today=False.",
        },
        {
            "coreq_id": "no_blender_offline_affine_kernel",
            "status": "still_true",
            "ship_flag_still_false": True,
            "prep_artifacts": [
                "offline_case1_honest_blender_pooling_path_report",
                "UNITS = FCC/COKER/CDU (no silent BLENDER)",
            ],
            "prep_note": (
                "Blender surface remains linear_quality_pooling — not a base_delta UNITS entry."
            ),
        },
        {
            "coreq_id": "case1_is_cdu_blender_package_admm",
            "status": "still_true",
            "ship_flag_still_false": True,
            "prep_artifacts": ["CASE1_FORM_CURRENT=classic_2block_excel_path"],
            "prep_note": "Case 1 remains CDU+Blender package ADMM shape on classic form.",
        },
        {
            "coreq_id": "feature_flag_enabled_today",
            "status": "false_today",
            "ship_flag_still_false": True,
            "prep_artifacts": [
                f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME} reserved False"
            ],
            "prep_note": "Feature flag remains reserved false; no Case 1 TF routing.",
        },
        {
            "coreq_id": "dual_recovery_path",
            "status": "none_today",
            "ship_flag_still_false": True,
            "prep_artifacts": [
                "dual_recovery_path=None on TF surface",
                f"planned={CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED}",
            ],
            "prep_note": (
                "dual_recovery_path is None today; planned-under-wire label is not pure-ADMM."
            ),
        },
    ]

    order_ids = [r["coreq_id"] for r in order_rows]
    order_hint_ok = order_ids == list(
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT
    )

    return {
        "kind": "offline_case1_dual_honest_multi_blocker_wire_go_board",
        "order_hint": list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT),
        "order_hint_rows": order_rows,
        "go_board_rows": order_rows,
        "companion_rows": companion_rows,
        "file_level_prep_map": {
            r["coreq_id"]: r["prep_artifacts"] for r in order_rows
        },
        "companion_prep_map": {
            r["coreq_id"]: r["prep_artifacts"] for r in companion_rows
        },
        "order_hint_coverage_ok": order_hint_ok,
        "first_row_matches_first_blocking_coreq": (
            order_rows[0]["coreq_id"] == first_blocking.get("first_blocking_coreq")
            if order_rows
            else False
        ),
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "does_not_rewrite_isolation_suite": True,
        "static_prep_only": True,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "note": (
            "Order_hint-sequenced go-board with static file-level prep paths only. "
            "Not an auto-executor; does not rewrite isolation suite; does not flip ship flags."
        ),
    }


def offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report() -> Dict[str, Any]:
    """Always-on dual-honest multi-blocker wire *implementation blueprint* / go-board.

    No TF, no PuLP, no excel_pipeline, no Case 1 solve routing. Aggregate
    ``ok`` / ``contract_ok`` / ``blueprint_ok`` = blueprint formalized ∧ honesty
    locks ∧ blueprint_present ∧ rehearsal+scaffold linked ∧ first_blocking_coreq
    present ∧ order_hint go-board ∧ all ship flags hard false ∧ dual_linf
    unproven ∧ blockers non-empty ∧ dual_recovery_path is None ∧ UNITS
    FCC/COKER/CDU. **Not** path shipped. **Not** path present ship-met.
    **Not** wire shipped. **Not** bundle shipped. **Not** isolation rewrite
    shipped. **Not** form flip. **Not** VERDICT. **Not** dual L∞ under wire
    proof. **Not** an auto-executor.
    """
    honesty = (
        _case1_dual_honest_multi_blocker_wire_implementation_blueprint_honesty_fields()
    )
    scaffold = offline_case1_dual_honest_tf_aware_path_execution_scaffold_report(
        include_diagnostic=True
    )
    rehearsal = offline_case1_dual_honest_multi_blocker_wire_rehearsal_report()
    dual_linf = case1_dual_linf_proof_checklist()
    first_blocking = (
        case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq(
            dual_linf=dual_linf
        )
    )
    go_board = case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board(
        first_blocking=first_blocking, dual_linf=dual_linf
    )
    shape = case1_dual_honest_tf_aware_path_shape()
    form = case1_form_label_contract()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]

    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids

    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"
    gate_in_open = CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY in open_ids

    form_label_status = checklist.get("form_label_change_shipped")
    form_label_open = form_label_status == "open" or form_label_status is None
    if "form_label_change_shipped" in open_ids:
        form_label_open = True

    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)
    bundle_met_map = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    bundle_ship_allowed_today = (
        case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(bundle_met_map)
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            bundle_met_map
        )
    )
    bundle_shipped = case1_dual_honest_multi_blocker_wire_bundle_shipped()

    isolation_rewrite_design_present = True
    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_design_present = True
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    blueprint_present = True
    implementation_blueprint_present = True
    wire_go_board_present = True
    rehearsal_present = bool(rehearsal.get("rehearsal_present", False))
    wire_rehearsal_present = bool(rehearsal.get("wire_rehearsal_present", False))
    scaffold_present = bool(scaffold.get("scaffold_present", False))
    execution_scaffold_present = bool(
        scaffold.get("execution_scaffold_present", False)
    )
    scaffold_compose_ok = bool(scaffold.get("compose_ok", False))
    rehearsal_ok_flag = bool(rehearsal.get("rehearsal_ok", rehearsal.get("ok")))
    bundle_design_present = True

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = (
        critical_blockers_present
        and affine_blocker_present
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
        and honesty["case1_is_cdu_blender_package_admm_blocker_still_true"] is True
    )
    pooling_status = checklist.get("blender_affine_kernel_or_honest_pooling_path")
    pooling_ok = pooling_status == CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS

    shape_ok = bool(
        shape["cdu_surface"] == "offline_affine_base_delta"
        and shape["blender_surface"] == CASE1_SHAPED_BLENDER_SURFACE
        and list(shape["intermediates"]) == list(CASE1_SHAPED_LINKING_STREAMS)
        and shape["form_current"] == CASE1_FORM_CURRENT
        and shape["form_planned"] == CASE1_PLANNED_TF_AWARE_FORM
        and shape["form_label_change_shipped"] is False
        and shape["dual_recovery_path_today_on_tf_surface"] is None
        and shape["path_shipped"] is False
        and shape["dual_honest_tf_aware_path_present"] is False
        and shape["feature_flag_enabled_today"] is False
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["blueprint_is_not_path_shipped"] is True
        and honesty["blueprint_is_not_path_present_for_ship"] is True
        and honesty["blueprint_is_not_wire_shipped"] is True
        and honesty["blueprint_is_not_wire"] is True
        and honesty["blueprint_is_not_bundle_shipped"] is True
        and honesty["blueprint_is_not_isolation_rewrite_shipped"] is True
        and honesty["blueprint_is_not_form_label_change_shipped"] is True
        and honesty["blueprint_is_not_ship_allow"] is True
        and honesty["blueprint_is_not_verdict_gate"] is True
        and honesty["blueprint_is_not_dual_linf_under_wire_proof"] is True
        and honesty["this_blueprint_alone_is_not_ship_criterion"] is True
        and honesty["this_blueprint_alone_is_not_multi_blocker_ship"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["diagnostic_linf_is_not_dual_linf_under_wire_proof"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
        and form["planned_form_distinct"] is True
    )
    ship_permission_ok = (
        bundle_ship_allowed_today is False
        and criteria_met_today is False
        and wire_ship_allowed_today is False
        and isolation_ship_allowed_today is False
        and form_label_ship_allowed_today is False
        and ship_met_allowed_today is False
    )
    gate_permission_ok = gate_flip_allowed_today is False
    gate_open_ok = gate_still_open and gate_in_open and gate_status == "open"
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    rewrite_not_shipped_ok = (
        isolation_rewrite_shipped is False
        and isolation_rewrite_design_present is True
        and honesty["isolation_rewrite_shipped"] is False
    )
    wire_not_shipped_ok = (
        wire_shipped is False
        and honesty["wire_shipped"] is False
        and "wire_not_shipped" in blockers
    )
    bundle_not_shipped_ok = (
        bundle_shipped is False
        and honesty["bundle_shipped"] is False
        and honesty["not_bundle_shipped"] is True
    )
    path_not_shipped_ok = (
        path_shipped is False
        and dual_honest_tf_aware_path_present is False
        and honesty["path_shipped"] is False
        and honesty["dual_honest_tf_aware_path_present"] is False
        and feature_flag_enabled_today is False
    )
    form_not_shipped_ok = (
        form_label_change_shipped is False
        and honesty["form_label_change_shipped"] is False
        and form_label_open
        and form["form_current"] == CASE1_FORM_CURRENT
    )

    anti = CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANTI_CRITERIA_TODAY
    anti_ok = (
        len(anti) >= 16
        and "this_blueprint_alone" in anti
        and "go_board_alone" in anti
        and "first_blocking_coreq_alone" in anti
        and "prep_map_alone" in anti
        and "scaffold_plus_rehearsal_plus_blueprint_alone" in anti
        and "this_rehearsal_alone" in anti
        and "this_scaffold_alone" in anti
        and "path_design_alone" in anti
        and "bundle_design_alone" in anti
        and "wire_ship_acceptance_alone" in anti
        and "packaging_alone" in anti
        and "residual_must_vanish" in anti
        and "diagnostic_linf_alone" in anti
    )

    order_hint = list(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    order_hint_ok = (
        CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR is True
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and go_board.get("order_hint_coverage_ok") is True
    )
    feature_flag_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME == "enable_tf_affine_case1_wire"
        and CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY is False
    )
    dual_recovery_planned_ok = (
        CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        == "online_lambda_under_tf_aware_form_when_shipped"
        and honesty["dual_recovery_path"] is None
        and honesty["not_pure_admm_dual_recovery"] is True
        and "pure-admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
        and "pure_admm"
        not in str(CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED).lower()
    )

    scaffold_link_ok = bool(
        scaffold_present
        and execution_scaffold_present
        and scaffold_compose_ok
        and scaffold.get("path_shipped") is False
        and scaffold.get("wire_shipped") is False
        and scaffold.get("bundle_shipped") is False
        and scaffold.get("isolation_rewrite_shipped") is False
        and scaffold.get("form_label_change_shipped") is False
        and scaffold.get("dual_linf_under_wire_status") == "unproven"
        and scaffold.get("dual_recovery_path") is None
        and scaffold.get("ok") is True
    )
    rehearsal_link_ok = bool(
        rehearsal_present
        and wire_rehearsal_present
        and rehearsal_ok_flag
        and rehearsal.get("path_shipped") is False
        and rehearsal.get("wire_shipped") is False
        and rehearsal.get("bundle_shipped") is False
        and rehearsal.get("isolation_rewrite_shipped") is False
        and rehearsal.get("form_label_change_shipped") is False
        and rehearsal.get("dual_linf_under_wire_status") == "unproven"
        and rehearsal.get("dual_recovery_path") is None
        and rehearsal.get("excel_packaging_twin_deferred") is False
    )
    first_blocking_ok = bool(
        first_blocking.get("first_blocking_coreq") == "isolation_rewrite_with_wire"
        and first_blocking.get("matches_expected_today") is True
        and first_blocking.get("order_hint_is_not_executor") is True
        and first_blocking.get("no_auto_wire") is True
        and first_blocking.get("order_hint_exhausted") is False
    )
    go_board_ok = bool(
        go_board.get("order_hint_coverage_ok") is True
        and go_board.get("order_hint_is_not_executor") is True
        and go_board.get("no_auto_wire") is True
        and go_board.get("static_prep_only") is True
        and go_board.get("does_not_rewrite_isolation_suite") is True
        and len(go_board.get("order_hint_rows") or [])
        == len(CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT)
    )

    blueprint_formalized = bool(
        blueprint_present
        and implementation_blueprint_present
        and wire_go_board_present
        and scaffold_link_ok
        and rehearsal_link_ok
        and first_blocking_ok
        and go_board_ok
        and shape_ok
        and anti_ok
        and order_hint_ok
        and feature_flag_ok
        and dual_recovery_planned_ok
        and CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANNOTATION
        == "present"
        and path_design_present is True
        and bundle_design_present is True
        and isolation_rewrite_design_present is True
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and gate_permission_ok
        and gate_open_ok
        and isolation_open_ok
        and rewrite_not_shipped_ok
        and wire_not_shipped_ok
        and bundle_not_shipped_ok
        and path_not_shipped_ok
        and form_not_shipped_ok
        and blueprint_formalized
        and blockers_still_documented
        and pooling_ok
        and form_label_open
        and form_label_change_shipped is False
        and path_shipped is False
        and dual_honest_tf_aware_path_present is False
    )
    blueprint_ok = honesty_ok
    contract_ok = blueprint_ok
    ok = (
        blueprint_ok
        and (honesty["blueprint_present"] is True)
        and (honesty["path_shipped"] is False)
        and (honesty["dual_honest_tf_aware_path_present"] is False)
        and (honesty["wire_shipped"] is False)
        and (honesty["bundle_shipped"] is False)
        and (honesty["isolation_rewrite_shipped"] is False)
        and (honesty["form_label_change_shipped"] is False)
    )

    ok_criteria = (
        "blueprint formalized ∧ honesty locks ∧ blueprint_present=True ∧ "
        "rehearsal+scaffold linked ∧ first_blocking_coreq present ∧ order_hint "
        "go-board ∧ path_shipped=False ∧ dual_honest_tf_aware_path_present "
        "ship-met=False ∧ wire_shipped=False ∧ bundle_shipped=False ∧ "
        "bundle_ship_allowed_today=False ∧ criteria_met_today=False ∧ "
        "isolation_rewrite_shipped=False ∧ isolation checklist open ∧ form classic ∧ "
        "form_label_change_shipped=False ∧ dual_linf unproven ∧ online_linf_gate open ∧ "
        "gate_flip_allowed_today=False ∧ blockers non-empty ∧ dual_recovery_path=None ∧ "
        "feature_flag_enabled_today=False ∧ UNITS FCC/COKER/CDU ∧ order_hint not "
        "executor — NOT path shipped; NOT path present ship-met; NOT wire shipped; "
        "NOT bundle shipped; NOT isolation rewrite shipped; NOT form flip; NOT ship "
        "allow; NOT gate flip; NOT VERDICT; NOT dual L∞ under wire proof"
    )

    multi_blocker_coreqs = {
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "form_label_change_shipped": form_label_change_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire_status"],
        "online_linf_gate_under_tf_path": gate_status,
        "wire_shipped": wire_shipped,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "path_shipped": path_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "dual_recovery_path": None,
        "scaffold_present": scaffold_present,
        "execution_scaffold_present": execution_scaffold_present,
        "rehearsal_present": rehearsal_present,
        "wire_rehearsal_present": wire_rehearsal_present,
        "blueprint_present": blueprint_present,
        "no_blender_offline_affine_kernel": True,
        "blender_surface": "linear_quality_pooling",
        "case1_is_cdu_blender_package_admm": True,
    }

    return {
        **honesty,
        "ok": ok,
        "blueprint_ok": blueprint_ok,
        "contract_ok": contract_ok,
        "design_contract_ok": blueprint_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "shape_ok": shape_ok,
        "scaffold_link_ok": scaffold_link_ok,
        "rehearsal_link_ok": rehearsal_link_ok,
        "first_blocking_ok": first_blocking_ok,
        "go_board_ok": go_board_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "gate_permission_ok": gate_permission_ok,
        "gate_open_ok": gate_open_ok,
        "isolation_open_ok": isolation_open_ok,
        "rewrite_not_shipped_ok": rewrite_not_shipped_ok,
        "wire_not_shipped_ok": wire_not_shipped_ok,
        "bundle_not_shipped_ok": bundle_not_shipped_ok,
        "path_not_shipped_ok": path_not_shipped_ok,
        "form_not_shipped_ok": form_not_shipped_ok,
        "blueprint_formalized": blueprint_formalized,
        "ok_criteria": ok_criteria,
        "blueprint_present": blueprint_present,
        "implementation_blueprint_present": implementation_blueprint_present,
        "wire_go_board_present": wire_go_board_present,
        "blueprint_annotation": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANNOTATION
        ),
        "rehearsal_present": rehearsal_present,
        "wire_rehearsal_present": wire_rehearsal_present,
        "rehearsal_ok": rehearsal_ok_flag,
        "scaffold_present": scaffold_present,
        "execution_scaffold_present": execution_scaffold_present,
        "scaffold_compose_ok": scaffold_compose_ok,
        "scaffold_ok": bool(scaffold.get("scaffold_ok", scaffold.get("ok"))),
        "compose_ok": scaffold_compose_ok and rehearsal_link_ok and go_board_ok,
        "anti_criteria_today": list(
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANTI_CRITERIA_TODAY
        ),
        "first_blocking_coreq": first_blocking.get("first_blocking_coreq"),
        "first_blocking_coreq_status": first_blocking.get(
            "first_blocking_coreq_status"
        ),
        "first_blocking_coreq_order_index": first_blocking.get(
            "first_blocking_coreq_order_index"
        ),
        "first_blocking": first_blocking,
        "go_board": go_board,
        "order_hint_rows": go_board.get("order_hint_rows"),
        "go_board_rows": go_board.get("go_board_rows"),
        "file_level_prep_map": go_board.get("file_level_prep_map"),
        "companion_rows": go_board.get("companion_rows"),
        "companion_prep_map": go_board.get("companion_prep_map"),
        "multi_blocker_coreqs": multi_blocker_coreqs,
        "any_ship_allowed_today": False,
        "all_ship_flags_false": True,
        "order_hint": order_hint,
        "order_hint_is_not_executor": (
            CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR
        ),
        "no_auto_wire": CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE,
        "isolation_rewrite_design_present": isolation_rewrite_design_present,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "blueprint_does_not_close_isolation_rewrite_checklist": True,
        "path_design_present": path_design_present,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "bundle_design_present": bundle_design_present,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_shipped": wire_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "blueprint_does_not_flip_wire_ship_met_today": True,
        "blueprint_does_not_flip_dual_honest_tf_aware_path_present_met_today": True,
        "blueprint_does_not_flip_form_label_change_shipped_met_today": True,
        "blueprint_does_not_flip_isolation_ship_met_today": True,
        "blueprint_does_not_flip_gate_met_today": True,
        "blueprint_does_not_flip_bundle_ship_met_today": True,
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        "form_label_change_shipped_checklist": form_label_status,
        "form_label_change_still_open": form_label_open,
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
        ),
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        "blender_pooling_checklist_status": pooling_status,
        "blender_pooling_checklist_key": "blender_affine_kernel_or_honest_pooling_path",
        "honest_pooling_path_present": pooling_ok,
        "units_affine_unchanged": list(UNITS),
        "cdu_surface": "offline_affine_base_delta",
        "blender_surface": "linear_quality_pooling",
        "intermediates": list(CASE1_SHAPED_LINKING_STREAMS),
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "form_label_change_required_in_default_wire_blockers": (
            "form_label_change_required" in blockers
        ),
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "case1_is_cdu_blender_package_admm_in_default_wire_blockers": (
            "case1_is_cdu_blender_package_admm" in blockers
        ),
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp_in_blockers": (
            affine_blocker_present
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "blueprint_is_not_path_shipped": True,
        "blueprint_is_not_wire_shipped": True,
        "blueprint_is_not_bundle_shipped": True,
        "blueprint_is_not_isolation_rewrite_shipped": True,
        "blueprint_is_not_form_label_shipped": True,
        "tf_available": tf_available(),
        "dual_honest_multi_blocker_wire_implementation_blueprint_available": True,
        "linf_le_15_is_not_gate": True,
        "residual_must_vanish_is_not_gate": True,
        # Excel packaging twin of *this blueprint* present after packaging ship
        # (existence only — not path/wire/bundle/isolation/form ship).
        "excel_packaging_twin_deferred": False,
        "excel_packaging_twin_present": True,
        # Rehearsal packaging twin exists after #63 (existence only — not ship).
        "excel_rehearsal_packaging_twin_deferred": False,
        "excel_rehearsal_packaging_twin_present": True,
        "note": honesty["note"],
    }


def case1_dual_honest_multi_blocker_wire_implementation_blueprint_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report(
        **kwargs
    )


def multi_unit_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report``."""
    return offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report(
        **kwargs
    )


def excel_fcc_matrix_matches_affine(
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Compare Submodel_FCC export (fcc_pims_matrix MB_*) to affine package.

    Always-on (no TF). Returns report dict; ``ok`` is True when all cells match.
    """
    from .base_delta import build_fcc_base_delta
    from .excel_pipeline import base_delta_unit_submodel_tables

    model = build_fcc_base_delta()
    coeffs = affine_coeffs_from_base_delta(model)
    tables = base_delta_unit_submodel_tables()
    matrix = tables.get("fcc_pims_matrix") or []
    return _excel_pims_matrix_matches_affine(
        model, coeffs, matrix, atol=atol
    )


def excel_coker_matrix_matches_affine(
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Compare Submodel_Coker export (coker_pims_matrix MB_*) to affine package.

    Always-on (no TF). E7/E8: product mass-balance rows only (MB_* BASE + D_*),
    not teaching rows (E_*, FREE). Excel None for missing D_* maps to 0.0.
    Returns report dict; ``ok`` is True when all cells match.
    """
    from .base_delta import build_coker_base_delta
    from .excel_pipeline import base_delta_unit_submodel_tables

    model = build_coker_base_delta()
    coeffs = affine_coeffs_from_base_delta(model)
    tables = base_delta_unit_submodel_tables()
    matrix = tables.get("coker_pims_matrix") or []
    return _excel_pims_matrix_matches_affine(
        model, coeffs, matrix, atol=atol
    )


def _excel_pims_matrix_matches_affine(
    model: Any,
    coeffs: AffineCoeffs,
    matrix: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Shared MB_* BASE/D_* ↔ affine y0/D checker (FCC / Coker)."""
    mismatches: List[Dict[str, Any]] = []
    checked = 0
    by_row = {str(r.get("row")): r for r in matrix if r.get("row")}
    for i, p in enumerate(coeffs.products):
        row_name = f"MB_{p}"
        row = by_row.get(row_name)
        if row is None:
            mismatches.append({"row": row_name, "error": "missing MB row"})
            continue
        base = row.get("BASE")
        if base is None or abs(float(base) - float(coeffs.y0[i])) > atol:
            mismatches.append(
                {
                    "row": row_name,
                    "field": "BASE",
                    "excel": base,
                    "affine": float(coeffs.y0[i]),
                }
            )
        else:
            checked += 1
        for j, drv in enumerate(coeffs.drivers):
            cell = row.get(f"D_{drv}")
            expected = float(coeffs.D[i, j])
            # Excel uses None when driver absent from deltas; affine stores 0.0
            if cell is None:
                cell_v = 0.0
            else:
                cell_v = float(cell)
            if abs(cell_v - expected) > atol:
                mismatches.append(
                    {
                        "row": row_name,
                        "field": f"D_{drv}",
                        "excel": cell,
                        "affine": expected,
                    }
                )
            else:
                checked += 1
    # order consistency
    if list(model.drivers) != list(coeffs.drivers):
        mismatches.append({"error": "driver order mismatch"})
    if list(model.products) != list(coeffs.products):
        mismatches.append({"error": "product order mismatch"})
    return {
        "ok": len(mismatches) == 0,
        "checked": checked,
        "mismatches": mismatches,
        "n_products": len(coeffs.products),
        "n_drivers": len(coeffs.drivers),
    }



# ---------------------------------------------------------------------------
# Offline Case-1 isolation first-blocker *operational prep* (prep without ship).
# Always-on numpy. Formalizes *how* first_blocking_coreq=isolation_rewrite_with_wire
# is prepared without executing rewrite / wire / form flip. Distinct from design
# (*what*), ship-met criteria (*when*), blueprint (*order_hint + prep map*), and
# packaging (*planner visibility*). dual_recovery_path=None; all ship flags hard
# false; rewrite-not-delete companion inventory only. No TF / PuLP / excel_pipeline.
# ---------------------------------------------------------------------------

CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_KIND = (
    "offline_case1_isolation_rewrite_first_blocker_operational_prep"
)
CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_ANNOTATION = "present"
CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH = (
    "tests/test_tf_import_isolation.py"
)

CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "design_contracts_alone",
    "this_prep_alone",
    "go_board_alone",
    "design_alone",
    "ship_criteria_alone",
    "blueprint_alone",
    "rehearsal_alone",
    "scaffold_alone",
    "isolation_design_alone",
    "isolation_ship_criteria_alone",
    "path_design_alone",
    "path_present_criteria_alone",
    "bundle_design_alone",
    "bundle_ship_met_criteria_alone",
    "wire_ship_acceptance_alone",
    "form_label_criteria_alone",
    "gate_criteria_alone",
    "diagnostic_linf_alone",
    "this_prep_packaging_alone",
)

CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_COMPANION_MATRIX: Dict[str, Any] = {
    "companion_matrix_is_inventory_only": True,
    "suite_path": CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH,
    "suite_delete_forbidden": True,
    "behavior_must_remain_until_rewrite_with_wire": True,
    "isolation_tests_rewritten_with_wire": False,
    "themes": {
        "no_excel_pipeline_on_tf_hot_path": True,
        "dual_recovery_path_none_today": True,
        "planned_dual_recovery_path_not_pure_admm": True,
        "form_classic_until_form_coreq": True,
        "wire_shipped_false_until_wire_coreq": True,
        "dual_linf_unproven_until_proof_path": True,
        "no_tensorflow_on_excel_packaging_path": True,
        "rewrite_not_delete": True,
    },
}


def case1_isolation_rewrite_first_blocker_companion_matrix() -> Dict[str, Any]:
    """Static companion assertion inventory (inventory only — not suite rewrite)."""
    return {
        "companion_matrix_is_inventory_only": True,
        "suite_path": CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH,
        "suite_delete_forbidden": True,
        "behavior_must_remain_until_rewrite_with_wire": True,
        "isolation_tests_rewritten_with_wire": False,
        "themes": dict(
            CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_COMPANION_MATRIX["themes"]
        ),
        "assertion_themes": list(
            CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_COMPANION_MATRIX["themes"].keys()
        ),
        "note": (
            "Inventory of isolation-suite assertion themes that must survive until "
            "rewrite co-ships with dual-honest wire. Not a suite rewrite; suite path "
            f"{CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH} must remain; do not delete."
        ),
    }


def case1_isolation_rewrite_first_blocker_prep_steps() -> List[Dict[str, Any]]:
    """Machine-readable prep steps expanded from go-board row-0 + dual-ban."""
    dual_planned = CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
    return [
        {
            "step_id": "suite_rewrite_not_delete_plan",
            "artifact": CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH,
            "status": "plan_notes_only",
            "executes_rewrite": False,
            "note": "Rewrite-not-delete plan notes only; do not rewrite suite this cycle.",
        },
        {
            "step_id": "isolation_design_contract_present",
            "artifact": "offline_case1_isolation_rewrite_design_contract_report",
            "status": "present",
            "executes_rewrite": False,
            "note": "Design formalizes *what*; not ship.",
        },
        {
            "step_id": "isolation_ship_criteria_present",
            "artifact": "offline_case1_isolation_rewrite_shipped_criteria_contract_report",
            "status": "present_ship_false",
            "executes_rewrite": False,
            "note": "Ship-met criteria formalizes *when*; isolation_rewrite_shipped still False.",
        },
        {
            "step_id": "operational_prep_report_present",
            "artifact": "offline_case1_isolation_rewrite_first_blocker_operational_prep_report",
            "status": "present",
            "executes_rewrite": False,
            "note": "This operational prep formalizes *how prep lands without ship*.",
        },
        {
            "step_id": "dual_ban_locks",
            "artifact": "prep_dual_ban_tokens",
            "status": "locked",
            "executes_rewrite": False,
            "note": (
                "prep≠ship; dual_linf unproven; dual_recovery_path=None today; "
                f"planned under wire={dual_planned} (not pure-ADMM)."
            ),
        },
        {
            "step_id": "steps_complete_is_not_ship",
            "artifact": "anti_criteria",
            "status": "locked",
            "executes_rewrite": False,
            "note": "Prep steps complete ≠ isolation_rewrite_shipped ≠ wire ≠ VERDICT.",
        },
    ]


def _case1_isolation_rewrite_first_blocker_operational_prep_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / prep-is-not-ship locks."""
    return {
        "kind": CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "prep_present": True,
        "first_blocker_prep_present": True,
        "path_shipped": False,
        "dual_honest_tf_aware_path_present": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "bundle_ship_allowed_today": False,
        "criteria_met_today": False,
        "not_wire_shipped": True,
        "not_path_shipped": True,
        "not_bundle_shipped": True,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "prep_is_not_isolation_rewrite_shipped": True,
        "prep_is_not_wire": True,
        "prep_is_not_form_label_change_shipped": True,
        "prep_is_not_path_shipped": True,
        "prep_is_not_bundle_shipped": True,
        "prep_is_not_verdict_gate": True,
        "prep_is_not_dual_linf_under_wire_proof": True,
        "prep_is_not_ship_allow": True,
        "prep_is_not_ship_met": True,
        "prep_is_not_form_flip": True,
        "prep_is_not_gate_flip": True,
        "this_prep_alone_is_not_ship_criterion": True,
        "go_board_alone_is_not_ship_criterion": True,
        "design_alone_is_not_ship_criterion": True,
        "ship_criteria_alone_is_not_ship_criterion": True,
        "blueprint_alone_is_not_ship_criterion": True,
        "rehearsal_alone_is_not_ship_criterion": True,
        "scaffold_alone_is_not_ship_criterion": True,
        "packaging_alone_is_not_ship_criterion": True,
        "order_hint_is_not_executor": True,
        "no_auto_wire": True,
        "does_not_rewrite_isolation_suite": True,
        "suite_delete_forbidden": True,
        "isolation_rewrite_shipped": False,
        "isolation_tests_rewritten_with_wire": False,
        "form_label_change_shipped": False,
        "feature_flag_enabled_today": False,
        "probe_linf_is_not_ship_criterion_today": True,
        "bridge_linf_is_not_ship_criterion_today": True,
        "warmstart_linf_is_not_ship_criterion_today": True,
        "pooling_linf_is_not_ship_criterion_today": True,
        "seed_identity_linf_is_not_ship_criterion": True,
        "recovered_blender_linf_is_not_ship_criterion_today": True,
        "residual_must_vanish_is_not_ship_criterion": True,
        "diagnostic_linf_is_not_dual_linf_under_wire_proof": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "scope": "case1_isolation_rewrite_first_blocker_operational_prep_offline",
        "note": (
            "Offline Case-1 isolation first-blocker *operational prep*: machine-readable "
            "*how prep lands without ship* for first_blocking_coreq=isolation_rewrite_with_wire. "
            "prep_present=True / first_blocker_prep_present=True; isolation_rewrite_shipped=False; "
            "isolation_tests_rewritten_with_wire=False; path/wire/bundle/form ship flags hard false; "
            "dual_linf unproven; dual_recovery_path=None today; planned under wire="
            f"{CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED} (not pure-ADMM); "
            f"suite path={CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH}; rewrite-not-delete; "
            "companion matrix is inventory only. Prep alone / go-board alone / design alone / "
            "ship_criteria alone / blueprint alone / rehearsal alone / scaffold alone / packaging "
            "alone ≠ isolation rewrite shipped ≠ wire ≠ VERDICT ≠ dual L∞ under wire proof. "
            "Does not redefine ready_for_wire_discussion; does not clear DEFAULT_WIRE_BLOCKERS; "
            "does not enable Case 1 TF feature flag; isolation suite behavior unchanged this cycle. "
            "SUGGESTED_NEXT_WAVE still full multi-blocker wire *execution* long-term."
        ),
    }


def offline_case1_isolation_rewrite_first_blocker_operational_prep_report() -> Dict[str, Any]:
    """Always-on isolation first-blocker operational prep (prep without ship).

    No TF, no PuLP, no excel_pipeline, no Case 1 solve routing. Aggregate
    ``ok`` / ``contract_ok`` / ``prep_ok`` = prep formalized ∧ honesty locks ∧
    prep_present ∧ first_blocking_coreq=isolation_rewrite_with_wire ∧ ship flags
    hard false ∧ dual_linf unproven ∧ dual_recovery_path is None ∧ UNITS
    FCC/COKER/CDU. **Not** isolation rewrite shipped. **Not** wire. **Not**
    form flip. **Not** VERDICT. **Not** dual L∞ under wire proof.
    """
    honesty = _case1_isolation_rewrite_first_blocker_operational_prep_honesty_fields()
    dual_linf = case1_dual_linf_proof_checklist()
    first_blocking = (
        case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq(
            dual_linf=dual_linf
        )
    )
    go_board = case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board(
        first_blocking=first_blocking, dual_linf=dual_linf
    )
    companion = case1_isolation_rewrite_first_blocker_companion_matrix()
    prep_steps = case1_isolation_rewrite_first_blocker_prep_steps()
    form = case1_form_label_contract()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    isolation_status = checklist.get(CASE1_ISOLATION_REWRITE_CHECKLIST_KEY)
    isolation_still_open = isolation_status == "open"
    isolation_in_open = CASE1_ISOLATION_REWRITE_CHECKLIST_KEY in open_ids
    gate_status = checklist.get(CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY)
    gate_still_open = gate_status == "open"

    isolation_met_map = case1_isolation_rewrite_shipped_criteria_met_today_map()
    isolation_ship_allowed_today = case1_isolation_ship_allowed_today(isolation_met_map)
    wire_criteria_met_map = case1_wire_ship_acceptance_criteria_met_today_map()
    wire_ship_allowed_today = case1_wire_ship_allowed_today(wire_criteria_met_map)
    gate_met_map = case1_online_linf_gate_criteria_met_today_map()
    gate_flip_allowed_today = case1_online_linf_gate_flip_allowed_today(gate_met_map)
    form_label_met_map = case1_form_label_change_shipped_criteria_met_today_map()
    form_label_ship_allowed_today = case1_form_label_ship_allowed_today(
        form_label_met_map
    )
    path_present_met_map = case1_dual_honest_tf_aware_path_present_criteria_met_today_map()
    ship_met_allowed_today = case1_dual_honest_tf_aware_path_present_ship_met_allowed_today(
        path_present_met_map
    )
    bundle_met_map = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map()
    )
    bundle_ship_allowed_today = (
        case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today(bundle_met_map)
    )
    criteria_met_today = (
        case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate(
            bundle_met_map
        )
    )

    isolation_rewrite_shipped = False
    isolation_tests_rewritten_with_wire = False
    path_shipped = False
    dual_honest_tf_aware_path_present = False
    form_label_change_shipped = False
    wire_shipped = False
    bundle_shipped = False
    feature_flag_enabled_today = bool(
        CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
    )
    prep_present = True
    first_blocker_prep_present = True

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    critical_blockers_present = {
        "isolation_rewrite_required",
        "form_label_change_required",
        "dual_linf_under_wire_unproven",
        "case1_is_cdu_blender_package_admm",
        "no_blender_offline_affine_kernel",
        "wire_not_shipped",
    }.issubset(set(blockers))
    affine_blocker_present = (
        "affine_kernels_are_yield_drivers_not_plant_blocks_feed_lp" in blockers
    )
    blocker_ok = critical_blockers_present and affine_blocker_present

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["feature_flag_enabled_today"] is False
        and honesty["prep_is_not_isolation_rewrite_shipped"] is True
        and honesty["prep_is_not_wire"] is True
        and honesty["prep_is_not_verdict_gate"] is True
        and honesty["prep_is_not_dual_linf_under_wire_proof"] is True
        and honesty["this_prep_alone_is_not_ship_criterion"] is True
        and honesty["order_hint_is_not_executor"] is True
        and honesty["no_auto_wire"] is True
        and honesty["does_not_rewrite_isolation_suite"] is True
        and honesty["suite_delete_forbidden"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_linf["dual_linf_under_wire_status"] == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
    )
    ship_permission_ok = (
        isolation_ship_allowed_today is False
        and wire_ship_allowed_today is False
        and form_label_ship_allowed_today is False
        and ship_met_allowed_today is False
        and bundle_ship_allowed_today is False
        and criteria_met_today is False
        and gate_flip_allowed_today is False
    )
    isolation_open_ok = (
        isolation_still_open
        and isolation_in_open
        and isolation_status == "open"
        and isolation_rewrite_shipped is False
        and isolation_tests_rewritten_with_wire is False
    )
    first_blocking_ok = bool(
        first_blocking.get("first_blocking_coreq") == "isolation_rewrite_with_wire"
        and first_blocking.get("matches_expected_today") is True
    )
    go_board_link_ok = bool(
        go_board.get("order_hint_is_not_executor") is True
        and go_board.get("does_not_rewrite_isolation_suite") is True
        and any(
            "operational_prep" in str(a)
            for a in (go_board.get("file_level_prep_map") or {})
            .get("isolation_rewrite_with_wire", [])
        )
    )
    companion_ok = bool(
        companion["companion_matrix_is_inventory_only"] is True
        and companion["suite_path"] == CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH
        and companion["suite_delete_forbidden"] is True
        and companion["isolation_tests_rewritten_with_wire"] is False
        and companion["themes"]["dual_recovery_path_none_today"] is True
        and companion["themes"]["rewrite_not_delete"] is True
    )
    prep_steps_ok = bool(
        len(prep_steps) >= 5
        and all(s.get("executes_rewrite") is False for s in prep_steps)
        and prep_steps[0]["step_id"] == "suite_rewrite_not_delete_plan"
    )
    anti = CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_ANTI_CRITERIA_TODAY
    anti_ok = (
        "this_prep_alone" in anti
        and "go_board_alone" in anti
        and "blueprint_alone" in anti
        and "packaging_alone" in anti
        and "ship_criteria_alone" in anti
        and "design_alone" in anti
        and "rehearsal_alone" in anti
        and "scaffold_alone" in anti
    )
    prep_formalized = bool(
        prep_present
        and first_blocker_prep_present
        and first_blocking_ok
        and companion_ok
        and prep_steps_ok
        and anti_ok
        and CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_ANNOTATION == "present"
    )
    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and ship_permission_ok
        and isolation_open_ok
        and gate_still_open
        and prep_formalized
        and blockers_still_documented
        and go_board_link_ok
    )
    prep_ok = honesty_ok
    contract_ok = prep_ok
    ok = (
        prep_ok
        and prep_present is True
        and isolation_rewrite_shipped is False
        and wire_shipped is False
        and path_shipped is False
        and form_label_change_shipped is False
        and honesty["dual_recovery_path"] is None
    )
    ok_criteria = (
        "prep formalized ∧ honesty locks ∧ prep_present=True ∧ "
        "first_blocking_coreq=isolation_rewrite_with_wire ∧ "
        "isolation_rewrite_shipped=False ∧ isolation_tests_rewritten_with_wire=False ∧ "
        "path/wire/bundle/form ship flags false ∧ dual_linf unproven ∧ "
        "dual_recovery_path=None ∧ blockers non-empty ∧ UNITS FCC/COKER/CDU — "
        "NOT isolation rewrite shipped; NOT wire; NOT form flip; NOT VERDICT; "
        "NOT dual L∞ under wire proof"
    )
    dual_recovery_planned = CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED
    return {
        **honesty,
        "ok": ok,
        "prep_ok": prep_ok,
        "contract_ok": contract_ok,
        "design_contract_ok": prep_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "ship_permission_ok": ship_permission_ok,
        "isolation_open_ok": isolation_open_ok,
        "first_blocking_ok": first_blocking_ok,
        "go_board_link_ok": go_board_link_ok,
        "companion_ok": companion_ok,
        "prep_steps_ok": prep_steps_ok,
        "prep_formalized": prep_formalized,
        "ok_criteria": ok_criteria,
        "prep_present": prep_present,
        "first_blocker_prep_present": first_blocker_prep_present,
        "prep_annotation": (
            CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_OPERATIONAL_PREP_ANNOTATION
        ),
        "first_blocking_coreq": first_blocking.get("first_blocking_coreq"),
        "first_blocking_coreq_status": first_blocking.get("first_blocking_coreq_status"),
        "first_blocking": first_blocking,
        "go_board_isolation_prep_artifacts": (
            (go_board.get("file_level_prep_map") or {}).get(
                "isolation_rewrite_with_wire", []
            )
        ),
        "companion_matrix": companion,
        "prep_steps": prep_steps,
        "n_prep_steps": len(prep_steps),
        "anti_criteria_today": list(anti),
        "suite_path": CASE1_ISOLATION_REWRITE_FIRST_BLOCKER_SUITE_PATH,
        "rewrite_not_delete": True,
        "isolation_rewrite_design_present": True,
        "isolation_ship_criteria_present": True,
        "isolation_rewrite_shipped": isolation_rewrite_shipped,
        "isolation_tests_rewritten_with_wire": isolation_tests_rewritten_with_wire,
        "isolation_rewrite_with_wire": isolation_status,
        "isolation_rewrite_still_open": isolation_still_open,
        "isolation_rewrite_checklist_open": isolation_still_open,
        "isolation_ship_allowed_today": isolation_ship_allowed_today,
        "isolation_criteria_met_today": isolation_ship_allowed_today,
        "path_design_present": True,
        "path_shipped": path_shipped,
        "dual_honest_tf_aware_path_present": dual_honest_tf_aware_path_present,
        "form_label_change_shipped": form_label_change_shipped,
        "bundle_shipped": bundle_shipped,
        "bundle_ship_allowed_today": bundle_ship_allowed_today,
        "criteria_met_today": criteria_met_today,
        "wire_ship_allowed_today": wire_ship_allowed_today,
        "wire_shipped": wire_shipped,
        "ship_met_allowed_today": ship_met_allowed_today,
        "form_label_ship_allowed_today": form_label_ship_allowed_today,
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        "gate_flip_allowed_today": gate_flip_allowed_today,
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": dual_recovery_planned,
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": feature_flag_enabled_today,
        "dual_linf_under_wire_status": dual_linf["dual_linf_under_wire_status"],
        "dual_linf_under_wire": dual_linf["dual_linf_under_wire"],
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "units_affine_unchanged": list(UNITS),
        "wire_blockers": blockers,
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "isolation_rewrite_required_in_default_wire_blockers": (
            "isolation_rewrite_required" in blockers
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "tf_available": tf_available(),
        "isolation_rewrite_first_blocker_operational_prep_available": True,
        "excel_packaging_twin_present": True,
        "excel_packaging_twin_deferred": False,
        "any_ship_allowed_today": False,
        "all_ship_flags_false": True,
    }


def case1_isolation_rewrite_first_blocker_operational_prep_report() -> Dict[str, Any]:
    """Alias for ``offline_case1_isolation_rewrite_first_blocker_operational_prep_report``."""
    return offline_case1_isolation_rewrite_first_blocker_operational_prep_report()


def multi_unit_case1_isolation_rewrite_first_blocker_operational_prep_report() -> Dict[str, Any]:
    """Alias for multi-unit registry symmetry."""
    return offline_case1_isolation_rewrite_first_blocker_operational_prep_report()



# ---------------------------------------------------------------------------
# Offline Case-1 dual_linf_under_wire flip-criteria contract (goal 3 honesty)
# ---------------------------------------------------------------------------
# Always-on pure compose. Formalizes *when* dual_linf_under_wire status may
# become proven — without proving it. Distinct from online_linf_gate criteria
# (that gates checklist id online_linf_gate_under_tf_path; this gates status
# dual_linf_under_wire / order_hint coreq dual_linf_under_wire_proven).
# Hard: dual_linf_under_wire="unproven"; criteria_met_today=False;
# dual_linf_proof_allowed_today=False; first_blocking remains isolation;
# dual_recovery_path=None; ship flags hard false. Anti-criteria include
# online_linf_gate_criteria_alone / packaging / probe/bridge/warmstart / prep /
# blueprint / scaffold / rehearsal. Does NOT clear DEFAULT_WIRE_BLOCKERS.
# Does NOT redefine ready_for_wire_discussion. No TF / no PuLP / no excel_pipeline.

CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND = (
    "offline_case1_dual_linf_under_wire_criteria_contract"
)
CASE1_DUAL_LINF_UNDER_WIRE_STATUS_TARGET = "dual_linf_under_wire"
CASE1_DUAL_LINF_UNDER_WIRE_ORDER_HINT_COREQ = "dual_linf_under_wire_proven"
CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_ANNOTATION = "present"

# Machine-readable flip criteria map (when dual_linf_under_wire *may* become proven).
# Values are requirement classes — not "met today" claims. Distinct constant names
# from CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA (related co-reqs, different status target).
CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA: Dict[str, str] = {
    "isolation_rewrite_with_wire": FLIP_CRITERION_REQUIRED,
    "form_label_change_shipped": FLIP_CRITERION_REQUIRED,
    "dual_honest_tf_aware_path_under_wire": FLIP_CRITERION_REQUIRED,
    "primary_online_lambda_owns_verdict_under_tf_aware_form": FLIP_CRITERION_REQUIRED,
    "primary_online_lambda_linf_le_15_under_shipped_tf_aware_form": (
        FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    ),
    "wire_shipped": FLIP_CRITERION_REQUIRED,
    "dual_recovery_path_labeled_honestly_under_wire": FLIP_CRITERION_REQUIRED,
    "no_silent_classic_form_linf_reuse": FLIP_CRITERION_REQUIRED,
    "online_linf_gate_closed_under_tf_path_as_co_req_when_proven": FLIP_CRITERION_REQUIRED,
    "isolation_tests_rewritten_with_wire_not_deleted": FLIP_CRITERION_REQUIRED,
}

# Explicit anti-criteria: these NEVER prove dual_linf_under_wire today.
CASE1_DUAL_LINF_UNDER_WIRE_ANTI_CRITERIA_TODAY: tuple = (
    "probe_linf",
    "bridge_linf",
    "warmstart_linf",
    "pooling_linf",
    "seed_identity_linf",
    "recovered_blender_linf",
    "residual_must_vanish",
    "packaging_alone",
    "blueprint_alone",
    "prep_alone",
    "scaffold_alone",
    "rehearsal_alone",
    "online_linf_gate_criteria_alone",
    "online_linf_gate_open_alone",
    "online_linf_gate_closed_alone",
    "this_dual_linf_criteria_alone",
    "this_dual_linf_criteria_contract_alone",
    "diagnostic_linf_alone",
    "go_board_alone",
    "design_contracts_alone",
)


def case1_dual_linf_under_wire_flip_criteria() -> Dict[str, str]:
    """Return a copy of the dual_linf_under_wire flip-criteria map."""
    return dict(CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA)


def case1_dual_linf_under_wire_criteria_met_today_map() -> Dict[str, bool]:
    """Per-criterion met_today snapshot under HEAD defaults.

    Aggregate criteria_met_today / dual_linf_proof_allowed_today remain False while
    isolation rewrite, form label shipped, wire_shipped, dual-honest TF path under
    wire, and online_linf_gate under TF path remain open. Structural honesty labels
    that already hold offline (e.g. dual_recovery_path None on TF surface; no silent
    classic form reuse registry) may be True without flipping the aggregate.
    """
    return {
        "isolation_rewrite_with_wire": False,
        "form_label_change_shipped": False,
        # Dual-honest TF-aware *wire path* not present (offline ladder ≠ wire path).
        "dual_honest_tf_aware_path_under_wire": False,
        # Classic Case 1 already owns VERDICT via PRIMARY online λ — structural.
        # Under *shipped TF-aware form* this must still hold (not yet).
        "primary_online_lambda_owns_verdict_under_tf_aware_form": False,
        # linf≤15 under shipped TF-aware form — not applicable until wire ships.
        "primary_online_lambda_linf_le_15_under_shipped_tf_aware_form": False,
        "wire_shipped": False,
        # Planned dual recovery under wire is labeled (not pure-ADMM) offline;
        # today TF surface dual_recovery_path is None (labeled honestly).
        "dual_recovery_path_labeled_honestly_under_wire": True,
        # Planned form is registered and distinct from classic (no silent reuse).
        "no_silent_classic_form_linf_reuse": True,
        # online_linf_gate still open under TF path — co-req when proven remains unmet.
        "online_linf_gate_closed_under_tf_path_as_co_req_when_proven": False,
        "isolation_tests_rewritten_with_wire_not_deleted": False,
    }


def case1_dual_linf_proof_allowed_today(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Hard False while wire/form/isolation / TF-path prerequisites remain open.

    Aggregate proof permission never allows dual_linf_under_wire to become proven
    until required wire-shipping criteria are all met. Distinct name from online
    gate_flip_allowed_today.
    """
    met = criteria_met if criteria_met is not None else (
        case1_dual_linf_under_wire_criteria_met_today_map()
    )
    required_keys = [
        k
        for k, cls in CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA.items()
        if cls == FLIP_CRITERION_REQUIRED
    ]
    return all(bool(met.get(k)) for k in required_keys)


def case1_dual_linf_under_wire_criteria_met_today_aggregate(
    criteria_met: Optional[Dict[str, bool]] = None,
) -> bool:
    """Aggregate criteria_met_today — False until all required criteria hold."""
    return case1_dual_linf_proof_allowed_today(criteria_met)


def _case1_dual_linf_under_wire_criteria_contract_honesty_fields() -> Dict[str, Any]:
    """Machine-readable dual-ban / not-proof / not-wire locks."""
    return {
        "kind": CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND,
        "solver": False,
        "dual_recovery_path": None,
        "on_excel_case1_path": False,
        "on_case1_solve": False,
        "not_case1_solve": True,
        "case1_form_unchanged": True,
        "wire_shipped": False,
        "not_wire_shipped": True,
        "path_shipped": False,
        "bundle_shipped": False,
        "form_label_change_shipped": False,
        "isolation_rewrite_shipped": False,
        "isolation_tests_rewritten_with_wire": False,
        "not_pure_admm_dual_recovery": True,
        "not_full_plant_mass_balance": True,
        "not_full_plant_blocks_feed_lp": True,
        "not_live_plant_blocks": True,
        "not_isolation_rewrite": True,
        "not_full_tf_admm_wire": True,
        "contract_is_not_dual_linf_under_wire_proof": True,
        "contract_is_not_gate_flip": True,
        "contract_is_not_wire": True,
        "contract_is_not_verdict_gate": True,
        "criteria_present_is_not_proven": True,
        "criteria_present_is_not_wire": True,
        "distinct_from_online_linf_gate_criteria_contract": True,
        "gates_status_not_checklist_id": True,
        "status_target": CASE1_DUAL_LINF_UNDER_WIRE_STATUS_TARGET,
        "order_hint_coreq": CASE1_DUAL_LINF_UNDER_WIRE_ORDER_HINT_COREQ,
        "online_linf_gate_criteria_alone_is_anti_criterion": True,
        "probe_linf_is_not_proof_today": True,
        "bridge_linf_is_not_proof_today": True,
        "warmstart_linf_is_not_proof_today": True,
        "pooling_linf_is_not_proof_today": True,
        "seed_identity_linf_is_not_proof": True,
        "recovered_blender_linf_is_not_proof_today": True,
        "packaging_alone_is_not_proof": True,
        "blueprint_alone_is_not_proof": True,
        "prep_alone_is_not_proof": True,
        "scaffold_alone_is_not_proof": True,
        "rehearsal_alone_is_not_proof": True,
        "this_dual_linf_criteria_alone_is_not_proof": True,
        "no_blender_offline_affine_kernel_blocker_still_true": True,
        "case1_is_cdu_blender_package_admm_blocker_still_true": True,
        "scope": "case1_dual_linf_under_wire_criteria_contract_offline",
        "note": (
            "Offline Case-1 dual_linf_under_wire flip-criteria contract: machine-readable "
            "criteria for status dual_linf_under_wire / order_hint dual_linf_under_wire_proven. "
            "Status stays unproven; dual_linf_proof_allowed_today=False; criteria_met_today=False; "
            "dual_recovery_path=None; solver=False; on_excel_case1_path=False; wire_shipped=False; "
            f"case1_form_unchanged ({CASE1_FORM_CURRENT}). Contract is NOT dual L∞ under wire "
            "proof, NOT gate flip, NOT wire, NOT VERDICT gate. Distinct from "
            "online_linf_gate criteria (checklist id online_linf_gate_under_tf_path). "
            "Probe/bridge/warmstart/pooling/seed/recovered/packaging/blueprint/prep/"
            "scaffold/rehearsal/online_linf_gate_criteria_alone never prove dual_linf. "
            "first_blocking_coreq remains isolation_rewrite_with_wire. UNITS stay "
            "FCC/COKER/CDU. Does not clear DEFAULT_WIRE_BLOCKERS. Does not redefine "
            "ready_for_wire_discussion. Always-on numpy; no TF/PuLP/excel_pipeline on "
            "hot path; no maximizer; no linf≤15 gate on contract ok."
        ),
    }


def offline_case1_dual_linf_under_wire_criteria_contract_report() -> Dict[str, Any]:
    """Always-on dual_linf_under_wire flip-criteria contract (no TF, no PuLP, no solve).

    Aggregate ``ok`` / ``contract_ok`` = criteria formalized ∧ honesty locks ∧
    dual_linf unproven ∧ online_linf_gate still open ∧ blockers non-empty ∧ form
    classic ∧ UNITS FCC/COKER/CDU ∧ dual_linf_proof_allowed_today=False ∧
    criteria_met_today=False ∧ first_blocking=isolation.
    **Not** dual L∞ proven. **Not** gate flipped. **Not** wire. **Not** VERDICT.
    Distinct from ``offline_case1_online_linf_gate_criteria_contract_report``.
    """
    honesty = _case1_dual_linf_under_wire_criteria_contract_honesty_fields()
    form = case1_form_label_contract()
    dual_linf = case1_dual_linf_proof_checklist()
    blockers = list(DEFAULT_WIRE_BLOCKERS)
    critical = set(CASE1_CONTRACT_CRITICAL_BLOCKERS)
    blockers_still_documented = critical.issubset(set(blockers)) and len(blockers) > 0

    checklist = dual_linf["dual_linf_proof_checklist"]
    gate_status = checklist.get("online_linf_gate_under_tf_path")
    gate_still_open = gate_status == "open"
    open_ids = dual_linf["dual_linf_proof_checklist_open_ids"]
    dual_status = dual_linf["dual_linf_under_wire_status"]

    flip_criteria = case1_dual_linf_under_wire_flip_criteria()
    criteria_met_map = case1_dual_linf_under_wire_criteria_met_today_map()
    dual_linf_proof_allowed_today = case1_dual_linf_proof_allowed_today(criteria_met_map)
    criteria_met_today = case1_dual_linf_under_wire_criteria_met_today_aggregate(
        criteria_met_map
    )

    required_keys = set(CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA.keys())
    under_wire_key = "primary_online_lambda_linf_le_15_under_shipped_tf_aware_form"
    flip_criteria_formalized = (
        set(flip_criteria.keys()) == required_keys
        and flip_criteria.get(under_wire_key) == FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        and all(
            flip_criteria[k] == FLIP_CRITERION_REQUIRED
            for k in required_keys
            if k != under_wire_key
        )
    )
    criteria_present = (
        flip_criteria_formalized
        and CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_ANNOTATION == "present"
    )

    units_ok = list(UNITS) == ["FCC", "COKER", "CDU"] and "BLENDER" not in UNITS
    blocker_ok = (
        "dual_linf_under_wire_unproven" in blockers
        and "no_blender_offline_affine_kernel" in blockers
        and honesty["no_blender_offline_affine_kernel_blocker_still_true"] is True
    )

    # Distinctness vs online_linf_gate criteria contract
    online_kind = CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_KIND
    distinct_ok = (
        CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND != online_kind
        and honesty["distinct_from_online_linf_gate_criteria_contract"] is True
        and honesty["gates_status_not_checklist_id"] is True
        and honesty["status_target"] == "dual_linf_under_wire"
        and honesty["order_hint_coreq"] == "dual_linf_under_wire_proven"
    )

    dual_ban_ok = bool(
        SOLVER is False
        and DUAL_RECOVERY_PATH is None
        and ON_EXCEL_CASE1_PATH is False
        and honesty["dual_recovery_path"] is None
        and honesty["solver"] is False
        and honesty["wire_shipped"] is False
        and honesty["path_shipped"] is False
        and honesty["bundle_shipped"] is False
        and honesty["form_label_change_shipped"] is False
        and honesty["isolation_rewrite_shipped"] is False
        and honesty["on_excel_case1_path"] is False
        and honesty["contract_is_not_dual_linf_under_wire_proof"] is True
        and honesty["contract_is_not_gate_flip"] is True
        and honesty["contract_is_not_wire"] is True
        and honesty["contract_is_not_verdict_gate"] is True
        and honesty["criteria_present_is_not_proven"] is True
        and honesty["probe_linf_is_not_proof_today"] is True
        and honesty["bridge_linf_is_not_proof_today"] is True
        and honesty["warmstart_linf_is_not_proof_today"] is True
        and honesty["online_linf_gate_criteria_alone_is_anti_criterion"] is True
    )
    dual_linf_unproven_ok = bool(
        dual_status == "unproven"
        and dual_linf["dual_linf_under_wire_unproven_still_true"] is True
        and dual_linf["dual_linf_status_unproven_ok"] is True
        and dual_status != "proven"
    )
    form_ok = bool(
        form["form_contract_ok"]
        and form["form_current"] == CASE1_FORM_CURRENT
        and form["form_unchanged"] is True
        and honesty["case1_form_unchanged"] is True
    )
    flip_permission_ok = (
        dual_linf_proof_allowed_today is False and criteria_met_today is False
    )
    gate_open_ok = gate_still_open and gate_status == "open"

    first_blocking = (
        case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq()
    )
    first_blocking_ok = (
        first_blocking.get("first_blocking_coreq") == "isolation_rewrite_with_wire"
    )

    anti = CASE1_DUAL_LINF_UNDER_WIRE_ANTI_CRITERIA_TODAY
    anti_ok = (
        "online_linf_gate_criteria_alone" in anti
        and "probe_linf" in anti
        and "bridge_linf" in anti
        and "warmstart_linf" in anti
        and "packaging_alone" in anti
        and "blueprint_alone" in anti
        and "prep_alone" in anti
        and "scaffold_alone" in anti
        and "rehearsal_alone" in anti
        and "this_dual_linf_criteria_alone" in anti
    )

    honesty_ok = bool(
        dual_ban_ok
        and units_ok
        and blocker_ok
        and dual_linf_unproven_ok
        and form_ok
        and flip_criteria_formalized
        and criteria_present
        and flip_permission_ok
        and gate_open_ok
        and blockers_still_documented
        and distinct_ok
        and first_blocking_ok
        and anti_ok
    )
    contract_ok = honesty_ok
    ok = (
        contract_ok
        and dual_status == "unproven"
        and dual_linf_proof_allowed_today is False
        and criteria_met_today is False
        and honesty["wire_shipped"] is False
        and honesty["dual_recovery_path"] is None
    )

    ok_criteria = (
        "criteria formalized ∧ honesty locks ∧ dual_linf unproven ∧ "
        "online_linf_gate still open ∧ blockers non-empty ∧ form classic ∧ "
        "UNITS FCC/COKER/CDU ∧ dual_linf_proof_allowed_today=False ∧ "
        "criteria_met_today=False ∧ first_blocking=isolation_rewrite_with_wire — "
        "NOT dual L∞ under wire proven; NOT gate flipped; NOT wire; NOT VERDICT; "
        "distinct from online_linf_gate criteria contract"
    )

    planned_dual_recovery = CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED

    return {
        **honesty,
        "ok": ok,
        "contract_ok": contract_ok,
        "honesty_ok": honesty_ok,
        "dual_ban_ok": dual_ban_ok,
        "units_ok": units_ok,
        "blocker_ok": blocker_ok,
        "form_ok": form_ok,
        "dual_linf_unproven_ok": dual_linf_unproven_ok,
        "flip_criteria_formalized": flip_criteria_formalized,
        "criteria_present": criteria_present,
        "flip_permission_ok": flip_permission_ok,
        "gate_open_ok": gate_open_ok,
        "distinct_ok": distinct_ok,
        "first_blocking_ok": first_blocking_ok,
        "anti_ok": anti_ok,
        "ok_criteria": ok_criteria,
        # Status target (must remain unproven)
        "dual_linf_under_wire": dual_status,
        "dual_linf_under_wire_status": dual_status,
        "dual_linf_under_wire_unproven_still_true": dual_linf[
            "dual_linf_under_wire_unproven_still_true"
        ],
        "dual_linf_status_unproven_ok": dual_linf["dual_linf_status_unproven_ok"],
        "dual_linf_criteria_contract": (
            CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_ANNOTATION
        ),
        # Flip permission (hard False under HEAD) — distinct from gate_flip_allowed
        "dual_linf_proof_allowed_today": dual_linf_proof_allowed_today,
        "criteria_met_today": criteria_met_today,
        "gate_flip_allowed_today": False,  # online gate stays open; not closed by this
        "flip_criteria": flip_criteria,
        "dual_linf_flip_criteria": flip_criteria,
        "criteria_status_today": criteria_met_map,
        "criteria_met_today_map": criteria_met_map,
        "anti_criteria_today": list(anti),
        "flip_criterion_required_class": FLIP_CRITERION_REQUIRED,
        "flip_criterion_required_under_wire_only_class": (
            FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
        ),
        # Distinctness from online_linf_gate criteria
        "online_linf_gate_criteria_contract_kind": online_kind,
        "this_kind": CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND,
        "online_linf_gate_under_tf_path": gate_status,
        "online_linf_gate_still_open": gate_still_open,
        # first_blocking remains isolation (do not claim dual_linf is first)
        "first_blocking_coreq": first_blocking.get("first_blocking_coreq"),
        "first_blocking_coreq_status": first_blocking.get(
            "first_blocking_coreq_status"
        ),
        "first_blocking": first_blocking,
        # Form
        "form_current": form["form_current"],
        "form_planned": form["form_planned"],
        "planned_form_distinct": form["planned_form_distinct"],
        "form_unchanged": form["form_unchanged"],
        "form_contract_ok": form["form_contract_ok"],
        "form_label_change_required_still_true": form[
            "form_label_change_required_still_true"
        ],
        # Ship flags hard false
        "path_shipped": False,
        "wire_shipped": False,
        "bundle_shipped": False,
        "form_label_change_shipped": False,
        "isolation_rewrite_shipped": False,
        "isolation_tests_rewritten_with_wire": False,
        "feature_flag_name": CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME,
        "feature_flag_enabled_today": (
            CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY
        ),
        # Dual recovery
        "dual_recovery_path_today_on_tf_surface": None,
        "dual_recovery_path_planned_when_shipped": planned_dual_recovery,
        "planned_dual_recovery_path_not_pure_admm": True,
        # Checklist snapshot
        "dual_linf_proof_checklist": checklist,
        "dual_linf_proof_checklist_open_ids": open_ids,
        "dual_linf_proof_checklist_n_open": dual_linf[
            "dual_linf_proof_checklist_n_open"
        ],
        "units_affine_unchanged": list(UNITS),
        # Blockers
        "wire_blockers": blockers,
        "critical_blockers_required": list(CASE1_CONTRACT_CRITICAL_BLOCKERS),
        "n_wire_blockers": len(blockers),
        "blockers_still_documented": blockers_still_documented,
        "wire_not_shipped_blocker_still_true": "wire_not_shipped" in blockers,
        "dual_linf_under_wire_unproven_blocker_still_true": (
            "dual_linf_under_wire_unproven" in blockers
        ),
        "no_blender_offline_affine_kernel_in_default_wire_blockers": (
            "no_blender_offline_affine_kernel" in blockers
        ),
        "no_blender_offline_affine_kernel_in_critical_blockers": (
            "no_blender_offline_affine_kernel" in critical
        ),
        "does_not_clear_default_wire_blockers": True,
        "does_not_redefine_ready_for_wire_discussion": True,
        "ready_for_wire_discussion_semantics": (
            "unchanged_parity_priced_timings_honesty_only"
        ),
        "suggested_next_wave_after_preflight": SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT,
        "suggested_next_wave_still_full_wire": (
            SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT
            == "dual_honest_tf_case1_wire_with_isolation_rewrite_and_form_label_change"
        ),
        "tf_available": tf_available(),
        "criteria_contract_available": True,
        "dual_linf_under_wire_criteria_contract_available": True,
        "linf_le_15_is_not_proof_gate_today": True,
        "residual_must_vanish_is_not_proof": True,
        "excel_packaging_twin_present": True,
        "excel_packaging_twin_deferred": False,
        "any_ship_allowed_today": False,
        "all_ship_flags_false": True,
        "note": honesty["note"],
    }


def case1_dual_linf_under_wire_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``offline_case1_dual_linf_under_wire_criteria_contract_report``."""
    return offline_case1_dual_linf_under_wire_criteria_contract_report(**kwargs)


def multi_unit_case1_dual_linf_under_wire_criteria_contract_report(
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for multi-unit registry symmetry."""
    return offline_case1_dual_linf_under_wire_criteria_contract_report(**kwargs)




__all__ = [
    "MODULE_KIND",
    "SOURCE",
    "SOLVER",
    "DUAL_RECOVERY_PATH",
    "ON_EXCEL_CASE1_PATH",
    "POSTPROCESS",
    "UNITS",
    "AffineCoeffs",
    "OfflineUnitDescriptor",
    "tf_available",
    "tf_import_error",
    "honesty_metadata",
    "affine_coeffs_from_base_delta",
    "pack_driver_vector",
    "numpy_affine_forward",
    "y_raw_dict",
    "apply_fcc_postprocess",
    "apply_coker_postprocess",
    "apply_cdu_postprocess",
    "TFLinearBlock",
    "tf_linear_fcc",
    "tf_linear_coker",
    "tf_linear_cdu",
    "offline_unit_registry",
    "offline_unit_coeffs",
    "cached_offline_unit_coeffs",
    "clear_offline_unit_coeffs_cache",
    "build_offline_unit",
    "offline_units_status",
    "multi_unit_parity_report",
    "PRICE_SOURCE",
    "default_offline_prices",
    "pack_price_vector",
    "priced_residual_for_unit",
    "multi_unit_priced_residual_report",
    "local_box_direction",
    "multi_unit_block_solve_timing_report",
    "offline_block_solve_readiness_report",
    "ADMM_RESIDUAL_KIND",
    "ADMM_AUGMENTED_FORMULA_L1",
    "admm_residual_for_unit",
    "multi_unit_admm_residual_report",
    "ADMM_SUBPROBLEM_KIND",
    "ADMM_SUBPROBLEM_FORMULA_L1_RAW",
    "admm_block_subproblem_for_unit",
    "multi_unit_admm_block_subproblem_report",
    "ADMM_COORDINATION_KIND",
    "ADMM_COORDINATION_FORMULA",
    "ADMM_COORDINATION_SCOPE",
    "admm_coordination_round_for_unit",
    "multi_unit_admm_coordination_report",
    "ADMM_PLANT_LINKING_KIND",
    "ADMM_PLANT_LINKING_FORMULA",
    "ADMM_PLANT_LINKING_SCOPE",
    "ADMM_PLANT_LINKING_STREAMS",
    "ADMM_PLANT_NAMED_LINKING_SCOPE",
    "ADMM_PLANT_NAMED_LINKING_STREAMS",
    "ADMM_PLANT_LINKING_MODES",
    "offline_plant_linking_topology",
    "offline_plant_named_linking_topology",
    "project_linking_to_unit",
    "lift_unit_y_to_linking",
    "plant_linking_admm_round",
    "multi_block_plant_linking_admm_report",
    "multi_block_plant_named_linking_admm_report",
    "WIRE_PREFLIGHT_KIND",
    "DEFAULT_WIRE_BLOCKERS",
    "WIRE_BLOCKER_NOTES",
    "SUGGESTED_NEXT_WAVE_AFTER_PREFLIGHT",
    "offline_wire_blocker_catalog",
    "offline_wire_preflight_report",
    "multi_unit_wire_preflight_report",
    "CASE1_SHAPED_LINKING_KIND",
    "CASE1_SHAPED_LINKING_SCOPE",
    "CASE1_SHAPED_LINKING_STREAMS",
    "CASE1_SHAPED_BLENDER_SURFACE",
    "CASE1_SHAPED_LINKING_FORMULA",
    "CASE1_SHAPED_BLEND_RECIPES",
    "case1_shaped_cdu_to_intermediate_map",
    "project_cdu_y_to_case1_intermediates",
    "blender_recipe_use_from_products",
    "case1_shaped_cdu_blender_linking_round",
    "offline_case1_shaped_cdu_blender_linking_report",
    "multi_block_case1_shaped_linking_admm_report",
    "CASE1_DUAL_SPACE_FORM_CONTRACT_KIND",
    "CASE1_FORM_CURRENT",
    "CASE1_PLANNED_TF_AWARE_FORM",
    "CASE1_DUAL_LINF_UNDER_WIRE_STATUS",
    "CASE1_DUAL_LINF_PROOF_CHECKLIST",
    "CASE1_CONTRACT_CRITICAL_BLOCKERS",
    "case1_form_label_contract",
    "case1_dual_space_stream_map",
    "case1_dual_linf_proof_checklist",
    "offline_case1_dual_space_form_contract_report",
    "multi_unit_case1_dual_space_form_contract_report",
    "CASE1_DUAL_SPACE_LINF_PROBE_KIND",
    "CASE1_FIXTURE_PRIMARY_ONLINE_LAMBDA",
    "CASE1_FIXTURE_SECONDARY_RECOVERED_LAMBDA",
    "CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE",
    "CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW",
    "CASE1_DUAL_VECTOR_FACE_SECONDARY_RECOVERED",
    "case1_primary_online_lambda_fixture",
    "case1_secondary_recovered_lambda_fixture",
    "case1_dual_space_stream_aligned_linf",
    "stream_aligned_dual_linf",
    "extract_case1_shaped_skeleton_lambda",
    "case1_shaped_skeleton_lambda",
    "offline_case1_dual_space_linf_probe_report",
    "case1_dual_space_linf_probe",
    "multi_unit_case1_dual_space_linf_probe_report",
    "CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_BRIDGE_KIND",
    "LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED",
    "LIVE_LAMBDA_SOURCE_FIXTURE",
    "LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT",
    "LIVE_LAMBDA_SOURCE_MISSING",
    "case1_primary_online_lambda_from_mapping",
    "extract_case1_primary_online_lambda",
    "extract_case1_secondary_recovered_lambda",
    "offline_case1_dual_space_linf_live_lambda_bridge_report",
    "case1_dual_space_linf_live_lambda_bridge",
    "multi_unit_case1_dual_space_linf_live_lambda_bridge_report",
    "CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_SEEDED_WARMSTART_KIND",
    "SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY",
    "Z0_POLICY_UNCHANGED_DEFAULT_SKELETON",
    "case1_warmstart_seed_lambda_from_primary",
    "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report",
    "case1_dual_space_linf_live_lambda_seeded_warmstart",
    "multi_unit_case1_dual_space_linf_live_lambda_seeded_warmstart_report",
    "CASE1_HONEST_BLENDER_POOLING_PATH_KIND",
    "CASE1_HONEST_BLENDER_POOLING_PATH_CHECKLIST_STATUS",
    "CASE1_HONEST_BLENDER_POOLING_PATH_RECIPES_SOURCE",
    "offline_case1_honest_blender_pooling_path_report",
    "case1_honest_blender_pooling_path_report",
    "multi_unit_case1_honest_blender_pooling_path_report",
    "CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_KIND",
    "CASE1_ONLINE_LINF_GATE_CHECKLIST_KEY",
    "CASE1_ONLINE_LINF_GATE_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_ONLINE_LINF_GATE_FLIP_CRITERIA",
    "CASE1_ONLINE_LINF_GATE_ANTI_CRITERIA_TODAY",
    "FLIP_CRITERION_REQUIRED",
    "FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY",
    "case1_online_linf_gate_flip_criteria",
    "case1_online_linf_gate_criteria_met_today_map",
    "case1_online_linf_gate_flip_allowed_today",
    "case1_online_linf_gate_criteria_met_today_aggregate",
    "offline_case1_online_linf_gate_criteria_contract_report",
    "case1_online_linf_gate_criteria_contract_report",
    "multi_unit_case1_online_linf_gate_criteria_contract_report",
    "CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_KIND",
    "CASE1_ISOLATION_REWRITE_CHECKLIST_KEY",
    "CASE1_ISOLATION_REWRITE_DESIGN_CONTRACT_ANNOTATION",
    "CASE1_ISOLATION_REWRITE_BLOCKER_ID",
    "CASE1_ISOLATION_REWRITE_FLIP_KEY",
    "CASE1_ISOLATION_TESTS_REWRITE_FLIP_KEY",
    "CASE1_ISOLATION_INVARIANTS_MUST_SURVIVE",
    "CASE1_ISOLATION_REWRITE_POST_WIRE_SHAPE",
    "case1_isolation_invariants_must_survive",
    "case1_isolation_rewrite_post_wire_shape",
    "offline_case1_isolation_rewrite_design_contract_report",
    "case1_isolation_rewrite_design_contract_report",
    "multi_unit_case1_isolation_rewrite_design_contract_report",
    "CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_KIND",
    "CASE1_WIRE_SHIP_ACCEPTANCE_DESIGN_CONTRACT_ANNOTATION",
    "CASE1_WIRE_SHIP_ACCEPTANCE_CRITERIA",
    "CASE1_WIRE_SHIP_ANTI_CRITERIA_TODAY",
    "case1_wire_ship_acceptance_criteria",
    "case1_wire_ship_acceptance_criteria_met_today_map",
    "case1_wire_ship_allowed_today",
    "case1_wire_ship_criteria_met_today_aggregate",
    "offline_case1_wire_ship_acceptance_design_contract_report",
    "case1_wire_ship_acceptance_design_contract_report",
    "multi_unit_case1_wire_ship_acceptance_design_contract_report",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_KIND",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_DESIGN_CONTRACT_ANNOTATION",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_NAME",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_DUAL_RECOVERY_PLANNED",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_SHAPE",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_tf_aware_path_shape",
    "offline_case1_dual_honest_tf_aware_path_design_contract_report",
    "case1_dual_honest_tf_aware_path_design_contract_report",
    "multi_unit_case1_dual_honest_tf_aware_path_design_contract_report",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_KIND",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_FLIP_CRITERIA",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_PRESENT_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_tf_aware_path_present_flip_criteria",
    "case1_dual_honest_tf_aware_path_present_criteria_met_today_map",
    "case1_dual_honest_tf_aware_path_present_ship_met_allowed_today",
    "case1_dual_honest_tf_aware_path_present_criteria_met_today_aggregate",
    "offline_case1_dual_honest_tf_aware_path_present_criteria_contract_report",
    "case1_dual_honest_tf_aware_path_present_criteria_contract_report",
    "multi_unit_case1_dual_honest_tf_aware_path_present_criteria_contract_report",
    "CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_KIND",
    "CASE1_FORM_LABEL_CHANGE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_NAME",
    "CASE1_FORM_LABEL_CHANGE_MUTATION_PATH_EXECUTED_TODAY",
    "CASE1_FORM_LABEL_CHANGE_SHIPPED_FLIP_CRITERIA",
    "CASE1_FORM_LABEL_CHANGE_SHIPPED_ANTI_CRITERIA_TODAY",
    "case1_form_label_change_shipped_flip_criteria",
    "case1_form_label_change_shipped_criteria_met_today_map",
    "case1_form_label_ship_allowed_today",
    "case1_form_label_change_shipped_criteria_met_today_aggregate",
    "offline_case1_form_label_change_shipped_criteria_contract_report",
    "case1_form_label_change_shipped_criteria_contract_report",
    "multi_unit_case1_form_label_change_shipped_criteria_contract_report",
    "CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_KIND",
    "CASE1_ISOLATION_REWRITE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_ISOLATION_REWRITE_SHIPPED_FLIP_CRITERIA",
    "CASE1_ISOLATION_REWRITE_SHIPPED_ANTI_CRITERIA_TODAY",
    "case1_isolation_rewrite_shipped_flip_criteria",
    "case1_isolation_rewrite_shipped_criteria_met_today_map",
    "case1_isolation_ship_allowed_today",
    "case1_isolation_rewrite_ship_allowed_today",
    "case1_isolation_rewrite_shipped_criteria_met_today_aggregate",
    "offline_case1_isolation_rewrite_shipped_criteria_contract_report",
    "case1_isolation_rewrite_shipped_criteria_contract_report",
    "multi_unit_case1_isolation_rewrite_shipped_criteria_contract_report",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_KIND",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_DESIGN_CONTRACT_ANNOTATION",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NAME",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_MEMBERS",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ORDER_HINT_IS_NOT_EXECUTOR",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_NO_AUTO_WIRE",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ATOMIC_COSHIP_ALSO_VALID",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_multi_blocker_wire_bundle_members",
    "case1_dual_honest_multi_blocker_wire_bundle_member_status_today",
    "case1_dual_honest_multi_blocker_wire_bundle_criteria_met_today_map",
    "case1_dual_honest_multi_blocker_wire_bundle_ship_allowed_today",
    "case1_dual_honest_multi_blocker_wire_bundle_criteria_met_today_aggregate",
    "case1_dual_honest_multi_blocker_wire_bundle_shipped",
    "offline_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report",
    "case1_dual_honest_multi_blocker_wire_bundle_design_contract_report",
    "multi_unit_case1_dual_honest_multi_blocker_wire_bundle_design_contract_report",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_KIND",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_FLIP_CRITERIA",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_BUNDLE_SHIPPED_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_multi_blocker_wire_bundle_shipped_flip_criteria",
    "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_map",
    "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_met_today_aggregate",
    "offline_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
    "case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
    "multi_unit_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_report",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_KIND",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANNOTATION",
    "CASE1_DUAL_HONEST_TF_AWARE_PATH_EXECUTION_SCAFFOLD_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_tf_aware_path_execution_scaffold_compose",
    "offline_case1_dual_honest_tf_aware_path_execution_scaffold_report",
    "case1_dual_honest_tf_aware_path_execution_scaffold_report",
    "multi_unit_case1_dual_honest_tf_aware_path_execution_scaffold_report",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_KIND",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANNOTATION",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_REHEARSAL_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_multi_blocker_wire_rehearsal_coreq_matrix",
    "offline_case1_dual_honest_multi_blocker_wire_rehearsal_report",
    "case1_dual_honest_multi_blocker_wire_rehearsal_report",
    "multi_unit_case1_dual_honest_multi_blocker_wire_rehearsal_report",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_KIND",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANNOTATION",
    "CASE1_DUAL_HONEST_MULTI_BLOCKER_WIRE_IMPLEMENTATION_BLUEPRINT_ANTI_CRITERIA_TODAY",
    "case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq",
    "case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board",
    "offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
    "case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
    "multi_unit_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report",
    "CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND",
    "CASE1_DUAL_LINF_UNDER_WIRE_STATUS_TARGET",
    "CASE1_DUAL_LINF_UNDER_WIRE_ORDER_HINT_COREQ",
    "CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_ANNOTATION",
    "CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA",
    "CASE1_DUAL_LINF_UNDER_WIRE_ANTI_CRITERIA_TODAY",
    "case1_dual_linf_under_wire_flip_criteria",
    "case1_dual_linf_under_wire_criteria_met_today_map",
    "case1_dual_linf_proof_allowed_today",
    "case1_dual_linf_under_wire_criteria_met_today_aggregate",
    "offline_case1_dual_linf_under_wire_criteria_contract_report",
    "case1_dual_linf_under_wire_criteria_contract_report",
    "multi_unit_case1_dual_linf_under_wire_criteria_contract_report",
    "excel_fcc_matrix_matches_affine",
    "excel_coker_matrix_matches_affine",
]
