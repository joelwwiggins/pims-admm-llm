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
            "UNITS); does not clear DEFAULT_WIRE_BLOCKERS."
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
) -> Dict[str, Any]:
    """Compose timing + parity_ok + priced_ok under dual-ban honesty locks.

    One call answers \"ready for wire discussion?\" without re-implementing
    parity/priced math. Does **not** mean wire is shipped or duals are owned.

    ``admm_residual_ok``, ``admm_block_subproblem_ok``, ``admm_coordination_ok``,
    ``admm_plant_linking_ok``, ``admm_plant_named_linking_ok``, and
    ``admm_case1_shaped_linking_ok`` are
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
    base["note"] = (
        "Offline block-solve readiness report: cached multi-unit timing + "
        "parity_ok + priced_ok under dual-ban honesty. "
        "ready_for_wire_discussion is structural readiness only (parity∧priced"
        "∧timings∧honesty) — wire is a separate checklist + form label change; "
        "dual_recovery_path remains None; on_excel_case1_path=False; "
        "timings/prices/gradients/ADMM residuals are NOT ADMM λ / Case 1 shadows; "
        "not Case 1 solve wall time; not a solve. Not pure-ADMM dual recovery. "
        "admm_residual_ok, admm_block_subproblem_ok, admm_coordination_ok, "
        "admm_plant_linking_ok, admm_plant_named_linking_ok, and "
        "admm_case1_shaped_linking_ok are additive "
        "pre-wire checklist items (synthetic λ,z,ρ residual / block subproblem / "
        "multi-round coordination / multi-block plant-linking synthetic + plant-named "
        "topology modes / Case-1-shaped CDU↔Blender offline skeleton; coordination is "
        "per-unit synthetic not plant linking; plant-linking modes are offline demos; "
        "Case-1-shaped skeleton is offline-only (not wire, not Case 1 duals, not full "
        "plant mass balance) and do not redefine ready_for_wire_discussion."
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
        "plant_linking/plant_named gates) and lists machine-readable wire_blockers true at "
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
    "excel_fcc_matrix_matches_affine",
    "excel_coker_matrix_matches_affine",
]
