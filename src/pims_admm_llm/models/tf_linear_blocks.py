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
            "not Case 1 wall time and not ADMM duals."
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
) -> Dict[str, Any]:
    """Compose timing + parity_ok + priced_ok under dual-ban honesty locks.

    One call answers \"ready for wire discussion?\" without re-implementing
    parity/priced math. Does **not** mean wire is shipped or duals are owned.
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
    base["note"] = (
        "Offline block-solve readiness report: cached multi-unit timing + "
        "parity_ok + priced_ok under dual-ban honesty. "
        "ready_for_wire_discussion is structural readiness only — wire is a "
        "separate checklist + form label change; dual_recovery_path remains "
        "None; on_excel_case1_path=False; timings/prices/gradients are NOT "
        "ADMM λ / Case 1 shadows; not Case 1 solve wall time; not a solve. "
        "Not pure-ADMM dual recovery."
    )
    return base


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
    "excel_fcc_matrix_matches_affine",
    "excel_coker_matrix_matches_affine",
]
