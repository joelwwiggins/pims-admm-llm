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

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

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
        "formula": "y_raw = y0 + D @ (x - x0)  # pre-postprocess exact linear",
        "note": (
            "Optional exact-linear surface only (FCC + COKER + CDU offline kernels). "
            "Not Excel Case 1 solver; not ADMM dual recovery; not a learned model. "
            "Full evaluate() = affine + numpy postprocess (clamp/renorm) outside TF. "
            "Coker renorm always engages → raw ≠ evaluate even at reference. "
            "CDU has nested cut_points_f.* drivers in x0; Submodel_CDU is classic "
            "TECH+A export (not a PIMS MB_* matrix twin like FCC/Coker)."
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
    "build_offline_unit",
    "offline_units_status",
    "multi_unit_parity_report",
    "excel_fcc_matrix_matches_affine",
    "excel_coker_matrix_matches_affine",
]
