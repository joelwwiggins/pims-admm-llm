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
# Multi-unit offline surface: FCC + Coker shells share this module; postprocess
# stays numpy/Python outside any TF graph (never dual recovery).
POSTPROCESS = "numpy_outside_tf"
UNITS = ("FCC", "COKER")

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
            "Optional exact-linear surface only (FCC + COKER offline kernels). "
            "Not Excel Case 1 solver; not ADMM dual recovery; not a learned model. "
            "Full evaluate() = affine + numpy postprocess (coke clamp/renorm) "
            "outside TF. Coker renorm always engages → raw affine ≠ evaluate even "
            "at reference."
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
    """Flatten reference_feed + numeric reference_conditions (FCC evaluate rules)."""
    ref_flat: Dict[str, float] = {}
    for k, v in (getattr(model, "reference_feed", None) or {}).items():
        if isinstance(v, (int, float)):
            ref_flat[str(k)] = float(v)
    for k, v in (getattr(model, "reference_conditions", None) or {}).items():
        if isinstance(v, (int, float)):
            ref_flat[str(k)] = float(v)
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


class TFLinearBlock:
    """Exact-linear block: y_raw = y0 + D @ (x - x0) with float64 TF constants.

    Postprocess is intentionally **not** in the graph. Call
    ``apply_fcc_postprocess`` / ``apply_coker_postprocess`` separately for full
    evaluate parity.
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
    "tf_available",
    "tf_import_error",
    "honesty_metadata",
    "affine_coeffs_from_base_delta",
    "pack_driver_vector",
    "numpy_affine_forward",
    "y_raw_dict",
    "apply_fcc_postprocess",
    "apply_coker_postprocess",
    "TFLinearBlock",
    "tf_linear_fcc",
    "tf_linear_coker",
    "excel_fcc_matrix_matches_affine",
    "excel_coker_matrix_matches_affine",
]
