"""Optional TensorFlow exact-linear block helpers (charter direction).

This module is **not** on the Excel Case 1 / PuLP ADMM path. It exists so later
cycles can host an exact affine copy of base_delta (``y = y0 + D @ dx``) behind
an optional dependency.

Hard rules for this surface:
- Never import tensorflow at package import time in a way that breaks Case 1.
- Feasibility ownership stays with existing solvers (CBC / package ADMM).
- Dual recovery is **not** claimed here (``dual_recovery_path`` is always None).
- LLM / advisory layers must not treat this module as a solver bypass.

Install (optional)::

    pip install -e \".[tf]\"

On hosts without a TF wheel (e.g. some Jetson images), leave TF uninstalled;
Case 1 Excel smoke must remain green.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Honesty metadata shared by future block objects and tests (E6/E14).
MODULE_KIND = "exact_linear_copy_surface"
SOURCE = "base_delta"  # intended source of truth for coefficients
SOLVER = False
DUAL_RECOVERY_PATH: Optional[str] = None
ON_EXCEL_CASE1_PATH = False

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
        "tf_available": tf_available(),
        "note": (
            "Optional exact-linear surface only. Not Excel Case 1 solver; "
            "not ADMM dual recovery; not a learned model."
        ),
    }


__all__ = [
    "MODULE_KIND",
    "SOURCE",
    "SOLVER",
    "DUAL_RECOVERY_PATH",
    "ON_EXCEL_CASE1_PATH",
    "tf_available",
    "tf_import_error",
    "honesty_metadata",
]
