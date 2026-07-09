"""Shadow price / marginal value reporting (PIMS-style economic interpretation)."""

from .shadow_prices import (
    LinearityCheck,
    MarginalValueRow,
    ShadowPriceReport,
    build_shadow_price_report,
    format_report_text,
    run_linearity_checks,
)

__all__ = [
    "LinearityCheck",
    "MarginalValueRow",
    "ShadowPriceReport",
    "build_shadow_price_report",
    "format_report_text",
    "run_linearity_checks",
]
