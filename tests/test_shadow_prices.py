"""Worker 7: shadow price table + linearity smoke tests."""

from __future__ import annotations

from pims_admm_llm.models.data import load_crude_data
from pims_admm_llm.models.blocks import solve_monolithic
from pims_admm_llm.reporting.shadow_prices import (
    build_shadow_price_report,
    format_report_text,
    run_linearity_checks,
)


def test_monolithic_optimal_and_duals():
    data = load_crude_data()
    r = solve_monolithic(data)
    assert r.status == "Optimal"
    assert r.objective > 0
    assert "cdu_capacity" in r.duals
    assert "crude_supply_WTI_light" in r.duals
    assert "tank_naphtha" in r.duals


def test_shadow_report_has_categories():
    report = build_shadow_price_report(run_linearity=True)
    cats = {row.category for row in report.table}
    assert "intermediate" in cats
    assert "capacity" in cats
    assert "crude" in cats
    assert "product" in cats
    text = format_report_text(report)
    assert "MARGINAL VALUE" in text
    assert "Linearity summary" in text
    # small-delta checks should mostly pass
    small = [c for c in report.linearity if abs(c.delta_rhs) <= 1.0]
    assert small, "expected small-delta linearity checks"
    assert sum(1 for c in small if c.passed) >= max(1, len(small) - 1)


def test_linearity_cdu_only():
    data = load_crude_data()
    base = solve_monolithic(data)
    checks = run_linearity_checks(
        data, base, deltas=[("cdu_capacity", 0.5), ("cdu_capacity", 1.0)]
    )
    assert all(c.passed for c in checks)
