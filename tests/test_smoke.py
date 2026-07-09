"""Scaffold smoke tests — package import + data load + optional LP solve."""

from __future__ import annotations


def test_version():
    import pims_admm_llm

    assert pims_admm_llm.__version__ == "0.1.0"


def test_load_crude_data():
    from pims_admm_llm.models import load_crude_data

    data = load_crude_data()
    assert len(data.crudes) >= 1
    assert data.cdu_capacity_kbd > 0
    assert "gasoline" in data.products


def test_monolithic_solve():
    from pims_admm_llm.models import load_crude_data, solve_monolithic

    result = solve_monolithic(load_crude_data(), msg=False)
    assert result.status == "Optimal"
    assert result.objective is not None
