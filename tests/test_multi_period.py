"""W4 multi-period inventory tanks: start inventory + end carries smoke."""

from __future__ import annotations

import pytest

from pims_admm_llm.models.assay_loader import load_assays_json
from pims_admm_llm.models.multi_period import TANK_KEYS, solve_multi_period


def test_multi_period_smoke_feasible():
    res = solve_multi_period(n_periods=2, inventory_mode=True)
    assert res.feasible
    assert res.status == "Optimal"
    assert res.n_periods == 2
    assert res.inventory_mode is True
    assert res.objective != 0.0
    assert len(res.tank_start) == 2
    assert len(res.tank_end) == 2
    assert len(res.carries) == 2


def test_start_inventory_from_assays():
    assays = load_assays_json()
    res = solve_multi_period(assays, n_periods=2, inventory_mode="multi_period")
    assert res.feasible
    tanks = assays.get("tanks") or {}
    expected = {
        "gasoil": float(tanks["tank_gasoil"]["start_kbd"]),
        "resid": float(tanks["tank_resid"]["start_kbd"]),
        "fcc_naph": float(tanks["tank_fcc_naph"]["start_kbd"]),
        "coker_naph": float(tanks["tank_coker_naph"]["start_kbd"]),
        "reformate": float(tanks["tank_reformate"]["start_kbd"]),
    }
    for k, v in expected.items():
        assert res.tank_start[0][k] == pytest.approx(v, abs=1e-6)
    # meta mirrors opening inventory
    assert res.meta["start0"]["gasoil"] == pytest.approx(expected["gasoil"], abs=1e-6)


def test_end_carries_equal_next_start():
    """Time coupling: end of period t is start of period t+1."""
    res = solve_multi_period(n_periods=3, inventory_mode=True)
    assert res.feasible
    for t in range(res.n_periods - 1):
        for k in TANK_KEYS:
            assert res.tank_end[t][k] == pytest.approx(res.tank_start[t + 1][k], abs=1e-6)
            assert res.carries[t][k] == pytest.approx(res.tank_end[t][k], abs=1e-6)
    # Multi-period plant stays multi-unit active (inventory may optimally end at 0)
    res2 = solve_multi_period(n_periods=2, inventory_mode=True)
    assert res2.feasible
    assert res2.period_unit_feeds[0]["cdu_charge"] > 0
    assert res2.period_unit_feeds[0]["fcc_feed"] > 0
    # Coupling still holds even when carry is zero
    for k in TANK_KEYS:
        assert res2.tank_end[0][k] == pytest.approx(res2.tank_start[1][k], abs=1e-6)


def test_pass_mode_zeroes_inventory():
    res = solve_multi_period(n_periods=2, inventory_mode=False)
    assert res.feasible
    assert res.inventory_mode is False
    for t in range(res.n_periods):
        assert sum(res.tank_start[t].values()) == pytest.approx(0.0, abs=1e-6)
        assert sum(res.tank_end[t].values()) == pytest.approx(0.0, abs=1e-6)


def test_tight_later_period_uses_units():
    """With crude cut in later periods, plant still runs (inventory/production mix)."""
    res = solve_multi_period(
        n_periods=2,
        inventory_mode=True,
        crude_scale=[1.0, 0.35],
    )
    assert res.feasible
    assert res.period_unit_feeds[0]["cdu_charge"] > 0
    # period 1 may still charge some crude (scaled) and/or draw inventory
    assert res.period_unit_feeds[1]["cdu_charge"] >= 0
    total_prod = sum(sum(p.values()) for p in res.period_products)
    assert total_prod > 0


def test_string_inventory_mode_multi_period():
    res = solve_multi_period(n_periods=2, inventory_mode="multi_period")
    assert res.feasible and res.inventory_mode is True
    res_off = solve_multi_period(n_periods=2, inventory_mode="pass")
    assert res_off.feasible and res_off.inventory_mode is False
