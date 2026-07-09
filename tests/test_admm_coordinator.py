"""Worker 3: ADMM coordinator core tests vs monolithic duals."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.admm import ADMMConfig, ADMMCoordinator, run_admm, solve_simple_monolithic
from pims_admm_llm.models import load_crude_data


@pytest.fixture(scope="module")
def data():
    return load_crude_data()


@pytest.fixture(scope="module")
def mono(data):
    return solve_simple_monolithic(data)


@pytest.fixture(scope="module")
def admm_result(data):
    cfg = ADMMConfig(
        backend="qp_l2",
        rho=3.0,
        max_iter=80,
        dual_step=0.5,
        stable_crude_iters=15,
        recover_primal=True,
        verbose=False,
    )
    return run_admm(data, cfg)


def test_admm_converges(admm_result):
    assert admm_result.converged or admm_result.status in ("converged", "max_iter")
    assert admm_result.iterations >= 1
    assert admm_result.recovered is True
    assert sum(admm_result.crude_rates.values()) > 1.0


def test_objective_matches_monolithic(admm_result, mono):
    assert admm_result.objective == pytest.approx(mono.objective, rel=1e-6, abs=1e-4)


def test_crude_slate_matches_monolithic(admm_result, mono, data):
    for c in data.crudes:
        assert admm_result.crude_rates.get(c.name, 0.0) == pytest.approx(
            mono.crude_rates.get(c.name, 0.0), abs=1e-3
        )


def test_products_match_monolithic(admm_result, mono, data):
    for name in data.products:
        assert admm_result.product_rates.get(name, 0.0) == pytest.approx(
            mono.product_rates.get(name, 0.0), abs=1e-3
        )


def test_duals_match_monolithic_balance(admm_result, mono, data):
    """Recovered duals must match monolithic balance duals (maximize-form)."""
    for n in data.intermediates:
        admm_d = admm_result.duals_like_monolithic[f"balance_{n}"]
        mono_d = mono.duals[f"balance_{n}"]
        assert admm_d == pytest.approx(mono_d, abs=1e-3), (n, admm_d, mono_d)


def test_economic_shadow_prices_positive_for_valuable_streams(admm_result):
    for n in ("naphtha", "distillate", "gasoil"):
        assert admm_result.economic_shadow_prices[n] >= -1e-6
        assert admm_result.economic_shadow_prices[n] > 1.0


def test_online_duals_present(admm_result, data):
    for n in data.intermediates:
        assert n in admm_result.online_duals
        assert math.isfinite(admm_result.online_duals[n])


def test_coordinator_step_history(data):
    cfg = ADMMConfig(max_iter=5, recover_primal=True, stable_crude_iters=100, verbose=False)
    coord = ADMMCoordinator(data, cfg)
    result = coord.run()
    assert len(result.history) == result.iterations
    assert result.history[0].iteration == 0


def test_pulp_l1_backend_runs(data):
    cfg = ADMMConfig(
        backend="pulp_l1",
        rho=1.0,
        max_iter=5,
        recover_primal=False,
        stable_crude_iters=100,
        verbose=False,
    )
    res = run_admm(data, cfg)
    assert res.iterations == 5
    assert res.backend == "pulp_l1"
    assert set(res.shadow_prices) == set(data.intermediates)
