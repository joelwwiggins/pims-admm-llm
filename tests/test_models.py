"""Worker 2: toy refinery block-angular LP models + data loaders."""

from __future__ import annotations


def test_load_crude_data_path_fixed():
    from pims_admm_llm.models import default_data_path, load_crude_data, validate_refinery_data

    path = default_data_path()
    assert path.is_file(), f"missing data at {path}"
    data = load_crude_data()
    assert len(data.crudes) == 3
    assert data.cdu_capacity_kbd == 120.0
    assert set(data.intermediates) == {"naphtha", "distillate", "gasoil", "residue"}
    assert "gasoline" in data.products
    assert "naphtha" in data.inventory
    assert "fuel_gas" in data.utilities
    # utility intensities present
    assert data.crudes[0].utility_use.get("fuel_gas", 0) > 0
    issues = validate_refinery_data(data)
    assert not issues, issues


def test_block_angular_structure():
    from pims_admm_llm.models import (
        BlockNames,
        describe_block_angular_structure,
        load_crude_data,
    )

    data = load_crude_data()
    desc = describe_block_angular_structure(data)
    assert desc["blocks"] == BlockNames.all()
    assert len(desc["linking_constraints"]["inventory_balance"]) == 4
    assert len(desc["linking_constraints"]["utility_balance"]) == 3


def test_monolithic_full_model_optimal():
    from pims_admm_llm.models import load_crude_data, solve_monolithic

    data = load_crude_data()
    res = solve_monolithic(data, msg=False)
    assert res.status == "Optimal"
    assert res.objective > 0
    # some crude charged
    assert sum(res.crude_rates.values()) > 0
    # products made
    assert sum(res.product_rates.values()) > 0
    # inventory and utilities populated
    assert res.inventory_end
    assert res.utility_supply
    # duals present for inventory or balance
    dual_keys = list(res.duals.keys())
    assert any(k.startswith("inv_balance_") or k.startswith("balance_") for k in dual_keys)
    assert any(
            k.startswith("util_balance_") or k.startswith("utility_cap_")
            for k in dual_keys
        )


def test_monolithic_classic_two_block():
    """Classic CDU+Blender form (no tanks/utilities) for ADMM dual comparison."""
    from pims_admm_llm.models import load_crude_data, solve_monolithic

    data = load_crude_data()
    res = solve_monolithic(
        data, msg=False, include_inventory=False, include_utilities=False
    )
    assert res.status == "Optimal"
    assert any(k.startswith("balance_") for k in res.duals)


def test_pulp_subproblems_solve():
    from pims_admm_llm.models import load_crude_data, solve_all_subproblems

    data = load_crude_data()
    # mid-range intermediate prices encourage production + blending
    prices = {i: 70.0 for i in data.intermediates}
    util_prices = {u: data.utilities[u].cost_usd_per_unit for u in data.utility_names}
    results = solve_all_subproblems(
        data, intermediate_prices=prices, utility_prices=util_prices, rho=1.0
    )
    assert set(results.keys()) == {"CDU", "Inventory", "Blender", "Utilities"}
    for name, r in results.items():
        assert r.status == "Optimal", f"{name}: {r.status} {r.message}"
        assert r.solve_time_s >= 0


def test_cdu_yields_material_balance():
    from pims_admm_llm.models import load_crude_data, solve_cdu_subproblem

    data = load_crude_data()
    prices = {i: 80.0 for i in data.intermediates}
    r = solve_cdu_subproblem(data, intermediate_prices=prices)
    assert r.status == "Optimal"
    total_crude = sum(v for k, v in r.primals.items() if k.startswith("crude_"))
    total_prod = sum(r.linking_intermediates.values())
    # yields sum ~1 so prod ≈ crude
    assert abs(total_prod - total_crude) < 1e-4


def test_inventory_capacity_respected():
    from pims_admm_llm.models import load_crude_data, solve_inventory_subproblem

    data = load_crude_data()
    # force large inflow consensus; end inv must stay within capacity
    big_in = {i: 1000.0 for i in data.intermediates}
    r = solve_inventory_subproblem(
        data,
        intermediate_prices={i: 50.0 for i in data.intermediates},
        consensus_in=big_in,
        consensus_out={i: 0.0 for i in data.intermediates},
        rho=0.0,
    )
    assert r.status == "Optimal"
    for i in data.intermediates:
        end = r.primals.get(f"inv_end_{i}", 0.0)
        assert end <= data.inventory[i].capacity_kbd + 1e-6
