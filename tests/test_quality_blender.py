"""Delta-base / index quality blender unit + plant integration tests."""

from __future__ import annotations

import copy

import pulp
import pytest

from pims_admm_llm.models.assay_loader import load_routing
from pims_admm_llm.models.full_plant import solve_full_plant
from pims_admm_llm.models.quality_blender import (
    GASOLINE_COMPONENT_DEFAULTS,
    GasolineQualityConfig,
    QualityComponent,
    add_gasoline_quality_constraints,
    blend_quality_closed_form,
    component_deltas,
    load_component_qualities,
    resolve_base,
    ron_blending_index,
    ron_from_blending_index,
)


def test_ron_index_roundtrip():
    for r in (70.0, 87.0, 93.0, 100.0):
        bi = ron_blending_index(r, mode="ethyl")
        assert ron_from_blending_index(bi, mode="ethyl") == pytest.approx(r, abs=1e-9)
        assert ron_blending_index(r, mode="identity") == r


def test_delta_base_closed_form_matches_linear():
    comps = {
        "reformate": QualityComponent("reformate", 100.0, 0.0005),
        "fcc_naphtha": QualityComponent("fcc_naphtha", 93.0, 0.005),
        "cdu_naphtha_light": QualityComponent("cdu_naphtha_light", 72.0, 0.005),
    }
    vols = {"reformate": 40.0, "fcc_naphtha": 40.0, "cdu_naphtha_light": 20.0}
    cfg = GasolineQualityConfig(model="delta_base", base_stream="reformate", min_ron=87.0)
    out = blend_quality_closed_form(vols, comps, cfg)
    # pure volume average RON
    linear = (100 * 40 + 93 * 40 + 72 * 20) / 100.0
    linear_s = (0.0005 * 40 + 0.005 * 40 + 0.005 * 20) / 100.0
    assert out["ron"] == pytest.approx(linear, abs=1e-9)
    assert out["sulfur_wt"] == pytest.approx(linear_s, abs=1e-12)
    assert out["base_stream"] == "reformate"
    deltas = component_deltas(comps, out["base_ron"], out["base_sulfur_wt"])
    assert deltas["reformate"]["delta_ron"] == pytest.approx(0.0)
    assert deltas["fcc_naphtha"]["delta_ron"] == pytest.approx(-7.0)


def test_index_blend_differs_from_linear():
    comps = {
        "reformate": QualityComponent("reformate", 100.0, 0.0005),
        "cdu_naphtha_light": QualityComponent("cdu_naphtha_light", 72.0, 0.005),
    }
    vols = {"reformate": 50.0, "cdu_naphtha_light": 50.0}
    linear = blend_quality_closed_form(
        vols, comps, GasolineQualityConfig(model="delta_base", base_stream="reformate")
    )
    indexed = blend_quality_closed_form(
        vols,
        comps,
        GasolineQualityConfig(model="index", base_stream="reformate", ron_index_mode="ethyl"),
    )
    assert linear["ron"] == pytest.approx(86.0, abs=1e-9)
    # Ethyl-style rational index is nonlinear — not equal to volume average
    assert indexed["ron"] != pytest.approx(linear["ron"], abs=0.05)
    assert indexed["model"] == "index"


def test_pulp_delta_base_constraint_feasible_at_spec():
    """LP with only blend + delta-base RON/S finds a feasible recipe."""
    comps = load_component_qualities(
        load_routing(),
        ["reformate", "fcc_naphtha", "cdu_naphtha_light"],
        defaults=GASOLINE_COMPONENT_DEFAULTS,
    )
    cfg = GasolineQualityConfig(model="delta_base", base_stream="reformate", min_ron=87.0, max_sulfur_wt=0.01)
    prob = pulp.LpProblem("q_mvp", pulp.LpMaximize)
    x = {s: pulp.LpVariable(f"x_{s}", lowBound=0) for s in comps}
    gas = pulp.LpVariable("gas", lowBound=0)
    prob += gas == pulp.lpSum(x.values()), "pool"
    # maximize low-RON component to stress the constraint
    prob += x["cdu_naphtha_light"] + 0.5 * x["fcc_naphtha"] + 0.1 * x["reformate"]
    # bound total
    prob += gas <= 100
    meta = add_gasoline_quality_constraints(
        prob, product_var=gas, volume_vars=x, components=comps, cfg=cfg
    )
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[status] == "Optimal"
    vols = {s: float(pulp.value(x[s]) or 0) for s in x}
    blend = blend_quality_closed_form(vols, comps, cfg)
    assert blend["ron"] + 1e-6 >= cfg.min_ron
    assert blend["sulfur_wt"] <= cfg.max_sulfur_wt + 1e-9
    assert meta.model == "delta_base"
    assert meta.base_stream == "reformate"


def test_routing_quality_model_config():
    r = load_routing()
    assert "quality_model" in r
    cfg = GasolineQualityConfig.from_routing(r)
    assert cfg.model.replace("-", "_") in ("delta_base", "linear", "index")
    assert cfg.base_stream == "reformate"
    assert cfg.min_ron == pytest.approx(87.0)


def test_full_plant_uses_delta_base_meta():
    res = solve_full_plant()
    assert res.feasible
    q = res.meta.get("quality") or {}
    assert q.get("model") in ("delta_base", "linear", "index")
    assert q.get("base_stream") == "reformate"
    assert "deltas" in q and "reformate" in q["deltas"]
    assert "qual_gas_min_ron" in res.quality_duals
    assert "qual_gas_max_s" in res.quality_duals
    # still meets product volume
    assert res.products.get("gasoline", 0) > 0


def test_full_plant_index_mode_solves():
    routing = load_routing()
    routing = copy.deepcopy(routing)
    routing.setdefault("quality_model", {}).setdefault("gasoline", {})["model"] = "index"
    routing["quality_model"]["gasoline"]["ron_index_mode"] = "ethyl"
    routing["product_quality_specs"]["gasoline"]["model"] = "index"
    res = solve_full_plant(routing=routing)
    assert res.feasible
    assert res.meta["quality"]["model"] == "index"
    assert res.meta["quality"]["ron_index_mode"] == "ethyl"
    assert res.products.get("gasoline", 0) > 0


def test_resolve_base_fallback():
    comps = {
        "fcc_naphtha": QualityComponent("fcc_naphtha", 93.0, 0.005),
        "cdu_naphtha_light": QualityComponent("cdu_naphtha_light", 72.0, 0.005),
    }
    cfg = GasolineQualityConfig(base_stream="reformate")  # missing
    label, br, bs = resolve_base(comps, cfg)
    # falls back to highest RON present
    assert label == "fcc_naphtha"
    assert br == 93.0
