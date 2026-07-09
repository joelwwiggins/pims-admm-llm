"""Recursive multi-level quality v1: tank quality → pool deltas."""

from __future__ import annotations

import copy

import pytest

from pims_admm_llm.models.assay_loader import load_routing
from pims_admm_llm.models.full_plant import solve_full_plant
from pims_admm_llm.models.quality_blender import (
    GASOLINE_COMPONENT_DEFAULTS,
    GasolineQualityConfig,
    QualityComponent,
    blend_quality_closed_form,
    load_component_qualities,
)
from pims_admm_llm.models.quality_recursive import (
    QualityProps,
    TankInflow,
    TransformSpec,
    build_default_gasoline_quality_graph,
    component_overrides_from_recursive,
    compute_tank_quality,
    evaluate_from_plant_result,
    evaluate_recursive_quality,
    patch_routing_component_properties,
    resolve_gasoline_components,
    solve_full_plant_with_recursive_quality,
    successive_recursive_refine,
    volume_weighted_quality,
)


def test_volume_weighted_two_stream():
    q, v = volume_weighted_quality(
        [
            (50.0, QualityProps(80.0, 0.02)),
            (50.0, QualityProps(100.0, 0.00)),
        ]
    )
    assert v == pytest.approx(100.0)
    assert q.ron == pytest.approx(90.0)
    assert q.sulfur_wt == pytest.approx(0.01)


def test_tank_with_heel_dilution():
    sources = {"fcc_naphtha": QualityComponent("fcc_naphtha", 93.0, 0.005)}
    heel = QualityComponent("heel", 88.0, 0.01)
    q = compute_tank_quality(
        [TankInflow("fcc_naphtha", 40.0)],
        sources,
        tank_name="tank_fcc_naph",
        heel_volume=10.0,
        heel_quality=heel,
    )
    assert q.ron == pytest.approx((10 * 88.0 + 40 * 93.0) / 50.0)
    assert q.ron != pytest.approx(93.0)


def test_identity_single_inflow_zero_heel():
    sources = {"fcc_naphtha": QualityComponent("fcc_naphtha", 93.0, 0.005)}
    q = compute_tank_quality(
        [TankInflow("fcc_naphtha", 25.0)],
        sources,
        tank_name="tank_fcc_naph",
        heel_volume=0.0,
    )
    assert q.ron == pytest.approx(93.0)
    assert q.sulfur_wt == pytest.approx(0.005)


def test_hdt_transform_absolute():
    t = TransformSpec(name="soft_hdt", ron=74.0, sulfur_wt=0.008)
    out = t.apply(QualityProps(72.0, 0.25))
    assert out.ron == pytest.approx(74.0)
    assert out.sulfur_wt == pytest.approx(0.008)


def test_recursive_eval_multi_source_tank_updates_deltas():
    routing = load_routing()
    volumes = {
        "fcc_naphtha": 30.0,
        "cdu_naphtha_light": 20.0,
        "reformate": 40.0,
        "coker_naphtha": 0.0,
        "cdu_naphtha_heavy": 0.0,
        "coker_naphtha_hdt": 0.0,
    }
    rec = evaluate_recursive_quality(
        routing,
        volumes,
        multi_source_inflows={
            "tank_fcc_naph": [
                TankInflow("fcc_naphtha", 30.0),
                TankInflow("cdu_naphtha_light", 20.0),
            ]
        },
        blend_volumes={
            "reformate": 40.0,
            "fcc_naphtha": 50.0,
            "cdu_naphtha_light": 0.0,
            "cdu_naphtha_heavy": 0.0,
            "coker_naphtha_hdt": 0.0,
        },
    )
    # Mixed tank: (30*93 + 20*72)/50 = 84.6
    assert rec.node_qualities["tank_fcc_naph"].ron == pytest.approx(84.6)
    assert rec.component_qualities["fcc_naphtha"].ron == pytest.approx(84.6)
    assert rec.deltas["fcc_naphtha"]["delta_ron"] == pytest.approx(84.6 - rec.base_ron)
    assert rec.model == "recursive_multi_level_v1"


def test_recursive_heel_lowers_blend_vs_fixed():
    routing = load_routing()
    volumes = {
        "fcc_naphtha": 40.0,
        "reformate": 30.0,
        "cdu_naphtha_light": 20.0,
        "cdu_naphtha_heavy": 0.0,
        "coker_naphtha": 10.0,
        "heel_fcc_naph": 10.0,
    }
    heel_q = {"tank_fcc_naph": QualityComponent("heel", 85.0, 0.02)}
    rec = evaluate_recursive_quality(
        routing,
        volumes,
        heel_qualities=heel_q,
        blend_volumes={
            "reformate": 30.0,
            "cdu_naphtha_light": 20.0,
            "cdu_naphtha_heavy": 0.0,
            "fcc_naphtha": 50.0,
            "coker_naphtha_hdt": 10.0,
        },
    )
    tank = rec.node_qualities["tank_fcc_naph"]
    assert tank.ron == pytest.approx((10 * 85.0 + 40 * 93.0) / 50.0)
    fixed = load_component_qualities(
        routing,
        list(rec.component_qualities.keys()),
        defaults=GASOLINE_COMPONENT_DEFAULTS,
    )
    fixed_blend = blend_quality_closed_form(
        {
            "reformate": 30.0,
            "cdu_naphtha_light": 20.0,
            "cdu_naphtha_heavy": 0.0,
            "fcc_naphtha": 50.0,
            "coker_naphtha_hdt": 10.0,
        },
        fixed,
        GasolineQualityConfig.from_routing(routing),
    )
    assert rec.blend["ron"] < fixed_blend["ron"] - 0.1


def test_resolve_fixed_assay_default():
    routing = load_routing()
    comps, meta = resolve_gasoline_components(routing, recursive_quality=False)
    assert meta["recursive_quality"] is False
    fixed = load_component_qualities(
        routing, list(comps.keys()), defaults=GASOLINE_COMPONENT_DEFAULTS
    )
    assert comps["fcc_naphtha"].ron == pytest.approx(fixed["fcc_naphtha"].ron)


def test_resolve_recursive_true():
    routing = load_routing()
    comps, meta = resolve_gasoline_components(
        routing,
        recursive_quality=True,
        multi_source_inflows={
            "tank_fcc_naph": [
                ("fcc_naphtha", 30.0),
                ("cdu_naphtha_light", 20.0),
            ]
        },
        volumes={"fcc_naphtha": 30.0, "cdu_naphtha_light": 20.0, "reformate": 1.0},
    )
    assert meta["recursive_quality"] is True
    assert meta["mode"] == "multi_level_volume_weighted"
    assert comps["fcc_naphtha"].ron == pytest.approx(84.6)


def test_successive_refine_patches_routing():
    routing = load_routing()
    volumes = {
        "fcc_naphtha": 40.0,
        "reformate": 40.0,
        "cdu_naphtha_light": 20.0,
        "coker_naphtha": 0.0,
        "heel_fcc_naph": 20.0,
    }
    heel_q = {"tank_fcc_naph": QualityComponent("heel", 80.0, 0.02)}
    patched, rec = successive_recursive_refine(
        routing, volumes, heel_qualities=heel_q
    )
    assert patched["quality_model"]["recursive_quality"] is True
    orig = float(routing["component_properties"]["fcc_naphtha"]["ron"])
    new = float(patched["component_properties"]["fcc_naphtha"]["ron"])
    assert new != pytest.approx(orig)
    assert new == pytest.approx(rec.component_qualities["fcc_naphtha"].ron)
    # original not mutated
    assert float(routing["component_properties"]["fcc_naphtha"]["ron"]) == pytest.approx(
        orig
    )


def test_patch_routing_preserves_other_streams():
    routing = load_routing()
    n_before = len(routing["component_properties"])
    patched = patch_routing_component_properties(
        routing, {"fcc_naphtha": {"ron": 91.0, "sulfur_wt": 0.006}}
    )
    assert len(patched["component_properties"]) == n_before
    assert patched["component_properties"]["reformate"]["ron"] == routing[
        "component_properties"
    ]["reformate"]["ron"]
    assert patched["component_properties"]["fcc_naphtha"]["ron"] == 91.0
    assert float(routing["component_properties"]["fcc_naphtha"]["ron"]) == pytest.approx(
        93.0
    )


def test_default_graph_has_tanks():
    g = build_default_gasoline_quality_graph(load_routing())
    names = set(g.node_names())
    assert "tank_fcc_naph" in names
    assert "coker_naphtha_hdt" in names
    assert "reformate" in names


def test_plant_result_recursive_eval_smoke():
    routing = load_routing()
    res = solve_full_plant()
    assert res.feasible
    rec = evaluate_from_plant_result(res, routing)
    assert rec.base_ron > 0
    assert "fcc_naphtha" in rec.component_qualities
    assert "deltas" in rec.as_dict()


def test_solve_full_plant_with_recursive_quality_flag():
    res0 = solve_full_plant()
    assert res0.feasible
    res1 = solve_full_plant_with_recursive_quality(max_refine_steps=1)
    assert res1.feasible
    qr = res1.meta.get("quality_recursive") or {}
    assert qr.get("enabled") is True
    res2 = solve_full_plant_with_recursive_quality(recursive_quality=False)
    assert (res2.meta.get("quality_recursive") or {}).get("enabled") is False
    # base path unchanged
    assert res0.meta.get("quality", {}).get("model") in ("delta_base", "linear", "index")


def test_component_overrides_helper():
    routing = load_routing()
    rec = evaluate_recursive_quality(
        routing,
        {"fcc_naphtha": 10.0, "reformate": 10.0},
        multi_source_inflows={
            "tank_fcc_naph": [TankInflow("fcc_naphtha", 10.0)]
        },
    )
    ov = component_overrides_from_recursive(rec)
    assert "fcc_naphtha" in ov and "ron" in ov["fcc_naphtha"]
