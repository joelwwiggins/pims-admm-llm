"""Process-network agents: real area inputs, pushback, FCC↔Reformer octane couple."""

from __future__ import annotations

import pytest

from pims_admm_llm.agents.process_network import (
    build_area_situations,
    evaluate_octane_couple,
    run_process_network_round,
)
from pims_admm_llm.models.assay_loader import load_assays_json
from pims_admm_llm.models.full_plant import solve_full_plant


@pytest.fixture(scope="module")
def plant_default():
    return solve_full_plant()


@pytest.fixture(scope="module")
def plant_pool():
    return solve_full_plant(process_pool_modes=True)


def test_area_situations_have_real_feeds(plant_default):
    assays = load_assays_json()
    sit = build_area_situations(plant_default, assays=assays)
    assert set(sit) >= {"CDU", "FCC", "Coker", "Reformer", "Blender"}
    assert sit["CDU"].util_frac["cdu"] > 0.5
    assert sit["FCC"].feeds["fcc_feed"] > 1.0
    assert "gasoline" in sit["Blender"].products


def test_default_plan_coker_idle_pushback(plant_default):
    """Default plan dumps resid to FO → Coker should push back if idled."""
    assays = load_assays_json()
    r = run_process_network_round(plant_default, assays=assays)
    assert r.plant_feasible
    coker = next(a for a in r.areas if a.area == "Coker")
    codes = {p.code for p in coker.pushbacks}
    # Default plant has coker_feed=0 and resid_frac_fo=1
    if plant_default.unit_feeds.get("coker_feed", 0) < 1e-6:
        assert "coker_idled_all_fo" in codes
        assert r.reoptimize_recommended is True


def test_octane_couple_present(plant_default):
    assays = load_assays_json()
    r = run_process_network_round(plant_default, assays=assays)
    assert r.cross_unit_couples
    couple = r.cross_unit_couples[0]
    assert couple["couple_id"] == "fcc_reformer_blender_octane"
    assert "FCC" in couple["areas"] and "Reformer" in couple["areas"]
    # RON dual is active on default plan
    assert abs(float(plant_default.quality_duals.get("qual_gas_min_ron", 0))) > 1e-6
    assert couple["binding"] is True


def test_blender_ron_pushback(plant_default):
    assays = load_assays_json()
    r = run_process_network_round(plant_default, assays=assays)
    blender = next(a for a in r.areas if a.area == "Blender")
    assert any(p.code == "gasoline_ron_binding" for p in blender.pushbacks)
    assert blender.wiggle_room == "none"
    assert any("Reformer" in p.related_areas for p in blender.pushbacks)


def test_fcc_knows_octane_couple_with_reformer(plant_default):
    assays = load_assays_json()
    r = run_process_network_round(plant_default, assays=assays)
    fcc = next(a for a in r.areas if a.area == "FCC")
    blob = " ".join(fcc.cross_unit_notes + fcc.soft_suggestions).lower()
    assert "reformer" in blob or "octane" in blob or "ron" in blob


def test_process_pool_changes_coker_story(plant_pool):
    assays = load_assays_json()
    r = run_process_network_round(plant_pool, assays=assays)
    coker = next(a for a in r.areas if a.area == "Coker")
    # With process-pool, coker usually has feed
    if plant_pool.unit_feeds.get("coker_feed", 0) > 1.0:
        assert coker.inputs.feeds["coker_feed"] > 1.0
        assert not any(p.code == "coker_idled_all_fo" for p in coker.pushbacks)
    assert plant_pool.meta.get("process_pool", {}).get("enabled") is True


def test_master_summary_nonempty(plant_default):
    r = run_process_network_round(plant_default)
    assert "Plant obj" in r.master_summary
    assert len(r.plan_feedback) >= 1
    d = r.to_dict()
    assert d["n_areas"] >= 5
    assert d["n_pushbacks"] >= 1


def test_closed_loop_applies_process_pool_and_improves_or_reduces_pushbacks(
    plant_default,
):
    from pims_admm_llm.agents.process_network import run_closed_loop

    assays = load_assays_json()
    cl = run_closed_loop(plant_default, assays=assays, max_rounds=3)
    assert cl.baseline is not None
    # Default plan has coker idle pushback → closed loop should act
    if plant_default.unit_feeds.get("coker_feed", 0) < 1e-6:
        assert cl.applied is True
        assert cl.n_rounds >= 2
        assert cl.replan is not None
        assert cl.plant_replan is not None
        assert cl.delta.get("coker_feed_best", cl.delta.get("coker_feed_1", 0)) >= (
            cl.delta.get("coker_feed_0", 0)
        )
        assert "CDU" in cl.node_badges or "COKER" in cl.node_badges
        badges = cl.node_badges
        assert any(b.get("status") in ("alarm", "watch", "ok") for b in badges.values())
        assert len(cl.transcript) == cl.n_rounds
        assert cl.transcript[0]["kind"] == "baseline"
        assert cl.stop_reason in (
            "no_hard_pushbacks",
            "no_new_actions",
            "hard_pushbacks_not_improving",
            "hard_pushbacks_worsened",
            "max_rounds",
        )
    d = cl.to_dict()
    assert "baseline" in d and "delta" in d
    assert "n_rounds" in d and "transcript" in d


def test_multi_round_escalates_pool_then_two_pass(plant_default):
    """Escalation ladder: r1 process-pool → r2 two-pass if RON still tight."""
    from pims_admm_llm.agents.process_network import run_closed_loop

    assays = load_assays_json()
    cl = run_closed_loop(plant_default, assays=assays, max_rounds=3)
    if not cl.applied:
        pytest.skip("no replan on this plant")
    codes = [a["code"] for a in cl.actions]
    assert "enable_process_pool" in codes
    # If more than one replan hop, second should be two-pass escalate
    replan_steps = [t for t in cl.transcript if t["kind"] == "replan"]
    assert len(replan_steps) >= 1
    if len(replan_steps) >= 2:
        assert "process_pool_two_pass_for_octane" in codes
        assert replan_steps[1]["solve_kwargs"].get("process_pool_two_pass") is True


def test_max_rounds_one_is_single_replan(plant_default):
    from pims_admm_llm.agents.process_network import run_closed_loop

    assays = load_assays_json()
    cl = run_closed_loop(plant_default, assays=assays, max_rounds=1)
    if plant_default.unit_feeds.get("coker_feed", 0) < 1e-6:
        assert cl.applied
        assert cl.n_rounds == 2  # baseline + 1 replan
        assert cl.max_rounds == 1


def test_node_badges_map_areas():
    from pims_admm_llm.agents.process_network import node_badges_from_round

    plant = solve_full_plant()
    r = run_process_network_round(plant)
    badges = node_badges_from_round(r)
    assert "BLENDER" in badges
    assert badges["BLENDER"]["wiggle_room"] in ("none", "limited", "ample")
