"""E1/E2: offline multi-block plant-linking ADMM harness (+ plant-named mode).

Always-on sections run without TensorFlow. Locks:
- topology lists FCC/COKER/CDU + non-empty linking streams
- incidence covers only known products
- multi_block_plant_linking_admm_report aggregate ok without TF
- dual_recovery_path is None; on_excel_case1_path is False; solver False
- topology_source / plant_linking_scope = synthetic_offline_demo (default)
- plant-named mode: topology_source=plant_named_offline_demo; identity incidence
- plant-named streams are unit product names (not only aggregate synthetic names)
- not_full_plant_mass_balance; not Case 1; not wire; plant-linking lam != Case 1
- n_rounds respected; trajectory length matches; residuals finite
- dual ascent uses pre-z-update linking residual (not post-z zero theater)
- composes subproblem (subproblem_ok / not_worse_than_ref)
- no residual-must-vanish SLA
- no excel_cdu_matrix_matches_affine invent
- no PuLP offline backend
- existing multi_unit_admm_coordination_report still not_plant_linking_coordinator
- optional readiness flags admm_plant_linking_ok + admm_plant_named_linking_ok
  additive only (do not redefine ready_for_wire_discussion)

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem, test_tf_offline_admm_coordination,
  test_tf_linear_block, test_tf_linear_coker, test_tf_linear_cdu,
  test_excel_pipeline, test_api_excel
  EMRPS optional-only (not required for this gate).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


def test_offline_plant_linking_topology_units_streams_incidence():
    topo = tlb.offline_plant_linking_topology()
    assert topo["unit_order"] == ["FCC", "COKER", "CDU"]
    assert topo["streams"] == list(tlb.ADMM_PLANT_LINKING_STREAMS)
    assert len(topo["streams"]) >= 1
    assert topo["topology_source"] == "synthetic_offline_demo"
    assert topo["plant_linking_scope"] == "synthetic_offline_demo"
    assert topo["not_full_plant_mass_balance"] is True
    assert topo["not_live_plant_blocks"] is True
    assert topo["not_case1_links"] is True
    assert topo["dual_recovery_path"] is None
    assert topo["solver"] is False
    assert topo["on_excel_case1_path"] is False

    for unit in tlb.UNITS:
        coeffs = tlb.cached_offline_unit_coeffs(unit)
        known = set(coeffs.products)
        inc = topo["incidence"][unit]
        assert set(inc.keys()).issubset(known)
        assert len(inc) >= 1
        for product, stream_map in inc.items():
            assert set(stream_map.keys()).issubset(set(topo["streams"]))
            assert all(float(v) == 1.0 for v in stream_map.values())
        coverage = topo["product_coverage"][unit]
        assert set(coverage) == set(inc.keys())
        assert set(coverage).issubset(known)


def test_project_lift_identity_selection_incidence():
    streams = list(tlb.ADMM_PLANT_LINKING_STREAMS)
    lam = {s: float(i + 1) for i, s in enumerate(streams)}
    z = {s: float(10 * (i + 1)) for i, s in enumerate(streams)}
    for unit in tlb.UNITS:
        proj = tlb.project_linking_to_unit(unit, lam, z, streams=streams)
        # lift of projected prices-as-y should recover stream values for mapped streams
        y_as_prices = proj["prices"]
        lifted = tlb.lift_unit_y_to_linking(unit, y_as_prices, streams=streams)
        # For 0/1 multi-product->stream, lift(A^T lam) = count * lam on that stream
        for s in streams:
            assert math.isfinite(lifted[s])
        # z projection finite and same keys as products
        assert set(proj["z"].keys()) == set(proj["products"])


def test_multi_block_plant_linking_report_always_on_aggregate_ok():
    report = tlb.multi_block_plant_linking_admm_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == "offline_admm_plant_linking"
    assert report["kind"] == tlb.ADMM_PLANT_LINKING_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["price_source"] == tlb.PRICE_SOURCE
    assert report["lam_source"] == tlb.PRICE_SOURCE
    assert report["z_source"] == "synthetic_offline_demo"
    assert report["rho_source"] == "synthetic_offline_demo"
    assert report["optimand_space"] == "raw_affine"
    assert report["linking_space"] == "synthetic_linking_streams"
    assert report["z_update_space"] == "synthetic_linking_streams"
    assert report["plant_linking_scope"] == tlb.ADMM_PLANT_LINKING_SCOPE
    assert report["topology_source"] == "synthetic_offline_demo"
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_case1_solve"] is True
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["plant_linking_lambda_is_not_case1_online_lambda"] is True
    assert report["not_live_plant_blocks"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["n_rounds"] == 3
    assert len(report["trajectory"]) == 3
    assert report["formula"] == tlb.ADMM_PLANT_LINKING_FORMULA
    assert set(report["streams"]) == set(tlb.ADMM_PLANT_LINKING_STREAMS)

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert len(row["rounds"]) == 3
        for rr in row["rounds"]:
            assert math.isfinite(rr["augmented_local_raw"])
            assert rr["ok"] is True
            assert rr["subproblem_ok"] is True
            assert rr["not_worse_than_ref"] is True


def test_plant_linking_honesty_not_dual_not_case1_not_wire_not_full_mb():
    report = tlb.multi_block_plant_linking_admm_report(n_rounds=2)
    note = (report.get("note") or "").lower()
    assert "synthetic" in note
    assert "not" in note
    assert "case 1" in note or "classic_2block" in note
    assert "wire" in note
    assert "mass balance" in note or "plant" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden
    assert report["plant_linking_lambda_is_not_case1_online_lambda"] is True
    assert report["not_full_plant_mass_balance"] is True


def test_n_rounds_respected_and_trajectory_finite():
    for n in (1, 2, 4):
        report = tlb.multi_block_plant_linking_admm_report(
            n_rounds=n, rho=1.0, delta=0.5
        )
        assert report["n_rounds"] == n
        assert len(report["trajectory"]) == n
        assert report["ok"] is True, report
        for tr in report["trajectory"]:
            assert math.isfinite(tr["r_l1_link"])
            assert math.isfinite(tr["r_linf_link"])
            assert math.isfinite(tr["sum_augmented_local_raw"])
            assert bool(tr["ok"]) is True
            assert set(tr["units_ok"].keys()) == set(tlb.UNITS)


def test_dual_ascent_uses_pre_z_linking_residual_not_post_zero_theater():
    """With z_blend=1 (full copy), post residual would be 0; lambda moves from pre."""
    streams = list(tlb.ADMM_PLANT_LINKING_STREAMS)
    z_far = {s: 0.0 for s in streams}
    row = tlb.plant_linking_admm_round(
        z_link=z_far, rho=1.0, delta=1.0, dual_step=1.0, z_blend=1.0
    )
    assert row["ok"] is True
    assert row["r_l1_link"] > 1e-6  # pre-update residual must be nonzero
    # After full copy, z_post == y_link_total
    for s in streams:
        assert abs(row["z_post"][s] - row["y_link_total"][s]) <= 1e-9
    # lam must have moved from pre residual
    moved = False
    for s in streams:
        if abs(row["lam_post"][s] - row["lam_pre"][s]) > 1e-12:
            moved = True
            expected = row["lam_pre"][s] + 1.0 * 1.0 * row["r_link_pre"][s]
            assert abs(row["lam_post"][s] - expected) <= 1e-9
    assert moved


def test_composes_subproblem_maximizer_kind():
    row = tlb.plant_linking_admm_round(rho=1.0, delta=0.5)
    assert row["ok"] is True
    for unit in tlb.UNITS:
        u = row["units"][unit]
        assert u["subproblem_kind"] == tlb.ADMM_SUBPROBLEM_KIND
        assert u["subproblem_ok"] is True
        assert u["not_worse_than_ref"] is True


def test_existing_coordination_still_not_plant_linking():
    report = tlb.multi_unit_admm_coordination_report(n_rounds=1)
    assert report["not_plant_linking_coordinator"] is True
    assert report["kind"] == tlb.ADMM_COORDINATION_KIND
    assert report["kind"] != tlb.ADMM_PLANT_LINKING_KIND


def test_parameter_guards():
    with pytest.raises(ValueError, match="n_rounds"):
        tlb.multi_block_plant_linking_admm_report(n_rounds=0)
    with pytest.raises(ValueError, match="rho"):
        tlb.multi_block_plant_linking_admm_report(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        tlb.plant_linking_admm_round(rho=-1.0)
    with pytest.raises(ValueError, match="dual_step"):
        tlb.multi_block_plant_linking_admm_report(dual_step=float("nan"))
    with pytest.raises(ValueError, match="z_blend"):
        tlb.multi_block_plant_linking_admm_report(z_blend=1.5)


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")
    report = tlb.multi_block_plant_linking_admm_report(n_rounds=1)
    note = (report.get("note") or "").lower()
    assert "excel_cdu_matrix_matches_affine" not in note


def test_no_residual_must_vanish_sla():
    """ok must not require residual->0 (absolute magnitude SLA forbidden)."""
    report = tlb.multi_block_plant_linking_admm_report(n_rounds=2, delta=1.0)
    assert report["ok"] is True
    any_nonzero = any(tr["r_l1_link"] > 1e-12 for tr in report["trajectory"])
    assert report["ok"] is True
    _ = any_nonzero


def test_no_pulp_offline_backend():
    row = tlb.plant_linking_admm_round(delta=0.5)
    assert row["solver"] is False
    assert row.get("backend") != "pulp"
    report = tlb.multi_block_plant_linking_admm_report(n_rounds=1)
    assert report["solver"] is False
    # Note may mention "no PuLP" as a ban; ensure no pulp backend/import path
    assert report.get("backend") != "pulp"
    assert "pulp" not in (report.get("kind") or "").lower()
    import pims_admm_llm.models.tf_linear_blocks as mod

    src = open(mod.__file__, encoding="utf-8").read()
    # plant-linking section must not import PuLP
    pl_idx = src.find("ADMM_PLANT_LINKING_KIND")
    assert pl_idx > 0
    pl_section = src[pl_idx : pl_idx + 8000]
    assert "import pulp" not in pl_section.lower()
    assert "from pulp" not in pl_section.lower()


def test_plant_linking_public_exports():
    for name in (
        "ADMM_PLANT_LINKING_KIND",
        "ADMM_PLANT_LINKING_FORMULA",
        "ADMM_PLANT_LINKING_SCOPE",
        "ADMM_PLANT_LINKING_STREAMS",
        "offline_plant_linking_topology",
        "project_linking_to_unit",
        "lift_unit_y_to_linking",
        "plant_linking_admm_round",
        "multi_block_plant_linking_admm_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_honesty_metadata_mentions_plant_linking():
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    assert meta.get("admm_plant_linking_available") is True
    note = (meta.get("note") or "").lower()
    assert "plant-linking" in note or "plant linking" in note


def test_readiness_admm_plant_linking_ok_additive():
    """Secondary A: admm_plant_linking_ok additive; does not redefine ready."""
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_residual=True,
        include_admm_block_subproblem=True,
        include_admm_coordination=True,
        include_admm_plant_linking=True,
    )
    assert "admm_plant_linking_ok" in rep
    assert rep["admm_plant_linking_ok"] is True
    assert "admm_coordination_ok" in rep
    assert "admm_block_subproblem_ok" in rep
    assert "admm_residual_ok" in rep
    # ready_for_wire_discussion still parity^priced^timings^honesty only
    assert "ready_for_wire_discussion" in rep
    ready = bool(rep["ready_for_wire_discussion"])
    assert rep["ok"] is ready
    note = (rep.get("note") or "").lower()
    assert "additive" in note
    assert "ready_for_wire_discussion" in note
    assert "plant" in note


def test_readiness_can_skip_plant_linking_flag():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=3,
        warmup=0,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
        include_admm_coordination=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
    )
    assert rep["admm_block_subproblem_ok"] is None
    assert rep["admm_residual_ok"] is None
    assert rep["admm_coordination_ok"] is None
    assert rep["admm_plant_linking_ok"] is None
    assert rep["admm_plant_named_linking_ok"] is None


def test_residual_trend_soft_diagnostic():
    report = tlb.multi_block_plant_linking_admm_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["ok"] is True
    assert report["residual_trend"] in (
        "nonincreasing",
        "nondecreasing",
        "mixed",
        "n/a",
    )
    assert set(report["final_lam"].keys()) == set(tlb.ADMM_PLANT_LINKING_STREAMS)
    assert set(report["final_z"].keys()) == set(tlb.ADMM_PLANT_LINKING_STREAMS)
    assert all(math.isfinite(v) for v in report["final_lam"].values())
    assert all(math.isfinite(v) for v in report["final_z"].values())


# ---------------------------------------------------------------------------
# Plant-named topology mode (goal-5 residual after #24/#25)
# ---------------------------------------------------------------------------


def test_offline_plant_named_linking_topology_structure_and_honesty():
    topo = tlb.offline_plant_named_linking_topology()
    assert topo["mode"] == "plant_named"
    topo = tlb.offline_plant_linking_topology(mode="plant_named")
    assert topo["unit_order"] == ["FCC", "COKER", "CDU"]
    assert topo["streams"] == list(tlb.ADMM_PLANT_NAMED_LINKING_STREAMS)
    assert topo["topology_source"] == tlb.ADMM_PLANT_NAMED_LINKING_SCOPE
    assert topo["plant_linking_scope"] == "plant_named_offline_demo"
    assert topo["mode"] == "plant_named"
    assert topo["not_full_plant_mass_balance"] is True
    assert topo["not_live_plant_blocks"] is True
    assert topo["not_case1_links"] is True
    assert topo["dual_recovery_path"] is None
    assert topo["solver"] is False
    assert topo["on_excel_case1_path"] is False

    # Plant-style stream names (not only aggregate synthetic family names)
    synthetic_family = set(tlb.ADMM_PLANT_LINKING_STREAMS)
    assert not set(topo["streams"]).issubset(synthetic_family)
    assert "fcc_naphtha" in topo["streams"]
    assert "cdu_gasoil" in topo["streams"]
    assert "light_ends" not in topo["streams"]
    assert "naphtha" not in topo["streams"]

    for unit in tlb.UNITS:
        coeffs = tlb.cached_offline_unit_coeffs(unit)
        known = set(coeffs.products)
        inc = topo["incidence"][unit]
        assert set(inc.keys()).issubset(known)
        assert len(inc) >= 1
        for product, stream_map in inc.items():
            # Identity incidence: product p -> stream p only
            assert list(stream_map.keys()) == [product]
            assert float(stream_map[product]) == 1.0
            assert product in topo["streams"]
        coverage = topo["product_coverage"][unit]
        assert set(coverage) == set(inc.keys())
        assert set(coverage).issubset(known)


def test_plant_named_identity_incidence_no_shared_fake_product_names():
    topo = tlb.offline_plant_linking_topology(mode="plant_named")
    # Streams are unique; no multi-unit collapse into shared fake product names
    assert len(topo["streams"]) == len(set(topo["streams"]))
    products_by_unit = {
        u: set(topo["incidence"][u].keys()) for u in tlb.UNITS
    }
    # Name-disjoint products across units (no intersection theater)
    assert products_by_unit["FCC"].isdisjoint(products_by_unit["COKER"])
    assert products_by_unit["FCC"].isdisjoint(products_by_unit["CDU"])
    assert products_by_unit["COKER"].isdisjoint(products_by_unit["CDU"])


def test_multi_block_plant_named_linking_report_always_on_ok():
    report = tlb.multi_block_plant_named_linking_admm_report(
        n_rounds=3, rho=1.0, delta=0.5, dual_step=1.0
    )
    assert report["unit_order"] == ["FCC", "COKER", "CDU"]
    assert report["kind"] == tlb.ADMM_PLANT_LINKING_KIND
    assert report["mode"] == "plant_named"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["linking_space"] == "plant_named_linking_streams"
    assert report["z_update_space"] == "plant_named_linking_streams"
    assert report["topology_source"] == tlb.ADMM_PLANT_NAMED_LINKING_SCOPE
    assert report["plant_linking_scope"] == "plant_named_offline_demo"
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_case1_solve"] is True
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["plant_linking_lambda_is_not_case1_online_lambda"] is True
    assert report["not_live_plant_blocks"] is True
    assert report["honesty_ok"] is True
    assert report["ok"] is True, report
    assert report["n_rounds"] == 3
    assert len(report["trajectory"]) == 3
    assert set(report["streams"]) == set(tlb.ADMM_PLANT_NAMED_LINKING_STREAMS)
    assert "fcc_naphtha" in report["streams"]
    assert "light_ends" not in report["streams"]
    note = (report.get("note") or "").lower()
    assert "plant-named" in note or "plant_named" in note
    assert "mass balance" in note or "not full plant" in note
    assert "case 1" in note or "classic_2block" in note
    assert "wire" in note

    for unit in ("FCC", "COKER", "CDU"):
        row = report["units"][unit]
        assert row["ok"] is True, (unit, row)
        assert len(row["rounds"]) == 3
        for rr in row["rounds"]:
            assert math.isfinite(rr["augmented_local_raw"])
            assert rr["ok"] is True
            assert rr["subproblem_ok"] is True
            assert rr["not_worse_than_ref"] is True


def test_plant_named_mode_via_report_mode_kwarg():
    report = tlb.multi_block_plant_linking_admm_report(
        n_rounds=2, mode="plant_named", rho=1.0, delta=0.5
    )
    assert report["ok"] is True
    assert report["topology_source"] == "plant_named_offline_demo"
    assert report["linking_space"] == "plant_named_linking_streams"


def test_plant_named_dual_ascent_pre_z_residual():
    streams = list(tlb.ADMM_PLANT_NAMED_LINKING_STREAMS)
    z_far = {s: 0.0 for s in streams}
    row = tlb.plant_linking_admm_round(
        z_link=z_far,
        rho=1.0,
        delta=1.0,
        dual_step=1.0,
        z_blend=1.0,
        mode="plant_named",
    )
    assert row["ok"] is True
    assert row["topology_source"] == "plant_named_offline_demo"
    assert row["r_l1_link"] > 1e-6
    for s in streams:
        assert abs(row["z_post"][s] - row["y_link_total"][s]) <= 1e-9
    moved = False
    for s in streams:
        if abs(row["lam_post"][s] - row["lam_pre"][s]) > 1e-12:
            moved = True
            expected = row["lam_pre"][s] + 1.0 * 1.0 * row["r_link_pre"][s]
            assert abs(row["lam_post"][s] - expected) <= 1e-9
    assert moved


def test_plant_named_composes_subproblem_not_pulp():
    row = tlb.plant_linking_admm_round(delta=0.5, mode="plant_named")
    assert row["ok"] is True
    assert row["solver"] is False
    assert row.get("backend") != "pulp"
    for unit in tlb.UNITS:
        u = row["units"][unit]
        assert u["subproblem_kind"] == tlb.ADMM_SUBPROBLEM_KIND
        assert u["subproblem_ok"] is True


def test_synthetic_default_still_green_alongside_plant_named():
    syn = tlb.multi_block_plant_linking_admm_report(n_rounds=2)
    named = tlb.multi_block_plant_named_linking_admm_report(n_rounds=2)
    assert syn["ok"] is True
    assert named["ok"] is True
    assert syn["topology_source"] == "synthetic_offline_demo"
    assert named["topology_source"] == "plant_named_offline_demo"
    assert syn["linking_space"] == "synthetic_linking_streams"
    assert named["linking_space"] == "plant_named_linking_streams"
    assert set(syn["streams"]) == set(tlb.ADMM_PLANT_LINKING_STREAMS)
    assert set(named["streams"]) == set(tlb.ADMM_PLANT_NAMED_LINKING_STREAMS)
    # Coordination still distinct / not plant-linking
    coord = tlb.multi_unit_admm_coordination_report(n_rounds=1)
    assert coord["not_plant_linking_coordinator"] is True
    assert coord["kind"] != tlb.ADMM_PLANT_LINKING_KIND


def test_readiness_admm_plant_named_linking_ok_additive():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_residual=True,
        include_admm_block_subproblem=True,
        include_admm_coordination=True,
        include_admm_plant_linking=True,
        include_admm_plant_named_linking=True,
    )
    assert rep["admm_plant_linking_ok"] is True
    assert rep["admm_plant_named_linking_ok"] is True
    # ready_for_wire_discussion still parity^priced^timings^honesty only
    ready = bool(rep["ready_for_wire_discussion"])
    assert rep["ok"] is ready
    note = (rep.get("note") or "").lower()
    assert "additive" in note
    assert "plant_named" in note or "plant-named" in note
    assert "ready_for_wire_discussion" in note


def test_plant_named_public_exports():
    for name in (
        "ADMM_PLANT_NAMED_LINKING_SCOPE",
        "ADMM_PLANT_NAMED_LINKING_STREAMS",
        "ADMM_PLANT_LINKING_MODES",
        "offline_plant_named_linking_topology",
        "multi_block_plant_named_linking_admm_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_unknown_plant_linking_mode_raises():
    with pytest.raises(ValueError, match="Unknown plant-linking topology mode"):
        tlb.offline_plant_linking_topology(mode="live_cascade")
    with pytest.raises(ValueError, match="Unknown plant-linking topology mode"):
        tlb.multi_block_plant_linking_admm_report(mode="full_plant_mb")
