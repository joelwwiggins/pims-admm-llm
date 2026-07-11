"""Wave4: every unit yield stream routes somewhere; poolers; process conditions."""

from __future__ import annotations

from pims_admm_llm.models.assay_loader import load_assays_json, load_routing
from pims_admm_llm.models.full_plant import build_yield_tables, solve_full_plant
from pims_admm_llm.models.unit_specs import (
    UNIT_YIELD_STREAMS,
    default_process_conditions,
    unit_catalog,
    validate_yields_cover_catalog,
)
from pims_admm_llm.models.yields import fcc_yields, coker_yields, reformer_yields
from pims_admm_llm.models.properties import FeedProperties


def test_unit_catalog_fcc_has_full_slate():
    cat = unit_catalog()
    fcc_streams = [r["stream"] for r in cat["units"]["FCC"]["yield_streams"]]
    for s in ("fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry", "fcc_coke"):
        assert s in fcc_streams
    pc = default_process_conditions("FCC")
    assert "riser_outlet_temp_f" in pc
    assert "catalyst_to_oil" in pc


def test_routing_has_arcs_for_each_catalog_stream():
    routing = load_routing()
    arc_streams = {a["stream"] for a in routing.get("arcs") or []}
    # Streams produced by conversion units must appear on ≥1 arc
    required = set()
    for unit in ("FCC", "COKER", "REFORMER"):
        for row in UNIT_YIELD_STREAMS[unit]:
            required.add(row["stream"])
    missing = sorted(s for s in required if s not in arc_streams)
    assert not missing, f"streams without routing arcs: {missing}"


def test_feed_poolers_and_direct_bypass_arcs():
    routing = load_routing()
    by_id = {a["id"]: a for a in routing.get("arcs") or []}
    # Pool + direct for FCC and coker
    assert "go_to_fcc" in by_id
    assert "go_direct_to_fcc" in by_id
    assert "resid_to_coker" in by_id
    assert "resid_direct_to_coker" in by_id
    assert "sr_heavy_to_ref_pool" in by_id
    assert "ref_pool_to_reformer" in by_id
    assert routing.get("feed_poolers")
    assert "process_conditions" in routing
    assert "FCC" in routing["process_conditions"]


def test_yield_tables_cover_catalog_and_process_conditions():
    assays = load_assays_json()
    routing = load_routing()
    y = build_yield_tables(assays, routing=routing)
    assert not y.get("catalog_gaps"), y.get("catalog_gaps")
    assert set(y["fcc"]) >= {
        "fcc_dry_gas",
        "fcc_lpg",
        "fcc_naphtha",
        "fcc_lco",
        "fcc_slurry",
        "fcc_coke",
    }
    assert "process_conditions" in y
    assert y["process_conditions"]["FCC"]["riser_outlet_temp_f"] == routing["process_conditions"]["FCC"]["riser_outlet_temp_f"]


def test_process_conditions_shift_fcc_severity():
    feed = FeedProperties(
        name="go",
        api=25.0,
        sulfur_wt=0.5,
        ccr_wt=1.0,
        nitrogen_ppm=800,
        paraffins_vol=0.3,
        naphthenes_vol=0.35,
        aromatics_vol=0.35,
    )
    mild = fcc_yields(feed, {"riser_outlet_temp_f": 940.0, "catalyst_to_oil": 5.0})
    severe = fcc_yields(feed, {"riser_outlet_temp_f": 1020.0, "catalyst_to_oil": 8.0})
    assert severe["fcc_naphtha"] >= mild["fcc_naphtha"] - 1e-9
    assert severe["fcc_dry_gas"] + severe["fcc_lpg"] >= mild["fcc_dry_gas"] + mild["fcc_lpg"] - 1e-9


def test_full_plant_routes_all_fcc_product_streams():
    import copy

    # Multi-unit slate: lower FO so resid→coker is optimal (default FO idles coker)
    assays = load_assays_json()
    assays = copy.deepcopy(assays)
    assays["products"]["fuel_oil"]["price_usd_per_bbl"] = 50.0
    res = solve_full_plant(assays)
    assert res.feasible
    assert res.status == "Optimal"
    # Production present
    for s in ("fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry", "fcc_coke"):
        assert res.streams.get(s, 0) > 0, s
    for s in ("coker_dry_gas", "coker_lpg", "coker_naphtha", "coker_gasoil", "coker_coke"):
        assert res.streams.get(s, 0) > 0, s
    for s in ("reformate", "reformer_h2", "reformer_lights", "cdu_offgas"):
        assert res.streams.get(s, 0) > 0, s
    # Routed (not all disposed)
    assert res.arc_flows.get("fcc_dry_gas_to_fuel", 0) > 0
    assert res.arc_flows.get("fcc_lpg_to_lpg", 0) > 0 or res.products.get("lpg", 0) > 0
    assert res.arc_flows.get("fcc_coke_to_regen", 0) > 0
    assert res.arc_flows.get("coker_coke_to_sales", 0) > 0
    # Process conditions accessible on result
    assert "FCC" in (res.meta.get("process_conditions") or {})
    assert res.meta["process_conditions"]["FCC"].get("riser_outlet_temp_f")
    # Pool or direct feed paths exist
    assert res.unit_feeds["fcc_feed"] > 0
    assert (
        res.unit_feeds["fcc_feed_via_pool"] + res.unit_feeds["fcc_feed_direct"]
        == res.unit_feeds["fcc_feed"]
    )
    assert (
        res.unit_feeds["coker_feed_via_pool"] + res.unit_feeds["coker_feed_direct"]
        == res.unit_feeds["coker_feed"]
    )


def test_validate_yields_cover_catalog_helper():
    feed = FeedProperties("x", 25, 0.5, 2, 1000, 0.3, 0.3, 0.4)
    gaps = validate_yields_cover_catalog(
        {
            "FCC": fcc_yields(feed),
            "COKER": coker_yields(feed),
            "REFORMER": reformer_yields(feed),
            "CDU": {
                "cdu_naphtha": 0.2,
                "cdu_distillate": 0.25,
                "cdu_gasoil": 0.3,
                "cdu_resid": 0.25,
                "cdu_offgas": 0.01,
            },
            "HDT_NAPH": {"hdt_naphtha": 0.98, "hdt_lights": 0.02},
        }
    )
    assert gaps == []
