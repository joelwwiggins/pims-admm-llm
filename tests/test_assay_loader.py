"""W1: assay JSON/Excel loader + property-driven yields dual-path."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ASSAYS = REPO / "data" / "assays" / "crudes.json"
INTER = REPO / "data" / "assays" / "intermediates.json"
ROUTING = REPO / "data" / "routing.json"
SYNTH = REPO / "data" / "synthetic_crudes.json"


def test_assay_files_exist():
    assert ASSAYS.is_file(), ASSAYS
    assert INTER.is_file(), INTER
    assert ROUTING.is_file(), ROUTING


def test_load_assays_json_fields():
    from pims_admm_llm.models.assay_loader import load_assays_json

    data = load_assays_json(ASSAYS)
    assert len(data["crudes"]) >= 3
    c0 = data["crudes"][0]
    for key in (
        "name",
        "api",
        "sulfur_wt",
        "ccr_wt",
        "nitrogen_ppm",
        "aromatics_vol",
        "paraffins_vol",
        "naphthenes_vol",
        "price_usd_per_bbl",
        "tbp_cut_vol",
    ):
        assert key in c0, key
    tbp = c0["tbp_cut_vol"]
    s = sum(tbp.values())
    assert abs(s - 1.0) < 0.05, s
    pna = c0["paraffins_vol"] + c0["naphthenes_vol"] + c0["aromatics_vol"]
    assert abs(pna - 1.0) < 0.05, pna


def test_load_intermediates_json():
    from pims_admm_llm.models.assay_loader import (
        intermediate_properties_list,
        load_intermediates_json,
    )

    pkg = load_intermediates_json(INTER)
    names = {r["name"] for r in pkg["intermediates"]}
    assert "cdu_gasoil" in names
    assert "cdu_resid" in names
    assert "fcc_naphtha" in names
    assert "reformate" in names
    props = intermediate_properties_list(pkg)
    assert len(props) == len(pkg["intermediates"])
    go = next(p for p in props if p.name == "cdu_gasoil")
    assert go.api > 0 and go.sulfur_wt > 0


def test_routing_linking_streams():
    from pims_admm_llm.models.assay_loader import load_routing

    r = load_routing(ROUTING)
    assert "CDU" in r["units"]
    assert "FCC" in r["units"]
    assert "cdu_gasoil" in r["linking_streams"]
    assert any(x["from"] == "CDU" and x["to"] == "TANK_GASOIL" for x in r["routes"])


def test_property_driven_cdu_yields():
    from pims_admm_llm.models.assay_loader import cdu_yields_classic, load_assays_json
    from pims_admm_llm.models.properties import crude_to_props
    from pims_admm_llm.models.yields import cdu_yields_from_assay

    assays = load_assays_json(ASSAYS)
    wti = next(c for c in assays["crudes"] if c["name"] == "WTI_light")
    maya = next(c for c in assays["crudes"] if c["name"] == "Maya_heavy")
    y_wti = cdu_yields_from_assay(crude_to_props(wti), wti["tbp_cut_vol"])
    y_maya = cdu_yields_from_assay(crude_to_props(maya), maya["tbp_cut_vol"])
    # light crude → more naphtha, less resid than heavy
    assert y_wti["cdu_naphtha"] > y_maya["cdu_naphtha"]
    assert y_wti["cdu_resid"] < y_maya["cdu_resid"]
    classic = cdu_yields_classic(wti)
    assert abs(sum(classic.values()) - 1.0) < 1e-9
    assert set(classic) == {"naphtha", "distillate", "gasoil", "residue"}


def test_fcc_coker_reformer_yields_respond_to_props():
    from pims_admm_llm.models.properties import FeedProperties
    from pims_admm_llm.models.yields import coker_yields, fcc_yields, reformer_yields

    light_go = FeedProperties(name="light", api=28, ccr_wt=0.5, sulfur_wt=0.5)
    heavy_go = FeedProperties(name="heavy", api=18, ccr_wt=4.0, sulfur_wt=2.5)
    assert fcc_yields(light_go)["fcc_naphtha"] > fcc_yields(heavy_go)["fcc_naphtha"]

    low_ccr = FeedProperties(name="lc", api=12, ccr_wt=8.0)
    high_ccr = FeedProperties(name="hc", api=10, ccr_wt=18.0)
    liq_low = sum(coker_yields(low_ccr).values())
    liq_high = sum(coker_yields(high_ccr).values())
    assert liq_low > liq_high

    good_n = FeedProperties(
        name="good",
        naphthenes_vol=0.4,
        aromatics_vol=0.35,
        paraffins_vol=0.25,
        nitrogen_ppm=100,
    )
    bad_n = FeedProperties(
        name="bad",
        naphthenes_vol=0.2,
        aromatics_vol=0.2,
        paraffins_vol=0.6,
        nitrogen_ppm=3000,
    )
    assert reformer_yields(good_n)["reformate"] > reformer_yields(bad_n)["reformate"]


def test_synthetic_consumers_unchanged():
    """Default load_crude_data still reads synthetic_crudes.json and matches MVP."""
    from pims_admm_llm.models import load_crude_data, validate_refinery_data

    data = load_crude_data()  # default path
    assert data.cdu_capacity_kbd == 120.0
    assert len(data.crudes) == 3
    assert data.crudes[0].name == "WTI_light"
    # fixed yields from synthetic file
    assert abs(data.crudes[0].yields["naphtha"] - 0.28) < 1e-9
    issues = validate_refinery_data(data)
    assert not issues, issues


def test_dual_path_load_assays_as_refinery_data():
    from pims_admm_llm.models import load_crude_data, solve_monolithic, validate_refinery_data
    from pims_admm_llm.models.assay_loader import assays_to_refinery_data, load_assays_json

    assays = load_assays_json(ASSAYS)
    data = assays_to_refinery_data(assays)
    assert len(data.crudes) >= 3
    assert data.cdu_capacity_kbd == 140.0
    issues = validate_refinery_data(data)
    assert not issues, issues
    # via load_crude_data dual path
    data2 = load_crude_data(ASSAYS)
    assert len(data2.crudes) == len(data.crudes)
    res = solve_monolithic(data2, msg=False)
    assert res.status == "Optimal"
    assert res.objective > 0


def test_excel_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from pims_admm_llm.models.assay_loader import (
        load_assays_excel,
        load_assays_json,
        write_template_excel,
    )

    xlsx = tmp_path / "assays_template.xlsx"
    write_template_excel(xlsx)
    assert xlsx.is_file()
    loaded = load_assays_excel(xlsx)
    orig = load_assays_json(ASSAYS)
    assert len(loaded["crudes"]) == len(orig["crudes"])
    names = {c["name"] for c in loaded["crudes"]}
    assert "WTI_light" in names
    assert "Maya_heavy" in names
    wti = next(c for c in loaded["crudes"] if c["name"] == "WTI_light")
    assert abs(wti["api"] - 39.6) < 1e-6
    assert abs(wti["sulfur_wt"] - 0.24) < 1e-6
    assert "tbp_cut_vol" in wti
    assert abs(wti["tbp_cut_vol"]["naphtha_ibp_350f"] - 0.28) < 1e-6
    # intermediates sheet present
    assert loaded.get("intermediates")
    assert any(i["name"] == "cdu_gasoil" for i in loaded["intermediates"])
    assert openpyxl is not None  # silence unused if importorskip returns module


def test_sample_assay_numbers_print():
    """Sanity dump for handoff (raw sample numbers)."""
    from pims_admm_llm.models.assay_loader import (
        cdu_yields_classic,
        load_assays_json,
        load_intermediates_json,
    )

    assays = load_assays_json(ASSAYS)
    print("CRUDES:")
    for c in assays["crudes"]:
        y = cdu_yields_classic(c)
        print(
            f"  {c['name']}: API={c['api']} S={c['sulfur_wt']} CCR={c['ccr_wt']} "
            f"N={c['nitrogen_ppm']} price={c['price_usd_per_bbl']} "
            f"yields={ {k: round(v, 4) for k, v in y.items()} }"
        )
    inter = load_intermediates_json(INTER)
    print(f"INTERMEDIATES: {len(inter['intermediates'])} streams")
    for row in inter["intermediates"][:3]:
        print(
            f"  {row['name']}: API={row['api']} S={row['sulfur_wt']} "
            f"CCR={row['ccr_wt']} A={row['aromatics_vol']}"
        )
