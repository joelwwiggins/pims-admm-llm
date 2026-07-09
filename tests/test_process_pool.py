"""Wave5 W2A: process-pool MIP mode selection (FCC ROT / coker recycle bands)."""

from __future__ import annotations

import pytest


def test_mode_catalogs_nonempty():
    from pims_admm_llm.models.process_pool import (
        list_coker_recycle_modes,
        list_fcc_rot_modes,
    )

    fcc = list_fcc_rot_modes()
    cok = list_coker_recycle_modes()
    assert len(fcc) == 3
    assert len(cok) == 3
    assert {m["id"] for m in fcc} == {"rot_low", "rot_mid", "rot_high"}
    assert {m["id"] for m in cok} == {"rec_low", "rec_mid", "rec_high"}
    assert all("conditions" in m for m in fcc + cok)


def test_yield_tables_differ_by_severity():
    from pims_admm_llm.models.process_pool import build_fcc_mode_yield_tables

    tables = build_fcc_mode_yield_tables()
    # Higher ROT severity should not collapse to identical naphtha yield
    assert tables["rot_high"]["fcc_naphtha"] != tables["rot_low"]["fcc_naphtha"]
    # All modes expose full FCC slate
    for mid, y in tables.items():
        for key in ("fcc_dry_gas", "fcc_lpg", "fcc_naphtha", "fcc_lco", "fcc_slurry", "fcc_coke"):
            assert key in y, mid
            assert y[key] >= 0.0


def test_solve_selects_exactly_one_mode_per_unit():
    from pims_admm_llm.models.process_pool import solve_process_pool_mip

    r = solve_process_pool_mip()
    assert r.feasible
    assert r.status == "Optimal"
    assert r.fcc_mode in {"rot_low", "rot_mid", "rot_high"}
    assert r.coker_mode in {"rec_low", "rec_mid", "rec_high"}
    assert abs(sum(r.fcc_mode_selection.values()) - 1.0) < 1e-6
    assert abs(sum(r.coker_mode_selection.values()) - 1.0) < 1e-6
    # Exactly one binary active per unit
    assert sum(1 for v in r.fcc_mode_selection.values() if v > 0.5) == 1
    assert sum(1 for v in r.coker_mode_selection.values() if v > 0.5) == 1
    assert r.n_binaries >= 6  # 3 FCC + 3 coker mode binaries
    assert r.objective > 0
    assert r.fcc_feed == pytest.approx(40.0)
    assert r.coker_feed == pytest.approx(25.0)


def test_fix_mode_forces_selection():
    from pims_admm_llm.models.process_pool import solve_process_pool_mip

    r = solve_process_pool_mip(fix_fcc_mode="rot_low", fix_coker_mode="rec_high")
    assert r.feasible
    assert r.fcc_mode == "rot_low"
    assert r.coker_mode == "rec_high"
    assert r.fcc_mode_selection["rot_low"] == pytest.approx(1.0)
    assert r.coker_mode_selection["rec_high"] == pytest.approx(1.0)


def test_attach_process_pool_to_plant_yields():
    from pims_admm_llm.models.process_pool import (
        attach_process_pool_to_plant_yields,
        solve_process_pool_mip,
    )

    r = solve_process_pool_mip(fix_fcc_mode="rot_mid", fix_coker_mode="rec_mid")
    assert r.feasible
    merged = attach_process_pool_to_plant_yields({"cdu": {"cdu_naphtha": 0.2}}, r)
    assert "fcc" in merged and "coker" in merged
    assert merged["process_pool"]["fcc_mode"] == "rot_mid"
    assert merged["process_pool"]["coker_mode"] == "rec_mid"
    assert merged["fcc"]["fcc_naphtha"] == pytest.approx(r.yields_selected["fcc"]["fcc_naphtha"])


def test_process_pool_library_note():
    from pims_admm_llm.models.process_pool import build_process_pool_yield_library

    lib = build_process_pool_yield_library()
    assert "SOS1" in lib["note"] or "binary" in lib["note"].lower()
    assert set(lib["fcc_yields_by_mode"]) == {"rot_low", "rot_mid", "rot_high"}
