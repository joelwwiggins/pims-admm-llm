"""Excel PIMS-shaped → mono + ADMM → lean results workbook tests."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openpyxl")

from pims_admm_llm.models.excel_pipeline import (
    ensure_template,
    load_pims_excel,
    run_excel_pipeline,
    write_results_excel,
)
from pims_admm_llm.models.assay_loader import write_template_excel

# Goal: ≤15 sheets, one tab per unit submodel, PIMS How-To 07 FCC/Coker
GOAL_MAX_SHEETS = 15
REQUIRED = (
    "How_to_read",
    "Submodel_Index",
    "Calc_Yields",
    "Calc_Blend",
    "Submodel_CDU",
    "Submodel_Blender",
    "Submodel_FCC",
    "Submodel_Coker",
    "Submodel_Linking",
    "Summary",
    "Rates",
    "Shadows",
    "Calc_Check",
)
BANNED_PREFIXES = ("Submodel_FCC_", "Submodel_Coker_", "Live_")
BANNED_EXACT = {
    "Submodel_CDU_Tech",
    "Submodel_CDU_A",
    "Submodel_Blender_Tech",
    "Submodel_Blender_A",
    "Submodel_Linking_B",
    "Calc_BlockAngular",
    "Calc_BA_Map",
    "Calc_BA_Legend",
    "Calc_Process",
    "Calc_Blocks",
    "Calc_Bounds",
    "Calc_Objective",
    "Calc_ModelNote",
    "Calc_Equations",
    "Calc_Linking",
    "Crudes_mono",
    "Products_mono",
    "Crudes_admm",
    "Products_admm",
    "Inter_prod_mono",
    "Inter_use_mono",
}


def test_excel_pipeline_end_to_end(tmp_path):
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    xlsx_out = tmp_path / "results.xlsx"
    json_out = tmp_path / "results.json"
    report = run_excel_pipeline(
        xlsx_in, results_xlsx=xlsx_out, results_json=json_out
    )
    assert xlsx_out.is_file() and json_out.is_file()
    assert report["mono"]["feasible"] and report["admm"]["feasible"]
    assert report["mono"]["objective"] > 0 and report["admm"]["objective"] > 0
    assert report["comparison"]["objective_gap_rel"] <= 0.005 + 1e-9
    assert report["comparison"]["dual_linf_online"] <= 15.0
    assert report["verdict"].startswith("PASS")
    assert report["meta"]["admm_config"]["rho"] == 8.0
    # E14: feasibility stays classic 2-block; TF must not own duals or form.
    assert report.get("model", {}).get("form") == "classic_2block_excel_path"
    path = str(report.get("admm", {}).get("dual_recovery_path") or "")
    assert "online_lambda" in path
    assert "tf_block" not in path.lower()
    assert "tensorflow" not in path.lower()
    assert "tf_block" not in report
    assert report.get("tf_block") is None


def test_excel_pipeline_shadows_prefer_online_lambda(tmp_path):
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    mono_sh = report["mono"]["shadow_prices"]
    online = report["admm"]["shadow_prices"]
    recovered = report["admm"]["shadow_prices_recovered"]
    assert report["comparison"]["dual_linf_online"] <= report["comparison"][
        "dual_linf_recovered"
    ] + 1e-9
    for s in mono_sh:
        assert s in online and s in recovered


def test_write_results_excel_lean_goal(tmp_path):
    """Agentic goal: ≤15 sheets, one unit tab each, PIMS FCC/Coker matrix."""
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    out = tmp_path / "out.xlsx"
    write_results_excel(out, report)
    import openpyxl

    wb = openpyxl.load_workbook(out)
    names = wb.sheetnames

    assert len(names) <= GOAL_MAX_SHEETS, f"n_sheets={len(names)} {names}"
    assert names[0] == "How_to_read"
    for name in REQUIRED:
        assert name in names, name

    banned = [
        s
        for s in names
        if s in BANNED_EXACT or any(s.startswith(p) for p in BANNED_PREFIXES)
    ]
    assert banned == [], f"banned tabs present: {banned}"

    # Index lists single unit sheets
    ih = [c.value for c in wb["Submodel_Index"][1]]
    blocks = {
        r[ih.index("block")].value
        for r in wb["Submodel_Index"].iter_rows(min_row=2)
        if r[0].value
    }
    assert {"CDU", "BLENDER", "FCC", "COKER", "LINKING"} <= blocks

    # CDU merged sections
    cdu_vals = [r[0].value for r in wb["Submodel_CDU"].iter_rows(min_col=1, max_col=1)]
    assert any(v and "TECH" in str(v) for v in cdu_vals)
    assert any(v and "A —" in str(v) or (v and str(v).startswith("===") and "A" in str(v)) for v in cdu_vals)

    # FCC PIMS matrix
    fh = [c.value for c in wb["Submodel_FCC"][1]]
    assert "FEED_FFD" in fh and "BASE" in fh
    assert any(str(h).startswith("D_") for h in fh if h)
    fcc = {
        r[fh.index("row")].value: dict(zip(fh, [c.value for c in r]))
        for r in wb["Submodel_FCC"].iter_rows(min_row=2)
        if r[fh.index("row")].value
    }
    assert fcc["E_BASE_REF"]["FEED_FFD"] == 1.0
    assert fcc["E_BASE_REF"]["BASE"] == -1.0
    assert float(fcc["MB_fcc_naphtha"]["BASE"]) > 0.3
    assert (fcc.get("E_api_REF") or {}).get("FEED_FFD") == -999.0

    ch = [c.value for c in wb["Submodel_Coker"][1]]
    assert "FEED_CFD" in ch and "BASE" in ch
    crow = [r[ch.index("row")].value for r in wb["Submodel_Coker"].iter_rows(min_row=2)]
    assert "E_BASE_REF" in crow and "MB_coker_naphtha" in crow

    # Rates comparison
    rh = [c.value for c in wb["Rates"][1]]
    assert "mono_kbd" in rh and "admm_kbd" in rh

    # Check all ok
    chk_h = [c.value for c in wb["Calc_Check"][1]]
    ok_i = chk_h.index("ok")
    for row in wb["Calc_Check"].iter_rows(min_row=2, values_only=True):
        if row[0]:
            assert row[ok_i] is True, row

    # yield_sum formula still on Calc_Yields
    yh = [c.value for c in wb["Calc_Yields"][1]]
    ys = yh.index("yield_sum")
    cell = wb["Calc_Yields"].cell(2, ys + 1).value
    assert isinstance(cell, str) and cell.startswith("="), cell

    assert report.get("model", {}).get("form") == "classic_2block_excel_path"

    # E1/E11: How_to three-path honesty (export ≠ offline TF ≠ Case1 duals)
    how = {
        str(r[0].value): str(r[1].value or "")
        for r in wb["How_to_read"].iter_rows(min_row=2, max_col=2)
        if r[0].value
    }
    three = how.get("fcc_three_path", "")
    assert "base_delta export" in three or "BASE/D_*" in three
    assert "offline TF" in three or "tf_linear" in three
    assert "classic_2block" in three
    dual_note = how.get("duals_online_lambda", "")
    assert "online" in dual_note.lower()
    # E1/E2: Coker three-path + renorm-outside-affine honesty
    coker = how.get("coker_three_path", "")
    assert coker, "How_to_read must include coker_three_path"
    assert "BASE/D_*" in coker or "base_delta export" in coker
    assert "tf_linear_coker" in coker or "offline TF" in coker
    assert "classic_2block" in coker
    assert "renorm" in coker.lower() or "postprocess" in coker.lower()
    assert "evaluate" in coker.lower() or "reference" in coker.lower()
    # E1/E2: CDU three-path — TECH+A classic ≠ offline TF ≠ duals (not PIMS MB_*)
    cdu = how.get("cdu_three_path", "")
    assert cdu, "How_to_read must include cdu_three_path"
    assert "TECH" in cdu and "A" in cdu
    assert "tf_linear_cdu" in cdu or "offline TF" in cdu or "offline" in cdu.lower()
    assert "not" in cdu.lower() and ("MB_*" in cdu or "BASE/DELTA" in cdu or "PIMS" in cdu or "How-To 07" in cdu)
    assert "online" in cdu.lower() or "dual" in cdu.lower()
    assert "postprocess" in cdu.lower() or "renorm" in cdu.lower() or "affine" in cdu.lower()

    # E1/E2: multi-unit offline TF status (static How_to; not on Case 1 solve)
    tf_off = how.get("tf_offline_units", "")
    assert tf_off, "How_to_read must include tf_offline_units"
    low = tf_off.lower()
    assert "fcc" in low and "coker" in low and "cdu" in low
    assert "classic_2block" in low or "not on this case 1" in low or "not on" in low
    assert "dual_recovery_path" in low or "dual" in low
    assert "primary" in low
    assert "solver=false" in low or "solver=False" in tf_off
    assert "none" in low  # dual_recovery_path=None on TF surface

    # Optional secondary: priced residual readiness (static How_to; still not Case 1)
    tf_priced = how.get("tf_offline_priced", "")
    assert tf_priced, "How_to_read must include tf_offline_priced"
    plow = tf_priced.lower()
    assert "priced residual" in plow or "priced" in plow
    assert "fcc" in plow and "coker" in plow and "cdu" in plow
    assert "not on" in plow or "classic_2block" in plow
    assert "dual" in plow and ("none" in plow or "primary" in plow)
    assert "not" in plow and ("shadow" in plow or "λ" in plow or "lambda" in plow or "admm" in plow)

    # Optional secondary: offline block-solve timing readiness (static How_to)
    tf_timing = how.get("tf_offline_timing", "")
    assert tf_timing, "How_to_read must include tf_offline_timing"
    tlow = tf_timing.lower()
    assert "timing" in tlow or "block-solve" in tlow or "block solve" in tlow
    assert "fcc" in tlow and "coker" in tlow and "cdu" in tlow
    assert "not on" in tlow or "classic_2block" in tlow
    assert "dual" in tlow and ("none" in tlow or "primary" in tlow)
    assert "not" in tlow and (
        "shadow" in tlow or "λ" in tlow or "lambda" in tlow or "wall" in tlow
    )

    # Optional secondary: offline ADMM residual harness (static How_to; still not Case 1)
    tf_admm = how.get("tf_offline_admm_residual", "")
    assert tf_admm, "How_to_read must include tf_offline_admm_residual"
    alow = tf_admm.lower()
    assert "admm" in alow and ("residual" in alow or "consensus" in alow)
    assert "fcc" in alow and "coker" in alow and "cdu" in alow
    assert "not on" in alow or "classic_2block" in alow
    assert "dual" in alow and ("none" in alow or "primary" in alow)
    assert "synthetic" in alow or "λ" in alow or "lambda" in alow
    assert "not" in alow and ("wire" in alow or "online" in alow or "shadow" in alow)

    # Optional secondary: offline ADMM block subproblem maximizer (static How_to)
    tf_sub = how.get("tf_offline_admm_block_subproblem", "")
    assert tf_sub, "How_to_read must include tf_offline_admm_block_subproblem"
    slow = tf_sub.lower()
    assert "admm" in slow and "subproblem" in slow
    assert "raw" in slow
    assert "fcc" in slow and "coker" in slow and "cdu" in slow
    assert "not on" in slow or "classic_2block" in slow
    assert "dual" in slow and ("none" in slow or "primary" in slow)
    assert "not" in slow and "wire" in slow
    assert "synthetic" in slow or "λ" in slow or "lambda" in slow

    # Offline multi-round ADMM coordination How_to (static; still not Case 1)
    tf_coord = how.get("tf_offline_admm_coordination", "")
    assert tf_coord, "How_to_read must include tf_offline_admm_coordination"
    clow = tf_coord.lower()
    assert "admm" in clow and "coordination" in clow
    assert "fcc" in clow and "coker" in clow and "cdu" in clow
    assert "not on" in clow or "classic_2block" in clow
    assert "dual" in clow and ("none" in clow or "primary" in clow)
    assert "not" in clow and "wire" in clow
    assert "synthetic" in clow or "λ" in clow or "lambda" in clow
    assert "plant" in clow or "linking" in clow or "per-unit" in clow or "per unit" in clow

    # Offline multi-block plant-linking ADMM How_to (static; synthetic topology; still not Case 1)
    tf_plant = how.get("tf_offline_admm_plant_linking", "")
    assert tf_plant, "How_to_read must include tf_offline_admm_plant_linking"
    plow = tf_plant.lower()
    assert "admm" in plow and ("plant-linking" in plow or "plant linking" in plow)
    assert "fcc" in plow and "coker" in plow and "cdu" in plow
    assert "not on" in plow or "classic_2block" in plow
    assert "dual" in plow and ("none" in plow or "primary" in plow)
    assert "not" in plow and "wire" in plow
    assert "synthetic" in plow
    assert "full plant" in plow or "mass balance" in plow
    assert "incidence" in plow or "linking" in plow

    # Offline multi-block plant-named linking ADMM How_to (static; plant_named; still not Case 1)
    tf_named = how.get("tf_offline_admm_plant_named_linking", "")
    assert tf_named, "How_to_read must include tf_offline_admm_plant_named_linking"
    nlow = tf_named.lower()
    assert "admm" in nlow and ("plant-named" in nlow or "plant named" in nlow)
    assert "fcc" in nlow and "coker" in nlow and "cdu" in nlow
    assert "not on" in nlow or "classic_2block" in nlow
    assert "dual" in nlow and ("none" in nlow or "primary" in nlow)
    assert "not" in nlow and "wire" in nlow
    assert "plant_named" in nlow or "plant-named" in nlow or "plant named" in nlow
    assert "full plant" in nlow or "mass balance" in nlow
    assert "identity" in nlow or "incidence" in nlow
    assert "synthetic" in nlow  # distinct-from-synthetic language


def test_format_tf_offline_units_howto_pure():
    """Static helper: no solve, isolation-safe contract strings."""
    from pims_admm_llm.models.excel_pipeline import format_tf_offline_units_howto

    d = format_tf_offline_units_howto()
    assert d["topic"] == "tf_offline_units"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    one = d["planner_one_liner"].lower()
    assert "offline" in one
    assert "classic_2block" in one
    assert "primary" in one
    assert "dual" in one


def test_format_tf_offline_priced_howto_pure():
    """Static priced residual How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import format_tf_offline_priced_howto

    d = format_tf_offline_priced_howto()
    assert d["topic"] == "tf_offline_priced"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["price_source"] == "synthetic_offline_demo"
    one = d["planner_one_liner"].lower()
    assert "priced residual" in one or "priced" in one
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one


def test_format_tf_offline_timing_howto_pure():
    """Static timing readiness How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import format_tf_offline_timing_howto

    d = format_tf_offline_timing_howto()
    assert d["topic"] == "tf_offline_timing"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    one = d["planner_one_liner"].lower()
    assert "timing" in one or "block-solve" in one
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one


def test_format_tf_offline_admm_residual_howto_pure():
    """Static ADMM residual How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import (
        format_tf_offline_admm_residual_howto,
    )

    d = format_tf_offline_admm_residual_howto()
    assert d["topic"] == "tf_offline_admm_residual"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["price_source"] == "synthetic_offline_demo"
    assert d["lam_source"] == "synthetic_offline_demo"
    one = d["planner_one_liner"].lower()
    assert "admm" in one and ("residual" in one or "consensus" in one)
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one and "wire" in one
    assert "synthetic" in one


def test_format_tf_offline_admm_block_subproblem_howto_pure():
    """Static ADMM block subproblem How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import (
        format_tf_offline_admm_block_subproblem_howto,
    )

    d = format_tf_offline_admm_block_subproblem_howto()
    assert d["topic"] == "tf_offline_admm_block_subproblem"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["optimand_space"] == "raw_affine"
    assert d["price_source"] == "synthetic_offline_demo"
    assert d["lam_source"] == "synthetic_offline_demo"
    one = d["planner_one_liner"].lower()
    assert "admm" in one and "subproblem" in one
    assert "raw" in one
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one and "wire" in one
    assert "synthetic" in one


def test_format_tf_offline_admm_coordination_howto_pure():
    """Static multi-round ADMM coordination How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import (
        format_tf_offline_admm_coordination_howto,
    )

    d = format_tf_offline_admm_coordination_howto()
    assert d["topic"] == "tf_offline_admm_coordination"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["optimand_space"] == "raw_affine"
    assert d["coordination_scope"] == "per_unit_synthetic_offline"
    assert d["price_source"] == "synthetic_offline_demo"
    assert d["lam_source"] == "synthetic_offline_demo"
    one = d["planner_one_liner"].lower()
    assert "admm" in one and "coordination" in one
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one and "wire" in one
    assert "synthetic" in one
    assert "plant" in one or "linking" in one or "per-unit" in one or "per unit" in one


def test_format_tf_offline_admm_plant_linking_howto_pure():
    """Static multi-block plant-linking ADMM How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import (
        format_tf_offline_admm_plant_linking_howto,
    )

    d = format_tf_offline_admm_plant_linking_howto()
    assert d["topic"] == "tf_offline_admm_plant_linking"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["optimand_space"] == "raw_affine"
    assert d["plant_linking_scope"] == "synthetic_offline_demo"
    assert d["topology_source"] == "synthetic_offline_demo"
    assert d["linking_space"] == "synthetic_linking_streams"
    assert d["not_full_plant_mass_balance"] == "true"
    assert d["price_source"] == "synthetic_offline_demo"
    assert d["lam_source"] == "synthetic_offline_demo"
    one = d["planner_one_liner"].lower()
    assert "admm" in one and ("plant-linking" in one or "plant linking" in one)
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one and "wire" in one
    assert "synthetic" in one
    assert "full plant" in one or "mass balance" in one
    assert "coordination" in one  # distinct-from-coordination language


def test_format_tf_offline_admm_plant_named_linking_howto_pure():
    """Static multi-block plant-named linking ADMM How_to: isolation-safe; no TF import."""
    from pims_admm_llm.models.excel_pipeline import (
        format_tf_offline_admm_plant_named_linking_howto,
    )

    d = format_tf_offline_admm_plant_named_linking_howto()
    assert d["topic"] == "tf_offline_admm_plant_named_linking"
    assert "FCC" in d["units"] and "COKER" in d["units"] and "CDU" in d["units"]
    assert d["on_case1_solve"] == "false"
    assert d["form"] == "classic_2block_excel_path"
    assert d["dual_recovery_path"] == "None"
    assert d["solver"] == "false"
    assert d["on_excel_case1_path"] == "false"
    assert d["optimand_space"] == "raw_affine"
    assert d["plant_linking_scope"] == "plant_named_offline_demo"
    assert d["topology_source"] == "plant_named_offline_demo"
    assert d["linking_space"] == "plant_named_linking_streams"
    assert d["not_full_plant_mass_balance"] == "true"
    assert d["z_update_space"] == "plant_named_linking_streams"
    one = d["planner_one_liner"].lower()
    assert "admm" in one and ("plant-named" in one or "plant named" in one)
    assert "classic_2block" in one or "not on" in one
    assert "primary" in one
    assert "dual" in one
    assert "not" in one and "wire" in one
    assert "full plant" in one or "mass balance" in one
    assert "identity" in one or "incidence" in one
    assert "synthetic" in one  # distinct-from-synthetic language
    assert "coordination" in one  # distinct-from-coordination language


def test_excel_fcc_export_matches_affine_coeffs():
    """E10 always-on: matrix builder MB_* == affine package (no TF, no solve)."""
    from pims_admm_llm.models.tf_linear_blocks import excel_fcc_matrix_matches_affine

    report = excel_fcc_matrix_matches_affine(atol=1e-12)
    assert report["ok"], report.get("mismatches")


def test_excel_pipeline_case1_tf_non_wiring_contract(tmp_path):
    """E14 permanent gate: Case 1 form + duals stay free of TF ownership claims."""
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)

    assert report["mono"]["feasible"] and report["admm"]["feasible"]
    assert report["comparison"]["objective_gap_rel"] <= 0.005 + 1e-9
    assert report["comparison"]["dual_linf_online"] <= 15.0
    assert report["verdict"].startswith("PASS")

    assert report.get("model", {}).get("form") == "classic_2block_excel_path"

    admm = report["admm"]
    path = str(admm.get("dual_recovery_path") or "")
    assert path, "dual_recovery_path must be labeled"
    assert "online_lambda" in path
    # Do not claim pure-ADMM as dual recovery (path may say package-admm backend).
    assert "pure_admm" not in path.lower()
    assert "pure-admm-dual" not in path.lower().replace("_", "-")
    # TF must never be presented as dual recovery or a report-level dual owner.
    for blob in (path, str(report.get("tf_block")), str(admm.get("tf_block"))):
        assert "tf_block" not in blob.lower()
        assert "tensorflow" not in blob.lower()
    assert "tf_block" not in report
    assert "tf_block" not in admm
    # Primary shadows remain free online economic λ (not a TF artifact).
    assert isinstance(admm.get("shadow_prices"), dict) and admm["shadow_prices"]
    assert isinstance(admm.get("shadow_prices_recovered"), dict)


def test_excel_dual_honesty_primary_secondary(tmp_path):
    """E1/E2: Shadows/How_to/Summary PRIMARY online vs SECONDARY recovered; online-only gate."""
    from pims_admm_llm.models.excel_pipeline import (
        _verdict,
        format_dual_honesty_summary,
    )

    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    out = tmp_path / "dual_honesty.xlsx"
    write_results_excel(out, report)
    import openpyxl

    wb = openpyxl.load_workbook(out)
    assert len(wb.sheetnames) <= GOAL_MAX_SHEETS

    # --- Report comparison roles (JSON honesty feed) ---
    cmp_ = report["comparison"]
    assert cmp_["dual_gate"] == "online_lambda"
    assert cmp_["verdict_dual_gate"] == "online_only"
    assert cmp_["dual_linf_online_role"] == "PRIMARY"
    assert cmp_["dual_linf_recovered_role"] == "SECONDARY"
    assert cmp_["recovered_secondary"] is True
    assert cmp_["dual_linf_online"] <= 15.0
    # Recovered may be large / ≥ online; never require recovered ≤15.
    assert cmp_["dual_linf_recovered"] + 1e-9 >= cmp_["dual_linf_online"]

    # Online-only VERDICT math lock (anti-recovered-PASS regression).
    assert report["verdict"].startswith("PASS")
    soft = _verdict(
        report["mono"],
        report["admm"],
        cmp_["objective_gap_rel"],
        dual_linf=100.0,  # would fail if this were the gate input
    )
    assert soft.startswith("PASS_SOFT") or soft.startswith("FAIL")
    assert "dual L∞" in soft or "dual" in soft.lower()

    dual = format_dual_honesty_summary(report)
    assert dual["primary_role"] == "PRIMARY"
    assert dual["secondary_role"] == "SECONDARY"
    assert dual["verdict_dual_gate"] == "online_only"
    assert "online_lambda" in dual["dual_recovery_path"]
    assert "PRIMARY" in dual["planner_one_liner"] and "SECONDARY" in dual["planner_one_liner"]

    # --- Shadows surface ---
    sh_vals = []
    for row in wb["Shadows"].iter_rows(values_only=True):
        sh_vals.extend(str(c) for c in row if c is not None)
    blob = " | ".join(sh_vals)
    assert "PRIMARY" in blob
    assert "SECONDARY" in blob
    assert "admm_online_econ" in blob
    assert "admm_recovered_econ" in blob
    assert "online_only" in blob or "verdict_dual_gate" in blob
    assert "online_lambda" in blob
    assert any("dual_L" in s and "online" in s.lower() for s in sh_vals) or (
        "dual_L∞_online_vs_mono" in blob or "dual_linf_online" in blob.lower()
    )
    assert any("recovered" in s.lower() and "dual" in s.lower() for s in sh_vals) or (
        "dual_L∞_recovered_vs_mono" in blob
    )

    # --- How_to this-run numbers + gate language ---
    how = {
        str(r[0].value): str(r[1].value or "")
        for r in wb["How_to_read"].iter_rows(min_row=2, max_col=2)
        if r[0].value
    }
    dual_note = how.get("duals_online_lambda", "")
    assert "PRIMARY" in dual_note and "SECONDARY" in dual_note
    assert "online_lambda" in dual_note
    # this-run number presence (formatted L∞ from report)
    online_s = dual["dual_linf_online"]
    assert online_s in dual_note or online_s in how.get("duals_primary_secondary", "")
    dps = how.get("duals_primary_secondary", "")
    assert dps and "PRIMARY" in dps and "SECONDARY" in dps
    assert "online_only" in dps or "gates VERDICT" in dps or "tol" in dps
    assert "not pure-ADMM" in dps.lower() or "not pure-admm" in dps.lower()
    assert "not TF" in dps or "not TF dual" in dps
    sb = how.get("solve_boundary", "")
    assert "PRIMARY" in sb and "SECONDARY" in sb

    # --- Summary gate notes ---
    summary = {
        str(r[0].value): r[1].value
        for r in wb["Summary"].iter_rows(min_row=2, max_col=2)
        if r[0].value
    }
    assert summary.get("dual_gate") == "online_lambda"
    assert summary.get("verdict_dual_gate") == "online_only"
    assert summary.get("recovered_secondary") is True
    online_role = str(summary.get("dual_linf_online_role") or "")
    rec_role = str(summary.get("dual_linf_recovered_role") or "")
    assert "PRIMARY" in online_role and "VERDICT" in online_role
    assert "SECONDARY" in rec_role
    assert "online_lambda" in str(summary.get("dual_recovery_path") or "")


def test_format_dual_honesty_summary_pure():
    """DRY helper works without a full solve."""
    from pims_admm_llm.models.excel_pipeline import format_dual_honesty_summary

    fake = {
        "admm": {"dual_recovery_path": "package-admm/qp_l2+online_lambda_shadows"},
        "comparison": {"dual_linf_online": 2.66, "dual_linf_recovered": 112.0},
    }
    d = format_dual_honesty_summary(fake)
    assert d["primary_role"] == "PRIMARY"
    assert d["secondary_role"] == "SECONDARY"
    assert d["verdict_dual_gate"] == "online_only"
    assert "2.66" in d["dual_linf_online"]
    assert "112" in d["dual_linf_recovered"]
    assert "online_lambda" in d["dual_recovery_path"]
    assert "PRIMARY" in d["shadows_role_banner"] and "SECONDARY" in d["shadows_role_banner"]


def test_planner_honesty_glance_package(tmp_path):
    """E1: Index OFFLINE_TF + Summary strip + Calc_Check audits + meta.planner_honesty.

    After #26 multi-block plant-named linking harness: glance package also locks
    multi-block plant-named linking readiness (plant product streams + identity
    incidence; plant_named_offline_demo; not full plant MB; plant-named λ ≠ duals)
    alongside synthetic plant-linking + multi-round coordination + residual +
    block subproblem + priced + timing (static; isolation-safe). Dual
    PRIMARY/SECONDARY and classic form remain non-regression contracts. Coordination
    surface remains distinct (not_plant_linking_coordinator language preserved).
    Synthetic plant-linking packaging remains green.
    """
    from pims_admm_llm.models.excel_pipeline import (
        format_planner_honesty_package,
        planner_honesty_check_rows,
    )

    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    out = tmp_path / "honesty_glance.xlsx"
    write_results_excel(out, report)
    import openpyxl

    wb = openpyxl.load_workbook(out)
    assert len(wb.sheetnames) <= GOAL_MAX_SHEETS

    # --- Pure helpers ---
    pkg = format_planner_honesty_package(report)
    assert pkg["index_row"]["block"] == "OFFLINE_TF"
    assert "NOT" in pkg["index_row"]["what"] and "FCC" in pkg["index_row"]["what"]
    assert "COKER" in pkg["index_row"]["what"] and "CDU" in pkg["index_row"]["what"]
    what_l = pkg["index_row"]["what"].lower()
    assert "priced" in what_l and "timing" in what_l
    assert "admm residual" in what_l
    assert "block subproblem" in what_l
    assert "coordination" in what_l
    assert "multi-round" in what_l or "multi round" in what_l
    assert "synthetic" in what_l
    assert "readiness" in what_l
    assert "raw affine" in what_l or "raw" in what_l
    assert "not duals" in what_l or "prices not duals" in what_l
    assert "not wire" in what_l
    # Positive plant-linking readiness packaging (not only the coordination ban phrase)
    assert "plant-linking readiness" in what_l or "plant linking readiness" in what_l
    assert "multi-block plant-linking" in what_l or "multi-block plant linking" in what_l
    assert "linking topology" in what_l or "shared λ" in what_l or "incidence" in what_l
    assert "not full plant" in what_l or "full plant mb" in what_l
    assert "per-unit coordination" in what_l or "coordination ≠ plant linking" in what_l
    # Positive plant-named readiness packaging (discriminators — not synthetic-only phrases)
    assert "plant-named linking readiness" in what_l or "plant named linking readiness" in what_l
    assert "plant_named_offline_demo" in what_l or "plant product streams" in what_l
    assert "identity incidence" in what_l or "plant product" in what_l
    assert pkg["meta"]["form"] == "classic_2block_excel_path"
    assert pkg["meta"]["dual_gate"] == "online_lambda"
    assert pkg["meta"]["verdict_dual_gate"] == "online_only"
    assert pkg["meta"]["on_excel_case1_path"] is False
    assert pkg["meta"]["tf_on_excel_case1_path"] is False
    assert "FCC" in pkg["meta"]["offline_tf_units"]
    assert "online_lambda" in str(pkg["meta"]["dual_recovery_path"])
    assert pkg["meta"]["offline_tf_priced_ready"] is True
    assert pkg["meta"]["offline_tf_timing_ready"] is True
    assert pkg["meta"]["offline_tf_admm_residual_ready"] is True
    assert pkg["meta"]["offline_tf_admm_block_subproblem_ready"] is True
    assert pkg["meta"]["offline_tf_admm_coordination_ready"] is True
    assert pkg["meta"]["offline_tf_admm_plant_linking_ready"] is True
    assert pkg["meta"]["offline_tf_admm_plant_named_linking_ready"] is True
    assert "priced" in str(pkg["meta"]["offline_tf_priced"]).lower()
    assert "timing" in str(pkg["meta"]["offline_tf_timing"]).lower()
    admm_note = str(pkg["meta"]["offline_tf_admm_residual"]).lower()
    assert "admm residual" in admm_note or "synthetic" in admm_note
    assert "not" in admm_note and (
        "dual" in admm_note or "wire" in admm_note or "online" in admm_note
    )
    sub_note = str(pkg["meta"]["offline_tf_admm_block_subproblem"]).lower()
    assert "block subproblem" in sub_note or "subproblem" in sub_note
    assert "synthetic" in sub_note or "raw" in sub_note
    assert "not" in sub_note and (
        "dual" in sub_note or "wire" in sub_note or "online" in sub_note
    )
    coord_note = str(pkg["meta"]["offline_tf_admm_coordination"]).lower()
    assert "coordination" in coord_note
    assert "synthetic" in coord_note or "multi-round" in coord_note
    assert "not" in coord_note and (
        "dual" in coord_note or "wire" in coord_note or "online" in coord_note
    )
    assert "plant" in coord_note or "linking" in coord_note or "per-unit" in coord_note
    plant_note = str(pkg["meta"]["offline_tf_admm_plant_linking"]).lower()
    assert "plant-linking" in plant_note or "plant linking" in plant_note
    assert "synthetic" in plant_note
    assert "not" in plant_note and (
        "dual" in plant_note or "wire" in plant_note or "online" in plant_note
    )
    assert "full plant" in plant_note or "mass balance" in plant_note
    assert "coordination" in plant_note  # distinct-from-coordination
    named_note = str(pkg["meta"]["offline_tf_admm_plant_named_linking"]).lower()
    assert "plant-named" in named_note or "plant named" in named_note
    assert "plant_named_offline_demo" in named_note or "plant product" in named_note
    assert "not" in named_note and (
        "dual" in named_note or "wire" in named_note or "online" in named_note
    )
    assert "full plant" in named_note or "mass balance" in named_note
    assert "synthetic" in named_note  # distinct-from-synthetic
    assert "coordination" in named_note  # distinct-from-coordination
    readiness_note = str(pkg["meta"]["offline_tf_readiness_note"]).lower()
    assert "not" in readiness_note
    assert "admm residual" in readiness_note
    assert "block subproblem" in readiness_note
    assert "coordination" in readiness_note
    assert "plant-linking" in readiness_note or "plant linking" in readiness_note
    assert "plant-named" in readiness_note or "plant named" in readiness_note
    one_l = str(pkg["meta"]["planner_one_liner"]).lower()
    assert "priced" in one_l and "timing" in one_l
    assert "admm residual" in one_l
    assert "block subproblem" in one_l
    assert "coordination" in one_l
    assert "plant-linking" in one_l or "plant linking" in one_l
    assert "plant-named" in one_l or "plant named" in one_l
    assert "PRIMARY" in pkg["meta"]["dual_linf_online_role"]
    assert "SECONDARY" in pkg["meta"]["dual_linf_recovered_role"]
    assert pkg.get("tf_offline_admm_block_subproblem") is not None
    assert pkg["tf_offline_admm_block_subproblem"]["topic"] == "tf_offline_admm_block_subproblem"
    assert pkg.get("tf_offline_admm_coordination") is not None
    assert pkg["tf_offline_admm_coordination"]["topic"] == "tf_offline_admm_coordination"
    assert pkg.get("tf_offline_admm_plant_linking") is not None
    assert pkg["tf_offline_admm_plant_linking"]["topic"] == "tf_offline_admm_plant_linking"
    assert pkg.get("tf_offline_admm_plant_named_linking") is not None
    assert pkg["tf_offline_admm_plant_named_linking"]["topic"] == "tf_offline_admm_plant_named_linking"
    summary_keys = {k for k, _ in pkg["summary_pairs"]}
    assert {
        "offline_tf_priced",
        "offline_tf_timing",
        "offline_tf_admm_residual",
        "offline_tf_admm_block_subproblem",
        "offline_tf_admm_coordination",
        "offline_tf_admm_plant_linking",
        "offline_tf_admm_plant_named_linking",
        "offline_tf_readiness_note",
        "offline_tf_units",
        "dual_gate",
        "model_form",
    } <= summary_keys

    rows = planner_honesty_check_rows(report)
    names = {r["check"] for r in rows}
    assert {
        "form_classic_2block",
        "dual_gate_online_only",
        "offline_tf_not_on_case1",
        "offline_tf_priced_not_duals",
        "offline_tf_timing_not_case1",
        "offline_tf_admm_residual_not_duals",
        "offline_tf_admm_block_subproblem_not_duals",
        "offline_tf_admm_coordination_not_duals",
        "offline_tf_admm_plant_linking_not_duals",
        "offline_tf_admm_plant_named_linking_not_duals",
    } <= names
    assert all(r["ok"] is True for r in rows)

    # --- meta.planner_honesty on report ---
    ph = report["meta"]["planner_honesty"]
    assert ph["form"] == "classic_2block_excel_path"
    assert ph["dual_gate"] == "online_lambda"
    assert ph["verdict_dual_gate"] == "online_only"
    assert ph["dual_linf_online_role"] == "PRIMARY"
    assert ph["dual_linf_recovered_role"] == "SECONDARY"
    assert ph["on_excel_case1_path"] is False
    assert ph["tf_on_excel_case1_path"] is False
    assert "FCC" in ph["offline_tf_units"] and "COKER" in ph["offline_tf_units"]
    assert "CDU" in ph["offline_tf_units"]
    assert "online_lambda" in str(ph["dual_recovery_path"])
    assert ph["offline_tf_priced_ready"] is True
    assert ph["offline_tf_timing_ready"] is True
    assert ph["offline_tf_admm_residual_ready"] is True
    assert ph["offline_tf_admm_block_subproblem_ready"] is True
    assert ph["offline_tf_admm_coordination_ready"] is True
    assert ph["offline_tf_admm_plant_linking_ready"] is True
    assert ph["offline_tf_admm_plant_named_linking_ready"] is True
    assert "priced" in str(ph["offline_tf_priced"]).lower()
    assert "timing" in str(ph["offline_tf_timing"]).lower()
    assert "not duals" in str(ph["offline_tf_priced"]).lower() or "not admm" in str(
        ph["offline_tf_priced"]
    ).lower()
    assert "not case 1" in str(ph["offline_tf_timing"]).lower() or "wall" in str(
        ph["offline_tf_timing"]
    ).lower()
    ph_admm = str(ph["offline_tf_admm_residual"]).lower()
    assert "synthetic" in ph_admm or "admm residual" in ph_admm
    assert "not" in ph_admm and (
        "dual" in ph_admm or "wire" in ph_admm or "online" in ph_admm
    )
    ph_sub = str(ph["offline_tf_admm_block_subproblem"]).lower()
    assert "subproblem" in ph_sub
    assert "not" in ph_sub and (
        "dual" in ph_sub or "wire" in ph_sub or "online" in ph_sub
    )
    ph_coord = str(ph["offline_tf_admm_coordination"]).lower()
    assert "coordination" in ph_coord
    assert "not" in ph_coord and (
        "dual" in ph_coord or "wire" in ph_coord or "online" in ph_coord
    )
    assert "plant" in ph_coord or "linking" in ph_coord or "per-unit" in ph_coord
    ph_plant = str(ph["offline_tf_admm_plant_linking"]).lower()
    assert "plant-linking" in ph_plant or "plant linking" in ph_plant
    assert "synthetic" in ph_plant
    assert "not" in ph_plant and (
        "dual" in ph_plant or "wire" in ph_plant or "online" in ph_plant
    )
    assert "full plant" in ph_plant or "mass balance" in ph_plant
    ph_named = str(ph["offline_tf_admm_plant_named_linking"]).lower()
    assert "plant-named" in ph_named or "plant named" in ph_named
    assert "plant_named_offline_demo" in ph_named or "plant product" in ph_named
    assert "not" in ph_named and (
        "dual" in ph_named or "wire" in ph_named or "online" in ph_named
    )
    assert "full plant" in ph_named or "mass balance" in ph_named

    # --- Submodel_Index OFFLINE_TF readiness ---
    ih = [c.value for c in wb["Submodel_Index"][1]]
    bi = ih.index("block")
    wi = ih.index("what")
    index_rows = {
        r[bi].value: str(r[wi].value or "")
        for r in wb["Submodel_Index"].iter_rows(min_row=2)
        if r[bi].value
    }
    assert "OFFLINE_TF" in index_rows
    ot = index_rows["OFFLINE_TF"].lower()
    assert "fcc" in ot and "coker" in ot and "cdu" in ot
    assert "not" in ot and ("case 1" in ot or "classic" in ot)
    assert "none" in ot or "dual_recovery_path" in ot
    assert "priced" in ot and "timing" in ot
    assert "admm residual" in ot
    assert "block subproblem" in ot
    assert "coordination" in ot
    assert "synthetic" in ot
    assert "readiness" in ot
    assert "not wire" in ot or "not pure-admm" in ot
    assert "plant-linking readiness" in ot or "plant linking readiness" in ot
    assert "multi-block plant-linking" in ot or "multi-block plant linking" in ot
    assert "not full plant" in ot or "full plant mb" in ot
    assert "plant-named linking readiness" in ot or "plant named linking readiness" in ot
    assert "plant_named_offline_demo" in ot or "plant product streams" in ot
    # FCC/COKER export-vs-live wording
    assert "export" in index_rows["FCC"].lower() or "teaching" in index_rows["FCC"].lower()
    assert "not live" in index_rows["FCC"].lower() or "not" in index_rows["FCC"].lower()
    assert {"CDU", "BLENDER", "FCC", "COKER", "LINKING", "MASTER_ADMM", "OFFLINE_TF"} <= set(
        index_rows
    )

    # --- Summary honesty strip ---
    summary = {
        str(r[0].value): r[1].value
        for r in wb["Summary"].iter_rows(min_row=2, max_col=2)
        if r[0].value
    }
    assert summary.get("model_form") == "classic_2block_excel_path"
    assert summary.get("dual_gate") == "online_lambda"
    assert summary.get("verdict_dual_gate") == "online_only"
    assert summary.get("tf_on_excel_case1_path") is False
    offline_units = str(summary.get("offline_tf_units") or "")
    assert "FCC" in offline_units and "COKER" in offline_units and "CDU" in offline_units
    assert "PRIMARY" in str(summary.get("dual_linf_online_role") or "")
    assert "SECONDARY" in str(summary.get("dual_linf_recovered_role") or "")
    note = str(summary.get("offline_tf_note") or summary.get("index_offline_tf_note") or "")
    assert "not" in note.lower() and ("case 1" in note.lower() or "classic" in note.lower())
    assert "priced" in note.lower() and "timing" in note.lower()
    assert "admm residual" in note.lower()
    assert "block subproblem" in note.lower()
    assert "coordination" in note.lower()
    assert "plant-linking" in note.lower() or "plant linking" in note.lower()
    assert "plant-named" in note.lower() or "plant named" in note.lower()
    priced_s = str(summary.get("offline_tf_priced") or "").lower()
    timing_s = str(summary.get("offline_tf_timing") or "").lower()
    admm_s = str(summary.get("offline_tf_admm_residual") or "").lower()
    sub_s = str(summary.get("offline_tf_admm_block_subproblem") or "").lower()
    coord_s = str(summary.get("offline_tf_admm_coordination") or "").lower()
    plant_s = str(summary.get("offline_tf_admm_plant_linking") or "").lower()
    named_s = str(summary.get("offline_tf_admm_plant_named_linking") or "").lower()
    assert "priced" in priced_s
    assert "not" in priced_s and ("dual" in priced_s or "shadow" in priced_s or "λ" in priced_s or "lambda" in priced_s)
    assert "timing" in timing_s
    assert "not" in timing_s and ("case 1" in timing_s or "wall" in timing_s or "dual" in timing_s)
    assert "admm residual" in admm_s or "synthetic" in admm_s
    assert "not" in admm_s and (
        "dual" in admm_s or "wire" in admm_s or "online" in admm_s
    )
    assert "subproblem" in sub_s
    assert "not" in sub_s and (
        "dual" in sub_s or "wire" in sub_s or "online" in sub_s
    )
    assert "coordination" in coord_s
    assert "not" in coord_s and (
        "dual" in coord_s or "wire" in coord_s or "online" in coord_s
    )
    assert "plant" in coord_s or "linking" in coord_s or "per-unit" in coord_s
    assert "plant-linking" in plant_s or "plant linking" in plant_s
    assert "synthetic" in plant_s
    assert "not" in plant_s and (
        "dual" in plant_s or "wire" in plant_s or "online" in plant_s
    )
    assert "full plant" in plant_s or "mass balance" in plant_s
    assert "plant-named" in named_s or "plant named" in named_s
    assert "plant_named_offline_demo" in named_s or "plant product" in named_s
    assert "not" in named_s and (
        "dual" in named_s or "wire" in named_s or "online" in named_s
    )
    assert "full plant" in named_s or "mass balance" in named_s

    # --- Calc_Check honesty audit rows (all ok) ---
    chk_h = [c.value for c in wb["Calc_Check"][1]]
    ci = chk_h.index("check")
    oi = chk_h.index("ok")
    checks = {
        r[ci]: r[oi]
        for r in wb["Calc_Check"].iter_rows(min_row=2, values_only=True)
        if r[ci]
    }
    assert checks.get("form_classic_2block") is True
    assert checks.get("dual_gate_online_only") is True
    assert checks.get("offline_tf_not_on_case1") is True
    assert checks.get("offline_tf_priced_not_duals") is True
    assert checks.get("offline_tf_timing_not_case1") is True
    assert checks.get("offline_tf_admm_residual_not_duals") is True
    assert checks.get("offline_tf_admm_block_subproblem_not_duals") is True
    assert checks.get("offline_tf_admm_coordination_not_duals") is True
    assert checks.get("offline_tf_admm_plant_linking_not_duals") is True
    assert checks.get("offline_tf_admm_plant_named_linking_not_duals") is True
    for name, ok in checks.items():
        assert ok is True, (name, ok)

    # How_to offline + dual keys preserved (units + priced + timing + residual +
    # subproblem + multi-round coordination + multi-block plant-linking + plant-named)
    how = {
        str(r[0].value): str(r[1].value or "")
        for r in wb["How_to_read"].iter_rows(min_row=2, max_col=2)
        if r[0].value
    }
    assert how.get("tf_offline_units")
    assert how.get("tf_offline_priced")
    assert how.get("tf_offline_timing")
    assert how.get("tf_offline_admm_residual")
    assert how.get("tf_offline_admm_block_subproblem")
    assert how.get("tf_offline_admm_coordination")
    assert how.get("tf_offline_admm_plant_linking")
    assert how.get("tf_offline_admm_plant_named_linking")
    assert "PRIMARY" in how.get("duals_online_lambda", "") or "PRIMARY" in how.get(
        "duals_primary_secondary", ""
    )


def test_planner_honesty_check_rows_pure():
    """Honesty audits fail closed on form / dual-gate drift (no solve needed)."""
    from pims_admm_llm.models.excel_pipeline import planner_honesty_check_rows

    good = {
        "model": {"form": "classic_2block_excel_path"},
        "comparison": {
            "dual_gate": "online_lambda",
            "verdict_dual_gate": "online_only",
            "dual_linf_online_role": "PRIMARY",
        },
        "admm": {"dual_recovery_path": "package-admm/qp_l2+online_lambda_shadows"},
    }
    rows_good = planner_honesty_check_rows(good)
    names = {r["check"] for r in rows_good}
    assert {
        "form_classic_2block",
        "dual_gate_online_only",
        "offline_tf_not_on_case1",
        "offline_tf_priced_not_duals",
        "offline_tf_timing_not_case1",
        "offline_tf_admm_residual_not_duals",
        "offline_tf_admm_block_subproblem_not_duals",
        "offline_tf_admm_coordination_not_duals",
        "offline_tf_admm_plant_linking_not_duals",
        "offline_tf_admm_plant_named_linking_not_duals",
    } <= names
    assert all(r["ok"] for r in rows_good)

    bad_form = dict(good)
    bad_form["model"] = {"form": "wired_tf_path"}
    rows = {r["check"]: r["ok"] for r in planner_honesty_check_rows(bad_form)}
    assert rows["form_classic_2block"] is False
    assert rows["offline_tf_not_on_case1"] is True
    assert rows["offline_tf_priced_not_duals"] is True
    assert rows["offline_tf_timing_not_case1"] is True
    assert rows["offline_tf_admm_residual_not_duals"] is True
    assert rows["offline_tf_admm_block_subproblem_not_duals"] is True
    assert rows["offline_tf_admm_coordination_not_duals"] is True
    assert rows["offline_tf_admm_plant_linking_not_duals"] is True
    assert rows["offline_tf_admm_plant_named_linking_not_duals"] is True


def test_format_planner_honesty_package_priced_timing_pure():
    """Pure formatter exposes priced + timing + residual + subproblem + coordination + plant-linking + plant-named readiness without TF or full solve."""
    from pims_admm_llm.models.excel_pipeline import format_planner_honesty_package

    fake = {
        "model": {"form": "classic_2block_excel_path"},
        "comparison": {
            "dual_gate": "online_lambda",
            "verdict_dual_gate": "online_only",
            "dual_linf_online": 2.66,
            "dual_linf_recovered": 112.0,
            "dual_linf_online_role": "PRIMARY",
            "dual_linf_recovered_role": "SECONDARY",
        },
        "admm": {"dual_recovery_path": "package-admm/qp_l2+online_lambda_shadows"},
    }
    pkg = format_planner_honesty_package(fake)
    what = pkg["index_row"]["what"].lower()
    assert "priced" in what and "timing" in what and "readiness" in what
    assert "admm residual" in what and "synthetic" in what
    assert "block subproblem" in what
    assert "coordination" in what
    assert "plant-linking readiness" in what or "plant linking readiness" in what
    assert "plant-named linking readiness" in what or "plant named linking readiness" in what
    assert "plant_named_offline_demo" in what or "plant product streams" in what
    assert "not" in what and "case 1" in what
    assert "not full plant" in what or "full plant mb" in what
    meta = pkg["meta"]
    assert meta["offline_tf_priced_ready"] is True
    assert meta["offline_tf_timing_ready"] is True
    assert meta["offline_tf_admm_residual_ready"] is True
    assert meta["offline_tf_admm_block_subproblem_ready"] is True
    assert meta["offline_tf_admm_coordination_ready"] is True
    assert meta["offline_tf_admm_plant_linking_ready"] is True
    assert meta["offline_tf_admm_plant_named_linking_ready"] is True
    assert meta["tf_dual_recovery_path"] is None
    assert meta["form"] == "classic_2block_excel_path"
    assert "priced" in meta["planner_one_liner"].lower()
    assert "timing" in meta["planner_one_liner"].lower()
    assert "admm residual" in meta["planner_one_liner"].lower()
    assert "block subproblem" in meta["planner_one_liner"].lower()
    assert "coordination" in meta["planner_one_liner"].lower()
    assert "plant-linking" in meta["planner_one_liner"].lower() or "plant linking" in meta[
        "planner_one_liner"
    ].lower()
    assert "plant-named" in meta["planner_one_liner"].lower() or "plant named" in meta[
        "planner_one_liner"
    ].lower()
    admm_note = str(meta["offline_tf_admm_residual"]).lower()
    assert "synthetic" in admm_note or "admm residual" in admm_note
    assert "not" in admm_note
    sub_note = str(meta["offline_tf_admm_block_subproblem"]).lower()
    assert "subproblem" in sub_note
    assert "not" in sub_note
    coord_note = str(meta["offline_tf_admm_coordination"]).lower()
    assert "coordination" in coord_note
    assert "not" in coord_note
    assert "plant" in coord_note or "linking" in coord_note or "per-unit" in coord_note
    plant_note = str(meta["offline_tf_admm_plant_linking"]).lower()
    assert "plant-linking" in plant_note or "plant linking" in plant_note
    assert "synthetic" in plant_note
    assert "not" in plant_note
    assert "full plant" in plant_note or "mass balance" in plant_note
    named_note = str(meta["offline_tf_admm_plant_named_linking"]).lower()
    assert "plant-named" in named_note or "plant named" in named_note
    assert "plant_named_offline_demo" in named_note or "plant product" in named_note
    assert "not" in named_note
    assert "full plant" in named_note or "mass balance" in named_note
    # anti-claim: must not claim wire shipped or full plant mass balance shipped
    readiness = str(meta["offline_tf_readiness_note"]).lower()
    assert "admm residual" in readiness
    assert "block subproblem" in readiness
    assert "coordination" in readiness
    assert "plant-linking" in readiness or "plant linking" in readiness
    assert "plant-named" in readiness or "plant named" in readiness
    assert "not wire shipped" in readiness or "not on classic case 1" in readiness
    assert "wire shipped" not in readiness.replace("not wire shipped", "")
    assert "full plant mass balance shipped" not in readiness
    assert pkg["tf_offline_admm_block_subproblem"]["topic"] == "tf_offline_admm_block_subproblem"
    assert pkg["tf_offline_admm_block_subproblem"]["dual_recovery_path"] == "None"
    assert pkg["tf_offline_admm_coordination"]["topic"] == "tf_offline_admm_coordination"
    assert pkg["tf_offline_admm_coordination"]["dual_recovery_path"] == "None"
    assert pkg["tf_offline_admm_plant_linking"]["topic"] == "tf_offline_admm_plant_linking"
    assert pkg["tf_offline_admm_plant_linking"]["dual_recovery_path"] == "None"
    assert pkg["tf_offline_admm_plant_linking"]["not_full_plant_mass_balance"] == "true"
    assert pkg["tf_offline_admm_plant_named_linking"]["topic"] == "tf_offline_admm_plant_named_linking"
    assert pkg["tf_offline_admm_plant_named_linking"]["dual_recovery_path"] == "None"
    assert pkg["tf_offline_admm_plant_named_linking"]["topology_source"] == "plant_named_offline_demo"
    assert pkg["tf_offline_admm_plant_named_linking"]["linking_space"] == "plant_named_linking_streams"
    assert pkg["tf_offline_admm_plant_named_linking"]["not_full_plant_mass_balance"] == "true"


def test_load_pims_excel_has_crudes(tmp_path):
    xlsx = tmp_path / "t.xlsx"
    write_template_excel(xlsx)
    pkg = load_pims_excel(xlsx)
    assert pkg["crudes"] and pkg["products"] and pkg["capacities"]


def test_ensure_template(tmp_path):
    path = tmp_path / "crudes_template.xlsx"
    out = ensure_template(path)
    assert out.is_file() and out.stat().st_size > 1000
