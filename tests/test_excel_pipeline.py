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


def test_load_pims_excel_has_crudes(tmp_path):
    xlsx = tmp_path / "t.xlsx"
    write_template_excel(xlsx)
    pkg = load_pims_excel(xlsx)
    assert pkg["crudes"] and pkg["products"] and pkg["capacities"]


def test_ensure_template(tmp_path):
    path = tmp_path / "crudes_template.xlsx"
    out = ensure_template(path)
    assert out.is_file() and out.stat().st_size > 1000
