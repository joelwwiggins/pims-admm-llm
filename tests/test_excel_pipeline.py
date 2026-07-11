"""Excel PIMS-shaped → mono + ADMM → results workbook MVP tests."""
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


def test_excel_pipeline_end_to_end(tmp_path):
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    assert xlsx_in.is_file()

    xlsx_out = tmp_path / "results.xlsx"
    json_out = tmp_path / "results.json"
    report = run_excel_pipeline(
        xlsx_in,
        results_xlsx=xlsx_out,
        results_json=json_out,
    )

    assert xlsx_out.is_file()
    assert json_out.is_file()
    assert report["mono"]["feasible"] is True
    assert report["admm"]["feasible"] is True
    assert report["mono"]["objective"] > 0
    assert report["admm"]["objective"] > 0
    # Tuned ρ=8 path: gap ~0.02% on template
    assert report["comparison"]["objective_gap_rel"] <= 0.005 + 1e-9
    # Online λ tracks mono duals; recovered blender duals may differ
    assert report["comparison"]["dual_linf_online"] <= 15.0
    assert report["verdict"].startswith("PASS")
    assert report["meta"]["n_crudes"] >= 3
    assert sum(report["mono"]["crude_rates"].values()) > 1.0
    assert sum(report["mono"]["product_rates"].values()) > 1.0
    assert "online_lambda" in report["admm"]["dual_recovery_path"]
    assert report["meta"]["admm_config"]["rho"] == 8.0


def test_excel_pipeline_shadows_prefer_online_lambda(tmp_path):
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    mono_sh = report["mono"]["shadow_prices"]
    online = report["admm"]["shadow_prices"]
    recovered = report["admm"]["shadow_prices_recovered"]
    # Online L∞ should beat recovered blender dual L∞ on multi-crude template
    assert report["comparison"]["dual_linf_online"] <= report["comparison"][
        "dual_linf_recovered"
    ] + 1e-9
    # All streams reported
    for s in mono_sh:
        assert s in online
        assert s in recovered


def test_write_results_excel_sheets(tmp_path):
    xlsx_in = tmp_path / "model.xlsx"
    write_template_excel(xlsx_in)
    report = run_excel_pipeline(xlsx_in)
    out = tmp_path / "out.xlsx"
    write_results_excel(out, report)
    import openpyxl

    wb = openpyxl.load_workbook(out)
    for name in ("Summary", "Crudes_mono", "Products_mono", "Shadows"):
        assert name in wb.sheetnames
    sh = wb["Shadows"]
    headers = [c.value for c in sh[1]]
    assert "admm_online_econ" in headers
    assert "admm_recovered_econ" in headers


def test_load_pims_excel_has_crudes(tmp_path):
    xlsx = tmp_path / "t.xlsx"
    write_template_excel(xlsx)
    pkg = load_pims_excel(xlsx)
    assert pkg["crudes"]
    assert pkg["products"]
    assert pkg["capacities"]


def test_ensure_template(tmp_path):
    path = tmp_path / "crudes_template.xlsx"
    out = ensure_template(path)
    assert out.is_file()
    assert out.stat().st_size > 1000
