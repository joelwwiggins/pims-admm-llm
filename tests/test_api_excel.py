"""API coverage for Excel PIMS MVP endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("fastapi")
pytest.importorskip("openpyxl")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from pims_admm_llm.models.assay_loader import write_template_excel  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_excel_template_download(client: TestClient):
    r = client.get("/api/excel/template")
    assert r.status_code == 200
    assert "spreadsheetml" in r.headers.get("content-type", "")
    assert len(r.content) > 1000


def test_excel_solve_json(client: TestClient, tmp_path: Path):
    xlsx = tmp_path / "model.xlsx"
    write_template_excel(xlsx)
    with xlsx.open("rb") as f:
        r = client.post(
            "/api/excel/solve",
            files={
                "file": (
                    "model.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["verdict"].startswith("PASS")
    assert body["mono"]["feasible"] is True
    assert body["admm"]["feasible"] is True
    assert body["comparison"]["objective_gap_rel"] <= 0.01 + 1e-9
    assert body["meta"]["results_xlsx"]
    assert body["meta"]["download_results_xlsx"].startswith("/api/excel/results?path=")
    # Honesty glance strip for API consumers (presentation only).
    ph = body["meta"].get("planner_honesty") or {}
    assert ph.get("form") == "classic_2block_excel_path" or ph.get("model_form") == (
        "classic_2block_excel_path"
    )
    assert ph.get("dual_gate") == "online_lambda"
    assert ph.get("verdict_dual_gate") == "online_only"
    assert ph.get("on_excel_case1_path") is False
    assert "FCC" in str(ph.get("offline_tf_units") or "")

    # download written results
    base = Path(body["meta"]["results_xlsx"]).name
    r2 = client.get("/api/excel/results", params={"path": base})
    assert r2.status_code == 200
    assert len(r2.content) > 500


def test_excel_solve_rejects_non_xlsx(client: TestClient):
    r = client.post(
        "/api/excel/solve",
        files={"file": ("notes.txt", b"not excel", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_excel_results_path_traversal_blocked(client: TestClient):
    r = client.get("/api/excel/results", params={"path": "../secrets.xlsx"})
    assert r.status_code == 400


def test_root_lists_excel_routes(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "excel_solve" in body
    assert "excel_template" in body
