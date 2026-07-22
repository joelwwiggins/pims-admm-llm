"""E2: offline Case-1 dual-space L∞ live-λ bridge / capture harness.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the bridge hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- checklist online_linf_gate_under_tf_path remains open
- bridge_ok = extract ∧ source documented ∧ probe honesty — NOT linf <= 15
- bridge ≠ VERDICT gate; bridge ≠ dual L∞ under wire proof
- fixture never labeled as caller_supplied / package_extract live
- no excel_pipeline import on tf_linear_blocks bridge hot path
- no form mutation; no BLENDER in UNITS; blockers still documented
"""

from __future__ import annotations

import inspect

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


CRITICAL_BLOCKERS = {
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}

STREAMS = ["naphtha", "distillate", "gasoil", "residue"]


def test_primary_from_mapping_aligned():
    m = {s: -10.0 - i for i, s in enumerate(STREAMS)}
    rep = tlb.case1_primary_online_lambda_from_mapping(m)
    assert rep["extract_ok"] is True
    assert rep["stream_alignment_ok"] is True
    assert set(rep["lambda"]) == set(STREAMS)
    assert rep["dual_recovery_path"] is None
    assert rep["source"] == tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
    for s in STREAMS:
        assert rep["lambda"][s] == pytest.approx(m[s], abs=1e-12)


def test_primary_from_mapping_missing_key_fails():
    m = {s: 1.0 for s in STREAMS if s != "residue"}
    rep = tlb.case1_primary_online_lambda_from_mapping(m)
    assert rep["extract_ok"] is False
    assert "residue" in rep["missing_streams"]
    assert rep["dual_recovery_path"] is None


def test_package_extract_prefers_admm_online_duals_raw():
    raw = {s: -20.0 - i for i, s in enumerate(STREAMS)}
    eco = {s: -v for s, v in raw.items()}
    package = {
        "admm": {
            "online_duals": raw,
            "shadow_prices": eco,
            "shadow_prices_recovered": {s: 100.0 for s in STREAMS},
        },
        "verdict": "PASS",
    }
    rep = tlb.extract_case1_primary_online_lambda(package)
    assert rep["extract_ok"] is True
    assert rep["source"] == tlb.LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
    assert rep["source_path"] == "admm.online_duals"
    assert rep["dual_vector_face"] == tlb.CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE
    for s in STREAMS:
        assert rep["lambda"][s] == pytest.approx(raw[s], abs=1e-12)


def test_package_extract_economic_face():
    raw = {s: -5.0 for s in STREAMS}
    eco = {s: 5.0 for s in STREAMS}
    package = {"admm": {"online_duals": raw, "shadow_prices": eco}}
    rep = tlb.extract_case1_primary_online_lambda(
        package, face=tlb.CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW
    )
    assert rep["extract_ok"] is True
    assert rep["source_path"] == "admm.shadow_prices"
    for s in STREAMS:
        assert rep["lambda"][s] == pytest.approx(5.0, abs=1e-12)


def test_package_extract_economic_from_raw_negation():
    raw = {s: -7.0 for s in STREAMS}
    package = {"admm": {"online_duals": raw}}  # no shadow_prices
    rep = tlb.extract_case1_primary_online_lambda(
        package, face=tlb.CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW
    )
    assert rep["extract_ok"] is True
    assert rep["face_converted"] is True
    for s in STREAMS:
        assert rep["lambda"][s] == pytest.approx(7.0, abs=1e-12)


def test_secondary_recovered_extract_diagnostic_only():
    sec = {s: 50.0 + i for i, s in enumerate(STREAMS)}
    package = {"admm": {"shadow_prices_recovered": sec}}
    rep = tlb.extract_case1_secondary_recovered_lambda(package)
    assert rep["extract_ok"] is True
    assert rep["secondary_recovered_is_not_gate"] is True
    assert rep["dual_recovery_path"] is None
    for s in STREAMS:
        assert rep["lambda"][s] == pytest.approx(sec[s], abs=1e-12)


def test_bridge_identical_primary_skeleton_linf_zero_still_unproven():
    vec = {s: 2.0 for s in STREAMS}
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        case1_primary_online_lambda=vec,
        skeleton_lambda=vec,
    )
    assert report["kind"] == tlb.CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_BRIDGE_KIND
    assert report["bridge_ok"] is True
    assert report["ok"] is True
    assert report["extract_ok"] is True
    assert report["linf"] == pytest.approx(0.0, abs=1e-12)
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_under_tf_path_open"] is True
    assert report["bridge_is_not_verdict_gate"] is True
    assert report["bridge_is_not_dual_linf_under_wire_proof"] is True
    assert report["probe_is_not_verdict_gate"] is True
    assert "NOT linf<=15" in report["ok_criteria"] or "not" in report["ok_criteria"].lower()


def test_bridge_known_gap_expected_linf_not_gated_on_15():
    primary = {s: 0.0 for s in STREAMS}
    skeleton = {s: 0.0 for s in STREAMS}
    skeleton["gasoil"] = 100.0  # >> 15
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        case1_primary_online_lambda=primary,
        skeleton_lambda=skeleton,
    )
    assert report["bridge_ok"] is True  # never requires linf<=15
    assert report["linf"] == pytest.approx(100.0, abs=1e-12)
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["wire_shipped"] is False


def test_bridge_package_shaped_extract():
    raw = {s: -1.0 for s in STREAMS}
    package = {
        "admm": {
            "online_duals": raw,
            "shadow_prices": {s: 1.0 for s in STREAMS},
            "shadow_prices_recovered": {s: 9.0 for s in STREAMS},
        }
    }
    skeleton = {s: 0.0 for s in STREAMS}
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        case1_package=package,
        skeleton_lambda=skeleton,
    )
    assert report["bridge_ok"] is True
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
    assert report["primary_source_path"] == "admm.online_duals"
    assert report["linf"] == pytest.approx(1.0, abs=1e-12)
    assert report["case1_secondary_recovered_lambda"] is not None
    assert report["secondary_recovered_is_not_gate"] is True
    assert report["secondary_gap_diagnostic"] is not None


def test_fixture_fallback_labeled_not_live():
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        allow_fixture_fallback=True,
        skeleton_n_rounds=1,
    )
    assert report["bridge_ok"] is True
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_FIXTURE
    assert report["used_fixture_fallback"] is True
    assert report["fixture_is_not_live"] is True
    assert report["live_lambda_source"] != tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
    assert report["dual_linf_under_wire_status"] == "unproven"


def test_missing_primary_without_fallback_not_ok():
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        allow_fixture_fallback=False,
    )
    assert report["extract_ok"] is False
    assert report["bridge_ok"] is False
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_MISSING
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"


def test_critical_blockers_still_present():
    report = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    assert report["bridge_ok"] is True
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert report["form_current"] == "tf_affine_cdu_blender_shaped_excel_path"
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]


def test_bridge_composes_existing_probe_no_second_engine():
    src = inspect.getsource(tlb.offline_case1_dual_space_linf_live_lambda_bridge_report)
    assert "offline_case1_dual_space_linf_probe_report" in src
    # Bridge should not reimplement max abs gap itself as primary math path
    # (case1_dual_space_stream_aligned_linf only via probe).
    assert "case1_dual_space_stream_aligned_linf" not in src


def test_no_excel_pipeline_pulp_tensorflow_on_bridge_hot_path():
    funcs = [
        tlb.case1_primary_online_lambda_from_mapping,
        tlb.extract_case1_primary_online_lambda,
        tlb.extract_case1_secondary_recovered_lambda,
        tlb.offline_case1_dual_space_linf_live_lambda_bridge_report,
        tlb._case1_dual_space_linf_live_lambda_bridge_honesty_fields,
    ]
    blob = "".join(inspect.getsource(f) for f in funcs)
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "from pulp" not in blob
    assert "from tensorflow" not in blob
    assert "import excel_pipeline" not in blob
    assert "from excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "from .excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob


def test_aliases_and_exports():
    for name in (
        "CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_BRIDGE_KIND",
        "LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED",
        "LIVE_LAMBDA_SOURCE_FIXTURE",
        "LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT",
        "LIVE_LAMBDA_SOURCE_MISSING",
        "case1_primary_online_lambda_from_mapping",
        "extract_case1_primary_online_lambda",
        "extract_case1_secondary_recovered_lambda",
        "offline_case1_dual_space_linf_live_lambda_bridge_report",
        "case1_dual_space_linf_live_lambda_bridge",
        "multi_unit_case1_dual_space_linf_live_lambda_bridge_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    b = tlb.case1_dual_space_linf_live_lambda_bridge(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    c = tlb.multi_unit_case1_dual_space_linf_live_lambda_bridge_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_bridge=True,
        include_admm_case1_dual_space_linf_probe=True,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_dual_space_linf_live_lambda_bridge_ok" in rep
    assert rep["admm_case1_dual_space_linf_live_lambda_bridge_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_bridge_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_bridge=False,
        include_admm_case1_dual_space_linf_probe=False,
        include_admm_case1_dual_space_form_contract=False,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert rep["admm_case1_dual_space_linf_live_lambda_bridge_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_bridge_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_bridge=True,
        include_admm_case1_dual_space_linf_probe=True,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert pf["wire_shipped"] is False
    assert pf["dual_recovery_path"] is None
    assert CRITICAL_BLOCKERS.issubset(set(pf["wire_blockers"]))
    assert pf.get("admm_case1_dual_space_linf_live_lambda_bridge_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_bridge():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_space_linf_live_lambda_bridge_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "live" in note and ("bridge" in note or "λ" in note or "lambda" in note)


def test_probe_non_regression_still_green():
    probe = tlb.offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
    assert probe["probe_ok"] is True
    assert probe["dual_linf_under_wire_status"] == "unproven"
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
