"""E1/E2: offline Case-1 dual-space L∞ live-λ-seeded skeleton warm-start.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the warm-start hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
  ALWAYS (even if L∞ 0 or ≤15)
- checklist online_linf_gate_under_tf_path remains open
- warmstart_ok = extract ∧ source ∧ seed_policy ∧ rounds ∧ dual-ban — NOT linf<=15
- warm-start ≠ VERDICT gate; warm-start ≠ dual L∞ under wire proof
- seed identity linf_at_seed is NOT dual L∞ under wire proof
- fixture never labeled as caller_supplied / package_extract live
- no excel_pipeline import on tf_linear_blocks warm-start hot path
- no form mutation; no BLENDER in UNITS; blockers still documented
- ready_for_wire_discussion meaning unchanged (additive readiness flag only)

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
    tests/test_tf_offline_case1_dual_space_linf_probe.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_bridge.py \\
    tests/test_tf_offline_case1_dual_space_form_contract.py \\
    tests/test_tf_offline_case1_shaped_linking.py \\
    tests/test_tf_offline_wire_preflight.py \\
    tests/test_tf_offline_case1_dual_space_linf_live_lambda_seeded_warmstart.py -q
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
    "form_label_change_required",
    "dual_linf_under_wire_unproven",
    "case1_is_cdu_blender_package_admm",
    "no_blender_offline_affine_kernel",
    "wire_not_shipped",
}

STREAMS = ["naphtha", "distillate", "gasoil", "residue"]


def test_seed_helper_aligns_primary():
    m = {s: -10.0 - i for i, s in enumerate(STREAMS)}
    rep = tlb.case1_warmstart_seed_lambda_from_primary(m)
    assert rep["seed_ok"] is True
    assert rep["stream_alignment_ok"] is True
    assert rep["seed_policy"] == tlb.SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY
    assert rep["dual_recovery_path"] is None
    assert rep["seeded_lambda_is_probe_input_only"] is True
    for s in STREAMS:
        assert rep["lam0"][s] == pytest.approx(m[s], abs=1e-12)


def test_seed_helper_missing_key_fails():
    m = {s: 1.0 for s in STREAMS if s != "residue"}
    rep = tlb.case1_warmstart_seed_lambda_from_primary(m)
    assert rep["seed_ok"] is False
    assert "residue" in rep["missing_streams"]
    assert rep["dual_recovery_path"] is None


def test_warmstart_caller_supplied_primary_finite_post_round_linf():
    vec = {s: -2.0 - i for i, s in enumerate(STREAMS)}
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda=vec,
        n_rounds=1,
    )
    assert report["kind"] == tlb.CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_SEEDED_WARMSTART_KIND
    assert report["warmstart_ok"] is True
    assert report["ok"] is True
    assert report["extract_ok"] is True
    assert report["seed_ok"] is True
    assert report["rounds_ran"] is True
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
    assert report["seed_policy"] == tlb.SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY
    assert report["z0_policy"] == tlb.Z0_POLICY_UNCHANGED_DEFAULT_SKELETON
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_under_tf_path_open"] is True
    assert report["warmstart_is_not_verdict_gate"] is True
    assert report["warmstart_is_not_dual_linf_under_wire_proof"] is True
    assert report["seed_identity_linf_is_not_proof"] is True
    assert report["seeded_lambda_is_probe_input_only"] is True
    assert report["skeleton_lambda_is_not_case1_online_lambda"] is True
    assert report["finite_linf"] is True
    assert report["linf_post_rounds"] == report["linf"]
    assert report["linf_at_seed"] == pytest.approx(0.0, abs=1e-9)
    assert report["seed_identity_linf_is_not_proof"] is True
    for s in STREAMS:
        assert report["lam0"][s] == pytest.approx(vec[s], abs=1e-12)
        assert report["case1_primary_online_lambda"][s] == pytest.approx(
            vec[s], abs=1e-12
        )


def test_warmstart_identity_seed_linf_zero_still_unproven():
    """Seed identity L∞≈0 must never flip dual_linf_under_wire or warmstart_ok gate."""
    vec = {s: 3.0 for s in STREAMS}
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda=vec,
        n_rounds=1,
    )
    assert report["warmstart_ok"] is True
    assert report["linf_at_seed"] == pytest.approx(0.0, abs=1e-9)
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["warmstart_is_not_dual_linf_under_wire_proof"] is True
    assert report["seed_identity_linf_is_not_proof"] is True
    assert "NOT linf<=15" in report["ok_criteria"] or "not" in report["ok_criteria"].lower()


def test_warmstart_ok_does_not_require_linf_le_15():
    # Large synthetic primary — post-round L∞ may be large; still warmstart_ok if honest.
    primary = {s: 0.0 for s in STREAMS}
    primary["gasoil"] = 200.0
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda=primary,
        n_rounds=1,
    )
    assert report["warmstart_ok"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["wire_shipped"] is False
    # Even if post-round L∞ is large or small, ok never depends on ≤15
    assert "NOT linf<=15" in report["ok_criteria"] or "not" in report[
        "ok_criteria"
    ].lower()


def test_warmstart_package_shaped_extract():
    raw = {s: -1.5 - i for i, s in enumerate(STREAMS)}
    package = {
        "admm": {
            "online_duals": raw,
            "shadow_prices": {s: -v for s, v in raw.items()},
            "shadow_prices_recovered": {s: 9.0 for s in STREAMS},
        },
        "verdict": "PASS",
    }
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_package=package,
        n_rounds=1,
        include_secondary_recovered=True,
    )
    assert report["warmstart_ok"] is True
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_PACKAGE_EXTRACT
    assert report["primary_source_path"] == "admm.online_duals"
    assert report["case1_secondary_recovered_lambda"] is not None
    assert report["secondary_recovered_is_not_gate"] is True
    for s in STREAMS:
        assert report["case1_primary_online_lambda"][s] == pytest.approx(
            raw[s], abs=1e-12
        )


def test_fixture_fallback_labeled_not_live():
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        allow_fixture_fallback=True,
        n_rounds=1,
    )
    assert report["warmstart_ok"] is True
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_FIXTURE
    assert report["used_fixture_fallback"] is True
    assert report["fixture_is_not_live"] is True
    assert report["live_lambda_source"] != tlb.LIVE_LAMBDA_SOURCE_CALLER_SUPPLIED
    assert report["dual_linf_under_wire_status"] == "unproven"


def test_missing_primary_without_fallback_not_ok():
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        allow_fixture_fallback=False,
        n_rounds=1,
    )
    assert report["extract_ok"] is False
    assert report["warmstart_ok"] is False
    assert report["ok"] is False
    assert report["live_lambda_source"] == tlb.LIVE_LAMBDA_SOURCE_MISSING
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"


def test_critical_blockers_still_present_after_warmstart_ok():
    report = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        n_rounds=1,
    )
    assert report["warmstart_ok"] is True
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["blockers_still_documented"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]


def test_warmstart_composes_existing_helpers_no_second_engine():
    src = inspect.getsource(
        tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report
    )
    assert "offline_case1_shaped_cdu_blender_linking_report" in src
    assert "offline_case1_dual_space_linf_probe_report" in src
    assert "case1_dual_space_stream_aligned_linf" in src  # seed identity only
    assert "extract_case1_shaped_skeleton_lambda" in src
    # Must not import excel_pipeline / tensorflow / pulp in body
    assert "import excel_pipeline" not in src
    assert "import tensorflow" not in src
    assert "import pulp" not in src


def test_no_excel_pipeline_pulp_tensorflow_on_warmstart_hot_path():
    funcs = [
        tlb.case1_warmstart_seed_lambda_from_primary,
        tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report,
        tlb._case1_dual_space_linf_live_lambda_seeded_warmstart_honesty_fields,
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
        "CASE1_DUAL_SPACE_LINF_LIVE_LAMBDA_SEEDED_WARMSTART_KIND",
        "SEED_POLICY_LAMBDA0_FROM_LIVE_PRIMARY",
        "Z0_POLICY_UNCHANGED_DEFAULT_SKELETON",
        "case1_warmstart_seed_lambda_from_primary",
        "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report",
        "case1_dual_space_linf_live_lambda_seeded_warmstart",
        "multi_unit_case1_dual_space_linf_live_lambda_seeded_warmstart_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        n_rounds=1,
    )
    b = tlb.case1_dual_space_linf_live_lambda_seeded_warmstart(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        n_rounds=1,
    )
    c = tlb.multi_unit_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        n_rounds=1,
    )
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=True,
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
    assert "admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok" in rep
    assert rep["admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_warmstart_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=False,
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
    assert rep["admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_warmstart_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_space_linf_live_lambda_seeded_warmstart=True,
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
    assert pf.get("admm_case1_dual_space_linf_live_lambda_seeded_warmstart_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_warmstart():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_space_linf_live_lambda_seeded_warmstart_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "warm" in note and ("seed" in note or "λ" in note or "lambda" in note)


def test_bridge_probe_non_regression_still_green():
    bridge = tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
        allow_fixture_fallback=True,
        skeleton_n_rounds=1,
    )
    assert bridge["bridge_ok"] is True
    assert bridge["dual_linf_under_wire_status"] == "unproven"
    probe = tlb.offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
    assert probe["probe_ok"] is True
    contract = tlb.offline_case1_dual_space_form_contract_report()
    assert contract["ok"] is True
