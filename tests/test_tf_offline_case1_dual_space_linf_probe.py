"""E1/E2: offline Case-1 dual-space L∞ probe / dual_linf proof-prep (always-on numpy).

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on the
probe hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- checklist online_linf_gate_under_tf_path remains open (probe ≠ proof under wire)
- probe_ok = finite ∧ aligned ∧ dual-ban — NOT linf <= 15; NOT VERDICT gate
- skeleton λ ≠ Case 1 PRIMARY/SECONDARY duals as dual recovery
- stream alignment naphtha/distillate/gasoil/residue (not plant-linking resid)
- identical vectors → L∞≈0; known gap → expected L∞
- SECONDARY recovered optional diagnostic does not become gate
- DEFAULT_WIRE_BLOCKERS still contain dual_linf_under_wire_unproven / wire_not_shipped
- additive readiness flag does not redefine ready_for_wire_discussion
- no residual-must-vanish; no recovered L∞ ≤15; no Case 1 form mutation
- no excel_cdu_matrix_matches_affine / excel_blender invent

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem, test_tf_offline_admm_coordination,
  test_tf_offline_admm_plant_linking, test_tf_offline_wire_preflight,
  test_tf_offline_case1_shaped_linking,
  test_tf_offline_case1_dual_space_form_contract,
  test_excel_pipeline, test_api_excel
  EMRPS optional-only (not required for this gate).
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


def test_stream_aligned_linf_identical_is_zero():
    a = {s: 1.5 for s in STREAMS}
    b = {s: 1.5 for s in STREAMS}
    gap = tlb.case1_dual_space_stream_aligned_linf(a, b)
    assert gap["stream_alignment_ok"] is True
    assert gap["finite_ok"] is True
    assert gap["linf"] == pytest.approx(0.0, abs=1e-12)
    assert gap["l1"] == pytest.approx(0.0, abs=1e-12)
    for s in STREAMS:
        assert gap["per_stream_abs"][s] == pytest.approx(0.0, abs=1e-12)
    # Alias
    gap2 = tlb.stream_aligned_dual_linf(a, b)
    assert gap2["linf"] == pytest.approx(0.0, abs=1e-12)


def test_stream_aligned_linf_known_gap():
    a = {s: 0.0 for s in STREAMS}
    b = {s: 0.0 for s in STREAMS}
    b["distillate"] = 3.0
    gap = tlb.case1_dual_space_stream_aligned_linf(a, b)
    assert gap["stream_alignment_ok"] is True
    assert gap["linf"] == pytest.approx(3.0, abs=1e-12)
    assert gap["l1"] == pytest.approx(3.0, abs=1e-12)
    assert gap["per_stream_abs"]["distillate"] == pytest.approx(3.0, abs=1e-12)
    assert gap["per_stream_abs"]["naphtha"] == pytest.approx(0.0, abs=1e-12)


def test_stream_aligned_missing_key_not_silent_success():
    a = {s: 1.0 for s in STREAMS}
    b = {s: 1.0 for s in STREAMS if s != "residue"}
    gap = tlb.case1_dual_space_stream_aligned_linf(a, b)
    assert gap["stream_alignment_ok"] is False
    assert "residue" in gap["missing_in_b"]
    assert "resid" not in gap["streams"]


def test_fixture_primary_online_lambda_keys_and_face():
    fix = tlb.case1_primary_online_lambda_fixture()
    assert set(fix) == set(STREAMS)
    assert all(fix[s] < 0 for s in STREAMS)  # raw online_duals face negative
    assert fix == tlb.CASE1_FIXTURE_PRIMARY_ONLINE_LAMBDA
    eco = tlb.case1_primary_online_lambda_fixture(
        face=tlb.CASE1_DUAL_VECTOR_FACE_ECONOMIC_SHADOW
    )
    for s in STREAMS:
        assert eco[s] == pytest.approx(-fix[s], abs=1e-12)
    sec = tlb.case1_secondary_recovered_lambda_fixture()
    assert set(sec) == set(STREAMS)


def test_skeleton_lambda_extract_keys_and_dual_ban():
    sk = tlb.extract_case1_shaped_skeleton_lambda(n_rounds=1)
    assert set(sk) == set(STREAMS)
    assert all(isinstance(v, float) for v in sk.values())
    # Parent skeleton report remains dual-ban
    rep = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    assert rep["dual_recovery_path"] is None
    assert rep.get("skeleton_lambda_is_not_case1_online_lambda", True) is True
    sk2 = tlb.case1_shaped_skeleton_lambda(skeleton_report=rep)
    assert set(sk2) == set(STREAMS)


def test_probe_report_honesty_locks_and_unproven():
    report = tlb.offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
    assert report["kind"] == "offline_case1_dual_space_linf_probe"
    assert report["kind"] == tlb.CASE1_DUAL_SPACE_LINF_PROBE_KIND
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["wire_shipped"] is False
    assert report["not_wire_shipped"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_isolation_rewrite"] is True
    assert report["not_full_tf_admm_wire"] is True
    assert report["skeleton_lambda_is_not_case1_online_lambda"] is True
    assert report["skeleton_lambda_is_not_case1_primary_or_secondary_duals"] is True
    assert report["probe_is_not_verdict_gate"] is True
    assert report["probe_is_not_dual_linf_under_wire_proof"] is True
    assert report["secondary_recovered_is_not_gate"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_under_tf_path_open"] is True
    assert report["package_dual_gate"] == "online_lambda"
    assert report["package_dual_secondary"] == "recovered_blender"
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["case1_form_unchanged"] is True
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["stream_alignment_ok"] is True
    assert report["finite_linf"] is True
    assert report["probe_ok"] is True
    assert report["ok"] is True, report
    assert set(report["streams"]) == set(STREAMS)
    assert "resid" not in report["streams"]
    assert report["dual_vector_face"] == tlb.CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE
    # probe_ok must not require linf<=15
    assert "NOT linf<=15" in report["ok_criteria"] or "not" in report["ok_criteria"].lower()
    diag = report["probe_linf_vs_gate_tol_diagnostic"]
    assert diag["is_not_verdict_gate"] is True
    assert diag["is_not_probe_ok_criterion"] is True


def test_probe_ok_even_when_linf_zero_stays_unproven():
    """Numeric L∞≈0 must NOT flip dual_linf_under_wire to proven."""
    vec = {s: 2.0 for s in STREAMS}
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda=vec,
        skeleton_lambda=vec,
    )
    assert report["probe_ok"] is True
    assert report["linf"] == pytest.approx(0.0, abs=1e-12)
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_under_tf_path_open"] is True
    assert "dual_linf_under_wire_unproven" in report["wire_blockers"]
    assert "wire_not_shipped" in report["wire_blockers"]
    assert report["probe_is_not_dual_linf_under_wire_proof"] is True
    assert report["probe_is_not_verdict_gate"] is True


def test_probe_known_gap_expected_linf():
    primary = {s: 0.0 for s in STREAMS}
    skeleton = {s: 0.0 for s in STREAMS}
    skeleton["gasoil"] = 4.5
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda=primary,
        skeleton_lambda=skeleton,
    )
    assert report["probe_ok"] is True
    assert report["linf"] == pytest.approx(4.5, abs=1e-12)
    assert report["per_stream_abs"]["gasoil"] == pytest.approx(4.5, abs=1e-12)


def test_probe_fixture_plus_skeleton_extract_path():
    report = tlb.offline_case1_dual_space_linf_probe_report(skeleton_n_rounds=1)
    assert report["case1_primary_online_lambda_source"] == "fixture"
    assert report["skeleton_lambda_source"] in (
        "case1_shaped_extract",
        "skeleton_report_final_lam",
    )
    assert set(report["case1_primary_online_lambda"]) == set(STREAMS)
    assert set(report["skeleton_lambda"]) == set(STREAMS)
    assert report["finite_linf"] is True
    assert report["probe_ok"] is True


def test_secondary_recovered_optional_non_gate():
    primary = {s: 1.0 for s in STREAMS}
    skeleton = {s: 1.0 for s in STREAMS}
    secondary = {s: 100.0 for s in STREAMS}
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda=primary,
        skeleton_lambda=skeleton,
        case1_secondary_recovered_lambda=secondary,
    )
    assert report["probe_ok"] is True
    assert report["linf"] == pytest.approx(0.0, abs=1e-12)
    assert report["secondary_recovered_is_not_gate"] is True
    assert report["case1_secondary_recovered_lambda"] is not None
    assert report["secondary_gap_diagnostic"] is not None
    assert report["secondary_gap_diagnostic"]["linf"] == pytest.approx(99.0, abs=1e-12)
    # Secondary gap does not affect probe_ok (PRIMARY aligned)
    assert report["package_dual_gate"] == "online_lambda"


def test_critical_blockers_still_present_after_probe_ok():
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    assert report["probe_ok"] is True
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert report["dual_linf_under_wire_unproven_blocker_still_true"] is True
    assert report["wire_not_shipped_blocker_still_true"] is True
    assert "form_label_change_required" in blockers
    assert "isolation_rewrite_required" not in blockers


def test_alias_and_no_tf_required():
    a = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    b = tlb.case1_dual_space_linf_probe(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    c = tlb.multi_unit_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] is True and b["ok"] is True and c["ok"] is True
    assert "tf_available" in a  # reported, not required


def test_source_does_not_import_pulp_tensorflow_or_excel_pipeline():
    src = inspect.getsource(tlb.offline_case1_dual_space_linf_probe_report)
    src2 = inspect.getsource(tlb.case1_dual_space_stream_aligned_linf)
    src3 = inspect.getsource(tlb.extract_case1_shaped_skeleton_lambda)
    src4 = inspect.getsource(tlb._case1_dual_space_linf_probe_honesty_fields)
    src5 = inspect.getsource(tlb.case1_primary_online_lambda_fixture)
    blob = src + src2 + src3 + src4 + src5
    assert "import pulp" not in blob
    assert "import tensorflow" not in blob
    assert "from pulp" not in blob
    assert "from tensorflow" not in blob
    # Import statements only — docstrings may mention excel_pipeline as a ban.
    assert "import excel_pipeline" not in blob
    assert "from excel_pipeline" not in blob
    assert "from pims_admm_llm.models.excel_pipeline" not in blob
    assert "from .excel_pipeline" not in blob
    assert "from pims_admm_llm.models import excel_pipeline" not in blob


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_probe=True,
        include_admm_case1_dual_space_form_contract=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_dual_space_linf_probe_ok" in rep
    assert rep["admm_case1_dual_space_linf_probe_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_probe_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_space_linf_probe=False,
        include_admm_case1_dual_space_form_contract=False,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert rep["admm_case1_dual_space_linf_probe_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_probe_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
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
    assert pf.get("admm_case1_dual_space_linf_probe_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_contract_and_skeleton_still_green():
    c = tlb.offline_case1_dual_space_form_contract_report()
    assert c["ok"] is True
    assert c["dual_linf_under_wire_status"] == "unproven"
    s = tlb.offline_case1_shaped_cdu_blender_linking_report(n_rounds=1)
    assert s["ok"] is True
    assert s["dual_recovery_path"] is None


def test_honesty_metadata_mentions_probe():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_space_linf_probe_available") is True
    assert meta["dual_recovery_path"] is None
    note = (meta.get("note") or "").lower()
    assert "linf" in note or "l∞" in note or "dual-space" in note


def test_exports_in_all():
    for name in (
        "CASE1_DUAL_SPACE_LINF_PROBE_KIND",
        "CASE1_FIXTURE_PRIMARY_ONLINE_LAMBDA",
        "CASE1_FIXTURE_SECONDARY_RECOVERED_LAMBDA",
        "CASE1_DUAL_VECTOR_FACE_PRIMARY_ONLINE",
        "case1_primary_online_lambda_fixture",
        "case1_secondary_recovered_lambda_fixture",
        "case1_dual_space_stream_aligned_linf",
        "stream_aligned_dual_linf",
        "extract_case1_shaped_skeleton_lambda",
        "case1_shaped_skeleton_lambda",
        "offline_case1_dual_space_linf_probe_report",
        "case1_dual_space_linf_probe",
        "multi_unit_case1_dual_space_linf_probe_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_no_excel_blender_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_blender_matrix_matches_affine")
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    assert report["excel_cdu_matrix_matches_affine"] is None
    assert report["excel_blender_matrix_matches_affine"] is None


def test_note_language_hard_negatives():
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda={s: 0.0 for s in STREAMS},
        skeleton_lambda={s: 0.0 for s in STREAMS},
    )
    note = (report.get("note") or "").lower()
    assert "probe" in note or "l∞" in note or "linf" in note
    assert "unproven" in note
    assert report["dual_recovery_path"] is None
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden


def test_misaligned_probe_not_ok():
    primary = {s: 1.0 for s in STREAMS}
    skeleton = {s: 1.0 for s in STREAMS if s != "naphtha"}
    report = tlb.offline_case1_dual_space_linf_probe_report(
        case1_primary_online_lambda=primary,
        skeleton_lambda=skeleton,
    )
    assert report["stream_alignment_ok"] is False
    assert report["probe_ok"] is False
    assert report["dual_linf_under_wire_status"] == "unproven"
