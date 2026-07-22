"""E1/E2: dual-honest offline wire-preflight report (compose + wire_blockers).

Always-on sections run without TensorFlow. Locks:
- offline_wire_preflight_report always-on; dual_recovery_path is None
- wire_shipped is False; not Case 1; not pure-ADMM dual recovery; not full plant MB
- wire_blockers non-empty with critical honesty ids true at HEAD
- composes readiness (parity/priced/timings/honesty + additive admm_* flags)
- ready_for_wire_discussion meaning unchanged (parity∧priced∧timings∧honesty)
- preflight_ok / blockers_documented separate from ready (not AND-ed into ready)
- no residual-must-vanish SLA; no ρ retune; no invent excel_cdu_matrix_matches_affine
- no live excel→tf preflight as primary (isolation stays green)

Regression list (run separately in CI / implementer validation):
  test_tf_import_isolation, test_tf_offline_registry, test_tf_offline_priced,
  test_tf_offline_timing, test_tf_offline_admm_residual,
  test_tf_offline_admm_block_subproblem, test_tf_offline_admm_coordination,
  test_tf_offline_admm_plant_linking,
  test_tf_linear_block, test_tf_linear_coker, test_tf_linear_cdu,
  test_excel_pipeline, test_api_excel
  EMRPS optional-only (not required for this gate).
"""

from __future__ import annotations

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


def test_offline_wire_blocker_catalog_stable_ids():
    cat = tlb.offline_wire_blocker_catalog()
    assert cat["kind"] == "offline_wire_blocker_catalog"
    assert cat["wire_shipped"] is False
    assert cat["dual_recovery_path"] is None
    assert cat["on_excel_case1_path"] is False
    assert cat["solver"] is False
    blockers = cat["wire_blockers"]
    assert isinstance(blockers, list) and len(blockers) >= 6
    assert CRITICAL_BLOCKERS.issubset(set(blockers))
    notes = cat["wire_blocker_notes"]
    for bid in CRITICAL_BLOCKERS:
        assert bid in notes
        assert isinstance(notes[bid], str) and len(notes[bid]) > 10
    # catalog matches module constants
    assert list(tlb.DEFAULT_WIRE_BLOCKERS) == blockers
    assert set(tlb.WIRE_BLOCKER_NOTES) >= CRITICAL_BLOCKERS


def test_offline_wire_preflight_report_honesty_and_blockers():
    report = tlb.offline_wire_preflight_report(
        readiness_n_repeats=12,
        readiness_warmup=1,
        include_box=True,
    )
    assert report["kind"] == tlb.WIRE_PREFLIGHT_KIND
    assert report["kind"] == "offline_wire_preflight"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["on_excel_case1_path"] is False
    assert report["not_case1_solve"] is True
    assert report["wire_shipped"] is False
    assert report["not_wire_shipped"] is True
    assert report["not_full_plant_mass_balance"] is True
    assert report["not_pure_admm_dual_recovery"] is True
    assert report["preflight_lambda_is_not_case1_online_lambda"] is True
    assert report["preflight_is_not_case1_primary_or_secondary_duals"] is True
    assert report["plant_linking_lambda_is_not_case1_online_lambda"] is True

    blockers = report["wire_blockers"]
    assert isinstance(blockers, list) and len(blockers) > 0
    assert CRITICAL_BLOCKERS.issubset(set(blockers))
    assert report["n_wire_blockers"] == len(blockers)
    assert report["blockers_documented"] is True
    notes = report["wire_blocker_notes"]
    for bid in CRITICAL_BLOCKERS:
        assert bid in notes

    assert report["ok"] is True, report
    assert report["preflight_ok"] is True
    assert report["compose_ok"] is True
    assert report["honesty_locks_ok"] is True
    assert report["ready_semantics_ok"] is True

    note = (report.get("note") or "").lower()
    assert "preflight" in note
    assert "wire" in note
    assert "not" in note
    # Must not claim dual recovery ownership
    for forbidden in ("online_lambda", "recovered_blender", "pure_admm"):
        assert report["dual_recovery_path"] != forbidden


def test_preflight_composes_readiness_gates_without_redefining_ready():
    report = tlb.offline_wire_preflight_report(
        readiness_n_repeats=12,
        readiness_warmup=1,
    )
    assert report["parity_ok"] is True
    assert report["priced_ok"] is True
    assert report["timings_ok"] is True
    assert report["honesty_ok"] is True
    # Structural ready still True when ladder green — blockers do NOT flip it
    assert report["ready_for_wire_discussion"] is True
    expected = bool(
        report["parity_ok"]
        and report["priced_ok"]
        and report["timings_ok"]
        and report["honesty_ok"]
    )
    assert report["ready_for_wire_discussion"] is expected

    assert report["admm_residual_ok"] is True
    assert report["admm_block_subproblem_ok"] is True
    assert report["admm_coordination_ok"] is True
    assert report["admm_plant_linking_ok"] is True
    assert report["admm_plant_named_linking_ok"] is True

    readiness = report["readiness"]
    assert readiness["kind"] == "offline_block_solve_readiness"
    assert readiness["ready_for_wire_discussion"] is report["ready_for_wire_discussion"]
    assert readiness["dual_recovery_path"] is None
    assert readiness["on_excel_case1_path"] is False
    assert readiness["solver"] is False


def test_preflight_skips_optional_gates_when_disabled():
    report = tlb.offline_wire_preflight_report(
        readiness_n_repeats=8,
        readiness_warmup=0,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
        include_admm_coordination=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
    )
    assert report["admm_residual_ok"] is None
    assert report["admm_block_subproblem_ok"] is None
    assert report["admm_coordination_ok"] is None
    assert report["admm_plant_linking_ok"] is None
    assert report["admm_plant_named_linking_ok"] is None
    # Structural ready still from parity/priced/timings/honesty
    assert report["ready_for_wire_discussion"] is True
    assert report["wire_shipped"] is False
    assert report["blockers_documented"] is True
    assert report["ok"] is True


def test_multi_unit_wire_preflight_alias():
    a = tlb.offline_wire_preflight_report(readiness_n_repeats=5, readiness_warmup=0)
    b = tlb.multi_unit_wire_preflight_report(readiness_n_repeats=5, readiness_warmup=0)
    assert a["kind"] == b["kind"]
    assert a["wire_blockers"] == b["wire_blockers"]
    assert a["wire_shipped"] is False and b["wire_shipped"] is False
    assert a["dual_recovery_path"] is None and b["dual_recovery_path"] is None


def test_honesty_metadata_mentions_wire_preflight():
    meta = tlb.honesty_metadata()
    assert meta.get("wire_preflight_available") is True
    assert meta["dual_recovery_path"] is None
    assert meta["on_excel_case1_path"] is False
    assert meta["solver"] is False
    note = (meta.get("note") or "").lower()
    assert "preflight" in note or "wire_blockers" in note or "wire preflight" in note


def test_exports_in_all():
    for name in (
        "WIRE_PREFLIGHT_KIND",
        "DEFAULT_WIRE_BLOCKERS",
        "WIRE_BLOCKER_NOTES",
        "offline_wire_blocker_catalog",
        "offline_wire_preflight_report",
        "multi_unit_wire_preflight_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)


def test_no_excel_cdu_matrix_matches_affine_invented():
    assert not hasattr(tlb, "excel_cdu_matrix_matches_affine")


def test_ready_not_and_with_blockers_at_head():
    """Critical semantic lock: blockers must not force ready_for_wire_discussion False.

    At HEAD, structural ladder is green so ready is True even though wire_blockers
    is non-empty and wire_shipped is False.
    """
    report = tlb.offline_wire_preflight_report(readiness_n_repeats=8, readiness_warmup=0)
    assert len(report["wire_blockers"]) > 0
    assert report["wire_shipped"] is False
    assert report["ready_for_wire_discussion"] is True
    assert report["preflight_ok"] is True
    # preflight_ok documents blockers; ready remains structural
    assert report["blockers_documented"] is True
    assert report["ready_for_wire_discussion"] is not (
        report["wire_shipped"]
    )  # ready != wire_shipped
