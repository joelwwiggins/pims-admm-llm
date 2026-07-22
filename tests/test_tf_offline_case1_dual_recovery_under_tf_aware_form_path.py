"""E1/E2: dual recovery under TF-aware form path policy (PR-B post-#83).

Locks:
- path_spec_present True; path_active_today False
- dual_recovery_path today None; module DUAL_RECOVERY_PATH None
- planned label online_lambda_under_tf_aware_form_when_shipped
- activates only wire_shipped AND flag (neither alone)
- never pure-admm / mono-oracle
- wire_shipped False; flag False; dual_linf unproven
- path policy ≠ wire ship ≠ dual_linf proof ≠ VERDICT
"""

from __future__ import annotations

import pytest

from pims_admm_llm.models import tf_linear_blocks as tlb


@pytest.fixture(autouse=True)
def _clear_coeffs_cache():
    tlb.clear_offline_unit_coeffs_cache()
    yield
    tlb.clear_offline_unit_coeffs_cache()


PLANNED = "online_lambda_under_tf_aware_form_when_shipped"


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_recovery_under_tf_aware_form_path_report()
    assert report["kind"] == "offline_case1_dual_recovery_under_tf_aware_form_path"
    assert report["path_spec_present"] is True
    assert report["path_active_today"] is False
    assert report["ok"] is True
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["dual_recovery_path_today"] is None
    assert report["dual_recovery_path_module_constant"] is None
    assert report["dual_recovery_path_planned_when_shipped"] == PLANNED
    assert report["wire_shipped"] is False
    assert report["wire_shipped_today"] is False
    assert report["feature_flag_enabled_today"] is False
    assert report["feature_flag_name"] == "enable_tf_affine_case1_wire"
    assert report["path_shipped"] is True
    assert report["form_label_change_shipped"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["bundle_shipped"] is False
    assert report["not_pure_admm"] is True
    assert report["not_mono_oracle_injection_into_pure_admm"] is True
    assert report["path_is_not_wire_ship"] is True
    assert report["path_is_not_dual_linf_proof"] is True
    assert report["path_is_not_flag_enable"] is True


def test_activation_policy_matrix():
    assert tlb.case1_tf_surface_dual_recovery_path() is None
    assert tlb.case1_tf_surface_dual_recovery_path(
        wire_shipped=False, flag_enabled=False
    ) is None
    assert tlb.case1_tf_surface_dual_recovery_path(
        wire_shipped=True, flag_enabled=False
    ) is None
    assert tlb.case1_tf_surface_dual_recovery_path(
        wire_shipped=False, flag_enabled=True
    ) is None
    assert (
        tlb.case1_tf_surface_dual_recovery_path(
            wire_shipped=True, flag_enabled=True
        )
        == PLANNED
    )
    report = tlb.offline_case1_dual_recovery_under_tf_aware_form_path_report()
    assert report["dual_recovery_path_when_wire_and_flag"] == PLANNED
    assert report["dual_recovery_path_when_flag_only"] is None
    assert report["dual_recovery_path_when_wire_only"] is None
    assert report["dual_recovery_path_when_neither"] is None


def test_flag_monkeypatch_alone_does_not_activate(monkeypatch):
    monkeypatch.setattr(
        tlb, "CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY", True
    )
    assert tlb.case1_tf_affine_wire_flag_enabled() is True
    assert tlb.case1_wire_shipped_today() is False
    assert tlb.case1_tf_surface_dual_recovery_path() is None
    assert tlb.DUAL_RECOVERY_PATH is None
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None


def test_wire_monkeypatch_alone_does_not_activate(monkeypatch):
    monkeypatch.setattr(tlb, "CASE1_WIRE_SHIPPED_TODAY", True)
    assert tlb.case1_wire_shipped_today() is True
    assert tlb.case1_tf_affine_wire_flag_enabled() is False
    assert tlb.case1_tf_surface_dual_recovery_path() is None


def test_both_gates_activate_planned_label(monkeypatch):
    monkeypatch.setattr(tlb, "CASE1_WIRE_SHIPPED_TODAY", True)
    monkeypatch.setattr(
        tlb, "CASE1_DUAL_HONEST_TF_AWARE_PATH_FEATURE_FLAG_ENABLED_TODAY", True
    )
    assert tlb.case1_tf_surface_dual_recovery_path() == PLANNED
    # Module constant stays dual-banned None even when policy would activate.
    assert tlb.DUAL_RECOVERY_PATH is None
    meta = tlb.honesty_metadata()
    assert meta["dual_recovery_path"] is None


def test_forbidden_labels_exclude_pure_admm_and_mono():
    forbidden = tlb.CASE1_DUAL_RECOVERY_UNDER_TF_AWARE_FORM_FORBIDDEN_LABELS
    assert "pure-admm" in forbidden
    assert "mono-oracle" in forbidden
    assert PLANNED not in forbidden
    assert "pure" not in PLANNED.lower()


def test_spec_and_aliases():
    spec = tlb.case1_dual_recovery_under_tf_aware_form_path_spec()
    assert spec["path_spec_present"] is True
    assert spec["path_active_today"] is False
    assert spec["dual_recovery_path_planned_when_shipped"] == PLANNED
    assert spec["activation_policy"] == "wire_shipped_and_flag_enabled"
    a = tlb.case1_dual_recovery_under_tf_aware_form_path_report()
    b = tlb.multi_unit_case1_dual_recovery_under_tf_aware_form_path_report()
    assert a["ok"] is True and b["ok"] is True
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_recovery_under_tf_aware_form_path_available") is True
    assert "case1_tf_surface_dual_recovery_path" in tlb.__all__
    assert "CASE1_WIRE_SHIPPED_TODAY" in tlb.__all__


def test_does_not_advance_dual_linf_or_first_blocking():
    report = tlb.offline_case1_dual_recovery_under_tf_aware_form_path_report()
    assert report["ok"] is True
    assert tlb.CASE1_DUAL_LINF_UNDER_WIRE_STATUS == "unproven"
    assert tlb.CASE1_WIRE_SHIPPED_TODAY is False
    fb = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_first_blocking_coreq()
    assert fb["first_blocking_coreq"] == "dual_linf_under_wire_proven"
    crit = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert crit["dual_linf_proof_allowed_today"] is False
    assert crit["wire_shipped"] is False
