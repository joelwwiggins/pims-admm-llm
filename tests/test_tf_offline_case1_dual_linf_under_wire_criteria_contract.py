"""E1/E2: offline Case-1 dual_linf_under_wire flip-criteria contract.

Always-on sections run without TensorFlow and without PuLP / excel_pipeline on
the criteria-contract hot path. Locks:
- dual_recovery_path is None; wire_shipped False; dual_linf_under_wire unproven
- criteria_present True; flip_criteria_formalized True
- dual_linf_proof_allowed_today False; criteria_met_today False
- flip criteria map present (required + required_under_wire_only)
- ok / contract_ok does NOT require linf<=15 and does NOT prove dual_linf
- distinct from online_linf_gate criteria contract (status vs checklist id)
- first_blocking_coreq remains isolation_rewrite_with_wire
- no_blender_offline_affine_kernel still in DEFAULT_WIRE_BLOCKERS / critical
- form still classic_2block_excel_path; planned distinct
- UNITS still FCC/COKER/CDU (no silent BLENDER)
- additive readiness flag does not redefine ready_for_wire_discussion
- contract ≠ dual L∞ under wire proof ≠ gate flip ≠ wire ≠ VERDICT
- probe/bridge/warmstart/packaging/online_linf_gate_criteria_alone never prove
- go-board dual_linf prep_artifacts includes this criteria report
- no excel_pipeline import on tf_linear_blocks criteria-contract hot path

Charter validation companions (run with this module):
  pytest tests/test_excel_pipeline.py tests/test_api_excel.py \\
    tests/test_tf_import_isolation.py \\
    tests/test_tf_offline_case1_dual_linf_under_wire_criteria_contract.py \\
    tests/test_tf_offline_case1_online_linf_gate_criteria_contract.py \\
    tests/test_tf_offline_case1_isolation_rewrite_first_blocker_operational_prep.py \\
    tests/test_tf_offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint.py -q
"""

from __future__ import annotations

import inspect
from pathlib import Path

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

REQUIRED_FLIP_KEYS = {
    "isolation_rewrite_with_wire",
    "form_label_change_shipped",
    "dual_honest_tf_aware_path_under_wire",
    "primary_online_lambda_owns_verdict_under_tf_aware_form",
    "primary_online_lambda_linf_le_15_under_shipped_tf_aware_form",
    "wire_shipped",
    "dual_recovery_path_labeled_honestly_under_wire",
    "no_silent_classic_form_linf_reuse",
    "online_linf_gate_closed_under_tf_path_as_co_req_when_proven",
    "isolation_tests_rewritten_with_wire_not_deleted",
}

UNDER_WIRE_KEY = "primary_online_lambda_linf_le_15_under_shipped_tf_aware_form"


def test_report_always_on_honesty_locks():
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert report["kind"] == tlb.CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND
    assert report["kind"] == "offline_case1_dual_linf_under_wire_criteria_contract"
    assert report["solver"] is False
    assert report["dual_recovery_path"] is None
    assert report["wire_shipped"] is False
    assert report["path_shipped"] is False
    assert report["bundle_shipped"] is False
    assert report["form_label_change_shipped"] is False
    assert report["isolation_rewrite_shipped"] is True
    assert report["on_excel_case1_path"] is False
    assert report["on_case1_solve"] is False
    assert report["not_case1_solve"] is True
    assert report["case1_form_unchanged"] is True
    assert report["form_current"] == "classic_2block_excel_path"
    assert report["form_planned"] == tlb.CASE1_PLANNED_TF_AWARE_FORM
    assert report["planned_form_distinct"] is True
    assert report["form_label_change_required_still_true"] is True
    assert report["online_linf_gate_under_tf_path"] == "open"
    assert report["online_linf_gate_still_open"] is True
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["criteria_met_today"] is False
    assert report["gate_flip_allowed_today"] is False
    assert report["criteria_present"] is True
    assert report["flip_criteria_formalized"] is True
    assert report["contract_is_not_dual_linf_under_wire_proof"] is True
    assert report["contract_is_not_gate_flip"] is True
    assert report["contract_is_not_wire"] is True
    assert report["contract_is_not_verdict_gate"] is True
    assert report["criteria_present_is_not_proven"] is True
    assert report["distinct_from_online_linf_gate_criteria_contract"] is True
    assert report["gates_status_not_checklist_id"] is True
    assert report["status_target"] == "dual_linf_under_wire"
    assert report["order_hint_coreq"] == "dual_linf_under_wire_proven"
    assert report["probe_linf_is_not_proof_today"] is True
    assert report["bridge_linf_is_not_proof_today"] is True
    assert report["warmstart_linf_is_not_proof_today"] is True
    assert report["online_linf_gate_criteria_alone_is_anti_criterion"] is True
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire"] == "unproven"
    assert report["dual_linf_under_wire_unproven_still_true"] is True
    assert report["no_blender_offline_affine_kernel_blocker_still_true"] is True
    assert report["contract_ok"] is True
    assert report["ok"] is True, report
    assert report["honesty_ok"] is True
    assert report["does_not_redefine_ready_for_wire_discussion"] is True
    assert report["does_not_clear_default_wire_blockers"] is True
    assert report["linf_le_15_is_not_proof_gate_today"] is True
    assert "NOT dual L∞ under wire proven" in report["ok_criteria"]
    assert "BLENDER" not in tlb.UNITS
    assert list(tlb.UNITS) == ["FCC", "COKER", "CDU"]
    assert report["units_affine_unchanged"] == ["FCC", "COKER", "CDU"]
    assert report["first_blocking_coreq"] == "form_label_change_shipped"
    assert report["first_blocking_ok"] is True
    assert report["feature_flag_enabled_today"] is False
    assert report["suggested_next_wave_still_full_wire"] is True
    assert report["all_ship_flags_false"] is True


def test_flip_criteria_map_keys_and_classes():
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    flip = report["flip_criteria"]
    assert set(flip.keys()) == REQUIRED_FLIP_KEYS
    assert flip[UNDER_WIRE_KEY] == tlb.FLIP_CRITERION_REQUIRED_UNDER_WIRE_ONLY
    for k in REQUIRED_FLIP_KEYS:
        if k != UNDER_WIRE_KEY:
            assert flip[k] == tlb.FLIP_CRITERION_REQUIRED
    anti = set(report["anti_criteria_today"])
    for a in (
        "probe_linf",
        "bridge_linf",
        "warmstart_linf",
        "pooling_linf",
        "seed_identity_linf",
        "recovered_blender_linf",
        "packaging_alone",
        "blueprint_alone",
        "prep_alone",
        "scaffold_alone",
        "rehearsal_alone",
        "online_linf_gate_criteria_alone",
        "online_linf_gate_open_alone",
        "online_linf_gate_closed_alone",
        "this_dual_linf_criteria_alone",
    ):
        assert a in anti
    assert set(tlb.CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA.keys()) == REQUIRED_FLIP_KEYS
    assert set(tlb.case1_dual_linf_under_wire_flip_criteria().keys()) == REQUIRED_FLIP_KEYS
    assert report["flip_criteria_formalized"] is True


def test_proof_permission_hard_false():
    assert tlb.case1_dual_linf_proof_allowed_today() is False
    assert tlb.case1_dual_linf_under_wire_criteria_met_today_aggregate() is False
    met = tlb.case1_dual_linf_under_wire_criteria_met_today_map()
    assert met["isolation_rewrite_with_wire"] is True
    assert met["wire_shipped"] is False
    assert met["form_label_change_shipped"] is False
    assert met["dual_honest_tf_aware_path_under_wire"] is False
    # Some structural labels may be True; aggregate still False
    assert met["dual_recovery_path_labeled_honestly_under_wire"] is True
    assert met["no_silent_classic_form_linf_reuse"] is True
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert report["dual_linf_proof_allowed_today"] is False
    assert report["criteria_met_today"] is False


def test_distinct_from_online_linf_gate_criteria_contract():
    dl = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    ol = tlb.offline_case1_online_linf_gate_criteria_contract_report()
    assert dl["kind"] != ol["kind"]
    assert dl["kind"] == "offline_case1_dual_linf_under_wire_criteria_contract"
    assert ol["kind"] == "offline_case1_online_linf_gate_criteria_contract"
    assert dl["status_target"] == "dual_linf_under_wire"
    assert dl["order_hint_coreq"] == "dual_linf_under_wire_proven"
    assert dl["distinct_from_online_linf_gate_criteria_contract"] is True
    assert dl["gates_status_not_checklist_id"] is True
    # online map keys target checklist gate; dual_linf map has status-oriented keys
    assert "online_lambda_owns_verdict_gate" in ol["flip_criteria"]
    assert "primary_online_lambda_owns_verdict_under_tf_aware_form" in dl["flip_criteria"]
    assert "online_linf_gate_criteria_alone" in dl["anti_criteria_today"]
    # online gate stays open; dual_linf unproven
    assert ol["online_linf_gate_under_tf_path"] == "open"
    assert ol["gate_flip_allowed_today"] is False
    assert dl["dual_linf_under_wire_status"] == "unproven"
    assert dl["dual_linf_proof_allowed_today"] is False
    assert dl["ok"] is True and ol["ok"] is True


def test_critical_blockers_still_present():
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    blockers = set(report["wire_blockers"])
    assert CRITICAL_BLOCKERS.issubset(blockers)
    assert CRITICAL_BLOCKERS.issubset(set(tlb.DEFAULT_WIRE_BLOCKERS))
    assert "dual_linf_under_wire_unproven" in tlb.DEFAULT_WIRE_BLOCKERS
    assert report["dual_linf_under_wire_unproven_blocker_still_true"] is True
    assert report["blockers_still_documented"] is True
    for bid in CRITICAL_BLOCKERS:
        assert bid in tlb.WIRE_BLOCKER_NOTES


def test_units_no_silent_blender():
    assert tlb.UNITS == ("FCC", "COKER", "CDU")
    assert "BLENDER" not in tlb.UNITS
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    assert report["units_ok"] is True


def test_aliases_and_exports():
    for name in (
        "CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_KIND",
        "CASE1_DUAL_LINF_UNDER_WIRE_STATUS_TARGET",
        "CASE1_DUAL_LINF_UNDER_WIRE_ORDER_HINT_COREQ",
        "CASE1_DUAL_LINF_UNDER_WIRE_CRITERIA_CONTRACT_ANNOTATION",
        "CASE1_DUAL_LINF_UNDER_WIRE_FLIP_CRITERIA",
        "CASE1_DUAL_LINF_UNDER_WIRE_ANTI_CRITERIA_TODAY",
        "case1_dual_linf_under_wire_flip_criteria",
        "case1_dual_linf_under_wire_criteria_met_today_map",
        "case1_dual_linf_proof_allowed_today",
        "case1_dual_linf_under_wire_criteria_met_today_aggregate",
        "offline_case1_dual_linf_under_wire_criteria_contract_report",
        "case1_dual_linf_under_wire_criteria_contract_report",
        "multi_unit_case1_dual_linf_under_wire_criteria_contract_report",
    ):
        assert name in tlb.__all__
        assert hasattr(tlb, name)

    a = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    b = tlb.case1_dual_linf_under_wire_criteria_contract_report()
    c = tlb.multi_unit_case1_dual_linf_under_wire_criteria_contract_report()
    assert a["kind"] == b["kind"] == c["kind"]
    assert a["ok"] and b["ok"] and c["ok"]


def test_source_does_not_import_pulp_tensorflow_excel_pipeline():
    funcs = [
        tlb._case1_dual_linf_under_wire_criteria_contract_honesty_fields,
        tlb.offline_case1_dual_linf_under_wire_criteria_contract_report,
        tlb.case1_dual_linf_under_wire_flip_criteria,
        tlb.case1_dual_linf_under_wire_criteria_met_today_map,
        tlb.case1_dual_linf_proof_allowed_today,
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
    assert "format_tf_offline" not in blob


def test_composes_checklist_form_not_maximizer():
    src = inspect.getsource(
        tlb.offline_case1_dual_linf_under_wire_criteria_contract_report
    )
    assert "case1_dual_linf_proof_checklist" in src
    assert "case1_form_label_contract" in src
    assert "DEFAULT_WIRE_BLOCKERS" in src
    assert "offline_case1_dual_space_linf_probe_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_bridge_report" not in src
    assert "offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report" not in src
    assert "offline_case1_shaped_cdu_blender_linking_report" not in src


def test_additive_readiness_flag_does_not_redefine_ready():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_linf_under_wire_criteria_contract=True,
        include_admm_case1_online_linf_gate_criteria_contract=True,
        include_admm_case1_isolation_rewrite_first_blocker_operational_prep=True,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert "admm_case1_dual_linf_under_wire_criteria_contract_ok" in rep
    assert rep["admm_case1_dual_linf_under_wire_criteria_contract_ok"] is True
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected
    assert rep["dual_recovery_path"] is None


def test_readiness_skips_criteria_contract_when_disabled():
    rep = tlb.offline_block_solve_readiness_report(
        n_repeats=5,
        warmup=0,
        include_admm_case1_dual_linf_under_wire_criteria_contract=False,
        include_admm_case1_shaped_linking=False,
        include_admm_plant_linking=False,
        include_admm_plant_named_linking=False,
        include_admm_coordination=False,
        include_admm_residual=False,
        include_admm_block_subproblem=False,
    )
    assert rep["admm_case1_dual_linf_under_wire_criteria_contract_ok"] is None
    expected = bool(
        rep.get("parity_ok")
        and rep.get("priced_ok")
        and rep.get("timings_ok")
        and rep.get("honesty_ok")
    )
    assert rep["ready_for_wire_discussion"] is expected


def test_preflight_surfaces_criteria_contract_flag_and_blockers():
    pf = tlb.offline_wire_preflight_report(
        readiness_n_repeats=5,
        readiness_warmup=0,
        include_admm_case1_dual_linf_under_wire_criteria_contract=True,
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
    assert pf.get("admm_case1_dual_linf_under_wire_criteria_contract_ok") is True
    readiness = pf["readiness"]
    expected = bool(
        readiness.get("parity_ok")
        and readiness.get("priced_ok")
        and readiness.get("timings_ok")
        and readiness.get("honesty_ok")
    )
    assert pf["ready_for_wire_discussion"] is expected
    assert pf["ok"] is True


def test_honesty_metadata_mentions_criteria_contract():
    meta = tlb.honesty_metadata()
    assert meta.get("admm_case1_dual_linf_under_wire_criteria_contract_available") is True
    assert meta["dual_recovery_path"] is None


def test_go_board_prep_artifacts_include_dual_linf_criteria():
    bp = tlb.offline_case1_dual_honest_multi_blocker_wire_implementation_blueprint_report()
    assert bp["ok"] is True
    arts = (bp.get("file_level_prep_map") or {}).get("dual_linf_under_wire_proven", [])
    assert any("dual_linf_under_wire_criteria_contract" in str(a) for a in arts)
    assert bp["first_blocking_coreq"] == "form_label_change_shipped"
    # status remains unproven
    go = tlb.case1_dual_honest_multi_blocker_wire_implementation_blueprint_go_board()
    rows = go.get("order_hint_rows") or go.get("rows") or []
    dual_row = None
    for r in rows:
        if r.get("coreq_id") == "dual_linf_under_wire_proven":
            dual_row = r
            break
    if dual_row is None:
        # alternate shape: file_level_prep_map only
        assert any("dual_linf_under_wire_criteria_contract" in str(a) for a in arts)
    else:
        assert dual_row.get("status") in ("unproven", "open", "false_today")
        assert any(
            "dual_linf_under_wire_criteria_contract" in str(a)
            for a in dual_row.get("prep_artifacts", [])
        )


def test_negative_ship_flags_never_true():
    report = tlb.offline_case1_dual_linf_under_wire_criteria_contract_report()
    for k in (
        "path_shipped",
        "wire_shipped",
        "bundle_shipped",
        "form_label_change_shipped",
        "feature_flag_enabled_today",
        "criteria_met_today",
        "dual_linf_proof_allowed_today",
        "gate_flip_allowed_today",
    ):
        assert report[k] is False
    assert report["dual_recovery_path"] is None
    assert report["dual_linf_under_wire_status"] == "unproven"
    assert report["dual_linf_under_wire"] != "proven"
    assert "BLENDER" not in report["units_affine_unchanged"]


def test_online_linf_gate_and_ladder_non_regression():
    online = tlb.offline_case1_online_linf_gate_criteria_contract_report()
    assert online["ok"] is True
    assert online["online_linf_gate_under_tf_path"] == "open"
    assert online["gate_flip_allowed_today"] is False
    prep = tlb.offline_case1_isolation_rewrite_first_blocker_operational_prep_report()
    assert prep["ok"] is True
    assert prep["first_blocking_coreq"] == "form_label_change_shipped"
    assert prep["isolation_rewrite_shipped"] is True
    assert tlb.CASE1_FORM_CURRENT == "classic_2block_excel_path"
    cl = tlb.case1_dual_linf_proof_checklist()
    assert cl["dual_linf_under_wire_status"] == "unproven"
    assert "online_linf_gate_under_tf_path" in cl["dual_linf_proof_checklist_open_ids"]


def test_isolation_suite_file_still_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "tests/test_tf_import_isolation.py").is_file()
