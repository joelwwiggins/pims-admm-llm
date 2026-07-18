#!/usr/bin/env python3
"""Excel PIMS-shaped → mono + ADMM → results Excel MVP demo.

Usage (repo root):
  source .venv/bin/activate
  export PYTHONPATH=src
  python -m demos.run_excel_pipeline_demo
  python -m demos.run_excel_pipeline_demo --input data/assays/crudes_template.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Excel PIMS → ADMM pipeline MVP")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="PIMS-shaped .xlsx (default: regenerate template from JSON assays)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "demos" / "output",
        help="Directory for results xlsx/json",
    )
    args = parser.parse_args(argv)

    from pims_admm_llm.models.excel_pipeline import ensure_template, run_excel_pipeline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.input is None:
        template = _REPO / "data" / "assays" / "crudes_template.xlsx"
        ensure_template(template)
        input_path = template
        print(f"Regenerated template: {template}")
    else:
        input_path = args.input
        if not input_path.is_file():
            print(f"ERROR: input not found: {input_path}", file=sys.stderr)
            return 2

    xlsx_out = args.out_dir / "excel_pipeline_results.xlsx"
    json_out = args.out_dir / "excel_pipeline_results.json"

    print("=" * 72)
    print("EXCEL PIMS → MONO + ADMM PIPELINE")
    print("=" * 72)
    print(f"Input:  {input_path}")
    print(f"Output: {xlsx_out}")
    print()

    report = run_excel_pipeline(
        input_path,
        results_xlsx=xlsx_out,
        results_json=json_out,
    )
    mono = report["mono"]
    admm = report["admm"]
    cmp_ = report["comparison"]
    meta = report["meta"]

    print(f"Crudes: {meta['n_crudes']}  CDU cap: {meta['cdu_capacity_kbd']} kbd")
    print(f"Mono:  status={mono['status']}  obj={mono['objective']:.6f}  "
          f"t={mono['wall_time_s']:.4f}s  feasible={mono['feasible']}")
    print(f"ADMM:  status={admm['status']}  obj={admm['objective']:.6f}  "
          f"iters={admm['iteration_count']}  t={admm['wall_time_s']:.4f}s  "
          f"rho={admm['rho']}  ||r||={admm['primal_residual']:.4g}")
    print(f"       dual_recovery_path={admm['dual_recovery_path']}")
    print(f"Gap:   abs={cmp_['objective_gap_abs']:.6f}  rel={cmp_['objective_gap_rel']:.6%}")
    print(
        f"Dual PRIMARY (online λ, gates VERDICT): L∞={cmp_.get('dual_linf_online')}  "
        f"[verdict_dual_gate=online_only]"
    )
    print(
        f"Dual SECONDARY (recovered blender face, not gate): L∞={cmp_.get('dual_linf_recovered')}  "
        f"[recovered_secondary=true]"
    )
    dual_strip = (meta.get("planner_honesty") or {}).get("dual_glance_strip")
    if dual_strip:
        print(f"Dual glance: {dual_strip}")
    ph = (meta.get("planner_honesty") or {})
    if ph.get("offline_tf_ladder_toc_ready"):
        print(
            "Offline TF ladder TOC: ready=true (How_to topic tf_offline_ladder_toc; "
            "ship=false dual-ban; blueprint_present≠wire ready; no Index growth)"
        )
    offline_units = ph.get("offline_tf_units") or "FCC,COKER,CDU"
    # Static readiness flags from meta only — never import tf_linear_blocks /
    # live residual, block subproblem, multi-round coordination, plant-linking,
    # plant-named, wire-preflight, Case-1-shaped skeleton, dual-space/form
    # contract, dual-space L∞ probe, dual-space L∞ live-λ bridge, dual-space
    # L∞ live-λ-seeded warm-start, honest blender pooling path, online_linf_gate
    # flip-criteria contract, isolation-rewrite design, form_label ship criteria,
    # isolation-rewrite ship criteria, multi-blocker bundle design, multi-blocker
    # bundle ship-met criteria, dual-honest path execution scaffold, rehearsal,
    # dual-honest multi-blocker wire implementation blueprint, or isolation
    # first-blocker operational prep, dual_linf_under_wire flip-criteria contract, or form_label second-coreq operational prep, path third-coreq operational prep, or dual_linf fourth-coreq operational prep reports.
    readiness_bits = []
    if ph.get("offline_tf_priced_ready"):
        readiness_bits.append("priced")
    if ph.get("offline_tf_timing_ready"):
        readiness_bits.append("timing")
    if ph.get("offline_tf_admm_residual_ready"):
        readiness_bits.append("admm_residual")
    if ph.get("offline_tf_admm_block_subproblem_ready"):
        readiness_bits.append("admm_block_subproblem")
    if ph.get("offline_tf_admm_coordination_ready"):
        readiness_bits.append("admm_coordination")
    if ph.get("offline_tf_admm_plant_linking_ready"):
        readiness_bits.append("admm_plant_linking")
    if ph.get("offline_tf_admm_plant_named_linking_ready"):
        readiness_bits.append("admm_plant_named_linking")
    if ph.get("offline_tf_wire_preflight_ready"):
        readiness_bits.append("wire_preflight")
    if ph.get("offline_tf_case1_shaped_linking_ready"):
        readiness_bits.append("case1_shaped_linking")
    if ph.get("offline_tf_case1_dual_space_form_contract_ready"):
        readiness_bits.append("case1_dual_space_form_contract")
    if ph.get("offline_tf_case1_dual_space_linf_probe_ready"):
        readiness_bits.append("case1_dual_space_linf_probe")
    if ph.get("offline_tf_case1_dual_space_linf_live_lambda_bridge_ready"):
        readiness_bits.append("case1_dual_space_linf_live_lambda_bridge")
    if ph.get("offline_tf_case1_dual_space_linf_live_lambda_seeded_warmstart_ready"):
        readiness_bits.append("case1_dual_space_linf_live_lambda_seeded_warmstart")
    if ph.get("offline_tf_case1_honest_blender_pooling_path_ready"):
        readiness_bits.append("case1_honest_blender_pooling_path")
    if ph.get("offline_tf_case1_online_linf_gate_criteria_contract_ready"):
        readiness_bits.append("case1_online_linf_gate_criteria_contract")
    if ph.get("offline_tf_case1_isolation_rewrite_design_contract_ready"):
        readiness_bits.append("case1_isolation_rewrite_design_contract")
    if ph.get("offline_tf_case1_wire_ship_acceptance_design_contract_ready"):
        readiness_bits.append("case1_wire_ship_acceptance_design_contract")
    if ph.get("offline_tf_case1_dual_honest_tf_aware_path_design_contract_ready"):
        readiness_bits.append("case1_dual_honest_tf_aware_path_design_contract")
    if ph.get("offline_tf_case1_dual_honest_tf_aware_path_present_criteria_contract_ready"):
        readiness_bits.append("case1_dual_honest_tf_aware_path_present_criteria_contract")
    if ph.get("offline_tf_case1_form_label_change_shipped_criteria_contract_ready"):
        readiness_bits.append("case1_form_label_change_shipped_criteria_contract")
    if ph.get("offline_tf_case1_isolation_rewrite_shipped_criteria_contract_ready"):
        readiness_bits.append("case1_isolation_rewrite_shipped_criteria_contract")
    if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ready"):
        readiness_bits.append("case1_dual_honest_multi_blocker_wire_bundle_design_contract")
    if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ready"):
        readiness_bits.append("case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract")
    if ph.get("offline_tf_case1_dual_honest_tf_aware_path_execution_scaffold_ready"):
        readiness_bits.append("case1_dual_honest_tf_aware_path_execution_scaffold")
    if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_rehearsal_ready"):
        readiness_bits.append("case1_dual_honest_multi_blocker_wire_rehearsal")
    if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ready"):
        readiness_bits.append("case1_dual_honest_multi_blocker_wire_implementation_blueprint")
    if ph.get("offline_tf_case1_isolation_rewrite_first_blocker_operational_prep_ready"):
        readiness_bits.append("case1_isolation_rewrite_first_blocker_operational_prep")
    if ph.get("offline_tf_case1_dual_linf_under_wire_criteria_contract_ready"):
        readiness_bits.append("case1_dual_linf_under_wire_criteria_contract")
    if ph.get("offline_tf_case1_form_label_second_coreq_operational_prep_ready"):
        readiness_bits.append("case1_form_label_second_coreq_operational_prep")
    if ph.get("offline_tf_case1_path_third_coreq_operational_prep_ready"):
        readiness_bits.append("case1_path_third_coreq_operational_prep")
    if ph.get("offline_tf_case1_dual_linf_fourth_coreq_operational_prep_ready"):
        readiness_bits.append("case1_dual_linf_fourth_coreq_operational_prep")
    if ph.get("offline_tf_case1_wire_fifth_coreq_operational_prep_ready"):
        readiness_bits.append("case1_wire_fifth_coreq_operational_prep")
    if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_ready"):
        readiness_bits.append("case1_dual_honest_multi_blocker_wire_bundle_operational_prep")
    if ph.get("offline_tf_case1_isolation_rewrite_first_blocker_execution_scaffold_ready"):
        readiness_bits.append("case1_isolation_rewrite_first_blocker_execution_scaffold")
    if ph.get("offline_tf_case1_form_label_second_coreq_execution_scaffold_ready"):
        readiness_bits.append("case1_form_label_second_coreq_execution_scaffold")
    if ph.get("offline_tf_case1_dual_linf_fourth_coreq_execution_scaffold_ready"):
        readiness_bits.append("case1_dual_linf_fourth_coreq_execution_scaffold")
    if ph.get("offline_tf_case1_wire_fifth_coreq_execution_scaffold_ready"):
        readiness_bits.append("case1_wire_fifth_coreq_execution_scaffold")
    readiness_pkg = "+".join(readiness_bits) if readiness_bits else "units_only"
    wire_note = (
        "wire_shipped=False; blockers documented; structural ready ≠ wire tomorrow"
        if ph.get("offline_tf_wire_preflight_ready")
        else "not wire shipped"
    )
    case1_shaped_note = (
        "Case-1-shaped skeleton packaged (linear_quality_pooling; "
        "naphtha/distillate/gasoil/residue; skeleton λ ≠ duals; skeleton ≠ wire)"
        if ph.get("offline_tf_case1_shaped_linking_ready")
        else "no case1_shaped packaging flag"
    )
    dual_space_note = (
        "dual-space/form contract packaged (planned≠classic form registered; "
        "streams aligned; dual_linf_under_wire=unproven; dual-ban; not wire)"
        if ph.get("offline_tf_case1_dual_space_form_contract_ready")
        else "no dual_space_form_contract packaging flag"
    )
    linf_probe_note = (
        "dual-space L∞ probe packaged (unproven; not VERDICT; not wire proof; "
        "dual-ban; skeleton λ ≠ Case 1 duals; wire_shipped=False)"
        if ph.get("offline_tf_case1_dual_space_linf_probe_ready")
        else "no dual_space_linf_probe packaging flag"
    )
    live_bridge_note = (
        "dual-space L∞ live-λ bridge packaged (source-labeled; unproven; not VERDICT; "
        "not wire proof; dual-ban; wire_shipped=False)"
        if ph.get("offline_tf_case1_dual_space_linf_live_lambda_bridge_ready")
        else "no dual_space_linf_live_lambda_bridge packaging flag"
    )
    live_warmstart_note = (
        "dual-space L∞ live-λ-seeded warm-start packaged (seed_policy; source labeled; "
        "seed≠proof; unproven; not VERDICT; not wire proof; dual-ban; wire_shipped=False)"
        if ph.get("offline_tf_case1_dual_space_linf_live_lambda_seeded_warmstart_ready")
        else "no dual_space_linf_live_lambda_seeded_warmstart packaging flag"
    )
    pooling_path_note = (
        "honest blender pooling path packaged (linear_quality_pooling; "
        "honest_pooling_path_present; not affine; not VERDICT; not wire; dual-ban; "
        "wire_shipped=False)"
        if ph.get("offline_tf_case1_honest_blender_pooling_path_ready")
        else "no case1_honest_blender_pooling_path packaging flag"
    )
    criteria_contract_note = (
        "gate-criteria contract packaged (gate open; flip=false; met=false; "
        "dual-ban; not VERDICT; not wire; wire_shipped=False)"
        if ph.get("offline_tf_case1_online_linf_gate_criteria_contract_ready")
        else "no case1_online_linf_gate_criteria_contract packaging flag"
    )
    isolation_design_note = (
        "isolation-rewrite design packaged (design_present; rewrite=false; "
        "checklist open; dual-ban; not VERDICT; not wire; wire_shipped=False)"
        if ph.get("offline_tf_case1_isolation_rewrite_design_contract_ready")
        else "no case1_isolation_rewrite_design_contract packaging flag"
    )
    wire_ship_design_note = (
        "wire-ship acceptance design packaged (design_present; ship_allowed=false; "
        "wire=false; dual-ban; not VERDICT; not ship allow)"
        if ph.get("offline_tf_case1_wire_ship_acceptance_design_contract_ready")
        else "no case1_wire_ship_acceptance_design_contract packaging flag"
    )
    path_design_note = (
        "dual-honest TF-aware path design packaged (path_design_present; "
        "path_shipped=false; ship-met=false; wire=false; dual-ban; not VERDICT; "
        "not path ship / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_tf_aware_path_design_contract_ready")
        else "no case1_dual_honest_tf_aware_path_design_contract packaging flag"
    )
    path_present_criteria_note = (
        "path-present ship-met criteria packaged (criteria_present; "
        "ship_met_allowed=false; ship-met=false; path_shipped=false; wire=false; "
        "dual-ban; not VERDICT; not ship-met / not path ship / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_tf_aware_path_present_criteria_contract_ready")
        else "no case1_dual_honest_tf_aware_path_present_criteria_contract packaging flag"
    )
    form_label_criteria_note = (
        "form_label ship criteria packaged (criteria_present; "
        "form_label_ship_allowed=false; form_label_change_shipped=false; form=classic; "
        "path_shipped=false; ship-met=false; wire=false; dual-ban; not VERDICT; "
        "not form flip / not form_label ship / not path ship / not ship allow)"
        if ph.get("offline_tf_case1_form_label_change_shipped_criteria_contract_ready")
        else "no case1_form_label_change_shipped_criteria_contract packaging flag"
    )
    isolation_ship_criteria_note = (
        "isolation ship criteria packaged (criteria_present; "
        "isolation_ship_allowed=false; isolation_rewrite_shipped=false; checklist open; "
        "rewrite-not-delete; form=classic; path_shipped=false; ship-met=false; wire=false; "
        "dual-ban; not VERDICT; not isolation rewrite ship / not form flip / not path ship / "
        "not ship allow)"
        if ph.get("offline_tf_case1_isolation_rewrite_shipped_criteria_contract_ready")
        else "no case1_isolation_rewrite_shipped_criteria_contract packaging flag"
    )
    bundle_design_note = (
        "multi-blocker wire bundle design packaged (bundle_design_present; "
        "bundle_shipped=false; bundle_ship_allowed=false; order_hint not executor; "
        "form=classic; path_shipped=false; ship-met=false; wire=false; dual-ban; "
        "not VERDICT; not bundle ship / not wire ship / not isolation rewrite ship / "
        "not form flip / not path ship / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_design_contract_ready")
        else "no case1_dual_honest_multi_blocker_wire_bundle_design_contract packaging flag"
    )
    bundle_criteria_note = (
        "multi-blocker wire bundle ship-met criteria packaged (criteria_present; "
        "bundle_shipped=false; bundle_ship_allowed=false; criteria_met=false; "
        "order_hint not executor; form=classic; path_shipped=false; ship-met=false; "
        "wire=false; dual-ban; not VERDICT; not bundle ship / not wire ship / "
        "not isolation rewrite ship / not form flip / not path ship / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract_ready")
        else "no case1_dual_honest_multi_blocker_wire_bundle_shipped_criteria_contract packaging flag"
    )
    scaffold_note = (
        "dual-honest TF-aware path execution scaffold packaged (scaffold_present; "
        "execution_scaffold_present; path_shipped=false; ship-met=false; "
        "wire=false; bundle_shipped=false; form=classic; dual-ban; not VERDICT; "
        "not path ship / not wire ship / not bundle ship / not isolation rewrite ship / "
        "not form flip / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_tf_aware_path_execution_scaffold_ready")
        else "no case1_dual_honest_tf_aware_path_execution_scaffold packaging flag"
    )
    rehearsal_note = (
        "dual-honest multi-blocker wire rehearsal packaged (rehearsal_present; "
        "wire_rehearsal_present; scaffold linked; path_shipped=false; ship-met=false; "
        "wire=false; bundle_shipped=false; form=classic; co-req dry-run static; "
        "dual-ban; not VERDICT; not path ship / not wire ship / not bundle ship / "
        "not isolation rewrite ship / not form flip / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_rehearsal_ready")
        else "no case1_dual_honest_multi_blocker_wire_rehearsal packaging flag"
    )
    blueprint_note = (
        "dual-honest multi-blocker wire implementation blueprint packaged "
        "(blueprint_present; first_blocking=isolation_rewrite_with_wire; "
        "rehearsal+scaffold linked; path_shipped=false; ship-met=false; "
        "wire=false; bundle_shipped=false; form=classic; go-board static; "
        "dual-ban; not VERDICT; not path ship / not wire ship / not bundle ship / "
        "not isolation rewrite ship / not form flip / not ship allow)"
        if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_implementation_blueprint_ready")
        else "no case1_dual_honest_multi_blocker_wire_implementation_blueprint packaging flag"
    )
    first_blocker_prep_note = (
        "isolation first-blocker operational prep packaged (prep_present; "
        "first_blocking=isolation_rewrite_with_wire; isolation_rewrite_shipped=false; "
        "rewrite-not-delete; dual_linf=unproven; dual_recovery_path=None; "
        "prep≠ship; dual-ban; not VERDICT; not wire; not dual L∞ under wire proof)"
        if ph.get("offline_tf_case1_isolation_rewrite_first_blocker_operational_prep_ready")
        else "no case1_isolation_rewrite_first_blocker_operational_prep packaging flag"
    )
    form_label_second_coreq_prep_note = (
        "form_label second-coreq operational prep packaged (prep_present; "
        "form=classic; form_label_change_shipped=false; form_label_ship_allowed=false; "
        "mutation not executed; first_blocking still isolation; prep≠ship; "
        "dual_recovery_path=None; dual-ban; not VERDICT; not form flip; not wire)"
        if ph.get("offline_tf_case1_form_label_second_coreq_operational_prep_ready")
        else "no case1_form_label_second_coreq_operational_prep packaging flag"
    )
    path_third_coreq_prep_note = (
        "path third-coreq operational prep packaged (prep_present; "
        "path_shipped=false; dual_honest_tf_aware_path_present ship-met=false; "
        "feature_flag=false; first_blocking still isolation; prep≠path ship; "
        "dual_recovery_path=None; dual-ban; not VERDICT; not wire; not ship-met)"
        if ph.get("offline_tf_case1_path_third_coreq_operational_prep_ready")
        else "no case1_path_third_coreq_operational_prep packaging flag"
    )
    dual_linf_fourth_coreq_prep_note = (
        "dual_linf fourth-coreq operational prep packaged (prep_present; "
        "dual_linf=unproven; proof_allowed=false; gate=open; first_blocking still "
        "isolation; prep≠proof; dual_recovery_path=None; dual-ban; not VERDICT; "
        "not gate flip; not dual_linf proven)"
        if ph.get("offline_tf_case1_dual_linf_fourth_coreq_operational_prep_ready")
        else "no case1_dual_linf_fourth_coreq_operational_prep packaging flag"
    )
    wire_fifth_coreq_prep_note = (
        "wire fifth-coreq operational prep packaged (prep_present; "
        "wire_shipped=false; wire_ship_allowed=false; first_blocking still "
        "isolation; prep≠ship; dual_recovery_path=None; dual-ban; not VERDICT; "
        "not dual_linf proven; not wire allow)"
        if ph.get("offline_tf_case1_wire_fifth_coreq_operational_prep_ready")
        else "no case1_wire_fifth_coreq_operational_prep packaging flag"
    )
    bundle_companion_prep_note = (
        "bundle companion operational prep packaged (prep_present; bundle_shipped=false; "
        "bundle_ship_allowed=false; land not executed; companion not order_hint primary; "
        "prep≠bundle; dual-ban; not VERDICT)"
        if ph.get("offline_tf_case1_dual_honest_multi_blocker_wire_bundle_operational_prep_ready")
        else "no case1_dual_honest_multi_blocker_wire_bundle_operational_prep packaging flag"
    )
    iso_rewrite_scaffold_note = (
        "isolation rewrite first-blocker execution scaffold packaged "
        "(scaffold_present; isolation_rewrite_shipped=false; "
        "isolation_ship_allowed=false; inventory only; first_blocking=isolation; "
        "scaffold≠ship; dual_recovery_path=None; dual-ban; not VERDICT; not wire; "
        "not dual_linf proven)"
        if ph.get("offline_tf_case1_isolation_rewrite_first_blocker_execution_scaffold_ready")
        else "no case1_isolation_rewrite_first_blocker_execution_scaffold packaging flag"
    )
    form_label_scaffold_note = (
        "form_label second-coreq execution scaffold packaged "
        "(scaffold_present; form_label_change_shipped=false; "
        "form_label_ship_allowed=false; form classic; mutation not executed; "
        "first_blocking still isolation; scaffold≠ship; dual_recovery_path=None; "
        "dual-ban; not VERDICT; not form flip; not wire; not dual_linf proven)"
        if ph.get("offline_tf_case1_form_label_second_coreq_execution_scaffold_ready")
        else "no case1_form_label_second_coreq_execution_scaffold packaging flag"
    )
    dual_linf_scaffold_note = (
        "dual_linf fourth-coreq execution scaffold packaged "
        "(scaffold_present; dual_linf unproven; dual_linf_proof_allowed=false; "
        "gate open; gate_flip_allowed=false; proof composition not executed; "
        "first_blocking still isolation; scaffold≠proof; dual_recovery_path=None; "
        "dual-ban; not VERDICT; not gate flip; not form flip; not wire; not dual_linf proven)"
        if ph.get("offline_tf_case1_dual_linf_fourth_coreq_execution_scaffold_ready")
        else "no case1_dual_linf_fourth_coreq_execution_scaffold packaging flag"
    )
    wire_scaffold_note = (
        "wire fifth-coreq execution scaffold packaged "
        "(scaffold_present; wire_shipped=false; wire_ship_allowed=false; "
        "wire_land not executed; dual_linf unproven; dual_linf_proof_allowed=false; "
        "gate open; first_blocking still isolation; scaffold≠wire; dual_recovery_path=None; "
        "dual-ban; not VERDICT; not gate flip; not form flip; not wire shipped)"
        if ph.get("offline_tf_case1_wire_fifth_coreq_execution_scaffold_ready")
        else "no case1_wire_fifth_coreq_execution_scaffold packaging flag"
    )
    print(
        f"Offline TF: units={offline_units}  readiness={readiness_pkg}  "
        f"on_excel_case1_path={ph.get('on_excel_case1_path', False)}  "
        f"(NOT on classic Case 1; dual_recovery_path=None on TF surface; "
        f"synthetic residual/subproblem/coordination/plant-linking/plant-named λ ≠ duals; "
        f"per-unit coordination ≠ plant linking; synthetic topology ≠ full plant MB; "
        f"plant-named offline demo ≠ full plant MB / ≠ live cascade; "
        f"preflight λ ≠ duals; {wire_note}; {case1_shaped_note}; {dual_space_note}; "
        f"{linf_probe_note}; {live_bridge_note}; {live_warmstart_note}; "
        f"{pooling_path_note}; {criteria_contract_note}; {isolation_design_note}; "
        f"{wire_ship_design_note}; {path_design_note}; {path_present_criteria_note}; "
        f"{form_label_criteria_note}; {isolation_ship_criteria_note}; "
        f"{bundle_design_note}; {bundle_criteria_note}; {scaffold_note}; {rehearsal_note}; "
        f"{blueprint_note}; {first_blocker_prep_note}; {form_label_second_coreq_prep_note}; {path_third_coreq_prep_note}; {dual_linf_fourth_coreq_prep_note}; {wire_fifth_coreq_prep_note}; {bundle_companion_prep_note}; {iso_rewrite_scaffold_note}; {form_label_scaffold_note}; {dual_linf_scaffold_note}; {wire_scaffold_note})"
    )
    print(f"Mono crudes:   { {k: round(v, 3) for k, v in mono['crude_rates'].items() if v > 1e-6} }")
    print(f"Mono products: { {k: round(v, 3) for k, v in mono['product_rates'].items() if v > 1e-6} }")
    print(f"Shadows mono:  { {k: round(v, 2) for k, v in mono['shadow_prices'].items()} }")
    print(f"Shadows ADMM:  { {k: round(v, 2) for k, v in admm['shadow_prices'].items()} }")
    rec = admm.get('shadow_prices_recovered') or {}
    if rec:
        print(f"Shadows recov: { {k: round(v, 2) for k, v in rec.items()} }")
    print()
    print(f"Results Excel: {xlsx_out}")
    print(f"Results JSON:  {json_out}")
    print("=" * 72)
    print(f"VERDICT: {report['verdict']}")

    # Post-solve diagnostic only (never gates VERDICT; never writes Excel).
    # Demo is allowed to import tf_linear_blocks; excel write path is not.
    try:
        from pims_admm_llm.models import tf_linear_blocks as _tlb

        bridge = _tlb.offline_case1_dual_space_linf_live_lambda_bridge_report(
            case1_package=report,
            skeleton_n_rounds=1,
            include_secondary_recovered=True,
        )
        print(
            f"Offline TF live-λ bridge (diagnostic only): "
            f"source={bridge.get('live_lambda_source')}  "
            f"L∞={bridge.get('linf')}  "
            f"bridge_ok={bridge.get('bridge_ok')}  "
            f"dual_linf_under_wire={bridge.get('dual_linf_under_wire_status')}  "
            f"online_linf_gate={bridge.get('online_linf_gate_under_tf_path')}  "
            f"dual_recovery_path={bridge.get('dual_recovery_path')}  "
            f"wire_shipped={bridge.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT dual L∞ under wire proof]"
        )
        warm = _tlb.offline_case1_dual_space_linf_live_lambda_seeded_warmstart_report(
            case1_package=report,
            n_rounds=1,
            include_secondary_recovered=True,
        )
        print(
            f"Offline TF live-λ-seeded warm-start (diagnostic only): "
            f"source={warm.get('live_lambda_source')}  "
            f"seed_policy={warm.get('seed_policy')}  "
            f"z0_policy={warm.get('z0_policy')}  "
            f"L∞_post={warm.get('linf_post_rounds')}  "
            f"L∞_seed={warm.get('linf_at_seed')}  "
            f"warmstart_ok={warm.get('warmstart_ok')}  "
            f"dual_linf_under_wire={warm.get('dual_linf_under_wire_status')}  "
            f"online_linf_gate={warm.get('online_linf_gate_under_tf_path')}  "
            f"dual_recovery_path={warm.get('dual_recovery_path')}  "
            f"wire_shipped={warm.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT dual L∞ under wire proof; "
            f"seed identity L∞ ≠ proof]"
        )
        pool = _tlb.offline_case1_honest_blender_pooling_path_report()
        print(
            f"Offline TF honest blender pooling path (diagnostic only): "
            f"surface={pool.get('blender_surface')}  "
            f"checklist={pool.get('blender_pooling_checklist_status')}  "
            f"pooling_path_ok={pool.get('pooling_path_ok')}  "
            f"affine_unit={pool.get('blender_is_base_delta_affine_unit')}  "
            f"no_blender_blocker={pool.get('no_blender_offline_affine_kernel_in_default_wire_blockers')}  "
            f"dual_linf_under_wire={pool.get('dual_linf_under_wire_status')}  "
            f"dual_recovery_path={pool.get('dual_recovery_path')}  "
            f"wire_shipped={pool.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT affine kernel; NOT wire proof]"
        )
        crit = _tlb.offline_case1_online_linf_gate_criteria_contract_report()
        print(
            f"Offline TF online_linf_gate criteria contract (diagnostic only): "
            f"gate={crit.get('online_linf_gate_under_tf_path')}  "
            f"flip_allowed={crit.get('gate_flip_allowed_today')}  "
            f"criteria_met={crit.get('criteria_met_today')}  "
            f"contract_ok={crit.get('contract_ok')}  "
            f"n_flip_criteria={len(crit.get('flip_criteria') or {})}  "
            f"dual_linf_under_wire={crit.get('dual_linf_under_wire_status')}  "
            f"dual_recovery_path={crit.get('dual_recovery_path')}  "
            f"wire_shipped={crit.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT gate flip; NOT dual L∞ under wire proof]"
        )
        design = _tlb.offline_case1_isolation_rewrite_design_contract_report()
        print(
            f"Offline TF isolation-rewrite design contract (diagnostic only): "
            f"design_present={design.get('isolation_rewrite_design_present')}  "
            f"rewrite_shipped={design.get('isolation_rewrite_shipped')}  "
            f"isolation_open={design.get('isolation_rewrite_with_wire')}  "
            f"design_contract_ok={design.get('design_contract_ok')}  "
            f"blocker_present={design.get('isolation_rewrite_required_in_default_wire_blockers')}  "
            f"dual_linf_under_wire={design.get('dual_linf_under_wire_status')}  "
            f"dual_recovery_path={design.get('dual_recovery_path')}  "
            f"wire_shipped={design.get('wire_shipped')}  "
            f"[NOT VERDICT gate; NOT isolation rewrite shipped; NOT wire]"
        )
        wire_ship = _tlb.offline_case1_wire_ship_acceptance_design_contract_report()
        print(
            f"Offline TF wire-ship acceptance design contract (diagnostic only): "
            f"design_present={wire_ship.get('design_present')}  "
            f"ship_allowed={wire_ship.get('wire_ship_allowed_today')}  "
            f"criteria_met={wire_ship.get('wire_ship_criteria_met_today')}  "
            f"wire_shipped={wire_ship.get('wire_shipped')}  "
            f"dual_linf_under_wire={wire_ship.get('dual_linf_under_wire_status')}  "
            f"form={wire_ship.get('form_current') or wire_ship.get('form')}  "
            f"isolation_rewrite_shipped={wire_ship.get('isolation_rewrite_shipped')}  "
            f"dual_recovery_path={wire_ship.get('dual_recovery_path')}  "
            f"[NOT VERDICT gate; NOT ship allow; NOT wire shipped]"
        )
    except Exception as exc:  # pragma: no cover - demo soft-skip
        print(
            f"Offline TF live-λ bridge/warm-start/pooling/criteria/design/wire-ship contract: "
            f"skipped ({exc})"
        )

    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
