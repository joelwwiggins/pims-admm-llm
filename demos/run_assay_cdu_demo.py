#!/usr/bin/env python3
"""Import assay → CDU with cut-point operational handles.

  PYTHONPATH=src python -m demos.run_assay_cdu_demo
  PYTHONPATH=src python -m demos.run_assay_cdu_demo --crude Cold_Lake_Blend \
      --naphtha-ep 220 --distillate-ep 370 --gasoil-ep 550
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.assay_swing import (
    cdu_cut_point_modes,
    import_crude_from_assays_package,
    list_importable_assays,
    solve_cdu_from_cut_points,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="CDU cut-point handles on imported assay")
    ap.add_argument("--crude", default="Cold_Lake_Blend")
    ap.add_argument("--charge", type=float, default=100.0)
    ap.add_argument("--naphtha-ep", type=float, default=200.0, help="Naphtha EP °C (handle)")
    ap.add_argument("--distillate-ep", type=float, default=370.0, help="Distillate EP °C (handle)")
    ap.add_argument("--gasoil-ep", type=float, default=550.0, help="Gasoil EP °C (handle)")
    ap.add_argument("--mode", choices=["cuts_light", "cuts_mid", "cuts_heavy"], default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("importable:", list_importable_assays())
        print("cut-point modes:", [m["id"] for m in cdu_cut_point_modes()])
        return 0

    if args.mode:
        cp = next(m["cut_points_c"] for m in cdu_cut_point_modes() if m["id"] == args.mode)
    else:
        cp = {
            "naphtha_ep_c": args.naphtha_ep,
            "distillate_ep_c": args.distillate_ep,
            "gasoil_ep_c": args.gasoil_ep,
        }

    assay = import_crude_from_assays_package(args.crude)
    print(f"assay={assay.name} cuts={len(assay.cuts)}")
    print(f"HANDLES cut_points_c={cp}")
    res = solve_cdu_from_cut_points(assay, cp, charge_kbd=args.charge)
    print(f"status={res.status} mb_ok={res.mass_balance['ok']} driver={res.mass_balance.get('driver')}")
    print("yields:", {k: round(v, 4) for k, v in res.product_yields_vol.items()})
    for p, props in res.product_properties.items():
        print(f"  {p:16s} API={props['api']:5.1f} S={props['sulfur_wt']:5.2f} CCR={props['ccr_wt']:5.2f}")
    if res.swing_allocations:
        print("boundary swings (cut-point driven):")
        for sid, a in res.swing_allocations.items():
            print(f"  {sid}: light_frac={a['light_frac']:.3f}")

    out = ROOT / "demos" / "output" / "assay_cdu_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    print(
        f"\nVERDICT cut_point_cdu crude={args.crude} naph_ep={cp['naphtha_ep_c']} "
        f"mb_ok={res.mass_balance['ok']} resid_y={res.product_yields_vol['cdu_resid']:.3f} path={out}"
    )
    return 0 if res.status == "Optimal" and res.mass_balance["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
