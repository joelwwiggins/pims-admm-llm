#!/usr/bin/env python3
"""Import assay → heart/swing CDU → mass balance + product properties.

  PYTHONPATH=src python -m demos.run_assay_cdu_demo
  PYTHONPATH=src python -m demos.run_assay_cdu_demo --crude Cold_Lake_Blend --charge 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.assay_swing import (
    import_crude_from_assays_package,
    list_importable_assays,
    solve_cdu_swing_cuts,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crude", default="Cold_Lake_Blend")
    ap.add_argument("--charge", type=float, default=100.0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("importable:", list_importable_assays())
        return 0

    assay = import_crude_from_assays_package(args.crude)
    print(f"assay={assay.name} ref={assay.reference} cuts={len(assay.cuts)} vol_sum={assay.total_vol():.4f}")
    res = solve_cdu_swing_cuts(assay, charge_kbd=args.charge, optimize=True)
    print(f"status={res.status} mb_ok={res.mass_balance['ok']}")
    print("yields vol frac:", {k: round(v, 4) for k, v in res.product_yields_vol.items()})
    print("rates kbd:", {k: round(v, 3) for k, v in res.product_rates_kbd.items()})
    print("properties:")
    for p, props in res.product_properties.items():
        print(
            f"  {p:16s} API={props['api']:5.1f}  S={props['sulfur_wt']:5.2f}  "
            f"CCR={props['ccr_wt']:5.2f}  Ni+V={props['metals_ni_v_ppm']:6.1f}"
        )
    print("swings:")
    for sid, a in res.swing_allocations.items():
        print(
            f"  {sid}: light_frac={a['light_frac']:.3f}  "
            f"{a['light_product']}={a['to_light_kbd']:.2f} / "
            f"{a['heavy_product']}={a['to_heavy_kbd']:.2f}"
        )
    print("mass_balance:", res.mass_balance)

    out = ROOT / "demos" / "output" / "assay_cdu_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    print(
        f"\nVERDICT assay_cdu crude={args.crude} status={res.status} "
        f"mb_ok={res.mass_balance['ok']} resid_y={res.product_yields_vol.get('cdu_resid', 0):.3f} "
        f"path={out}"
    )
    return 0 if res.status == "Optimal" and res.mass_balance["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
