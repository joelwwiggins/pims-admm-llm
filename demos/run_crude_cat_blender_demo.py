#!/usr/bin/env python3
"""Crude → CDU → tanks → FCC → blender (mono + ADMM).

  PYTHONPATH=src python -m demos.run_crude_cat_blender_demo
  PYTHONPATH=src python -m demos.run_crude_cat_blender_demo --crude Cold_Lake_Blend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.crude_cat_blender import compare_mono_admm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crude", default="WTI")
    ap.add_argument("--charge", type=float, default=100.0)
    args = ap.parse_args()

    c = compare_mono_admm(crude_name=args.crude, max_crude_kbd=args.charge)
    v = c["VERDICT"]
    print("=== crude→cat→blender ===")
    print("crude", args.crude, "charge_max", args.charge)
    print("mono_obj", v["mono_obj"])
    print("admm_obj", v["admm_obj"])
    print("gap_rel", c["obj_gap_rel"])
    print("mb_ok", c["mass_balance_ok"], "quality_ok", c["quality_ok"])
    print("streams crude", v["crude"], "gasoline", v["gasoline"], "fcc_mode", v["fcc_mode"])
    print("h2_kscf", v["h2_kscf"], "fuel_gas_mmbtu", v["fuel_gas_mmbtu"])
    mono = c["mono"]
    print("tanks", mono["tank"])
    print("products", mono["products"])
    print("purchases", mono["purchases"])
    print("quality", mono["quality"])

    out = ROOT / "demos" / "output" / "crude_cat_blender.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=2), encoding="utf-8")
    print(
        f"\nVERDICT crude_cat_blender crude={args.crude} mono={v['mono_obj']:.4f} "
        f"admm={v['admm_obj']:.4f} gap_rel={c['obj_gap_rel']:.6f} "
        f"mb_ok={c['mass_balance_ok']} path={out}"
    )
    return 0 if c["mass_balance_ok"] and c["obj_gap_rel"] < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
