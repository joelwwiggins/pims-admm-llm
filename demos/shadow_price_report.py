#!/usr/bin/env python3
"""Worker 7 demo: PIMS-style shadow price table + make-buy-sell + linearity.

Usage (from repo root, venv active):
  PYTHONPATH=src python demos/shadow_price_report.py
  PYTHONPATH=src python demos/shadow_price_report.py --with-admm
  PYTHONPATH=src python demos/shadow_price_report.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.models.data import load_crude_data
from pims_admm_llm.reporting.shadow_prices import (
    build_shadow_price_report,
    format_report_text,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PIMS-style shadow price report")
    p.add_argument("--with-admm", action="store_true", help="Also run ADMM and compare λ")
    p.add_argument("--json", type=str, default="", help="Optional JSON output path")
    p.add_argument("--no-linearity", action="store_true", help="Skip RHS linearity checks")
    p.add_argument(
        "--tight-tanks",
        action="store_true",
        help="Set zero intermediate hold (tanks already non-binding at opt hold=0)",
    )
    args = p.parse_args(argv)

    data = load_crude_data()
    if args.tight_tanks:
        # shrink tank capacities so duals can bind for demo sensitivity
        for stream, inv in (data.inventory or {}).items():
            inv.capacity_kbd = max(inv.start_kbd, 1.0)

    admm_result = None
    if args.with_admm:
        try:
            from pims_admm_llm.admm import ADMMConfig, run_admm

            admm_result = run_admm(data, ADMMConfig(max_iter=40, rho=5.0))
            print(
                f"[admm] status={admm_result.status} iters={admm_result.iterations} "
                f"obj≈{admm_result.objective:.4f} "
                f"λ={ {k: round(v, 4) for k, v in admm_result.shadow_prices.items()} }",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[admm] skipped: {exc}", file=sys.stderr)

    report = build_shadow_price_report(
        data=data,
        admm_result=admm_result,
        run_linearity=not args.no_linearity,
    )
    text = format_report_text(report)
    print(text)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote JSON: {out}", file=sys.stderr)

    # Exit non-zero only if critical failures
    lin_fail = [c for c in report.linearity if not c.passed]
    # Small deltas must pass; larger may fail on basis change — count only |ΔRHS|<=1 fails
    critical = [c for c in lin_fail if abs(c.delta_rhs) <= 1.0 + 1e-9]
    if report.baseline_status != "Optimal":
        return 2
    if critical:
        print(f"\nCRITICAL linearity failures: {len(critical)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
