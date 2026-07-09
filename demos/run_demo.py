"""MVP demo entrypoint — monolithic solve until ADMM/agents land.

Usage (from repo root with venv active):
  python -m demos.run_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without editable install during early scaffold
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    from pims_admm_llm.models import load_crude_data, solve_monolithic

    data = load_crude_data()
    result = solve_monolithic(data, msg=False)
    report = {
        "status": result.status,
        "objective_margin_usd_per_day_scale": result.objective,
        "solve_time_s": round(result.solve_time_s, 4),
        "crude_rates_kbd": {k: round(v, 4) for k, v in result.crude_rates.items()},
        "product_rates_kbd": {k: round(v, 4) for k, v in result.product_rates.items()},
        "intermediate_prod_kbd": {
            k: round(v, 4) for k, v in result.intermediate_prod.items()
        },
        "key_shadow_prices": {
            k: round(v, 6)
            for k, v in result.duals.items()
            if k.startswith(("cdu_", "balance_", "blend_"))
        },
        "note": "ADMM dual comparison + LLM agents arrive in later workers.",
    }
    print(json.dumps(report, indent=2))
    return 0 if result.status == "Optimal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
