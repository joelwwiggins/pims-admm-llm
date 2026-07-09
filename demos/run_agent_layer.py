#!/usr/bin/env python3
"""Standalone smoke for Worker 5: LLM multi-agent layer (stub by default).

Usage:
  python -m demos.run_agent_layer
  LLM_MODE=stub python demos/run_agent_layer.py
  # optional real LLM (OpenAI-compatible):
  # LLM_MODE=openai OPENAI_API_KEY=... OPENAI_BASE_URL=... LLM_MODEL=... python -m demos.run_agent_layer
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.agents.layer import demo_round  # noqa: E402
from pims_admm_llm.agents.prompts import BLOCK_PROMPTS, MASTER_PROMPT, list_blocks  # noqa: E402


def main() -> int:
    mode = os.environ.get("LLM_MODE", "stub")
    print("=== pims-admm-llm agent layer smoke ===")
    print("blocks:", list_blocks())
    print("prompt chars:", {b: len(BLOCK_PROMPTS[b]) for b in list_blocks()})
    print("master prompt chars:", len(MASTER_PROMPT))
    print("llm_mode:", mode)
    out = demo_round(llm_mode=mode, residual_norm=0.05, iteration=1)
    print(json.dumps(out, indent=2))
    # hard-constraint invariant: proposals come from mock solver, not LLM rewrite
    for p in out["proposals"]:
        assert "proposal" in p
        assert p["block"] in list_blocks()
    print("VERDICT: agent_layer_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
