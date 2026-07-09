#!/usr/bin/env python3
"""Standalone smoke for Worker 5: LLM multi-agent layer (stub by default).

Usage:
  python -m demos.run_agent_layer
  python demos/run_agent_layer.py --live          # live Grok/OpenAI if key present
  LLM_MODE=stub python demos/run_agent_layer.py

Live / env vars (optional):
  XAI_API_KEY       xAI Grok key (preferred for Wave2)
  OPENAI_API_KEY    OpenAI-compatible key
  LLM_API_KEY       generic key alias
  LLM_BASE_URL / OPENAI_BASE_URL
                    default https://api.x.ai/v1 when XAI_API_KEY is set,
                    else https://api.openai.com/v1
  LLM_MODEL / OPENAI_MODEL
                    default grok-4 (xAI) or gpt-4o-mini (OpenAI);
                    also try grok-3-mini
  LLM_MODE          stub | auto | live | openai

With --live and no API key, the demo skips the live call (exit 0) and prints
a skip notice so CI stays green offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.agents.layer import demo_round  # noqa: E402
from pims_admm_llm.agents.llm_client import (  # noqa: E402
    detect_llm_env,
    has_llm_api_key,
    make_llm_client,
)
from pims_admm_llm.agents.prompts import BLOCK_PROMPTS, MASTER_PROMPT, list_blocks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pims-admm-llm agent layer smoke")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use OpenAI-compat client (Grok/xAI when XAI_API_KEY set). Skips if no key.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override LLM mode: stub|auto|live|openai (default: env LLM_MODE or stub)",
    )
    args = parser.parse_args(argv)

    if args.mode:
        mode = args.mode
    elif args.live:
        mode = "live"
    else:
        mode = os.environ.get("LLM_MODE", "stub")

    env = detect_llm_env()
    print("=== pims-admm-llm agent layer smoke ===")
    print("blocks:", list_blocks())
    print("prompt chars:", {b: len(BLOCK_PROMPTS[b]) for b in list_blocks()})
    print("master prompt chars:", len(MASTER_PROMPT))
    print("llm_mode:", mode)
    print(
        "env_detection:",
        {
            "key_source": env.get("key_source"),
            "base_url": env.get("base_url"),
            "model": env.get("model"),
            "provider": env.get("provider"),
            "has_key": bool(env.get("api_key")),
        },
    )

    if mode in ("live", "openai", "real", "xai", "grok") and not has_llm_api_key():
        print(
            "SKIP live: no XAI_API_KEY / OPENAI_API_KEY / LLM_API_KEY set "
            "(stub path still available without --live)."
        )
        print("VERDICT: agent_layer_live_skipped")
        return 0

    # Resolve client early so misconfig surfaces clearly
    client = make_llm_client(mode)
    print("client:", type(client).__name__)
    if hasattr(client, "base_url"):
        print("client.base_url:", getattr(client, "base_url", None))
        print("client.model:", getattr(client, "model", None))

    out = demo_round(llm_mode=mode, residual_norm=0.05, iteration=1)
    print(json.dumps(out, indent=2))
    # hard-constraint invariant: proposals come from mock solver, not LLM rewrite
    for p in out["proposals"]:
        assert "proposal" in p
        assert p["block"] in list_blocks()
        # suggestions are the only LLM-owned soft channel
        assert isinstance(p.get("suggestions", []), list)
    print("VERDICT: agent_layer_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
