"""LLM agent stubs — deterministic intelligence layer without API calls.

Swap `reason` methods for real Grok/OpenAI calls when credentials exist.
Hard constraints always stay with the LP solver; agents only annotate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts import MASTER_PROMPT, format_subagent_prompt


@dataclass
class LLMSubAgent:
    block: str
    local_constraints: str = "capacity, yields, recipe mass balance"

    def reason(
        self,
        lambda_dict: Dict[str, float],
        z_dict: Dict[str, float],
        proposal: Dict[str, Any],
        use_real_llm: bool = False,
    ) -> Dict[str, Any]:
        prompt = format_subagent_prompt(
            self.block, lambda_dict, z_dict, proposal, self.local_constraints
        )
        if use_real_llm:
            # Placeholder for real provider integration
            return {
                "block": self.block,
                "proposal_summary": "real LLM not wired",
                "local_obj_note": "",
                "suggestion": "",
                "proposed_soft_penalty": {},
                "prompt_chars": len(prompt),
            }

        # Deterministic "expert operator" stub
        suggestion = ""
        if self.block == "CDU":
            heavy = proposal.get("crude_Maya_heavy", 0.0)
            if heavy and heavy > 20:
                suggestion = (
                    "Heavy crude slate high; real yields may drop naphtha vs linear vector — "
                    "consider soft penalty on residue if tank limited."
                )
            else:
                suggestion = "Linear yields OK near base slate; watch sulfur on medium/heavy mix."
        elif self.block == "Blender":
            suggestion = (
                "Recipe fractions are fixed linear; if octane/cetane soft specs bind, "
                "raise naphtha preference via soft dual nudge."
            )
        else:
            suggestion = "No special nonlinear note."

        return {
            "block": self.block,
            "proposal_summary": f"{self.block} solver status={proposal.get('_status')}",
            "local_obj_note": f"local_obj={proposal.get('_obj')}",
            "suggestion": suggestion,
            "proposed_soft_penalty": {},
            "prompt_chars": len(prompt),
        }


@dataclass
class MasterAgent:
    def reason(
        self,
        iteration: int,
        primal_res: float,
        dual_res: float,
        lambda_dict: Dict[str, float],
        z_dict: Dict[str, float],
        proposals: List[Dict[str, Any]],
        obj_approx: float,
        tol: float = 1e-3,
        use_real_llm: bool = False,
    ) -> Dict[str, Any]:
        prompt = MASTER_PROMPT.format(
            iteration=iteration,
            primal_res=primal_res,
            dual_res=dual_res,
            lambda_json=json.dumps(lambda_dict, indent=2),
            z_json=json.dumps(z_dict, indent=2),
            proposals_json=json.dumps(proposals, indent=2, default=str)[:4000],
            obj_approx=obj_approx,
        )
        action = "terminate" if primal_res < tol and dual_res < tol else "continue"
        highlights = {
            k: f"Marginal value ~ {v:.3f} USD/bbl if linking stream relaxed"
            for k, v in sorted(lambda_dict.items(), key=lambda kv: -abs(kv[1]))[:4]
        }
        brief = (
            f"Iter {iteration}: residual primal={primal_res:.4f}, dual={dual_res:.4f}. "
            f"Approx margin ${obj_approx:,.1f}/day (toy units kbd×USD/bbl). "
            f"Top shadow prices highlight binding intermediate economics for make-buy-sell."
        )
        return {
            "action": action,
            "rho_suggestion": None,
            "economic_brief": brief,
            "shadow_price_highlights": highlights,
            "prompt_chars": len(prompt),
        }
