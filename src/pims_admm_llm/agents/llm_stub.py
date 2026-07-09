"""Legacy LLM stubs (compat for any early imports).

Prefer llm_client.StubLLMClient + SubAgent/MasterCoordinatorAgent.
Hard constraints always stay with the LP solver; agents only annotate.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .llm_client import StubLLMClient
from .prompts import render_master_prompt, render_subagent_prompt
from .schemas import BlockName


def format_subagent_prompt(
    block: str,
    lambda_dict: Dict[str, float],
    z_dict: Dict[str, float],
    proposal: Dict[str, Any],
    local_constraints: str = "",
) -> str:
    return render_subagent_prompt(
        block,
        prices=lambda_dict,
        consensus=z_dict,
        local_solution=proposal,
        local_data={"constraints": local_constraints},
        iteration=0,
    )


class LLMSubAgent:
    def __init__(self, block: str, local_constraints: str = "capacity, yields") -> None:
        self.block = block
        self.local_constraints = local_constraints
        self._client = StubLLMClient()

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
            return {
                "block": self.block,
                "proposal_summary": "real LLM not wired in legacy stub",
                "suggestion": "",
                "proposed_soft_penalty": {},
                "prompt_chars": len(prompt),
            }
        raw = self._client.complete_json(prompt)
        sug = raw.get("suggestion") or {}
        return {
            "block": self.block,
            "proposal_summary": f"{self.block} status={proposal.get('_status')}",
            "local_obj_note": f"local_obj={proposal.get('_obj')}",
            "suggestion": sug.get("message", "") if isinstance(sug, dict) else str(sug),
            "proposed_soft_penalty": {},
            "prompt_chars": len(prompt),
            "structured": raw,
        }


class MasterAgent:
    def __init__(self) -> None:
        self._client = StubLLMClient()

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
        prompt = render_master_prompt(
            iteration=iteration,
            prices=lambda_dict,
            consensus=z_dict,
            residual_norm=primal_res,
            dual_residual_norm=dual_res,
            global_obj=obj_approx,
            proposals=proposals,
            tol=tol,
            max_iter=50,
        )
        action = "terminate" if primal_res < tol and dual_res < tol else "continue"
        if use_real_llm:
            return {
                "action": action,
                "economic_brief": "real LLM not wired in legacy stub",
                "prompt_chars": len(prompt),
            }
        raw = self._client.complete_json(prompt)
        return {
            "action": raw.get("action", action),
            "rho_suggestion": None,
            "economic_brief": raw.get("reasoning", ""),
            "shadow_price_highlights": raw.get("price_commentary") or {},
            "prompt_chars": len(prompt),
            "structured": raw,
        }


def default_legacy_agents() -> Dict[str, LLMSubAgent]:
    return {
        BlockName.CDU.value: LLMSubAgent(BlockName.CDU.value),
        BlockName.TANK.value: LLMSubAgent(BlockName.TANK.value),
        BlockName.BLENDER.value: LLMSubAgent(BlockName.BLENDER.value),
        BlockName.UTILITIES.value: LLMSubAgent(BlockName.UTILITIES.value),
    }
