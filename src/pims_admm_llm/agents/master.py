"""Master coordinator agent — interprets ADMM duals + sub-agent suggestions.

Does not own dual updates (admm/). Math residual gate is authoritative;
LLM only adds economic commentary and suggestion highlighting.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from .llm_client import LLMClient, make_llm_client
from .prompts import render_master_prompt
from .schemas import MasterDecision, SubAgentProposal


class MasterCoordinatorAgent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        enable_llm: bool = True,
        tol: float = 1e-3,
        max_iter: int = 50,
    ) -> None:
        self.llm = llm or make_llm_client()
        self.enable_llm = enable_llm
        self.tol = tol
        self.max_iter = max_iter

    def decide(
        self,
        *,
        iteration: int,
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        residual_norm: float,
        dual_residual_norm: float = 0.0,
        global_obj: float = 0.0,
        proposals: Sequence[SubAgentProposal | Mapping[str, Any]] = (),
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        new_prices: Optional[Mapping[str, float]] = None,
        consensus_targets: Optional[Mapping[str, float]] = None,
    ) -> MasterDecision:
        tol = self.tol if tol is None else tol
        max_iter = self.max_iter if max_iter is None else max_iter

        # Hard residual rule first (math over LLM)
        action_math = "terminate" if residual_norm <= tol else "continue"
        if iteration >= max_iter:
            action_math = "terminate"

        decision = MasterDecision(
            action=action_math,
            new_prices=dict(new_prices if new_prices is not None else prices),
            consensus_targets=dict(
                consensus_targets if consensus_targets is not None else consensus
            ),
            applied_suggestions=[],
            reasoning=(
                f"Math gate: residual={residual_norm:.6g} tol={tol} "
                f"iter={iteration}/{max_iter} → {action_math}"
            ),
            global_obj_estimate=float(global_obj),
            residual_norm=float(residual_norm),
            iteration=int(iteration),
        )

        if not self.enable_llm:
            return decision

        prompt = render_master_prompt(
            iteration=iteration,
            prices=prices,
            consensus=consensus,
            residual_norm=residual_norm,
            dual_residual_norm=dual_residual_norm,
            global_obj=global_obj,
            proposals=list(proposals),
            tol=tol,
            max_iter=max_iter,
        )
        try:
            raw = self.llm.complete_json(prompt)
        except Exception as exc:
            decision.reasoning = f"LLM error (math action kept={action_math}): {exc}"
            return decision

        action = str(raw.get("action", action_math)).lower().strip()
        if action not in ("continue", "terminate"):
            action = action_math
        # Prefer math terminate when residuals already good / max_iter
        if action_math == "terminate":
            action = "terminate"
        decision.action = action

        applied: List[dict] = []
        for item in raw.get("highlighted_suggestions") or raw.get("applied_suggestions") or []:
            if isinstance(item, dict):
                applied.append(item)
        # Also surface sub-agent suggestions for logging
        for p in proposals:
            if isinstance(p, SubAgentProposal):
                for s in p.suggestions:
                    applied.append(
                        {
                            "block": p.block,
                            "kind": s.kind,
                            "message": s.message,
                            "stream": s.stream,
                            "delta_frac": s.delta_frac,
                            "confidence": s.confidence,
                            "source": "subagent",
                        }
                    )
            elif isinstance(p, Mapping):
                for s in p.get("suggestions") or []:
                    if isinstance(s, dict):
                        s2 = dict(s)
                        s2.setdefault("block", p.get("block"))
                        s2["source"] = "subagent"
                        applied.append(s2)
        decision.applied_suggestions = applied

        reasoning = str(raw.get("reasoning", "") or raw.get("economic_brief", ""))
        price_commentary = raw.get("price_commentary") or {}
        if isinstance(price_commentary, dict) and price_commentary:
            decision.cuts = [f"{k}: {v}" for k, v in price_commentary.items()]
            if not reasoning:
                reasoning = "; ".join(f"{k}: {v}" for k, v in price_commentary.items())
        decision.reasoning = reasoning or decision.reasoning
        return decision
