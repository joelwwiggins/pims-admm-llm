"""Per-block SubAgent: local LP results + optional LLM nonlinear suggestions.

Hard constraints always stay with the LP solver (solve_fn / upstream ADMM).
The LLM only adds soft intelligence as structured JSON.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .llm_client import LLMClient, make_llm_client
from .prompts import render_subagent_prompt
from .schemas import BlockName, SubAgentProposal, Suggestion

# Optional local solver: (prices, consensus, **kw) -> dict with proposal fields
SolveFn = Callable[..., Mapping[str, Any]]


class SubAgent:
    """One refinery block agent (CDU / Tank / Blender / Utilities)."""

    def __init__(
        self,
        block: Optional[str] = None,
        llm: Optional[LLMClient] = None,
        local_data: Optional[Mapping[str, Any]] = None,
        solve_fn: Optional[SolveFn] = None,
        enable_llm: bool = True,
        name: Optional[str] = None,
    ) -> None:
        block = block or name
        if not block:
            raise ValueError("SubAgent requires block= or name=")
        self.block = block
        self.name = block  # alias for callers expecting .name
        self.llm = llm or make_llm_client()
        self.local_data = dict(local_data or {})
        self.solve_fn = solve_fn
        self.enable_llm = enable_llm

    def build_proposal(
        self,
        *,
        local_solution: Mapping[str, Any],
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        iteration: int = 0,
        linking_flows: Optional[Mapping[str, float]] = None,
        local_obj: float = 0.0,
        status: str = "Optimal",
        reduced_costs: Optional[Mapping[str, float]] = None,
        local_duals: Optional[Mapping[str, float]] = None,
        augment: bool = True,
    ) -> SubAgentProposal:
        """Build proposal from an already-solved local LP (or ADMM block result)."""
        proposal_nums = {
            k: float(v)
            for k, v in local_solution.items()
            if isinstance(v, (int, float))
        }
        prop = SubAgentProposal(
            block=self.block,
            proposal=proposal_nums,
            reduced_costs={k: float(v) for k, v in (reduced_costs or {}).items()},
            linking_flows={k: float(v) for k, v in (linking_flows or {}).items()},
            local_duals={k: float(v) for k, v in (local_duals or {}).items()},
            local_obj=float(local_obj),
            status=status,
            iteration=iteration,
        )
        if not augment or not self.enable_llm:
            return prop

        prompt = render_subagent_prompt(
            self.block,
            prices=prices,
            consensus=consensus,
            local_solution=dict(local_solution),
            local_data=self.local_data,
            iteration=iteration,
        )
        try:
            raw = self.llm.complete_json(prompt)
        except Exception as exc:
            prop.suggestions.append(
                Suggestion(
                    kind="other",
                    message=f"LLM error (ignored for feasibility): {exc}",
                    confidence=0.0,
                )
            )
            prop.note = "llm_error"
            return prop

        if isinstance(raw.get("suggestion"), dict) and raw["suggestion"].get("message"):
            prop.suggestions.append(Suggestion.from_dict(raw["suggestion"]))
        elif isinstance(raw.get("suggestions"), list):
            for s in raw["suggestions"]:
                if isinstance(s, dict):
                    prop.suggestions.append(Suggestion.from_dict(s))
        prop.note = str(raw.get("note", "") or "")
        return prop

    def solve_local(
        self,
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        *,
        iteration: int = 0,
        **solver_kwargs: Any,
    ) -> Dict[str, Any]:
        if self.solve_fn is None:
            return {
                "status": "NoSolver",
                "proposal": {},
                "reduced_costs": {},
                "local_obj": 0.0,
                "linking_flows": {},
                "local_duals": {},
            }
        raw = dict(self.solve_fn(prices, consensus, iteration=iteration, **solver_kwargs))
        return {
            "status": str(raw.get("status", "Optimal")),
            "proposal": dict(raw.get("proposal") or {}),
            "reduced_costs": dict(raw.get("reduced_costs") or {}),
            "local_obj": float(raw.get("local_obj", 0.0) or 0.0),
            "linking_flows": dict(raw.get("linking_flows") or raw.get("proposal") or {}),
            "local_duals": dict(raw.get("local_duals") or {}),
        }

    def propose(
        self,
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        *,
        iteration: int = 0,
        **solver_kwargs: Any,
    ) -> SubAgentProposal:
        """Solve (hard) then optionally LLM-augment (soft)."""
        local = self.solve_local(prices, consensus, iteration=iteration, **solver_kwargs)
        return self.build_proposal(
            local_solution=local.get("proposal") or {},
            prices=prices,
            consensus=consensus,
            iteration=iteration,
            linking_flows=local.get("linking_flows"),
            local_obj=float(local.get("local_obj", 0.0) or 0.0),
            status=str(local.get("status", "Optimal")),
            reduced_costs=local.get("reduced_costs"),
            local_duals=local.get("local_duals"),
            augment=self.enable_llm,
        )


def default_block_agents(
    llm: Optional[LLMClient] = None,
    *,
    enable_llm: bool = True,
    solvers: Optional[Mapping[str, SolveFn]] = None,
    local_data: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, SubAgent]:
    client = llm or make_llm_client()
    solvers = solvers or {}
    local_data = local_data or {}
    out: Dict[str, SubAgent] = {}
    for name in (
        BlockName.CDU.value,
        BlockName.TANK.value,
        BlockName.BLENDER.value,
        BlockName.UTILITIES.value,
    ):
        out[name] = SubAgent(
            name,
            llm=client,
            local_data=local_data.get(name),
            solve_fn=solvers.get(name),
            enable_llm=enable_llm,
        )
    return out
