"""Multi-agent layer: parallel block proposals + master decision + ADMM annotate.

Hard constraints stay inside each SubAgent.solve_fn / upstream ADMM solvers.
LLM only injects soft nonlinear/yield suggestions as structured JSON.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .llm_client import LLMClient, make_llm_client
from .master import MasterCoordinatorAgent
from .schemas import BlockName, MasterDecision, SubAgentProposal
from .subagent import DEFAULT_BLOCKS, SubAgent, default_block_agents


@dataclass
class MultiAgentLayerResult:
    proposals: List[SubAgentProposal] = field(default_factory=list)
    master: Optional[MasterDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "master": self.master.to_dict() if self.master else None,
            "decision": self.master.to_dict() if self.master else None,
            "suggestions_only": collect_suggestions(self.proposals),
        }


class MultiAgentLayer:
    """Facade used by demos / ADMM coordinator."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        agents: Optional[Dict[str, SubAgent]] = None,
        master: Optional[MasterCoordinatorAgent] = None,
        max_workers: int = 8,
    ) -> None:
        self.llm = llm or make_llm_client()
        self.agents = agents if agents is not None else default_block_agents(self.llm)
        self.master = master or MasterCoordinatorAgent(self.llm)
        self.max_workers = max_workers

    @classmethod
    def create(
        cls,
        *,
        llm_mode: str = "stub",
        enable_llm: bool = True,
        solvers: Optional[Mapping[str, Any]] = None,
        local_data: Optional[Mapping[str, Dict[str, Any]]] = None,
        blocks: Optional[Sequence[str]] = None,
        tol: float = 1e-3,
        max_iter: int = 50,
        max_workers: int = 8,
        llm: Optional[LLMClient] = None,
    ) -> "MultiAgentLayer":
        client = llm or make_llm_client(llm_mode)
        agents = default_block_agents(
            llm=client,
            enable_llm=enable_llm,
            solvers=solvers,
            local_data=local_data,
            blocks=list(blocks) if blocks is not None else None,
        )
        master = MasterCoordinatorAgent(
            client, enable_llm=enable_llm, tol=tol, max_iter=max_iter
        )
        return cls(llm=client, agents=agents, master=master, max_workers=max_workers)

    def propose_all(
        self,
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        *,
        iteration: int = 0,
        blocks: Optional[Sequence[str]] = None,
        parallel: bool = True,
        **solver_kwargs: Any,
    ) -> List[SubAgentProposal]:
        names = list(blocks) if blocks is not None else list(self.agents.keys())
        if not parallel or len(names) <= 1:
            return [
                self.agents[n].propose(
                    prices, consensus, iteration=iteration, **solver_kwargs
                )
                for n in names
                if n in self.agents
            ]

        results: Dict[str, SubAgentProposal] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {
                pool.submit(
                    self.agents[n].propose,
                    prices,
                    consensus,
                    iteration=iteration,
                    **solver_kwargs,
                ): n
                for n in names
                if n in self.agents
            }
            for fut in as_completed(futs):
                name = futs[fut]
                results[name] = fut.result()
        return [results[n] for n in names if n in results]

    def coordinate(
        self,
        prices: Mapping[str, float],
        consensus: Mapping[str, float],
        *,
        residual_norm: float,
        iteration: int = 0,
        dual_residual_norm: float = 0.0,
        global_obj: float = 0.0,
        new_prices: Optional[Mapping[str, float]] = None,
        consensus_targets: Optional[Mapping[str, float]] = None,
        blocks: Optional[Sequence[str]] = None,
        parallel: bool = True,
        **solver_kwargs: Any,
    ) -> Dict[str, Any]:
        proposals = self.propose_all(
            prices,
            consensus,
            iteration=iteration,
            blocks=blocks,
            parallel=parallel,
            **solver_kwargs,
        )
        decision = self.master.decide(
            iteration=iteration,
            prices=prices,
            consensus=consensus,
            proposals=proposals,
            residual_norm=residual_norm,
            dual_residual_norm=dual_residual_norm,
            global_obj=global_obj,
            new_prices=new_prices,
            consensus_targets=consensus_targets,
        )
        return {
            "proposals": [p.to_dict() for p in proposals],
            "decision": decision.to_dict(),
            "master": decision.to_dict(),
            "suggestions_only": collect_suggestions(proposals),
        }

    def annotate_admm(
        self,
        *,
        shadow_prices: Mapping[str, float],
        consensus_z: Mapping[str, float],
        crude_rates: Mapping[str, float],
        product_rates: Mapping[str, float],
        intermediate_prod: Mapping[str, float],
        intermediate_use: Mapping[str, float],
        residual_norm: float,
        dual_residual_norm: float = 0.0,
        objective: float = 0.0,
        iterations: int = 0,
        tol: float = 1e-3,
        max_iter: int = 50,
        include_tank_utilities: bool = True,
        include_conversion_units: bool = True,
        fcc_rates: Optional[Mapping[str, float]] = None,
        coker_rates: Optional[Mapping[str, float]] = None,
        reformer_rates: Optional[Mapping[str, float]] = None,
    ) -> MultiAgentLayerResult:
        """Post-solve annotation for ADMM/monolithic results."""
        cdu = self.agents["CDU"].build_proposal(
            local_solution={**dict(crude_rates), **dict(intermediate_prod)},
            prices=shadow_prices,
            consensus=consensus_z,
            iteration=iterations,
            linking_flows=intermediate_prod,
            local_obj=0.0,
            status="Optimal",
        )
        blender = self.agents["Blender"].build_proposal(
            local_solution={**dict(product_rates), **dict(intermediate_use)},
            prices=shadow_prices,
            consensus=consensus_z,
            iteration=iterations,
            linking_flows=intermediate_use,
            local_obj=0.0,
            status="Optimal",
        )
        proposals = [cdu, blender]
        if include_conversion_units:
            for key, rates, label in (
                ("FCC", fcc_rates or intermediate_prod, "FCC"),
                ("Coker", coker_rates or intermediate_prod, "Coker"),
                ("Reformer", reformer_rates or product_rates, "Reformer"),
            ):
                if key in self.agents:
                    proposals.append(
                        self.agents[key].build_proposal(
                            local_solution=dict(rates),
                            prices=shadow_prices,
                            consensus=consensus_z,
                            iteration=iterations,
                            linking_flows=dict(rates),
                            status="Optimal",
                        )
                    )
        if include_tank_utilities:
            if "Tank" in self.agents:
                tank = self.agents["Tank"].build_proposal(
                    local_solution=dict(intermediate_prod),
                    prices=shadow_prices,
                    consensus=consensus_z,
                    iteration=iterations,
                    linking_flows=intermediate_prod,
                    status="Optimal",
                )
                proposals.append(tank)
            if "Utilities" in self.agents:
                util = self.agents["Utilities"].build_proposal(
                    local_solution={
                        "charge_kbd": sum(float(v) for v in crude_rates.values())
                    },
                    prices=shadow_prices,
                    consensus=consensus_z,
                    iteration=iterations,
                    status="Optimal",
                )
                proposals.append(util)

        decision = self.master.decide(
            iteration=max(iterations - 1, 0) if iterations else 0,
            prices=shadow_prices,
            consensus=consensus_z,
            residual_norm=residual_norm,
            dual_residual_norm=dual_residual_norm,
            global_obj=objective,
            proposals=proposals,
            tol=tol,
            max_iter=max_iter,
        )
        return MultiAgentLayerResult(proposals=proposals, master=decision)


def collect_suggestions(proposals: Sequence[SubAgentProposal]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in proposals:
        for s in p.suggestions:
            d = s.to_dict()
            d["block"] = p.block
            out.append(d)
    return out


def inject_suggestions_into_solver_context(
    base_context: Mapping[str, Any],
    suggestions: Sequence[Mapping[str, Any]],
    *,
    apply_yield_nudge: bool = False,
) -> Dict[str, Any]:
    """Merge soft suggestions into next-solve context (never hard RHS by default)."""
    ctx = dict(base_context)
    ctx["llm_suggestions"] = [dict(s) for s in suggestions]
    if apply_yield_nudge:
        nudges: Dict[str, float] = dict(ctx.get("soft_yield_nudge") or {})
        for s in suggestions:
            if (
                s.get("kind") == "yield_nonlinear"
                and s.get("stream")
                and s.get("delta_frac") is not None
            ):
                stream = str(s["stream"])
                nudges[stream] = nudges.get(stream, 0.0) + float(s["delta_frac"])
        if nudges:
            ctx["soft_yield_nudge"] = nudges
    return ctx


def demo_round(
    *,
    llm_mode: str = "stub",
    prices: Optional[Mapping[str, float]] = None,
    consensus: Optional[Mapping[str, float]] = None,
    residual_norm: float = 0.05,
    iteration: int = 0,
) -> Dict[str, Any]:
    """Self-contained smoke demo of the agent layer (no ADMM required)."""
    prices = dict(
        prices
        or {
            "naphtha": 2.5,
            "distillate": 3.0,
            "gasoil": 1.2,
            "residue": -0.5,
            "fcc_naphtha": 2.2,
            "coker_naphtha": 2.0,
            "reformate": 3.5,
        }
    )
    consensus = dict(
        consensus
        or {
            "naphtha": 30.0,
            "distillate": 35.0,
            "gasoil": 25.0,
            "residue": 20.0,
            "fcc_naphtha": 15.0,
            "coker_naphtha": 8.0,
            "reformate": 18.0,
        }
    )

    def _mock(name: str):
        def fn(prices, consensus, **kw):
            flows = {
                k: float(v) * (0.95 if name == BlockName.BLENDER.value else 1.0)
                for k, v in consensus.items()
            }
            return {
                "status": "Optimal",
                "proposal": flows,
                "linking_flows": flows,
                "local_obj": sum(prices.get(k, 0.0) * v for k, v in flows.items()),
                "reduced_costs": {k: 0.01 for k in flows},
                "local_duals": {},
            }

        return fn

    solvers = {name: _mock(name) for name in DEFAULT_BLOCKS}
    layer = MultiAgentLayer.create(llm_mode=llm_mode, solvers=solvers)
    return layer.coordinate(
        prices,
        consensus,
        residual_norm=residual_norm,
        iteration=iteration,
        global_obj=1000.0,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(demo_round(), indent=2))
