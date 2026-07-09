"""Smoke tests for LLM multi-agent layer (stub mode, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pims_admm_llm.agents import (  # noqa: E402
    BlockName,
    MasterCoordinatorAgent,
    MultiAgentLayer,
    StubLLMClient,
    SubAgent,
    make_llm_client,
    render_master_prompt,
    render_subagent_prompt,
)
from pims_admm_llm.agents.layer import (  # noqa: E402
    collect_suggestions,
    demo_round,
    inject_suggestions_into_solver_context,
)
from pims_admm_llm.agents.schemas import (  # noqa: E402
    MasterDecision,
    SubAgentProposal,
    Suggestion,
    parse_json_object,
)
from pims_admm_llm.agents.prompts import BLOCK_PROMPTS, MASTER_PROMPT, list_blocks  # noqa: E402


def test_block_prompts_exist():
    blocks = list_blocks()
    assert blocks == ["CDU", "Tank", "Blender", "Utilities"]
    for b in blocks:
        assert b in BLOCK_PROMPTS
        assert "JSON" in BLOCK_PROMPTS[b] or "json" in BLOCK_PROMPTS[b]
    assert "Master Coordinator" in MASTER_PROMPT


def test_render_prompts_fill_placeholders():
    p = render_subagent_prompt(
        "CDU",
        prices={"naphtha": 1.0},
        consensus={"naphtha": 10.0},
        local_solution={"status": "Optimal", "local_obj": 1.0},
        local_data={"cdu_capacity_kbd": 120},
        iteration=3,
    )
    assert "naphtha" in p
    assert "3" in p
    m = render_master_prompt(
        iteration=2,
        prices={"naphtha": 1.0},
        consensus={"naphtha": 10.0},
        residual_norm=0.01,
        proposals=[],
    )
    assert "0.01" in m


def test_stub_llm_subagent_and_master():
    client = StubLLMClient()
    sub = client.complete_json(BLOCK_PROMPTS["CDU"])
    assert sub["block"] == "CDU"
    assert sub["suggestion"]["message"]
    master = client.complete_json(
        "You are the Refinery Master Coordinator\n"
        "Iteration: 0\nPrimal residual norm: 0.05\n"
        "Convergence tol: 0.001\nMax iterations: 50\n"
    )
    assert master["action"] == "continue"
    master2 = client.complete_json(
        "You are the Refinery Master Coordinator\n"
        "Iteration: 50\nPrimal residual norm: 1e-6\n"
        "Convergence tol: 0.001\nMax iterations: 50\n"
    )
    assert master2["action"] == "terminate"


def test_subagent_propose_without_solver():
    agent = SubAgent(name="CDU", llm=StubLLMClient(), enable_llm=True)
    prop = agent.propose({"naphtha": 1.0}, {"naphtha": 10.0}, iteration=1)
    assert prop.block == "CDU"
    assert prop.status == "NoSolver"
    assert len(prop.suggestions) == 1
    assert prop.suggestions[0].kind == "yield_nonlinear"
    # round-trip JSON
    again = SubAgentProposal.from_dict(json.loads(prop.to_json()))
    assert again.block == "CDU"


def test_subagent_solver_hard_constraints_untouched():
    def solve_fn(prices, consensus, **kw):
        return {
            "status": "Optimal",
            "proposal": {"naphtha": 12.0},
            "linking_flows": {"naphtha": 12.0},
            "local_obj": 99.0,
            "reduced_costs": {"naphtha": 0.0},
        }

    agent = SubAgent(name="CDU", llm=StubLLMClient(), solve_fn=solve_fn)
    prop = agent.propose({}, {}, iteration=0)
    # solver numbers preserved exactly
    assert prop.proposal["naphtha"] == 12.0
    assert prop.local_obj == 99.0
    # LLM only added soft suggestion
    assert prop.suggestions


def test_master_residual_gate():
    master = MasterCoordinatorAgent(llm=StubLLMClient(), tol=1e-3, max_iter=10)
    d = master.decide(
        iteration=10,
        prices={"naphtha": 1.0},
        consensus={"naphtha": 1.0},
        proposals=[],
        residual_norm=0.5,  # still large but max_iter hit
    )
    assert d.action == "terminate"
    d2 = master.decide(
        iteration=1,
        prices={"naphtha": 1.0},
        consensus={"naphtha": 1.0},
        proposals=[],
        residual_norm=1e-6,
    )
    assert d2.action == "terminate"


def test_multi_agent_layer_demo_round():
    out = demo_round(llm_mode="stub", residual_norm=0.05, iteration=0)
    assert "proposals" in out and "decision" in out
    assert len(out["proposals"]) == 4
    blocks = {p["block"] for p in out["proposals"]}
    assert blocks == {"CDU", "Tank", "Blender", "Utilities"}
    assert out["decision"]["action"] == "continue"
    assert out["suggestions_only"]
    # terminate path
    out2 = demo_round(llm_mode="stub", residual_norm=1e-6, iteration=0)
    assert out2["decision"]["action"] == "terminate"


def test_inject_suggestions_soft_only():
    base = {"rho": 1.0}
    sugs = [
        {
            "kind": "yield_nonlinear",
            "stream": "naphtha",
            "delta_frac": -0.015,
            "message": "haircut",
        }
    ]
    ctx = inject_suggestions_into_solver_context(base, sugs, apply_yield_nudge=True)
    assert ctx["rho"] == 1.0
    assert "llm_suggestions" in ctx
    assert ctx["soft_yield_nudge"]["naphtha"] == -0.015
    # without flag, no nudge map mutation beyond metadata
    ctx2 = inject_suggestions_into_solver_context(base, sugs, apply_yield_nudge=False)
    assert "soft_yield_nudge" not in ctx2


def test_parse_json_object_fences():
    raw = '```json\n{"action": "continue", "reasoning": "ok"}\n```'
    obj = parse_json_object(raw)
    assert obj["action"] == "continue"


def test_make_llm_client_stub():
    c = make_llm_client("stub")
    assert isinstance(c, StubLLMClient)


def test_master_decision_schema_roundtrip():
    d = MasterDecision(
        action="continue",
        new_prices={"naphtha": 1.5},
        reasoning="test",
        residual_norm=0.1,
        iteration=2,
    )
    d2 = MasterDecision.from_dict(json.loads(d.to_json()))
    assert d2.action == "continue"
    assert d2.new_prices["naphtha"] == 1.5


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
