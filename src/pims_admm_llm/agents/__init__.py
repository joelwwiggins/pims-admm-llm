"""LLM multi-agent layer for block-angular refinery planning.

Hard constraints always stay with the LP solvers (PuLP/CBC). Agents only
propose nonlinear/yield suggestions, warm-starts, and economic notes via
structured JSON. Optional real LLM (OpenAI-compatible) or deterministic stub.
"""

from .schemas import (
    BlockName,
    MasterDecision,
    SubAgentProposal,
    Suggestion,
)
from .prompts import (
    BLOCK_PROMPTS,
    MASTER_PROMPT,
    list_blocks,
    render_subagent_prompt,
    render_master_prompt,
)
from .llm_client import LLMClient, StubLLMClient, OpenAICompatClient, make_llm_client
from .subagent import SubAgent, default_block_agents
from .master import MasterCoordinatorAgent
from .layer import (
    MultiAgentLayer,
    MultiAgentLayerResult,
    collect_suggestions,
    demo_round,
    inject_suggestions_into_solver_context,
)

__all__ = [
    "BlockName",
    "MasterDecision",
    "SubAgentProposal",
    "Suggestion",
    "BLOCK_PROMPTS",
    "MASTER_PROMPT",
    "list_blocks",
    "render_subagent_prompt",
    "render_master_prompt",
    "LLMClient",
    "StubLLMClient",
    "OpenAICompatClient",
    "make_llm_client",
    "SubAgent",
    "default_block_agents",
    "MasterCoordinatorAgent",
    "MultiAgentLayer",
    "MultiAgentLayerResult",
    "collect_suggestions",
    "demo_round",
    "inject_suggestions_into_solver_context",
]
