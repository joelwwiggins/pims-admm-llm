"""LLM multi-agent layer for block-angular refinery planning.

Hard constraints always stay with the LP solvers (PuLP/CBC). Agents only
propose nonlinear/yield suggestions, warm-starts, and economic notes via
structured JSON. Optional real LLM (OpenAI-compatible / xAI Grok) or
deterministic stub.
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
from .llm_client import (
    LLMClient,
    StubLLMClient,
    OpenAICompatClient,
    make_llm_client,
    detect_llm_env,
    has_llm_api_key,
    XAI_DEFAULT_BASE_URL,
    XAI_DEFAULT_MODEL,
)
from .subagent import SubAgent, default_block_agents, DEFAULT_BLOCKS
from .master import MasterCoordinatorAgent
from .layer import (
    MultiAgentLayer,
    MultiAgentLayerResult,
    collect_suggestions,
    demo_round,
    inject_suggestions_into_solver_context,
)
from .process_network import (
    AreaInput,
    AreaReport,
    ClosedLoopResult,
    ProcessNetworkRound,
    Pushback,
    build_area_situations,
    format_closed_loop,
    format_process_network_round,
    node_badges_from_round,
    propose_replan_actions,
    run_closed_loop,
    run_process_network_round,
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
    "detect_llm_env",
    "has_llm_api_key",
    "XAI_DEFAULT_BASE_URL",
    "XAI_DEFAULT_MODEL",
    "SubAgent",
    "default_block_agents",
    "DEFAULT_BLOCKS",
    "MasterCoordinatorAgent",
    "MultiAgentLayer",
    "MultiAgentLayerResult",
    "collect_suggestions",
    "demo_round",
    "inject_suggestions_into_solver_context",
    "AreaInput",
    "AreaReport",
    "ClosedLoopResult",
    "ProcessNetworkRound",
    "Pushback",
    "build_area_situations",
    "format_closed_loop",
    "format_process_network_round",
    "node_badges_from_round",
    "propose_replan_actions",
    "run_closed_loop",
    "run_process_network_round",
]
