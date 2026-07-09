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
from .prompts import BLOCK_PROMPTS, MASTER_PROMPT, render_subagent_prompt, render_master_prompt
from .llm_client import LLMClient, StubLLMClient, OpenAICompatClient, make_llm_client
from .subagent import SubAgent
from .master import MasterCoordinatorAgent
from .layer import MultiAgentLayer

__all__ = [
    "BlockName",
    "MasterDecision",
    "SubAgentProposal",
    "Suggestion",
    "BLOCK_PROMPTS",
    "MASTER_PROMPT",
    "render_subagent_prompt",
    "render_master_prompt",
    "LLMClient",
    "StubLLMClient",
    "OpenAICompatClient",
    "make_llm_client",
    "SubAgent",
    "MasterCoordinatorAgent",
    "MultiAgentLayer",
]
