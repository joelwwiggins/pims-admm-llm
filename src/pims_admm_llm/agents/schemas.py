"""Structured JSON schemas for sub-agent proposals and master decisions.

All agent I/O is forced into these shapes so solvers can ignore free-form text
and only apply validated, soft suggestions (never hard-constraint overrides).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BlockName(str, Enum):
    CDU = "CDU"
    FCC = "FCC"
    COKER = "Coker"  # Delayed coker unit
    REFORMER = "Reformer"
    TANK = "Tank"
    BLENDER = "Blender"
    UTILITIES = "Utilities"


class SuggestionKind(str, Enum):
    YIELD_NONLINEAR = "yield_nonlinear"
    WARM_START = "warm_start"
    CAPACITY_HINT = "capacity_hint"
    BUSINESS_RULE = "business_rule"
    UNCERTAINTY = "uncertainty"
    OTHER = "other"


@dataclass
class Suggestion:
    """Soft intelligence the LLM may inject; solvers never treat as binding."""

    kind: str
    message: str
    stream: Optional[str] = None
    delta_frac: Optional[float] = None  # e.g. +0.02 yield nudge (informational)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Suggestion":
        return cls(
            kind=str(d.get("kind", SuggestionKind.OTHER.value)),
            message=str(d.get("message", "")),
            stream=d.get("stream"),
            delta_frac=(
                float(d["delta_frac"]) if d.get("delta_frac") is not None else None
            ),
            confidence=float(d.get("confidence", 0.5)),
        )


@dataclass
class SubAgentProposal:
    """One block's response after local LP solve + optional LLM augmentation.

    HARD RULE: ``proposal`` and ``local_obj`` come only from the solver / ADMM
    block result. The LLM may only append to ``suggestions`` (and ``note``).
    """

    block: str
    proposal: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    local_obj: float = 0.0
    linking_flows: Dict[str, float] = field(default_factory=dict)
    local_duals: Dict[str, float] = field(default_factory=dict)
    suggestions: List[Suggestion] = field(default_factory=list)
    note: str = ""
    status: str = "Optimal"
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block": self.block,
            "proposal": {k: float(v) for k, v in self.proposal.items()},
            "reduced_costs": {k: float(v) for k, v in self.reduced_costs.items()},
            "local_obj": float(self.local_obj),
            "linking_flows": {k: float(v) for k, v in self.linking_flows.items()},
            "local_duals": {k: float(v) for k, v in self.local_duals.items()},
            "suggestions": [s.to_dict() for s in self.suggestions],
            "note": self.note,
            "status": self.status,
            "iteration": int(self.iteration),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubAgentProposal":
        suggestions = [
            Suggestion.from_dict(s) if isinstance(s, dict) else Suggestion(kind="other", message=str(s))
            for s in d.get("suggestions", [])
        ]
        return cls(
            block=str(d.get("block", "")),
            proposal={k: float(v) for k, v in (d.get("proposal") or {}).items()},
            reduced_costs={k: float(v) for k, v in (d.get("reduced_costs") or {}).items()},
            local_obj=float(d.get("local_obj", 0.0) or 0.0),
            linking_flows={k: float(v) for k, v in (d.get("linking_flows") or {}).items()},
            local_duals={k: float(v) for k, v in (d.get("local_duals") or {}).items()},
            suggestions=suggestions,
            note=str(d.get("note", "")),
            status=str(d.get("status", "Optimal")),
            iteration=int(d.get("iteration", 0)),
        )


@dataclass
class MasterDecision:
    """Coordinator output after aggregating sub-agent proposals."""

    action: str  # "continue" | "terminate"
    new_prices: Dict[str, float] = field(default_factory=dict)
    consensus_targets: Dict[str, float] = field(default_factory=dict)
    cuts: List[str] = field(default_factory=list)
    applied_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    global_obj_estimate: float = 0.0
    residual_norm: float = 0.0
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "new_prices": {k: float(v) for k, v in self.new_prices.items()},
            "consensus_targets": {
                k: float(v) for k, v in self.consensus_targets.items()
            },
            "cuts": list(self.cuts),
            "applied_suggestions": list(self.applied_suggestions),
            "reasoning": self.reasoning,
            "global_obj_estimate": float(self.global_obj_estimate),
            "residual_norm": float(self.residual_norm),
            "iteration": int(self.iteration),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MasterDecision":
        return cls(
            action=str(d.get("action", "continue")),
            new_prices={k: float(v) for k, v in (d.get("new_prices") or {}).items()},
            consensus_targets={
                k: float(v) for k, v in (d.get("consensus_targets") or {}).items()
            },
            cuts=list(d.get("cuts") or []),
            applied_suggestions=list(d.get("applied_suggestions") or []),
            reasoning=str(d.get("reasoning", "")),
            global_obj_estimate=float(d.get("global_obj_estimate", 0.0) or 0.0),
            residual_norm=float(d.get("residual_norm", 0.0) or 0.0),
            iteration=int(d.get("iteration", 0)),
        )


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model text (tolerates fences / preamble)."""
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fences
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # brace scan
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from LLM output: {text[:200]!r}")
