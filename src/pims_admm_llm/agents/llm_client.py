"""LLM backends: deterministic stub + optional OpenAI-compatible HTTP client.

Stub is the default so demos and CI need no API keys. Real calls use the
OpenAI Chat Completions shape (works with OpenAI, Grok/xAI, Ollama, vLLM, etc.).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from .schemas import BlockName, Suggestion, SuggestionKind, parse_json_object


class LLMClient(ABC):
    @abstractmethod
    def complete_json(self, system_or_prompt: str, *, temperature: float = 0.2) -> Dict[str, Any]:
        """Return a parsed JSON object from the model (or stub)."""


class StubLLMClient(LLMClient):
    """Rule-based stub that emits plausible structured suggestions without network."""

    def complete_json(self, system_or_prompt: str, *, temperature: float = 0.2) -> Dict[str, Any]:
        text = system_or_prompt
        # Detect master vs sub-agent by prompt content
        if "Master Coordinator" in text or "highlighted_suggestions" in text:
            return self._master(text)
        return self._subagent(text)

    def _detect_block(self, text: str) -> str:
        # Prefer explicit role line near the top (Tank prompt also mentions CDU).
        m = re.search(
            r"You are the\s+(CDU|Tank Farm|Tank|Blender|Utilities)\b",
            text,
        )
        if m:
            name = m.group(1)
            if name in ("Tank Farm", "Tank"):
                return BlockName.TANK.value
            return name
        for name in (
            BlockName.UTILITIES.value,
            BlockName.BLENDER.value,
            BlockName.TANK.value,
            BlockName.CDU.value,
        ):
            if re.search(rf"\b{re.escape(name)}\b Block Agent", text):
                return name
        if "Tank Farm" in text:
            return BlockName.TANK.value
        return BlockName.CDU.value

    def _subagent(self, text: str) -> Dict[str, Any]:
        block = self._detect_block(text)
        catalog = {
            BlockName.CDU.value: Suggestion(
                kind=SuggestionKind.YIELD_NONLINEAR.value,
                message=(
                    "With heavier slate (lower API / higher S), linear yields understate "
                    "residue and overstate naphtha ~1–2 vol%. Consider a soft naphtha "
                    "yield haircut of 0.01–0.02 on Maya-like crudes before next cut."
                ),
                stream="naphtha",
                delta_frac=-0.015,
                confidence=0.72,
            ),
            BlockName.TANK.value: Suggestion(
                kind=SuggestionKind.CAPACITY_HINT.value,
                message=(
                    "Intermediate inventory headroom looks tight on distillate. Extra "
                    "5 kbd-days of tank working capacity would likely have positive "
                    "marginal value near current shadow prices."
                ),
                stream="distillate",
                delta_frac=None,
                confidence=0.65,
            ),
            BlockName.BLENDER.value: Suggestion(
                kind=SuggestionKind.BUSINESS_RULE.value,
                message=(
                    "Gasoline recipe is naphtha-heavy; mild distillate giveaway may be "
                    "hiding octane-pool flexibility. Soft note only — solver recipes stay binding."
                ),
                stream="gasoline",
                delta_frac=None,
                confidence=0.58,
            ),
            BlockName.UTILITIES.value: Suggestion(
                kind=SuggestionKind.UNCERTAINTY.value,
                message=(
                    "Fuel-gas balance may swing with heavier crude charge; watch steam "
                    "and fuel co-firing cost if residue disposition changes."
                ),
                stream="fuel_gas",
                delta_frac=None,
                confidence=0.55,
            ),
        }
        sug = catalog.get(block, catalog[BlockName.CDU.value])
        return {
            "block": block,
            "suggestion": sug.to_dict(),
            "note": f"stub-{block.lower()}",
        }

    def _master(self, text: str) -> Dict[str, Any]:
        # Parse residual if present
        residual = 1.0
        m = re.search(r"Primal residual norm:\s*([0-9.eE+-]+)", text)
        if m:
            try:
                residual = float(m.group(1))
            except ValueError:
                pass
        iteration = 0
        m2 = re.search(r"Iteration:\s*(\d+)", text)
        if m2:
            iteration = int(m2.group(1))
        tol = 1e-3
        m3 = re.search(r"Convergence tol:\s*([0-9.eE+-]+)", text)
        if m3:
            try:
                tol = float(m3.group(1))
            except ValueError:
                pass
        max_iter = 50
        m4 = re.search(r"Max iterations:\s*(\d+)", text)
        if m4:
            max_iter = int(m4.group(1))

        action = "terminate" if (residual <= tol or iteration >= max_iter) else "continue"
        return {
            "action": action,
            "reasoning": (
                f"ADMM residual={residual:.4g}, iter={iteration}/{max_iter}. "
                + (
                    "Within tolerance — publish plan and shadow prices for make-buy-sell."
                    if action == "terminate"
                    else "Continue ADMM; prices still adjusting linking balances."
                )
            ),
            "highlighted_suggestions": [
                {
                    "block": "CDU",
                    "message": "Nonlinear yield haircut on heavy crude naphtha",
                    "why": "Protects gasoline pool quality assumptions PIMS linearizes",
                },
                {
                    "block": "Tank",
                    "message": "Distillate tank headroom tight",
                    "why": "Signals value of extra intermediate storage vs spot sales",
                },
            ],
            "price_commentary": {
                "naphtha": "Shadow price ≈ marginal value of an extra barrel of naphtha to the blender.",
                "distillate": "Positive dual → extra distillate raises margin; check diesel demand headroom.",
            },
        }


class OpenAICompatClient(LLMClient):
    """Minimal OpenAI-compatible Chat Completions client (stdlib only)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "XAI_API_KEY"
        ) or os.environ.get("LLM_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or os.environ.get(
            "OPENAI_MODEL"
        ) or "gpt-4o-mini"
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError(
                "OpenAICompatClient requires OPENAI_API_KEY / XAI_API_KEY / LLM_API_KEY"
            )

    def complete_json(self, system_or_prompt: str, *, temperature: float = 0.2) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful refinery planning agent. "
                        "Respond with a single valid JSON object only."
                    ),
                },
                {"role": "user", "content": system_or_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {e.code}: {err_body[:500]}") from e

        content = payload["choices"][0]["message"]["content"]
        return parse_json_object(content)


def make_llm_client(
    mode: str = "auto",
    **kwargs: Any,
) -> LLMClient:
    """Factory.

    mode:
      - "stub": always StubLLMClient
      - "openai" / "real": OpenAICompatClient (requires key)
      - "auto": real if a key is present, else stub
    """
    mode = (mode or "auto").lower()
    if mode == "stub":
        return StubLLMClient()
    if mode in ("openai", "real", "live"):
        return OpenAICompatClient(**kwargs)
    # auto
    key = (
        kwargs.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if key:
        try:
            return OpenAICompatClient(api_key=key, **{k: v for k, v in kwargs.items() if k != "api_key"})
        except Exception:
            return StubLLMClient()
    return StubLLMClient()
