"""LLM backends: deterministic stub + optional OpenAI-compatible HTTP client.

Stub is the default so demos and CI need no API keys. Real calls use the
OpenAI Chat Completions shape (works with OpenAI, Grok/xAI, Ollama, vLLM, etc.).

Env detection (auto / live):
  API key (first match):  XAI_API_KEY | OPENAI_API_KEY | LLM_API_KEY
  Base URL:               LLM_BASE_URL | OPENAI_BASE_URL |
                          https://api.x.ai/v1  when XAI_API_KEY is the key source,
                          else https://api.openai.com/v1
  Model:                  LLM_MODEL | OPENAI_MODEL |
                          grok-4 (xAI) | gpt-4o-mini (OpenAI)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .schemas import BlockName, Suggestion, SuggestionKind, parse_json_object

# Public defaults for live Grok wiring
XAI_DEFAULT_BASE_URL = "https://api.x.ai/v1"
XAI_DEFAULT_MODEL = "grok-4"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
# Alternate xAI models users may set via LLM_MODEL
XAI_KNOWN_MODELS = ("grok-4", "grok-3-mini", "grok-3", "grok-2")


def detect_llm_env() -> Dict[str, Optional[str]]:
    """Return resolved env path for live LLM wiring (no network).

    Preference order for keys: XAI_API_KEY, OPENAI_API_KEY, LLM_API_KEY.
    When the chosen key is XAI_API_KEY (or LLM_API_KEY with no OpenAI base),
    defaults point at xAI Grok.
    """
    xai_key = os.environ.get("XAI_API_KEY") or None
    openai_key = os.environ.get("OPENAI_API_KEY") or None
    llm_key = os.environ.get("LLM_API_KEY") or None

    if xai_key:
        key, key_source = xai_key, "XAI_API_KEY"
    elif openai_key:
        key, key_source = openai_key, "OPENAI_API_KEY"
    elif llm_key:
        key, key_source = llm_key, "LLM_API_KEY"
    else:
        key, key_source = None, None

    explicit_base = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    explicit_model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or None
    )

    use_xai = False
    if key_source == "XAI_API_KEY":
        use_xai = True
    elif explicit_base and "x.ai" in explicit_base:
        use_xai = True
    elif key_source == "LLM_API_KEY" and not openai_key and not (
        explicit_base and "openai.com" in explicit_base
    ):
        # LLM_API_KEY alone → prefer xAI Grok defaults (Wave2 live path)
        use_xai = True

    if explicit_base:
        base_url = explicit_base.rstrip("/")
    elif use_xai or key is None:
        # Prefer xAI Grok defaults for this project when key is XAI/LLM or unset
        base_url = XAI_DEFAULT_BASE_URL
    else:
        base_url = OPENAI_DEFAULT_BASE_URL

    if explicit_model:
        model = explicit_model
    elif use_xai or key is None:
        model = XAI_DEFAULT_MODEL
    else:
        model = OPENAI_DEFAULT_MODEL

    if key and use_xai:
        provider: Optional[str] = "xai"
    elif key:
        provider = "openai"
    else:
        provider = None

    return {
        "api_key": key,
        "key_source": key_source,
        "base_url": base_url,
        "model": model,
        "provider": provider,
    }


def has_llm_api_key() -> bool:
    """True if any supported API key env var is set."""
    return bool(
        os.environ.get("XAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )


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
            r"You are the\s+(CDU|FCC|Delayed Coker|Coker|Reformer|Tank Farm|Tank|Blender|Utilities)\b",
            text,
        )
        if m:
            name = m.group(1)
            if name in ("Tank Farm", "Tank"):
                return BlockName.TANK.value
            if name in ("Delayed Coker", "Coker"):
                return BlockName.COKER.value
            return name
        for name in (
            BlockName.UTILITIES.value,
            BlockName.BLENDER.value,
            BlockName.REFORMER.value,
            BlockName.COKER.value,
            BlockName.FCC.value,
            BlockName.TANK.value,
            BlockName.CDU.value,
        ):
            if re.search(rf"\b{re.escape(name)}\b Block Agent", text):
                return name
        if "Tank Farm" in text or re.search(r"\bTank\b", text):
            return BlockName.TANK.value
        if re.search(r"\bFCC\b", text):
            return BlockName.FCC.value
        if re.search(r"\bCoker\b", text):
            return BlockName.COKER.value
        if re.search(r"\bReformer\b", text):
            return BlockName.REFORMER.value
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
            BlockName.FCC.value: Suggestion(
                kind=SuggestionKind.YIELD_NONLINEAR.value,
                message=(
                    "Gasoil CCR/N rising with heavier crude → conversion slip and more "
                    "slurry. Soft FCC naphtha yield haircut ~0.01–0.02 and watch cat "
                    "activity if feed metals climb."
                ),
                stream="fcc_naphtha",
                delta_frac=-0.015,
                confidence=0.68,
            ),
            BlockName.COKER.value: Suggestion(
                kind=SuggestionKind.CAPACITY_HINT.value,
                message=(
                    "Resid quality (high CCR) may force longer cycle / lower liquid "
                    "yields than the linear vector. Soft capacity headroom note; "
                    "solver feed caps remain binding."
                ),
                stream="coker_naphtha",
                delta_frac=None,
                confidence=0.62,
            ),
            BlockName.REFORMER.value: Suggestion(
                kind=SuggestionKind.YIELD_NONLINEAR.value,
                message=(
                    "Naphtha aromatics / N may understate severity need vs linear "
                    "reformate yield. Soft reformate yield nudge −0.01 if feed N high; "
                    "octane pool still owned by blender LP."
                ),
                stream="reformate",
                delta_frac=-0.01,
                confidence=0.64,
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
                    "block": "FCC",
                    "message": "Conversion slip on high-CCR gasoil",
                    "why": "Linear FCC yields miss severity/feed quality effects",
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
                "fcc_naphtha": "FCC naphtha dual feeds reformer/octane economics.",
            },
        }


class OpenAICompatClient(LLMClient):
    """Minimal OpenAI-compatible Chat Completions client (stdlib only).

    Supports OpenAI, xAI Grok (https://api.x.ai/v1), Ollama, vLLM, etc.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        env = detect_llm_env()
        self.api_key = api_key or env["api_key"]
        self.base_url = (base_url or env["base_url"] or OPENAI_DEFAULT_BASE_URL).rstrip("/")
        self.model = model or env["model"] or OPENAI_DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.provider = env.get("provider")
        if not self.api_key:
            raise ValueError(
                "OpenAICompatClient requires XAI_API_KEY / OPENAI_API_KEY / LLM_API_KEY"
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
                        "Respond with a single valid JSON object only. "
                        "Never invent hard constraint overrides; solvers own feasibility. "
                        "Do not rewrite proposal numbers or local_obj — only soft suggestions."
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
      - "stub": always StubLLMClient (default for offline/CI)
      - "openai" / "real" / "live": OpenAICompatClient (requires key; raises if missing)
      - "auto": real if a key is present, else stub
    """
    mode = (mode or "auto").lower()
    if mode == "stub":
        return StubLLMClient()
    if mode in ("openai", "real", "live", "xai", "grok"):
        return OpenAICompatClient(**kwargs)
    # auto
    key = kwargs.get("api_key") or detect_llm_env()["api_key"]
    if key:
        try:
            return OpenAICompatClient(
                api_key=key,
                **{k: v for k, v in kwargs.items() if k != "api_key"},
            )
        except Exception:
            return StubLLMClient()
    return StubLLMClient()
