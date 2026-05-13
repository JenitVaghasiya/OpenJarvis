"""LocalCloudAgent — shared base for hybrid local+cloud paradigm agents.

The hybrid paradigms (Minions, Conductor, Archon, Advisors, SkillOrchestra,
ToolOrchestra) all coordinate at least two models: a small **local** model
served by vLLM over an OpenAI-compatible endpoint, and a **cloud** model
reached via the Anthropic or OpenAI SDK.

Why not just use OpenJarvis's :class:`InferenceEngine` for both? Two reasons:

1. The reference hybrid adapters (``hybrid-local-cloud-compute/adapters/``) make
   raw SDK calls because some of them (Minions, Archon) construct external
   library objects that themselves create their own SDK clients. We mirror that
   here so the n=500 numbers stay reproducible during the port.
2. Cloud-side quirks (Opus 4.7 temperature stripping, GPT-5 family
   ``max_completion_tokens``) are paradigm-shaped — Minions needs structured
   outputs on the supervisor turn, SkillOrchestra needs them on the router,
   baseline_cloud does not. Keeping the SDK calls in the agent layer lets each
   paradigm decide the schema rather than fighting a shared engine API.

The base class therefore provides only:

- Standard ``run()`` contract returning an :class:`AgentResult` whose
  ``metadata`` carries the hybrid-result fields (``tokens_local``,
  ``tokens_cloud``, ``cost_usd``, ``latency_s``, ``traces``).
- ``_call_anthropic`` / ``_call_openai`` / ``_call_vllm`` helpers that handle
  Opus 4.7 temperature stripping, GPT-5 token-arg naming, vLLM
  ``enable_thinking`` kwargs, and basic token bookkeeping.
- ``_soft_fail_metadata`` for deterministic failure rows (e.g. Qwen JSON
  malformation) so the runner doesn't crash the whole cell.

Agents register themselves with ``@AgentRegistry.register("name")`` and become
discoverable via the existing SDK / CLI flow. The runner constructs them with
the cloud ``(engine, model)`` as the canonical pair, and paradigm-specific
kwargs (``local_model``, ``local_endpoint``, ``cloud_endpoint``, …) follow.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.agents.hybrid._prices import (
    NO_TEMP_PREFIXES,
    cost as estimate_cost,
    is_gpt5_family,
    supports_temperature,
)
from openjarvis.engine._stubs import InferenceEngine


# Anthropic server-side web_search: $10 per 1000 searches.
WEB_SEARCH_COST_PER_CALL = 0.01

ANTHROPIC_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8,
}


class LocalCloudAgent(BaseAgent):
    """Base for paradigm agents that coordinate a local + cloud model pair.

    Subclasses implement :meth:`_run_paradigm` rather than ``run`` so the
    base can wrap timing, metadata shaping, and soft-fail handling
    uniformly.

    The :meth:`run` contract takes the formatted task prompt as ``input``
    and reads paradigm-shaped data from ``context.metadata``:

    - ``context.metadata["task"]``: optional dict (the bench's raw task
      row, used by paradigms that look at hints / problem_statement / etc.).
    - ``context.metadata["task_id"]``: optional string identifier.

    Construction args:
    - ``engine``, ``model``: the cloud engine + model id (satisfies
      :class:`BaseAgent`'s contract; only used incidentally — we make raw
      SDK calls).
    - ``local_model``, ``local_endpoint``: vLLM-served local model and its
      OpenAI-compatible endpoint, e.g. ``"http://localhost:8001/v1"``.
    - ``cloud_endpoint``: ``"anthropic"`` or ``"openai"`` — picks the
      cloud SDK.
    - ``cfg``: paradigm-specific knobs (max_tokens, schemas, mode, …).
    """

    accepts_tools: bool = False

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        local_model: Optional[str] = None,
        local_endpoint: Optional[str] = None,
        cloud_endpoint: str = "anthropic",
        cfg: Optional[Dict[str, Any]] = None,
        bus: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._cloud_model = model
        self._cloud_endpoint = (cloud_endpoint or "anthropic").lower()
        self._local_model = local_model
        self._local_endpoint = local_endpoint
        self._cfg: Dict[str, Any] = dict(cfg or {})

    # ------------------------------------------------------------------
    # SDK call helpers — raw clients, paradigm-shaped quirks applied
    # ------------------------------------------------------------------

    @staticmethod
    def _call_anthropic(
        model: str,
        *,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
        output_config: Optional[dict] = None,
        timeout: float = 600.0,
        max_retries: int = 5,
    ) -> Tuple[str, int, int, int]:
        """Single Anthropic call. Returns (text, p_tok, c_tok, n_web_searches).

        Strips ``temperature`` for Opus 4.7+ (rejected by the API).
        """
        import anthropic

        client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        if supports_temperature(model):
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if output_config:
            kwargs["output_config"] = output_config
        msg = client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        srv = getattr(msg.usage, "server_tool_use", None)
        n_searches = getattr(srv, "web_search_requests", 0) if srv else 0
        return text, msg.usage.input_tokens, msg.usage.output_tokens, n_searches

    @staticmethod
    def _call_openai(
        model: str,
        *,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        response_format: Optional[dict] = None,
        timeout: float = 600.0,
    ) -> Tuple[str, int, int]:
        """Single OpenAI call. Returns (text, p_tok, c_tok)."""
        from openai import OpenAI

        client = OpenAI(timeout=timeout)
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if is_gpt5_family(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        p = getattr(u, "prompt_tokens", 0) if u else 0
        c = getattr(u, "completion_tokens", 0) if u else 0
        return text, p, c

    @staticmethod
    def _call_vllm(
        model: str,
        endpoint: str,
        *,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        timeout: float = 600.0,
    ) -> Tuple[str, int, int]:
        """Local vLLM (OpenAI-compatible) call. Returns (text, p_tok, c_tok)."""
        from openai import OpenAI

        client = OpenAI(base_url=endpoint, api_key="EMPTY", timeout=timeout)
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        p = getattr(u, "prompt_tokens", 0) if u else 0
        c = getattr(u, "completion_tokens", 0) if u else 0
        return text, p, c

    def _call_cloud(
        self,
        *,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[str, int, int]:
        """Dispatch a single cloud call by ``self._cloud_endpoint``.

        Returns (text, p_tok, c_tok). For Anthropic, the web_search count
        is discarded — paradigms that care should call ``_call_anthropic``
        directly.
        """
        if self._cloud_endpoint == "anthropic":
            text, p, c, _ = self._call_anthropic(
                self._cloud_model,
                user=user,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return text, p, c
        if self._cloud_endpoint == "openai":
            return self._call_openai(
                self._cloud_model,
                user=user,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        raise ValueError(f"unsupported cloud endpoint: {self._cloud_endpoint!r}")

    # ------------------------------------------------------------------
    # Result shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_fail_metadata(reason: str) -> Dict[str, Any]:
        """Metadata for a soft-fail row (Qwen JSON broke, Anthropic 400, etc.).

        The agent still returns an :class:`AgentResult` with empty content;
        the runner records it as score=0 without crashing the cell.
        """
        return {
            "tokens_local": 0,
            "tokens_cloud": 0,
            "cost_usd": 0.0,
            "latency_s": 0.0,
            "soft_error": reason,
            "traces": {"soft_error": reason},
        }

    @staticmethod
    def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return estimate_cost(model, prompt_tokens, completion_tokens)

    # ------------------------------------------------------------------
    # Run contract
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)
        t0 = time.time()
        try:
            answer, meta = self._run_paradigm(input, context, **kwargs)
        except Exception as exc:
            soft = self._is_soft_failure(exc)
            if soft is not None:
                meta = self._soft_fail_metadata(soft)
                self._emit_turn_end(soft_error=soft)
                return AgentResult(content="", metadata=meta, turns=0)
            raise
        meta.setdefault("latency_s", time.time() - t0)
        self._emit_turn_end(**{k: v for k, v in meta.items() if k != "traces"})
        return AgentResult(
            content=answer,
            metadata=meta,
            turns=int(meta.get("turns", 0) or 0),
        )

    @abstractmethod
    def _run_paradigm(
        self,
        input: str,
        context: Optional[AgentContext],
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run the paradigm. Return ``(final_answer, metadata)``.

        Metadata should include the hybrid-shape fields:
        ``tokens_local``, ``tokens_cloud``, ``cost_usd``, optional
        ``latency_s`` (the base fills it if absent), and a ``traces`` dict.
        """

    # Subclasses override to declare deterministic failure modes they
    # want the base to swallow into a soft-fail row.
    def _is_soft_failure(self, exc: BaseException) -> Optional[str]:
        return None


__all__ = [
    "ANTHROPIC_WEB_SEARCH_TOOL",
    "LocalCloudAgent",
    "NO_TEMP_PREFIXES",
    "WEB_SEARCH_COST_PER_CALL",
    "estimate_cost",
    "is_gpt5_family",
    "supports_temperature",
]
