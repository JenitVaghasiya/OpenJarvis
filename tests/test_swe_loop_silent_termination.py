"""Regression tests for silent-termination bugs in the cloud SWE-agent loop.

Pre-fix (observed 2026-05-15 on the n=100 skillorchestra-qwen-gpt5mini-swe
and skillorchestra-qwen-gemini25flash-swe cells):

- OpenAI loop: when ``finish_reason='length'`` truncated the model
  mid-response, ``tool_calls`` was empty and ``text`` was empty/short.
  The loop's ``if not tool_calls: break`` rule treated that as "model
  done" and exited with ``final_summary=""`` → answer became the
  ``[mini-swe-agent produced no summary text]`` placeholder → score 0.

- Gemini loop: when ``finish_reason=MALFORMED_FUNCTION_CALL`` (model
  tried to call ``bash`` but produced unparseable args), the response
  had no ``function_call`` parts and no text → same silent exit. This
  hit 24/100 tasks in the gemini-flash cell.

Both loops now inject a one-shot recovery nudge and continue the loop
instead of exiting, only treating natural ``stop`` / text-only ``STOP``
as termination.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------- OpenAI loop ----------

def _openai_message(
    *,
    content: str = "",
    tool_calls: List[Any] = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> Any:
    """Build a mock OpenAI ChatCompletion response."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _openai_tool_call(call_id: str, command: str) -> Any:
    """Build a mock tool_calls entry — the loop reads .id, .function.name,
    .function.arguments."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="bash", arguments=f'{{"command":"{command}"}}'),
    )


def test_openai_loop_recovers_from_length_truncation(tmp_path: Any) -> None:
    """When finish_reason='length' AND text is empty AND no tool_calls,
    the loop must NOT terminate — it should inject a recovery nudge and
    let the model retry. Pre-fix the loop exited silently with empty
    final_summary (silent failure on gpt-5-mini SWE cells)."""
    from openjarvis.agents.hybrid.mini_swe_agent import _loop_cloud_openai

    # Sequence: turn 1 truncated (length, empty), turn 2 normal stop with summary
    responses = [
        _openai_message(content="", tool_calls=None, finish_reason="length"),
        _openai_message(content="Done.", tool_calls=None, finish_reason="stop"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = responses

    with patch(
        "openjarvis.agents.hybrid.mini_swe_agent.OpenAI", create=True,
        return_value=mock_client,
    ), patch("openai.OpenAI", return_value=mock_client):
        result = _loop_cloud_openai(
            "fix the bug", tmp_path,
            model="gpt-5-mini", max_turns=5,
            bash_timeout=10, output_cap=1000, turn_max_tokens=100,
            trace_prefix="test",
        )

    # Must NOT have exited at turn 1 — should have retried.
    assert result["turns"] == 2, (
        f"Loop must retry after length-truncation with empty text; "
        f"got turns={result['turns']}"
    )
    assert result["final_summary"] == "Done."
    assert result["max_turns_hit"] is False


def test_openai_loop_terminates_normally_on_stop(tmp_path: Any) -> None:
    """Sanity guard: natural ``finish_reason='stop'`` with non-empty text
    still terminates the loop on turn 1."""
    from openjarvis.agents.hybrid.mini_swe_agent import _loop_cloud_openai

    responses = [
        _openai_message(content="Done.", tool_calls=None, finish_reason="stop"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = responses

    with patch("openai.OpenAI", return_value=mock_client):
        result = _loop_cloud_openai(
            "fix the bug", tmp_path,
            model="gpt-5-mini", max_turns=5,
            bash_timeout=10, output_cap=1000, turn_max_tokens=100,
            trace_prefix="test",
        )

    assert result["turns"] == 1
    assert result["final_summary"] == "Done."


# ---------- Gemini loop ----------

def _gemini_response(
    *,
    text: str = "",
    function_calls: List[Dict[str, Any]] = None,
    finish_reason: str = "FinishReason.STOP",
    prompt_tokens: int = 10,
    candidates_tokens: int = 5,
) -> Any:
    """Build a mock google-genai GenerateContentResponse."""
    parts = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for fc in function_calls or []:
        parts.append(SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(name=fc["name"], args=fc["args"]),
        ))
    content = SimpleNamespace(parts=parts, role="model")
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _gemini_mocks(responses: List[Any]) -> Dict[str, Any]:
    """Wire up sys.modules mocks for google.genai so the Gemini loop
    sees our scripted ``generate_content`` responses. Returns the dict
    suitable for ``patch.dict('sys.modules', ...)``.

    The loop's ``from google import genai`` resolves via the
    ``google`` parent module's ``genai`` attribute (NOT via
    sys.modules['google.genai']), so we also bind ``fake_genai`` onto
    the parent mock's attribute table — otherwise the import returns a
    fresh unconfigured MagicMock and our scripted responses are bypassed.
    """
    fake_client = MagicMock()
    fake_models = MagicMock()
    fake_models.generate_content = MagicMock(side_effect=responses)
    fake_client.models = fake_models
    fake_genai = MagicMock()
    fake_genai.Client = MagicMock(return_value=fake_client)
    fake_types = MagicMock()
    fake_genai.types = fake_types
    fake_google = MagicMock()
    fake_google.genai = fake_genai
    return {
        "google": fake_google,
        "google.genai": fake_genai,
        "google.genai.types": fake_types,
    }


def test_gemini_loop_recovers_from_malformed_function_call(tmp_path: Any) -> None:
    """When finish_reason includes MALFORMED_FUNCTION_CALL AND no text AND
    no function_calls, the loop must inject a recovery nudge and retry —
    NOT exit silently. Pre-fix this hit 24/100 tasks on gemini-flash."""
    from openjarvis.agents.hybrid import mini_swe_agent

    responses = [
        _gemini_response(
            text="", function_calls=None,
            finish_reason="FinishReason.MALFORMED_FUNCTION_CALL",
        ),
        _gemini_response(
            text="Fixed.", function_calls=None,
            finish_reason="FinishReason.STOP",
        ),
    ]
    with patch.dict("sys.modules", _gemini_mocks(responses)):
        result = mini_swe_agent._loop_cloud_gemini(
            "fix the bug", tmp_path,
            model="gemini-2.5-flash", max_turns=5,
            bash_timeout=10, output_cap=1000, turn_max_tokens=100,
            trace_prefix="test",
        )

    assert result["turns"] == 2, (
        f"Gemini loop must retry on MALFORMED_FUNCTION_CALL with empty text; "
        f"got turns={result['turns']}"
    )
    assert result["final_summary"] == "Fixed."


def test_gemini_loop_recovers_from_max_tokens(tmp_path: Any) -> None:
    """MAX_TOKENS truncation parallel of the OpenAI ``length`` recovery."""
    from openjarvis.agents.hybrid import mini_swe_agent

    responses = [
        _gemini_response(
            text="", function_calls=None,
            finish_reason="FinishReason.MAX_TOKENS",
        ),
        _gemini_response(
            text="Done.", function_calls=None,
            finish_reason="FinishReason.STOP",
        ),
    ]
    with patch.dict("sys.modules", _gemini_mocks(responses)):
        result = mini_swe_agent._loop_cloud_gemini(
            "fix the bug", tmp_path,
            model="gemini-2.5-flash", max_turns=5,
            bash_timeout=10, output_cap=1000, turn_max_tokens=100,
            trace_prefix="test",
        )

    assert result["turns"] == 2
    assert result["final_summary"] == "Done."


def test_gemini_loop_terminates_normally_on_stop_with_text(tmp_path: Any) -> None:
    """Sanity guard: text-only STOP still ends the loop (no recovery)."""
    from openjarvis.agents.hybrid import mini_swe_agent

    responses = [
        _gemini_response(
            text="All done.", function_calls=None,
            finish_reason="FinishReason.STOP",
        ),
    ]
    with patch.dict("sys.modules", _gemini_mocks(responses)):
        result = mini_swe_agent._loop_cloud_gemini(
            "fix the bug", tmp_path,
            model="gemini-2.5-flash", max_turns=5,
            bash_timeout=10, output_cap=1000, turn_max_tokens=100,
            trace_prefix="test",
        )

    assert result["turns"] == 1
    assert result["final_summary"] == "All done."
