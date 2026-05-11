"""Essential unit tests for core agent loop behavior."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def _ensure_local_import_bootstrap() -> None:
    """Make direct file execution work (PyCharm 'Run file') as well as pytest runs."""
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if "caterva2" not in sys.modules:
        fake_caterva2 = types.ModuleType("caterva2")

        class _FakeClient:
            def __init__(self, _url: str):
                self.url = _url

        class _FakeDataset:
            pass

        fake_caterva2.Client = _FakeClient
        fake_caterva2.Dataset = _FakeDataset
        sys.modules["caterva2"] = fake_caterva2

    if "caterva2_agent.prompts" not in sys.modules:
        fake_prompts = types.ModuleType("caterva2_agent.prompts")
        fake_prompts.SYSTEM_PROMPT = "test system prompt"
        sys.modules["caterva2_agent.prompts"] = fake_prompts

    if "caterva2_agent.config" not in sys.modules:
        fake_config = types.ModuleType("caterva2_agent.config")
        fake_config.CATERVA2_URLBASE = "http://test-caterva2.local"
        fake_config.MODEL_NAME = "test-model"
        fake_config.SYSTEM_PROMPT = "test system prompt"
        fake_config.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )
        )
        sys.modules["caterva2_agent.config"] = fake_config
        sys.modules["config"] = fake_config


_ensure_local_import_bootstrap()
from caterva2_agent import agent


def _fake_response(
    content: str | None,
    tool_calls: list | None,
    total_tokens: int = 10,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
):
    """Build a minimal Groq/OpenAI-like response object for unit tests."""
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return types.SimpleNamespace(choices=[choice], usage=usage)


def test_run_returns_final_answer_when_no_tool_calls(monkeypatch) -> None:
    # What this tests: the agent exits immediately when the model returns a direct final answer.
    # Why important: not every user request needs tools; forcing tool use would break normal chat behavior.
    test_agent = agent.Agent()
    monkeypatch.setattr(
        test_agent,
        "_call_llm_with_retry",
        lambda **_: _fake_response("Direct answer", tool_calls=None),
    )

    output = test_agent.run("hello")
    assert "Session Token Usage" in output
    assert "Input:" in output
    assert "Output:" in output
    assert "---" in output
    assert "Direct answer" in output


def test_run_tracks_prompt_and_completion_tokens(monkeypatch) -> None:
    # What this tests: session counters split input/output usage and include them in replies.
    # Why important: users need accurate per-direction token accounting for API cost tracking.
    test_agent = agent.Agent()
    monkeypatch.setattr(
        test_agent,
        "_call_llm_with_retry",
        lambda **_: _fake_response(
            "Usage-aware answer",
            tool_calls=None,
            total_tokens=19,
            prompt_tokens=12,
            completion_tokens=7,
        ),
    )

    output = test_agent.run("hello")
    assert "Session Token Usage" in output
    assert "Input:  12" in output
    assert "Output: 7" in output
    assert "---" in output


def test_run_executes_tool_call_then_returns_final_answer(monkeypatch) -> None:
    # What this tests: the full core loop (model asks for tool -> tool result appended -> final answer).
    # Why important: this is the minimum end-to-end proof that the agent architecture works correctly.
    test_agent = agent.Agent()

    tool_call = types.SimpleNamespace(
        id="tc1",
        function=types.SimpleNamespace(name="list_roots", arguments="{}"),
        model_dump=lambda: {
            "id": "tc1",
            "type": "function",
            "function": {"name": "list_roots", "arguments": "{}"},
        },
    )

    responses = [
        _fake_response(content=None, tool_calls=[tool_call]),
        _fake_response(content="Found one root: @public", tool_calls=None),
    ]
    iterator = iter(responses)
    monkeypatch.setattr(test_agent, "_call_llm_with_retry", lambda **_: next(iterator))
    monkeypatch.setattr(agent, "execute_tool", lambda name, args: json.dumps({"roots": ["@public"]}))

    output = test_agent.run("show roots")
    assert "Found one root: @public" in output
    assert any(m.get("role") == "tool" for m in test_agent.messages)


def test_run_sanitizes_large_tool_binary_payload_for_llm(monkeypatch) -> None:
    # What this tests: huge binary-like fields are redacted before entering LLM history.
    # Why important: prevents token bloat and keeps model context focused.
    test_agent = agent.Agent()

    tool_call = types.SimpleNamespace(
        id="tc-image",
        function=types.SimpleNamespace(name="render_projection", arguments="{}"),
        model_dump=lambda: {
            "id": "tc-image",
            "type": "function",
            "function": {"name": "render_projection", "arguments": "{}"},
        },
    )

    responses = [
        _fake_response(content=None, tool_calls=[tool_call]),
        _fake_response(content="Projection ready.", tool_calls=None),
    ]
    iterator = iter(responses)
    monkeypatch.setattr(test_agent, "_call_llm_with_retry", lambda **_: next(iterator))

    image_payload = "data:image/png;base64," + ("A" * 8000)
    monkeypatch.setattr(
        agent,
        "execute_tool",
        lambda _name, _args: json.dumps(
            {"status": "success", "image": image_payload, "format": "png"}
        ),
    )

    output = test_agent.run("render me a projection")
    assert "Projection ready." in output

    tool_messages = [m for m in test_agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    llm_payload = json.loads(tool_messages[0]["content"])
    assert "image" not in llm_payload
    assert llm_payload["image_available_in_notebook"] is True
    assert "_llm_sanitization" in llm_payload

    assert test_agent.last_tool_results[0]["name"] == "render_projection"
    assert test_agent.last_tool_results[0]["content"]["image"] == image_payload
