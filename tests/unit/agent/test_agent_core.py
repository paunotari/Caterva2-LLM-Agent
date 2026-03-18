"""Essential unit tests for core agent loop behavior."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def _ensure_local_import_bootstrap() -> None:
    """Make direct file execution work (PyCharm 'Run file') as well as pytest runs."""
    project_root = Path(__file__).resolve().parents[3]
    agent_dir = project_root / "caterva2_agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    if "config" not in sys.modules:
        fake_config = types.ModuleType("config")
        fake_config.CATERVA2_URLBASE = "http://test-caterva2.local"
        fake_config.MODEL_NAME = "test-model"
        fake_config.SYSTEM_PROMPT = "test system prompt"
        fake_config.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )
        )
        sys.modules["config"] = fake_config

    if "caterva2" not in sys.modules:
        fake_caterva2 = types.ModuleType("caterva2")

        class _FakeClient:
            def __init__(self, _url: str):
                self.url = _url

        fake_caterva2.Client = _FakeClient
        sys.modules["caterva2"] = fake_caterva2


_ensure_local_import_bootstrap()
import agent


def _fake_response(content: str | None, tool_calls: list | None, total_tokens: int = 10):
    """Build a minimal Groq/OpenAI-like response object for unit tests."""
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(total_tokens=total_tokens)
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
    assert "Direct answer" in output


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
