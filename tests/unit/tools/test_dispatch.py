"""Essential unit tests for tool dispatch safety behavior."""

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
import tools


def test_execute_tool_unknown_tool_returns_error_json() -> None:
    # What this tests: unknown tool names return structured JSON errors.
    # Why important: the LLM can hallucinate tool names; agent must fail safely, not crash.
    result = json.loads(tools.execute_tool("not_a_real_tool", {}))
    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_execute_tool_typeerror_is_reported_as_invalid_arguments() -> None:
    # What this tests: wrong/missing tool arguments are translated into a clear error.
    # Why important: this is the runtime guardrail that enforces the tool schema contract.
    result = json.loads(tools.execute_tool("list_datasets", {"wrong_arg": 1}))
    assert "error" in result
    assert "Invalid arguments" in result["error"]
