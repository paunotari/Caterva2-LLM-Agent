"""Essential unit tests for tool dispatch safety behavior."""

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

    # Set up fake caterva2 before importing tools
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

    # Set up fake config modules
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
        sys.modules["config"] = fake_config  # backwards compat


_ensure_local_import_bootstrap()
from caterva2_agent import tools


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


def test_execute_tool_upload_dataset_is_registered() -> None:
    # What this tests: upload_dataset is in the tool registry and can be called.
    # Why important: ensures the new tool is discoverable to the agent.
    assert "upload_dataset" in tools.TOOL_MAP
    assert callable(tools.TOOL_MAP["upload_dataset"])


def test_execute_tool_upload_dataset_in_tool_schemas() -> None:
    # What this tests: upload_dataset schema is included in TOOLS sent to LLM.
    # Why important: the LLM must know about the tool to call it.
    tool_names = [tool["function"]["name"] for tool in tools.TOOLS if "function" in tool]
    assert "upload_dataset" in tool_names

