"""Essential unit tests for root listing contract."""

from __future__ import annotations

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
from caterva2_agent.tools import browsing


class _FakeClient:
    def get_roots(self) -> dict:
        """Return fake roots matching Caterva2 API structure."""
        return {
            "@public": {"name": "@public"},
            "@examples": {"name": "@examples"},
            "@demo": {"name": "@demo"},
        }


def test_list_roots_returns_sorted_root_names(monkeypatch) -> None:
    # What this tests: roots are returned as a sorted list, preserving the '@' prefix.
    # Why important: root names with '@' are the API contract; dropping it breaks all downstream tool calls.
    monkeypatch.setattr(browsing, "_get_client", lambda: _FakeClient())
    result = browsing.list_roots()

    assert "roots" in result
    assert isinstance(result["roots"], list)
    # Verify sorting (alphabetical order)
    assert result["roots"] == ["@demo", "@examples", "@public"]
    # Verify '@' prefix is preserved
    assert all(name.startswith("@") for name in result["roots"])


def test_list_roots_empty_server(monkeypatch) -> None:
    # What this tests: empty roots list is valid (server with no data).
    # Why important: new/empty servers should not cause errors.
    
    class _EmptyClient:
        def get_roots(self) -> dict:
            return {}
    
    monkeypatch.setattr(browsing, "_get_client", lambda: _EmptyClient())
    result = browsing.list_roots()

    assert "roots" in result
    assert result["roots"] == []
