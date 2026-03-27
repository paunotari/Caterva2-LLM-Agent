"""Essential unit tests for dataset listing contract."""

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
    def get_list(self, _path: str) -> list[str]:
        return ["ds1", "ds2", "ds3", "ds4", "ds5"]


def test_list_datasets_pagination_fields(monkeypatch) -> None:
    # What this tests: pagination contract fields (`total`, `offset`, `has_more`, page size).
    # Why important: browsing is the core dataset-discovery flow; bad pagination causes silent regressions.
    monkeypatch.setattr(browsing, "_get_client", lambda: _FakeClient())
    result = browsing.list_datasets("@public", limit=2, offset=2)

    assert result["path"] == "@public"
    assert result["total"] == 5
    assert result["offset"] == 2
    assert result["has_more"] is True
    assert result["datasets"] == ["@public/ds3", "@public/ds4"]
