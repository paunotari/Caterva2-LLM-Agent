"""Essential unit tests for dataset listing contract."""

from __future__ import annotations

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


class _FakeClient:
    def get_list(self, _path: str) -> list[str]:
        return ["ds1", "ds2", "ds3", "ds4", "ds5"]


def test_list_datasets_pagination_fields(monkeypatch) -> None:
    # What this tests: pagination contract fields (`total`, `offset`, `has_more`, page size).
    # Why important: browsing is the core dataset-discovery flow; bad pagination causes silent regressions.
    monkeypatch.setattr(tools, "_get_client", lambda: _FakeClient())
    result = tools.list_datasets("@public", limit=2, offset=2)

    assert result["path"] == "@public"
    assert result["total"] == 5
    assert result["offset"] == 2
    assert result["has_more"] is True
    assert result["datasets"] == ["@public/ds3", "@public/ds4"]
