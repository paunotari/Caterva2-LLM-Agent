"""Essential unit tests for dataset info retrieval contract."""

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
    def get_info(self, _path: str) -> dict:
        """Return fake metadata matching Caterva2 API structure."""
        return {
            "shape": [100, 200],
            "dtype": "float32",
            "chunks": [50, 100],
            "blocks": [10, 10],
            "cratio": 3.14,
            "mtime": 1234567890,
        }


def test_get_dataset_info_returns_metadata_contract(monkeypatch) -> None:
    # What this tests: metadata response contains critical fields (shape, dtype, chunks, etc.).
    # Why important: users rely on this info to understand dataset structure; missing fields break workflows.
    monkeypatch.setattr(tools, "_get_client", lambda: _FakeClient())
    result = tools.get_dataset_info("@public/example.b2nd")

    assert result["path"] == "@public/example.b2nd"
    assert "info" in result
    info = result["info"]
    
    # Verify all critical metadata fields are present
    assert "shape" in info
    assert "dtype" in info
    assert "chunks" in info
    assert "blocks" in info
    assert "cratio" in info
    assert "mtime" in info
    
    # Verify values are correct (not just present)
    assert info["shape"] == [100, 200]
    assert info["dtype"] == "float32"
