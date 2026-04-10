"""Essential unit tests for load_dataset explicit materialization behavior."""

from __future__ import annotations

import sys
import types
from pathlib import Path
import numpy as np


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
from caterva2_agent.tools import data_access
from caterva2_agent.tools._base import ResolvedData, get_fetched_objects, clear_fetched_objects


class _SmallServerDataset:
    """Small server dataset that can be explicitly loaded."""

    def __init__(self):
        self.shape = (50,)
        self.dtype = "float32"
        self._data = np.arange(50, dtype=np.float32)

    def __getitem__(self, key):
        return self._data[key]


class _TooLargeServerDataset:
    """Large dataset that should fail size checks before any materialization."""

    def __init__(self):
        self.shape = (20_000_000,)  # float64 => ~152.6 MB
        self.dtype = "float64"

    def __getitem__(self, key):
        raise AssertionError("Should not materialize data above load size limit")


def _make_resolved(fake_dataset):
    """Helper to create a ResolvedData wrapping a fake dataset."""
    return ResolvedData(fake_dataset, source="server", name="@test/data.b2nd")


def test_load_dataset_materializes_small_server_dataset(monkeypatch) -> None:
    # What this tests: explicit loading registers data for notebook use.
    # Why important: this is the only path meant to materialize server data.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_SmallServerDataset()))
    clear_fetched_objects()

    result = data_access.load_dataset("@public/small.b2nd")

    assert "error" not in result
    assert result["status"] == "success"
    assert result["registered_in_notebook"] is True
    assert "@public/small.b2nd" in get_fetched_objects()
    assert "data" in result  # 50 elements <= inline payload threshold


def test_load_dataset_rejects_payload_above_100mb(monkeypatch) -> None:
    # What this tests: explicit load enforces the confirmed 100MB cap.
    # Why important: prevents unsafe materialization of large arrays.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_TooLargeServerDataset()))

    result = data_access.load_dataset("@public/too-large.b2nd")

    assert "error" in result
    assert "100 MB" in result["error"]
    assert "suggestion" in result
