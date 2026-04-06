"""Essential unit tests for dataset statistics contract."""

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
from caterva2_agent.tools import analysis
from caterva2_agent.tools._base import ResolvedData


class _FakeDataset:
    """Minimal Dataset mock with statistical methods."""
    
    def __init__(self):
        self.shape = (1000,)
        self.dtype = "int64"
    
    def min(self, axis=None):
        return 0
    
    def max(self, axis=None):
        return 999
    
    def mean(self, axis=None):
        return 499.5
    
    def std(self, axis=None):
        return 288.67
    
    def sum(self, axis=None):
        return 499500
    
    def var(self, axis=None):
        return 83333.25
    
    def argmin(self, axis=None):
        return 0
    
    def argmax(self, axis=None):
        return 999
    
    def any(self, axis=None):
        return True
    
    def all(self, axis=None):
        return False


def _make_resolved(fake_dataset):
    """Helper to create a ResolvedData wrapping a fake dataset."""
    return ResolvedData(fake_dataset, source='server', name='@test/data.b2nd')


def test_get_dataset_stats_returns_default_stats(monkeypatch) -> None:
    # What this tests: default stats (min, max, mean, std) are returned when no stats list provided.
    # Why important: users expect sensible defaults without specifying every stat.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset()))
    result = analysis.get_dataset_stats("@public/example.b2nd")

    assert result["name"] == "@public/example.b2nd"
    assert result["shape"] == [1000]
    assert result["dtype"] == "int64"
    assert "stats" in result
    
    # Default stats must be present
    stats = result["stats"]
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert "std" in stats
    
    # Values should match fake data
    assert stats["min"] == 0
    assert stats["max"] == 999


def test_get_dataset_stats_custom_stats_list(monkeypatch) -> None:
    # What this tests: only requested stats are computed and returned.
    # Why important: allows users to request specific stats without computing all.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset()))
    result = analysis.get_dataset_stats(
        "@public/example.b2nd",
        stats=["sum", "var", "argmin"]
    )

    stats = result["stats"]
    # Only requested stats present
    assert "sum" in stats
    assert "var" in stats
    assert "argmin" in stats
    # Default stats NOT present (not requested)
    assert "mean" not in stats
    assert "std" not in stats


def test_get_dataset_stats_invalid_stat_returns_error(monkeypatch) -> None:
    # What this tests: invalid stat names are rejected with clear error.
    # Why important: prevents silent failures or confusing exceptions.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset()))
    result = analysis.get_dataset_stats(
        "@public/example.b2nd",
        stats=["min", "invalid_stat"]
    )

    assert "error" in result
    assert "invalid_stat" in result["error"]
    assert "Unsupported" in result["error"]


def test_get_dataset_stats_includes_metadata(monkeypatch) -> None:
    # What this tests: response includes shape, dtype, and axis for context.
    # Why important: LLM needs metadata to interpret stats correctly.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset()))
    result = analysis.get_dataset_stats("@public/example.b2nd", axis=0)

    assert "shape" in result
    assert "dtype" in result
    assert "axis" in result
    assert result["axis"] == 0
