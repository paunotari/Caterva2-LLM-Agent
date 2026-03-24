"""Essential unit tests for get_slice tool contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path
import numpy as np


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

        class _FakeDataset:
            """Stub Dataset class for type annotation."""
            pass

        fake_caterva2.Client = _FakeClient
        fake_caterva2.Dataset = _FakeDataset
        sys.modules["caterva2"] = fake_caterva2


_ensure_local_import_bootstrap()
import tools


class _FakeDataset1D:
    """Minimal 1D Dataset mock with indexing support."""
    
    def __init__(self):
        self.shape = (1000,)
        self.dtype = "int64"
        self._data = np.arange(1000)
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeDataset2D:
    """Minimal 2D Dataset mock with indexing support."""
    
    def __init__(self):
        self.shape = (100, 50)
        self.dtype = "float64"
        self._data = np.arange(5000).reshape(100, 50)
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeDatasetHuge:
    """Dataset that would exceed element limits."""
    
    def __init__(self):
        self.shape = (1000, 1000, 1000)  # 1 billion elements
        self.dtype = "float32"
    
    def __getitem__(self, key):
        # Should never be called if limit check works
        raise AssertionError("Should not fetch data exceeding limit")


def test_get_slice_returns_data_1d(monkeypatch) -> None:
    # What this tests: basic 1D slice returns correct data and metadata.
    # Why important: core functionality for data retrieval.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd", slices="0:10")

    assert "error" not in result
    assert result["path"] == "@public/example.b2nd"
    assert result["dataset_shape"] == [1000]
    assert result["dtype"] == "int64"
    assert result["slice"] == "0:10"
    assert result["result_shape"] == [10]
    assert result["data"] == list(range(10))


def test_get_slice_returns_data_2d(monkeypatch) -> None:
    # What this tests: 2D slice with multi-dimension syntax works.
    # Why important: scientific data is often multi-dimensional.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset2D())
    result = tools.get_slice("@public/example.b2nd", slices="0:3, 0:2")

    assert "error" not in result
    assert result["dataset_shape"] == [100, 50]
    assert result["result_shape"] == [3, 2]
    # Data should be [[0,1], [50,51], [100,101]]
    expected = [[0, 1], [50, 51], [100, 101]]
    assert result["data"] == expected


def test_get_slice_default_slice_respects_limit(monkeypatch) -> None:
    # What this tests: when no slice provided, defaults to safe limit.
    # Why important: protects LLM context from huge data dumps.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd")  # No slice specified

    assert "error" not in result
    # Should return at most MAX_SLICE_ELEMENTS (1000 < 10000, so returns all)
    assert len(result["data"]) == 1000


def test_get_slice_rejects_oversized_request(monkeypatch) -> None:
    # What this tests: requests exceeding element limit are rejected with clear error.
    # Why important: prevents memory issues and LLM context overflow.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDatasetHuge())
    result = tools.get_slice("@public/huge.b2nd", slices="0:100, 0:200, 0:200")
    
    # 100 * 200 * 200 = 4,000,000 elements — should be rejected
    assert "error" in result
    assert "exceeding limit" in result["error"]
    assert "shape" in result  # Should include shape for context


def test_get_slice_invalid_syntax_returns_error(monkeypatch) -> None:
    # What this tests: invalid slice syntax returns clear error, not exception.
    # Why important: agent must handle bad LLM-generated input gracefully.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd", slices="not:valid:slice:syntax")

    assert "error" in result
    assert "Invalid slice" in result["error"]


def test_get_slice_too_many_dimensions_returns_error(monkeypatch) -> None:
    # What this tests: slice with more dimensions than dataset is rejected.
    # Why important: prevents confusing numpy errors.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd", slices="0:10, 0:5")  # 2D slice on 1D data

    assert "error" in result
    assert "Too many dimensions" in result["error"]


def test_get_slice_negative_indices(monkeypatch) -> None:
    # What this tests: Python-style negative indices work.
    # Why important: common user pattern for "last N elements".
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd", slices="-5:")

    assert "error" not in result
    assert result["data"] == [995, 996, 997, 998, 999]


def test_get_slice_step_syntax(monkeypatch) -> None:
    # What this tests: step parameter in slice (start:stop:step) works.
    # Why important: allows downsampling large arrays.
    monkeypatch.setattr(tools, "_get_dataset", lambda _path: _FakeDataset1D())
    result = tools.get_slice("@public/example.b2nd", slices="0:10:2")

    assert "error" not in result
    assert result["data"] == [0, 2, 4, 6, 8]
