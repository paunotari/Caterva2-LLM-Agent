"""Essential unit tests for get_slice tool contract."""

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


class _FakeDatasetLarge:
    """Large but manageable dataset used to validate summary-only responses."""
    
    def __init__(self):
        self.shape = (100, 200, 200)  # 4,000,000 elements
        self.dtype = "float32"
        self._data = np.zeros(self.shape, dtype=np.float32)
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeUploadClient:
    """Client double for verifying get_slice persistence uploads."""

    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def upload(self, obj, path: str, **kwargs):
        self.calls.append({"obj": obj, "path": path, "kwargs": kwargs})
        return types.SimpleNamespace(path=path)


def _make_resolved(fake_dataset):
    """Helper to create a ResolvedData wrapping a fake dataset."""
    return ResolvedData(fake_dataset, source='server', name='@test/data.b2nd')


def test_get_slice_returns_data_1d(monkeypatch) -> None:
    # What this tests: basic 1D slice returns correct data and metadata.
    # Why important: core functionality for data retrieval.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:10")

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
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset2D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:3, 0:2")

    assert "error" not in result
    assert result["dataset_shape"] == [100, 50]
    assert result["result_shape"] == [3, 2]
    # Data should be [[0,1], [50,51], [100,101]]
    expected = [[0, 1], [50, 51], [100, 101]]
    assert result["data"] == expected


def test_get_slice_default_slice_respects_limit(monkeypatch) -> None:
    # What this tests: when no slice is provided, tool uses the auto-preview slice.
    # Why important: avoids accidental full loads while preserving summary-first behavior.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd")  # No slice specified

    assert "error" not in result
    assert result["summary"]["num_elements"] == 1000
    assert "data" not in result


def test_get_slice_large_request_returns_summary_only(monkeypatch) -> None:
    # What this tests: large explicit slices are allowed but returned as summary-only payloads.
    # Why important: decouples operation size from LLM context size.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDatasetLarge()))
    result = data_access.get_slice("@public/large.b2nd", slices="0:100, 0:200, 0:200")

    assert "error" not in result
    assert result["summary"]["num_elements"] == 4_000_000
    assert "data" not in result
    assert "_hint" in result


def test_get_slice_invalid_syntax_returns_error(monkeypatch) -> None:
    # What this tests: invalid slice syntax returns clear error, not exception.
    # Why important: agent must handle bad LLM-generated input gracefully.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="not:valid:slice:syntax")

    assert "error" in result
    assert "Invalid slice" in result["error"]


def test_get_slice_too_many_dimensions_returns_error(monkeypatch) -> None:
    # What this tests: slice with more dimensions than dataset is rejected.
    # Why important: prevents confusing numpy errors.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:10, 0:5")  # 2D slice on 1D data

    assert "error" in result
    assert "Too many dimensions" in result["error"]


def test_get_slice_negative_indices(monkeypatch) -> None:
    # What this tests: Python-style negative indices work.
    # Why important: common user pattern for "last N elements".
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="-5:")

    assert "error" not in result
    assert result["data"] == [995, 996, 997, 998, 999]


def test_get_slice_step_syntax(monkeypatch) -> None:
    # What this tests: step parameter in slice (start:stop:step) works.
    # Why important: allows downsampling large arrays.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:10:2")

    assert "error" not in result
    assert result["data"] == [0, 2, 4, 6, 8]


def test_get_slice_includes_summary(monkeypatch) -> None:
    # What this tests: result includes pre-computed summary for LLM presentation.
    # Why important: enables LLM to present summaries instead of raw data dumps.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:50")

    assert "error" not in result
    assert "summary" in result
    
    summary = result["summary"]
    assert summary["num_elements"] == 50
    assert summary["min"] == 0
    assert summary["max"] == 49
    assert "mean" in summary
    assert "preview" in summary


def test_get_slice_large_result_includes_hint(monkeypatch) -> None:
    # What this tests: large results include hint for LLM to use summary.
    # Why important: guides LLM behavior for better UX.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset2D()))
    result = data_access.get_slice("@public/example.b2nd", slices="0:20, 0:10")  # 200 elements

    assert "error" not in result
    assert "_hint" in result
    assert "summary" in result["_hint"].lower()
    assert "data" not in result


def test_get_slice_does_not_auto_register_server_data(monkeypatch) -> None:
    # What this tests: get_slice no longer auto-materializes results in notebook namespace.
    # Why important: materialization must be explicit (load_dataset).
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    clear_fetched_objects()

    result = data_access.get_slice("@public/example.b2nd", slices="0:10")

    assert "error" not in result
    assert get_fetched_objects() == {}


def test_get_slice_persist_result_saves_server_slice_when_authenticated(monkeypatch) -> None:
    # What this tests: opt-in persistence stores server slice result in @personal.
    # Why important: enables chaining on persisted slices without making reads stateful by default.
    fake_client = _FakeUploadClient()
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    monkeypatch.setattr(
        data_access,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(data_access, "_get_client", lambda: fake_client)

    result = data_access.get_slice(
        "@public/example.b2nd",
        slices="0:10",
        persist_result=True,
    )

    assert "error" not in result
    assert result["stored_server_side"] is True
    assert result["result_path"].startswith("@personal/slices/")
    assert fake_client.calls[0]["path"] == result["result_path"]


def test_get_slice_persist_result_skips_when_not_authenticated(monkeypatch) -> None:
    # What this tests: opt-in persistence reports auth requirement but still returns slice data.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    monkeypatch.setattr(
        data_access,
        "get_client_auth_status",
        lambda: {"authenticated": False, "urlbase": "http://test", "username": None},
    )

    result = data_access.get_slice(
        "@public/example.b2nd",
        slices="0:10",
        persist_result=True,
    )

    assert "error" not in result
    assert result["stored_server_side"] is False
    assert "not authenticated" in result["persistence_note"]
