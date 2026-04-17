"""Essential unit tests for where_filter tool contract."""

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
from caterva2_agent.tools._base import (
    ResolvedData,
    get_fetched_objects,
    clear_fetched_objects,
    set_notebook_namespace,
)


class _FakeDataset1D:
    """Minimal 1D Dataset mock with indexing support."""
    
    def __init__(self):
        self.shape = (100,)
        self.dtype = "int64"
        self._data = np.arange(100)  # 0 to 99
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeDataset2D:
    """Minimal 2D Dataset mock with indexing support."""
    
    def __init__(self):
        self.shape = (10, 10)
        self.dtype = "float64"
        self._data = np.arange(100).reshape(10, 10)
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeElevationDataset:
    """Simulates elevation data for realistic use case testing."""
    
    def __init__(self):
        self.shape = (50,)
        self.dtype = "float64"
        # Elevation values ranging from 1000m to 4000m
        self._data = np.array([1000 + i * 60 for i in range(50)])  # 1000, 1060, ..., 3940
    
    def __getitem__(self, key):
        return self._data[key]


class _FakeServerCondition:
    """Boolean condition object with server-style where()."""

    def __init__(self, mask: np.ndarray, source_values: np.ndarray):
        self._mask = np.asarray(mask, dtype=bool)
        self._source_values = np.asarray(source_values)

    def where(self, value1=None, value2=None):
        true_values = self._source_values if value1 is None else np.asarray(value1)
        false_values = 0 if value2 is None else value2
        return _FakeServerOperand(np.where(self._mask, true_values, false_values))

    def __array__(self, dtype=None):
        return np.asarray(self._mask, dtype=dtype)


class _FakeServerOperand:
    """Server-side operand that supports comparisons and materialization."""

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __gt__(self, threshold):
        return _FakeServerCondition(self._data > threshold, self._data)

    def __ge__(self, threshold):
        return _FakeServerCondition(self._data >= threshold, self._data)

    def __lt__(self, threshold):
        return _FakeServerCondition(self._data < threshold, self._data)

    def __le__(self, threshold):
        return _FakeServerCondition(self._data <= threshold, self._data)

    def __eq__(self, threshold):
        return _FakeServerCondition(self._data == threshold, self._data)

    def __ne__(self, threshold):
        return _FakeServerCondition(self._data != threshold, self._data)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


class _FakeServerWhereDataset:
    """Dataset mock with server-style slice() + comparison support."""

    def __init__(self):
        self.shape = (100,)
        self.dtype = "int64"
        self._data = np.arange(100)

    def __getitem__(self, key):
        return self._data[key]

    def slice(self, key, as_blosc2=True):
        return _FakeServerOperand(self._data[key])

    def __gt__(self, threshold):
        return _FakeServerOperand(self._data) > threshold

    def __ge__(self, threshold):
        return _FakeServerOperand(self._data) >= threshold

    def __lt__(self, threshold):
        return _FakeServerOperand(self._data) < threshold

    def __le__(self, threshold):
        return _FakeServerOperand(self._data) <= threshold

    def __eq__(self, threshold):
        return _FakeServerOperand(self._data) == threshold

    def __ne__(self, threshold):
        return _FakeServerOperand(self._data) != threshold


def _make_resolved(fake_dataset):
    """Helper to create a ResolvedData wrapping a fake dataset."""
    return ResolvedData(fake_dataset, source='server', name='@test/data.b2nd')


# ---------------------------------------------------------------------------
# CORE FUNCTIONALITY TESTS
# ---------------------------------------------------------------------------

def test_where_filter_greater_than(monkeypatch) -> None:
    # What this tests: basic > operator works correctly.
    # Why important: most common filtering use case.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=90,
        slices="0:100"
    )

    assert "error" not in result
    assert result["condition"] == "data > 90"
    
    # Values 91-99 should pass (9 values), 0-90 should become 0
    data = result["data"]
    assert data[90] == 0   # 90 is NOT > 90
    assert data[91] == 91  # 91 IS > 90, returns original
    assert data[99] == 99
    
    # Check match summary
    assert result["match_summary"]["matched"] == 9  # 91-99


def test_where_filter_less_than_or_equal(monkeypatch) -> None:
    # What this tests: <= operator for range filtering.
    # Why important: validates all comparison operators work.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator="<=",
        threshold=10,
        slices="0:20"
    )

    assert "error" not in result
    # Values 0-10 should pass (11 values)
    assert result["match_summary"]["matched"] == 11


def test_where_filter_custom_replacement_values(monkeypatch) -> None:
    # What this tests: custom value_if_true and value_if_false work.
    # Why important: enables binary masking (1/0) and other transformations.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=50,
        value_if_true=1,
        value_if_false=0,
        slices="0:100"
    )

    assert "error" not in result
    data = result["data"]
    
    # All values should be 0 or 1
    assert all(v in [0, 1] for v in data)
    
    # Positions 51-99 should be 1 (49 values)
    assert sum(data) == 49
    assert result["value_if_true"] == 1
    assert result["value_if_false"] == 0


def test_where_filter_elevation_use_case(monkeypatch) -> None:
    # What this tests: realistic elevation filtering scenario.
    # Why important: validates the mountain peaks example from requirements.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeElevationDataset()))
    result = data_access.where_filter(
        path="@public/terrain.b2nd",
        operator=">",
        threshold=3000,
        # Keep original elevation where > 3000, 0 elsewhere (default)
    )

    assert "error" not in result
    data = result["data"]
    
    # All non-zero values should be > 3000
    non_zero = [v for v in data if v > 0]
    assert all(v > 3000 for v in non_zero)
    
    # Check that we found some peaks
    assert result["match_summary"]["matched"] > 0
    assert result["match_summary"]["percentage"] > 0


def test_where_filter_2d_dataset(monkeypatch) -> None:
    # What this tests: filtering works on multi-dimensional data.
    # Why important: scientific datasets are often multi-dimensional.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset2D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">=",
        threshold=50,
        slices="0:5, 0:5"  # 5x5 = 25 elements
    )

    assert "error" not in result
    assert result["result_shape"] == [5, 5]
    
    # In a 10x10 grid with values 0-99, first 5x5 has values 0-4, 10-14, 20-24, 30-34, 40-44
    # None of these are >= 50, so all should be 0
    data = result["data"]
    assert all(all(v == 0 for v in row) for row in data)


def test_where_filter_prefers_server_where_when_available(monkeypatch) -> None:
    # What this tests: server datasets use server-style where path when supported.
    # Why important: avoids forcing NumPy local filtering for server-backed data.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeServerWhereDataset()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=95,
        slices="0:100"
    )

    assert "error" not in result
    assert result["execution_mode"] == "server_where"
    assert result["match_summary"]["matched"] == 4
    assert result["data"][-1] == 99


def test_where_filter_falls_back_to_local_numpy_when_server_where_unavailable(monkeypatch) -> None:
    # What this tests: server path fallback is explicit and safe when capabilities are missing.
    # Why important: keeps tool robust across API/version differences.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=95,
        slices="0:100"
    )

    assert "error" not in result
    assert result["execution_mode"] == "local_numpy"
    assert result["match_summary"]["matched"] == 4


# ---------------------------------------------------------------------------
# EDGE CASES AND ERROR HANDLING
# ---------------------------------------------------------------------------

def test_where_filter_invalid_operator(monkeypatch) -> None:
    # What this tests: invalid operator returns clear error.
    # Why important: graceful handling of bad LLM input.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">>>",  # Invalid
        threshold=50
    )

    assert "error" in result
    assert "Invalid operator" in result["error"]


def test_where_filter_large_request_returns_summary_only(monkeypatch) -> None:
    # What this tests: large filter requests are allowed but returned as summary-only payloads.
    # Why important: decouples operation size from LLM context size.
    class _LargeDataset:
        def __init__(self):
            self.shape = (100, 200, 200)  # 4,000,000 elements
            self.dtype = "float32"
            self._data = np.arange(4_000_000, dtype=np.float32).reshape(self.shape)
        def __getitem__(self, key):
            return self._data[key]

    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_LargeDataset()))
    result = data_access.where_filter(
        path="@public/large.b2nd",
        operator=">",
        threshold=0,
        slices="0:100, 0:200, 0:200"  # 4M elements
    )

    assert "error" not in result
    assert result["summary"]["num_elements"] == 4_000_000
    assert "data" not in result
    assert "_hint" in result


def test_where_filter_equal_operator(monkeypatch) -> None:
    # What this tests: == operator for exact value matching.
    # Why important: useful for finding specific values.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator="==",
        threshold=50,
        value_if_true=999,
        value_if_false=0,
        slices="0:100"
    )

    assert "error" not in result
    data = result["data"]
    
    # Only position 50 should be 999
    assert data[50] == 999
    assert data[49] == 0
    assert data[51] == 0
    assert result["match_summary"]["matched"] == 1


def test_where_filter_not_equal_operator(monkeypatch) -> None:
    # What this tests: != operator for exclusion filtering.
    # Why important: useful for masking specific values.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator="!=",
        threshold=0,
        slices="0:10"
    )

    assert "error" not in result
    # Values 1-9 should match (9 values), value 0 should not
    assert result["match_summary"]["matched"] == 9


# ---------------------------------------------------------------------------
# OUTPUT CONTRACT TESTS
# ---------------------------------------------------------------------------

def test_where_filter_output_contract(monkeypatch) -> None:
    # What this tests: result contains all expected fields.
    # Why important: contract for agent to parse results.
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=50
    )

    assert "error" not in result
    
    # Required fields
    assert "path" in result
    assert "dataset_shape" in result
    assert "dtype" in result
    assert "condition" in result
    assert "slice_applied" in result
    assert "result_shape" in result
    assert "value_if_true" in result
    assert "value_if_false" in result
    assert "execution_mode" in result
    assert "match_summary" in result
    assert "summary" in result
    assert "data" in result
    
    # Match summary structure
    ms = result["match_summary"]
    assert "matched" in ms
    assert "total" in ms
    assert "percentage" in ms


def test_where_filter_large_result_includes_hint(monkeypatch) -> None:
    # What this tests: large results include hint for LLM.
    # Why important: guides LLM to present summary instead of data dump.
    
    class _LargerDataset:
        def __init__(self):
            self.shape = (500,)
            self.dtype = "float64"
            self._data = np.arange(500)
        def __getitem__(self, key):
            return self._data[key]
    
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_LargerDataset()))
    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=0
    )

    assert "error" not in result
    assert "_hint" in result
    assert "summary" in result["_hint"].lower()
    assert "data" not in result


def test_where_filter_does_not_auto_register_server_data(monkeypatch) -> None:
    # What this tests: where_filter no longer auto-materializes results in notebook namespace.
    # Why important: materialization must be explicit (load_dataset).
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_FakeDataset1D()))
    clear_fetched_objects()

    result = data_access.where_filter(
        path="@public/example.b2nd",
        operator=">",
        threshold=50,
        slices="0:100"
    )

    assert "error" not in result
    assert get_fetched_objects() == {}


def test_where_filter_local_numpy_input_uses_blosc2_path() -> None:
    # What this tests: local NumPy inputs are normalized and filtered via Blosc2-native path.
    # Why important: migration goal is Blosc2-first internal execution.
    local_np = np.arange(10, dtype=np.int64)
    set_notebook_namespace({"local_np": local_np})

    result = data_access.where_filter(
        path="local_np",
        operator=">",
        threshold=5,
        slices="0:10"
    )

    assert "error" not in result
    assert result["execution_mode"] == "blosc2_where"
    assert result["match_summary"]["matched"] == 4
