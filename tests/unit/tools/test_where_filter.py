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
from caterva2_agent.tools._base import ResolvedData


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


def test_where_filter_oversized_request_rejected(monkeypatch) -> None:
    # What this tests: requests exceeding element limit are rejected.
    # Why important: prevents memory issues.
    
    class _HugeDataset:
        def __init__(self):
            self.shape = (1000, 1000, 1000)
            self.dtype = "float32"
        def __getitem__(self, key):
            raise AssertionError("Should not fetch data")
    
    monkeypatch.setattr(data_access, "resolve_data", lambda _path: _make_resolved(_HugeDataset()))
    result = data_access.where_filter(
        path="@public/huge.b2nd",
        operator=">",
        threshold=0,
        slices="0:100, 0:200, 0:200"  # 4M elements
    )

    assert "error" in result
    assert "exceeding limit" in result["error"]


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
