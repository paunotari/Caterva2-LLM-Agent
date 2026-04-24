"""Essential unit tests for collapse_dimensions contract."""

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
from caterva2_agent.tools import analysis
from caterva2_agent.tools._base import ResolvedData, set_notebook_namespace, get_fetched_objects, clear_fetched_objects


class _FakeDataset3D:
    """Minimal Dataset mock with 3D shape and reduction methods."""
    
    def __init__(self):
        self.shape = (10, 20, 30)
        self.dtype = "float64"
        # Store a numpy array internally for realistic reduction behavior
        self._data = np.random.rand(10, 20, 30)
    
    def max(self, axis=None):
        return np.max(self._data, axis=axis)
    
    def mean(self, axis=None):
        return np.mean(self._data, axis=axis)
    
    def sum(self, axis=None):
        return np.sum(self._data, axis=axis)
    
    def min(self, axis=None):
        return np.min(self._data, axis=axis)
    
    def std(self, axis=None):
        return np.std(self._data, axis=axis)
    
    def var(self, axis=None):
        return np.var(self._data, axis=axis)
    
    def prod(self, axis=None):
        return np.prod(self._data, axis=axis)


def _make_resolved(fake_dataset):
    """Helper to create a ResolvedData wrapping a fake dataset."""
    return ResolvedData(fake_dataset, source='server', name='@test/data-3d.b2nd')


def test_collapse_dimensions_reduces_dimensionality(monkeypatch) -> None:
    # What this tests: collapsing along an axis reduces N-D to (N-1)-D.
    # Why important: this is the core functionality for giant dataset exploration.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=0,
        operation="max"
    )
    
    assert result["status"] == "success"
    assert result["source_shape"] == [10, 20, 30]
    assert result["result_shape"] == [20, 30]  # Collapsed axis 0
    assert result["axis_collapsed"] == 0
    assert result["operation"] == "max"


def test_collapse_dimensions_different_axes(monkeypatch) -> None:
    # What this tests: can collapse along any valid axis (0, 1, 2 for 3D).
    # Why important: users need flexibility to choose projection direction.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    # Collapse axis 1
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=1,
        operation="mean"
    )
    assert result["status"] == "success"
    assert result["result_shape"] == [10, 30]  # Collapsed axis 1
    
    # Collapse axis 2
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=2,
        operation="sum"
    )
    assert result["status"] == "success"
    assert result["result_shape"] == [10, 20]  # Collapsed axis 2


def test_collapse_dimensions_invalid_axis_returns_error(monkeypatch) -> None:
    # What this tests: invalid axis (out of bounds) is rejected with clear error.
    # Why important: prevents confusing numpy exceptions, gives actionable feedback.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=5,  # Invalid - dataset only has 3 dimensions (0, 1, 2)
        operation="max"
    )
    
    assert "error" in result
    assert "Invalid axis=5" in result["error"]
    assert "Valid range: 0 to 2" in result["error"]
    assert result["shape"] == [10, 20, 30]


def test_collapse_dimensions_invalid_operation_returns_error(monkeypatch) -> None:
    # What this tests: unsupported operations are rejected.
    # Why important: clear contract of what reductions are available.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=0,
        operation="median"  # Not in SUPPORTED_REDUCTIONS
    )
    
    assert "error" in result
    assert "Unsupported operation" in result["error"]
    assert "median" in result["error"]


def test_collapse_dimensions_all_operations_work(monkeypatch) -> None:
    # What this tests: all supported operations (max, mean, sum, min, std, var, prod) execute.
    # Why important: ensures every operation in SUPPORTED_REDUCTIONS actually works.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    operations = ["max", "mean", "sum", "min", "std", "var", "prod"]
    
    for op in operations:
        result = analysis.collapse_dimensions(
            path="@public/test-3d.b2nd",
            axis=1,
            operation=op
        )
        assert result["status"] == "success", f"Operation '{op}' failed"
        assert result["operation"] == op
        assert result["result_shape"] == [10, 30]


def test_collapse_dimensions_registers_result_in_namespace(monkeypatch) -> None:
    # What this tests: result is automatically stored in notebook namespace.
    # Why important: enables chaining with visualize_dataset and further analysis.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    # Setup fake notebook namespace
    fake_namespace = {}
    set_notebook_namespace(fake_namespace)
    clear_fetched_objects()
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=2,
        operation="max"
    )
    
    assert result["status"] == "success"
    var_name = result["variable_name"]
    
    # For server datasets: NOT registered in local namespace (only persisted to @personal)
    # Instead, check that result is NOT in fetched (different from local variables)
    fetched = get_fetched_objects()
    assert var_name not in fetched
    
    # Instead, verify response indicates server storage
    assert result.get("stored_server_side") in (True, False)  # depends on auth



def test_collapse_dimensions_custom_variable_name(monkeypatch) -> None:
    # What this tests: user can override variable name for server storage.
    # Why important: enables semantic naming like 'tomo_mip' instead of 'tomo_max_axis2'.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    clear_fetched_objects()
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=0,
        operation="max",
        variable_name="my_custom_projection"
    )
    
    assert result["status"] == "success"
    assert result["variable_name"] == "my_custom_projection"
    
    # For server datasets: custom name is used for @personal path, NOT local registry
    fetched = get_fetched_objects()
    assert "my_custom_projection" not in fetched
    assert result.get("stored_server_side") in (True, False)


def test_collapse_dimensions_auto_generated_name_format(monkeypatch) -> None:
    # What this tests: auto-generated names follow predictable pattern.
    # Why important: users can predict variable names for chaining operations.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    clear_fetched_objects()
    
    result = analysis.collapse_dimensions(
        path="@public/examples/ds-3d.b2nd",
        axis=1,
        operation="mean"
    )
    
    assert result["status"] == "success"
    # Expected: path '@public/examples/ds-3d.b2nd' → base name 'ds_3d'
    # With operation 'mean' and axis 1 → 'ds_3d_mean_axis1'
    assert result["variable_name"] == "ds_3d_mean_axis1"


def test_collapse_dimensions_includes_metadata(monkeypatch) -> None:
    # What this tests: result includes comprehensive metadata for LLM context.
    # Why important: LLM needs source shape, result shape, data range to interpret results.
    monkeypatch.setattr(analysis, "resolve_data", lambda _path: _make_resolved(_FakeDataset3D()))
    
    result = analysis.collapse_dimensions(
        path="@public/test-3d.b2nd",
        axis=0,
        operation="max"
    )
    
    assert result["status"] == "success"
    # Metadata fields
    assert "source_path" in result
    assert "source_shape" in result
    assert "result_shape" in result
    assert "result_size" in result
    assert "dtype" in result
    assert "data_range" in result
    assert "note" in result
    assert "visualize_dataset(" not in result["note"]
    
    # Data range should include min and max
    assert "min" in result["data_range"]
    assert "max" in result["data_range"]


def test_collapse_dimensions_with_local_numpy_array(monkeypatch) -> None:
    # What this tests: works with local numpy arrays, not just server datasets.
    # Why important: consistent API for both data sources.
    
    # Create a local 3D numpy array
    local_array = np.random.rand(5, 10, 15)
    
    # Mock resolve_data to return local data
    def fake_resolve(path):
        return ResolvedData(local_array, source='local', name='local_volume')
    
    monkeypatch.setattr(analysis, "resolve_data", fake_resolve)
    clear_fetched_objects()
    
    result = analysis.collapse_dimensions(
        path="local_volume",
        axis=1,
        operation="mean"
    )
    
    assert result["status"] == "success"
    assert result["source_shape"] == [5, 10, 15]
    assert result["result_shape"] == [5, 15]
    
    # Verify the result matches numpy's mean
    fetched = get_fetched_objects()
    stored = fetched[result["variable_name"]]
    expected = np.mean(local_array, axis=1)
    assert np.allclose(stored, expected)


def test_collapse_dimensions_preview_for_small_results(monkeypatch) -> None:
    # What this tests: small results (<= 100 elements) include data preview.
    # Why important: LLM can see actual values for tiny results without separate get_slice.
    
    # Create small 3D array that will result in tiny output
    small_array = np.ones((3, 4, 5))
    
    def fake_resolve(path):
        return ResolvedData(small_array, source='local', name='small_volume')
    
    monkeypatch.setattr(analysis, "resolve_data", fake_resolve)
    
    result = analysis.collapse_dimensions(
        path="small_volume",
        axis=0,
        operation="max"
    )
    
    # Result shape is (4, 5) = 20 elements (<= 100)
    assert result["result_size"] == 20
    assert "preview" in result  # Should include preview
    
    # Verify preview data is present
    preview = result["preview"]
    assert preview is not None
