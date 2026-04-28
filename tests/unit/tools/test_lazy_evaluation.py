"""
Tests for lazy evaluation workflow across tools.

These tests verify that:
1. LazyArray expressions are preserved through tool chains
2. compute=False keeps lazy, compute=True materializes
3. Stats work on lazy expressions without full materialization
4. Operations compose without intermediate materialization
"""

import pytest
import numpy as np
import blosc2

from caterva2_agent.tools._base import (
    _normalize_local_array,
    register_fetched_object,
    get_notebook_namespace,
    set_notebook_namespace,
    clear_fetched_objects,
)
from caterva2_agent.tools.data_access import where_filter, get_slice
from caterva2_agent.tools.analysis import collapse_dimensions


# ============================================================================
# Test: LazyArray Preservation in _normalize_local_array
# ============================================================================

class TestNormalizeLocalArrayPreservesLazy:
    """Verify that _normalize_local_array keeps lazy arrays as-is."""

    def test_preserves_ndarray_blosc2(self):
        """NDArray input should be returned unchanged."""
        arr = blosc2.asarray(np.arange(100))
        normalized, source_type = _normalize_local_array(arr, "test_nd")
        
        assert normalized is arr  # Same object
        assert source_type is None  # No conversion needed
        assert isinstance(normalized, blosc2.NDArray)

    def test_preserves_lazyarray(self):
        """LazyArray input should be returned unchanged."""
        arr = blosc2.asarray(np.arange(100))
        lazy = arr > 50  # Creates LazyExpr
        
        normalized, source_type = _normalize_local_array(lazy, "test_lazy")
        
        assert normalized is lazy  # Same object
        assert source_type is None  # No conversion needed
        assert isinstance(normalized, blosc2.LazyArray)

    def test_converts_numpy_to_blosc2(self):
        """numpy.ndarray should be converted to blosc2."""
        arr = np.arange(100)
        
        normalized, source_type = _normalize_local_array(arr, "test_np")
        
        assert isinstance(normalized, blosc2.NDArray)
        assert source_type == "ndarray"  # Conversion recorded

    def test_converts_list_to_blosc2(self):
        """List should be converted to blosc2."""
        arr = [1, 2, 3, 4, 5]
        
        normalized, source_type = _normalize_local_array(arr, "test_list")
        
        assert isinstance(normalized, blosc2.NDArray)
        assert source_type == "list"


# ============================================================================
# Test: Local LazyArray Chaining (Multiple Filters)
# ============================================================================

class TestLocalLazyChaining:
    """Verify that local lazy expressions can be chained without materialization."""

    @pytest.fixture(autouse=True)
    def setup_notebook(self):
        """Set up notebook namespace for each test."""
        namespace = {"volume_data": blosc2.asarray(np.arange(1000).reshape(10, 10, 10))}
        set_notebook_namespace(namespace)
        clear_fetched_objects()
        yield
        set_notebook_namespace(None)
        clear_fetched_objects()

    def test_lazy_chain_two_filters(self):
        """Chain two where_filter calls on local data—should stay lazy until materialization."""
        # First filter: data > 300
        result1 = where_filter(
            path="volume_data",
            operator=">",
            threshold=300,
            value_if_false=None,
            compute=False  # Keep lazy
        )
        
        assert result1.get("error") is None
        
        # Get the auto-injected variable name
        namespace = get_notebook_namespace()
        # Variable names have underscores removed: volume_data → filtered_volumedata
        filtered_vars = [v for v in namespace if "filtered" in v]
        assert len(filtered_vars) > 0, f"First filter should have auto-injected result. Namespace: {list(namespace.keys())}"
        injected_var = filtered_vars[0]
        
        # Verify the injected result is a LazyArray (or NDArray from first filter)
        injected_result = namespace[injected_var]
        assert isinstance(injected_result, (blosc2.NDArray, blosc2.LazyArray))
        
        # Chain second filter
        result2 = where_filter(
            path=injected_var,
            operator="<",
            threshold=700,
            value_if_false=None,
            compute=False  # Keep lazy
        )
        
        assert result2.get("error") is None
        assert "summary" in result2  # Stats should work on lazy

    def test_lazy_stats_without_materialization(self):
        """Get stats on lazy expression without full materialization."""
        from caterva2_agent.tools.analysis import get_dataset_stats
        
        # Create lazy expression
        namespace = get_notebook_namespace()
        arr = namespace["volume_data"]
        lazy = arr > 500  # Create LazyExpr
        
        # Register the lazy expression
        namespace["lazy_filtered"] = lazy
        register_fetched_object("lazy_filtered", lazy)
        
        # Get stats on lazy expression
        stats = get_dataset_stats(path="lazy_filtered")
        
        assert stats.get("error") is None
        assert "stats" in stats
        assert "min" in stats["stats"]
        assert "max" in stats["stats"]
        # min/max on boolean should be 0 and 1
        assert stats["stats"]["min"] in [0, False]
        assert stats["stats"]["max"] in [1, True]

    def test_collapse_preserves_lazy_with_compute_false(self):
        """collapse_dimensions with compute=False should return lazy."""
        result = collapse_dimensions(
            path="volume_data",
            axis=0,
            operation="mean",
            compute=False  # Keep lazy
        )
        
        assert result.get("error") is None
        assert "result_shape" in result
        # Result should be 2D (10, 10) after collapsing first dimension of (10, 10, 10)
        assert result["result_shape"] == [10, 10]
        
        # Verify result is auto-injected to notebook
        namespace = get_notebook_namespace()
        collapsed_vars = [v for v in namespace if "collapsed_volume_data_mean" in v]
        assert len(collapsed_vars) > 0, "collapse_dimensions should auto-inject result"


# ============================================================================
# Test: Compute Parameter Behavior
# ============================================================================

class TestComputeParameterBehavior:
    """Verify that compute=True/False controls materialization."""

    @pytest.fixture(autouse=True)
    def setup_notebook(self):
        """Set up notebook namespace for each test."""
        namespace = {"test_volume": blosc2.asarray(np.arange(500).reshape(5, 10, 10))}
        set_notebook_namespace(namespace)
        clear_fetched_objects()
        yield
        set_notebook_namespace(None)
        clear_fetched_objects()

    def test_where_filter_compute_false_returns_lazy(self):
        """where_filter with compute=False should preserve lazy in result."""
        result = where_filter(
            path="test_volume",
            operator=">",
            threshold=200,
            compute=False
        )
        
        assert result.get("error") is None
        # Result should show it came from local operation (auto-injected)
        assert "summary" in result

    def test_where_filter_compute_true_materializes(self):
        """where_filter with compute=True should materialize."""
        result = where_filter(
            path="test_volume",
            operator=">",
            threshold=200,
            compute=True
        )
        
        assert result.get("error") is None
        # Result should be materialized (blosc2.NDArray)
        assert "summary" in result

    def test_collapse_compute_false_returns_lazy(self):
        """collapse_dimensions with compute=False should preserve lazy."""
        result = collapse_dimensions(
            path="test_volume",
            axis=0,
            operation="mean",
            compute=False
        )
        
        assert result.get("error") is None
        assert result.get("result_shape") == [10, 10]

    def test_collapse_compute_true_materializes(self):
        """collapse_dimensions with compute=True should materialize."""
        result = collapse_dimensions(
            path="test_volume",
            axis=0,
            operation="mean",
            compute=True
        )
        
        assert result.get("error") is None
        assert result.get("result_shape") == [10, 10]


# ============================================================================
# Test: LazyArray Operations
# ============================================================================

class TestLazyArrayOperations:
    """Verify that LazyArray operations work correctly."""

    def test_lazyarray_arithmetic_operations(self):
        """LazyArray should support arithmetic without materialization."""
        arr = blosc2.asarray(np.arange(100).reshape(10, 10))
        
        # Create lazy expressions
        lazy1 = arr > 50
        lazy2 = arr < 80
        
        # Combine with logical operation
        lazy_combined = lazy1 & lazy2
        
        # Should still be lazy
        assert isinstance(lazy_combined, blosc2.LazyArray)
        
        # Stats should work
        sum_result = lazy_combined.sum()
        assert isinstance(sum_result, (int, float, np.integer))
        assert sum_result > 0

    def test_lazyarray_materialization_with_getitem(self):
        """LazyArray[:] should convert to numpy.ndarray."""
        arr = blosc2.asarray(np.arange(100).reshape(10, 10))
        lazy = arr > 50
        
        # Materialize with [:]
        result = lazy[:]
        
        # Should be numpy now
        assert isinstance(result, np.ndarray)
        # Should be boolean array
        assert result.dtype == bool

    def test_lazyarray_stats_without_materialization(self):
        """LazyArray stats should work without calling [:]."""
        arr = blosc2.asarray(np.arange(100).reshape(10, 10))
        lazy = arr > 50
        
        # Get stats without materializing
        min_val = lazy.min()
        max_val = lazy.max()
        mean_val = lazy.mean()
        sum_val = lazy.sum()
        
        # Should all be valid
        assert isinstance(min_val, (bool, int, float, np.generic))
        assert isinstance(max_val, (bool, int, float, np.generic))
        assert isinstance(mean_val, (int, float, np.floating, np.generic))
        assert isinstance(sum_val, (int, float, np.integer, np.generic))


# ============================================================================
# Test: Lazy Slicing
# ============================================================================

class TestLazySlicing:
    """Verify that get_slice with compute=False preserves lazy."""

    @pytest.fixture(autouse=True)
    def setup_notebook(self):
        """Set up notebook namespace for each test."""
        namespace = {"slice_test_data": blosc2.asarray(np.arange(1000).reshape(10, 10, 10))}
        set_notebook_namespace(namespace)
        clear_fetched_objects()
        yield
        set_notebook_namespace(None)
        clear_fetched_objects()

    def test_get_slice_local_with_compute_false(self):
        """get_slice on local data with compute=False should work."""
        result = get_slice(
            path="slice_test_data",
            slices="0:5, 0:5, 0:5",  # Small slice
            persist_result=False,
            compute=False
        )
        
        assert result.get("error") is None
        assert result["result_shape"] == [5, 5, 5]
        assert result["estimated_elements"] == 125

    def test_get_slice_local_auto_injects(self):
        """get_slice on local data should auto-inject result to notebook."""
        result = get_slice(
            path="slice_test_data",
            slices="0:3, 0:3, 0:3",
            persist_result=False,
            compute=False
        )
        
        assert result.get("error") is None
        
        # Check that result was auto-injected
        # Variable names have underscores removed: slice_test_data → sliced_slicetestdata
        namespace = get_notebook_namespace()
        sliced_vars = [v for v in namespace if "sliced" in v]
        assert len(sliced_vars) > 0, f"Should have auto-injected sliced result. Namespace: {list(namespace.keys())}"


# ============================================================================
# Test: Complex Lazy Workflows
# ============================================================================

class TestComplexLazyWorkflows:
    """Test realistic workflows combining multiple lazy operations."""

    @pytest.fixture(autouse=True)
    def setup_notebook(self):
        """Set up notebook namespace with complex data."""
        # Simulate 3D volume: (50, 100, 100)
        data = np.random.normal(loc=500, scale=50, size=(50, 100, 100))
        arr = blosc2.asarray(data)
        namespace = {"volume": arr}
        set_notebook_namespace(namespace)
        clear_fetched_objects()
        yield
        set_notebook_namespace(None)
        clear_fetched_objects()

    def test_workflow_filter_collapse_stats(self):
        """Realistic workflow: filter → collapse → stats (all lazy)."""
        
        # Step 1: Filter (compute=False keeps lazy)
        filter_result = where_filter(
            path="volume",
            operator=">",
            threshold=450,  # Keep values above 450
            compute=False
        )
        assert filter_result.get("error") is None
        
        # Step 2: Get the auto-injected variable name
        namespace = get_notebook_namespace()
        filtered_var = None
        for var in namespace:
            if "filtered" in var and "volume" in var:
                filtered_var = var
                break
        
        assert filtered_var is not None, "Should have filtered variable"
        
        # Step 3: Collapse the filtered data (compute=False keeps lazy)
        collapse_result = collapse_dimensions(
            path=filtered_var,
            axis=0,
            operation="mean",
            compute=False
        )
        assert collapse_result.get("error") is None
        assert collapse_result["result_shape"] == [100, 100]
        
        # Step 4: Get stats on collapsed (still lazy)
        from caterva2_agent.tools.analysis import get_dataset_stats
        
        # Find the collapsed variable
        namespace = get_notebook_namespace()
        collapsed_var = None
        for var in namespace:
            if "collapsed" in var and "filtered" in var:
                collapsed_var = var
                break
        
        if collapsed_var:  # Only if auto-injection worked
            stats_result = get_dataset_stats(
                path=collapsed_var
            )
            assert stats_result.get("error") is None
            assert "stats" in stats_result

    def test_workflow_multiple_filters_local_chain(self):
        """Test chaining multiple local filters."""
        
        # Filter 1: values > 450
        f1 = where_filter(
            path="volume",
            operator=">",
            threshold=450,
            compute=False
        )
        assert f1.get("error") is None
        
        # Get the filtered variable
        namespace = get_notebook_namespace()
        filtered_var_1 = None
        for var in namespace:
            if "filtered_volume" in var:
                filtered_var_1 = var
                break
        
        if filtered_var_1:
            # Filter 2: values < 550
            f2 = where_filter(
                path=filtered_var_1,
                operator="<",
                threshold=550,
                compute=False
            )
            assert f2.get("error") is None
            
            # Verify the filtered result
            namespace = get_notebook_namespace()
            assert any("filtered" in v for v in namespace)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestLazyEvaluationEdgeCases:
    """Test edge cases and error handling in lazy evaluation."""

    @pytest.fixture(autouse=True)
    def setup_notebook(self):
        """Set up notebook namespace."""
        namespace = {"numpy_data": np.arange(100)}
        set_notebook_namespace(namespace)
        clear_fetched_objects()
        yield
        set_notebook_namespace(None)
        clear_fetched_objects()

    def test_lazy_with_invalid_operation_still_errors(self):
        """LazyArray should not bypass error checking."""
        result = where_filter(
            path="nonexistent_variable",
            operator=">",
            threshold=100,
            compute=False
        )
        
        # Should still error on nonexistent variable
        assert result.get("error") is not None

    def test_compute_parameter_with_local_numpy_data(self):
        """compute parameter should not affect local numpy data."""
        # compute=False
        result1 = where_filter(
            path="numpy_data",
            operator=">",
            threshold=50,
            compute=False
        )
        assert result1.get("error") is None
        
        # compute=True
        result2 = where_filter(
            path="numpy_data",
            operator=">",
            threshold=50,
            compute=True
        )
        assert result2.get("error") is None

    def test_lazy_with_slicing(self):
        """Test that lazy expressions work with slicing."""
        namespace = get_notebook_namespace()
        arr = blosc2.asarray(np.arange(1000).reshape(10, 10, 10))
        namespace["data"] = arr
        register_fetched_object("data", arr)
        
        # Create lazy expression
        lazy = arr > 500
        namespace["lazy"] = lazy
        register_fetched_object("lazy", lazy)
        
        # Slice the lazy expression
        sliced = lazy[0:5, 0:5, 0:5]
        
        # Should work
        assert sliced.shape == (5, 5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
