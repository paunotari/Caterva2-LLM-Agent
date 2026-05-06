"""
Unit tests for visualization tool.

Tests the visualize_dataset function and its helper functions.
Uses pytest fixtures from conftest.py which mock external dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestDownsampling:
    """Tests for the _downsample_3d helper function."""
    
    def test_downsample_preserves_small_arrays(self):
        """Arrays already under max_elements should not be downsampled."""
        from caterva2_agent.tools.visualization import _downsample_3d
        
        data = np.random.rand(10, 10, 10)  # 1000 elements
        result, factor = _downsample_3d(data, max_elements=5000)
        
        assert result.shape == data.shape
        assert factor == 1.0
        np.testing.assert_array_equal(result, data)
    
    def test_downsample_reduces_large_arrays(self):
        """Arrays over max_elements should be downsampled."""
        from caterva2_agent.tools.visualization import _downsample_3d
        
        data = np.random.rand(100, 100, 100)  # 1,000,000 elements
        result, factor = _downsample_3d(data, max_elements=50000)
        
        # Should be much smaller
        assert result.size < data.size
        assert result.size <= 60000  # Some tolerance for zoom
        assert 0 < factor < 1


class TestVisualizationSchemas:
    """Tests for tool schema structure."""
    
    def test_visualize_dataset_schema_exists(self):
        """visualize_dataset should be in TOOLS list."""
        from caterva2_agent.tools.visualization import VISUALIZATION_TOOLS
        
        names = [t["function"]["name"] for t in VISUALIZATION_TOOLS]
        assert "visualize_dataset" in names
    
    def test_visualize_dataset_has_required_params(self):
        """Schema should require 'path' parameter."""
        from caterva2_agent.tools.visualization import VISUALIZATION_TOOLS
        
        viz_tool = VISUALIZATION_TOOLS[0]
        required = viz_tool["function"]["parameters"]["required"]
        
        assert "path" in required
    
    def test_visualize_dataset_has_optional_params(self):
        """Schema should have slices, colorscale, opacity, max_size, title."""
        from caterva2_agent.tools.visualization import VISUALIZATION_TOOLS
        
        viz_tool = VISUALIZATION_TOOLS[0]
        params = viz_tool["function"]["parameters"]["properties"]
        
        assert "slices" in params
        assert "colorscale" in params
        assert "opacity" in params
        assert "max_size" in params
        assert "title" in params


class TestPlotHelpers:
    """Tests for individual plot helper functions."""
    
    def test_plot_1d_returns_figure(self):
        """_plot_1d should return a Plotly Figure."""
        from caterva2_agent.tools.visualization import _plot_1d, DEFAULT_INTERACTIVE_HEIGHT
        import plotly.graph_objects as go
        
        data = np.array([1, 2, 3, 4, 5])
        fig = _plot_1d(data, title="Test", path="@test/data")
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One trace
        assert fig.layout.height == DEFAULT_INTERACTIVE_HEIGHT
    
    def test_plot_2d_returns_figure(self):
        """_plot_2d should return a Plotly Figure."""
        from caterva2_agent.tools.visualization import _plot_2d, DEFAULT_INTERACTIVE_HEIGHT
        import plotly.graph_objects as go
        
        data = np.array([[1, 2], [3, 4]])
        fig = _plot_2d(data, title="Test", colorscale="Viridis")
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One heatmap trace
        assert fig.layout.height == DEFAULT_INTERACTIVE_HEIGHT
    
    def test_plot_3d_volume_returns_figure(self):
        """_plot_3d_volume should return a Plotly Figure."""
        from caterva2_agent.tools.visualization import (
            _plot_3d_volume,
            DEFAULT_INTERACTIVE_HEIGHT,
        )
        import plotly.graph_objects as go
        
        data = np.random.rand(5, 5, 5)
        fig = _plot_3d_volume(data, title="Test", colorscale="Viridis", opacity=0.3)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One volume trace
        assert fig.layout.height == DEFAULT_INTERACTIVE_HEIGHT


class TestStructuredArrays:
    """Tests for handling structured/compound dtypes from Caterva2."""
    
    def test_structured_array_field_extraction(self):
        """Structured arrays should have first field extracted for visualization."""
        from caterva2_agent.tools.visualization import visualize_dataset
        from caterva2_agent.tools._base import set_notebook_namespace
        
        # Create a structured 2D array
        structured_data = np.zeros((5, 5), dtype=[('a', '<f4'), ('b', '<f8')])
        for i in range(5):
            for j in range(5):
                structured_data[i, j] = (i + j, i * j)
        
        # Mock local variable
        namespace = {'test_struct': structured_data}
        set_notebook_namespace(namespace)
        
        result = visualize_dataset('test_struct')
        
        # Should succeed and report heatmap
        assert result['status'] == 'success'
        assert result['visualization_type'] == 'heatmap'
        assert result['rendered_shape'] == [5, 5]
