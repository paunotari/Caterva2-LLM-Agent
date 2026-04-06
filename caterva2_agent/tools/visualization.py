"""
Visualization tools for rendering datasets as interactive Plotly plots.

Tools in this module:
- visualize_dataset: Auto-detect dimensionality and render appropriate plot

Supported visualizations:
- 1D: Line plot (x=index, y=value)
- 2D: Heatmap with colorscale
- 3D: Volume rendering (ideal for tomographies)
- ≥4D: Error with guidance to slice down

Output: Displays inline in Jupyter notebooks via fig.show()

Works with both:
- Server datasets (path starts with '@')
- Local variables (referenced by name)
"""

from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom

from ._base import resolve_data
from .data_access import _parse_slice_string

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Maximum elements to render for interactive performance
# 3D volume with 80×80×80 ≈ 512,000 elements — reasonable for Plotly
DEFAULT_MAX_SIZE = 500_000

# Default colorscale for heatmaps and volumes
DEFAULT_COLORSCALE = "Viridis"

# Default opacity for 3D volume rendering (lower = more transparent)
DEFAULT_OPACITY = 0.3


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

VISUALIZATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "visualize_dataset",
            "description": (
                "Visualize a dataset or local variable as an interactive Plotly plot. "
                "Auto-detects dimensionality: 1D → line plot, 2D → heatmap, 3D → volume rendering. "
                "For 3D tomographies, renders a volume with transparency to show internal structures. "
                "Large datasets are automatically downsampled for interactive performance. "
                "For ≥4D datasets, you MUST provide slices to reduce to ≤3D. "
                "Displays inline in Jupyter notebooks. "
                "Works with both server datasets (@path) and local variables (variable_name)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Server dataset path (e.g. '@public/examples/kevlar-tomo.b2nd') "
                            "OR local variable name (e.g. 'my_data'). "
                            "Use '@' prefix for server datasets, plain name for local variables."
                        )
                    },
                    "slices": {
                        "type": "string",
                        "description": (
                            "Optional slice specification. Required for ≥4D datasets to reduce dimensions. "
                            "Uses Python slice syntax: '0:100', ':, 0:50, 0:50' etc. "
                            "Also useful to visualize a specific region of interest."
                        )
                    },
                    "colorscale": {
                        "type": "string",
                        "description": (
                            "colorscale name for heatmaps and volumes. "
                            "Options: 'Viridis', 'Plasma', 'Inferno', 'Greys', 'Blues', 'Hot', etc. "
                            "Default: 'Viridis'"
                        )
                    },
                    "opacity": {
                        "type": "number",
                        "description": (
                            "Opacity for 3D volume rendering (0.0 to 1.0). "
                            "Lower values show more internal structure. "
                            "Default: 0.3"
                        )
                    },
                    "max_size": {
                        "type": "integer",
                        "description": (
                            "Maximum number of elements to render. "
                            "Larger datasets are downsampled for performance. "
                            "Default: 500,000. Increase for higher quality, decrease for faster rendering."
                        )
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Optional custom title for the plot. "
                            "If not provided, uses the dataset path or variable name."
                        )
                    }
                },
                "required": ["path"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _downsample_3d(data: np.ndarray, max_elements: int) -> tuple[np.ndarray, float]:
    """
    Downsample a 3D array to fit within max_elements while preserving aspect ratio.
    
    Uses scipy.ndimage.zoom for smooth interpolation.
    
    Args:
        data: 3D numpy array
        max_elements: Target maximum number of elements
    
    Returns:
        Tuple of (downsampled_array, zoom_factor_used)
    """
    current_size = data.size
    if current_size <= max_elements:
        return data, 1.0
    
    # Calculate uniform zoom factor to achieve target size
    # For 3D: if we zoom by factor f in each dimension, new_size = old_size * f^3
    zoom_factor = (max_elements / current_size) ** (1/3)
    
    # Apply zoom with order=1 (bilinear) for speed, order=3 (cubic) for quality
    downsampled = zoom(data, zoom_factor, order=1)
    
    return downsampled, zoom_factor


def _plot_1d(data: np.ndarray, title: str, path: str) -> go.Figure:
    """
    Create a 1D line plot.
    
    Args:
        data: 1D numpy array
        title: Plot title
        path: Dataset path for axis label
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Values',
        line=dict(width=1)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        template="plotly_white"
    )
    return fig


def _plot_2d(data: np.ndarray, title: str, colorscale: str) -> go.Figure:
    """
    Create a 2D heatmap.
    
    Args:
        data: 2D numpy array
        title: Plot title
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=data,
        colorscale=colorscale,
        colorbar=dict(title="Value")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )
    return fig


def _plot_3d_volume(
    data: np.ndarray,
    title: str,
    colorscale: str,
    opacity: float
) -> go.Figure:
    """
    Create a 3D volume rendering.
    
    Plotly's go.Volume requires explicit X, Y, Z meshgrid coordinates
    and a flattened value array.
    
    Args:
        data: 3D numpy array (Z, Y, X) or (depth, height, width)
        title: Plot title
        colorscale: Plotly colorscale name
        opacity: Overall opacity (0-1)
    
    Returns:
        Plotly Figure object
    """
    # Create meshgrid coordinates for the volume
    # data.shape = (nz, ny, nx)
    nz, ny, nx = data.shape
    
    # Create coordinate arrays
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    
    # Create meshgrid (returns arrays of shape (nz, ny, nx))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data.flatten(),
        isomin=data_min,
        isomax=data_max,
        surface=dict(count=21),  # Render 21 isosurfaces across the value range for full volume effect
        opacity=opacity,
        colorscale=colorscale,
        colorbar=dict(title="Value"),
        caps=dict(x_show=False, y_show=False, z_show=False)  # Hide caps for cleaner look
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'  # Preserve aspect ratio
        ),
        template="plotly_white"
    )
    
    return fig


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATION
# ---------------------------------------------------------------------------

def visualize_dataset(
    path: str,
    slices: str | None = None,
    colorscale: str | None = None,
    opacity: float | None = None,
    max_size: int | None = None,
    title: str | None = None
) -> Dict[str, Any]:
    """
    Visualize a dataset or local variable as an interactive Plotly plot.
    
    Auto-detects dimensionality and renders:
    - 1D: Line plot
    - 2D: Heatmap
    - 3D: Volume rendering (for tomographies)
    - ≥4D: Returns error with guidance
    
    Large datasets are automatically downsampled for interactive performance.
    The plot displays inline in Jupyter notebooks.
    
    Args:
        path: Server dataset path (e.g. '@public/examples/kevlar-tomo.b2nd')
              OR local variable name (e.g. 'my_data')
        slices: Optional slice specification to select a region
        colorscale: Colorscale name (default: 'Viridis')
        opacity: Volume opacity 0-1 (default: 0.3)
        max_size: Max elements to render (default: 500,000)
        title: Custom plot title (default: dataset path)
    
    Returns:
        Dict with visualization status and metadata, or 'error' on failure.
    """
    # Apply defaults (explicit types for clarity)
    colorscale_final: str = colorscale or DEFAULT_COLORSCALE
    opacity_final: float = opacity if opacity is not None else DEFAULT_OPACITY
    max_size_final: int = max_size or DEFAULT_MAX_SIZE
    
    is_local = not path.startswith("@")
    source_type = "local variable" if is_local else "server dataset"
    
    print(f"→ Visualizing {source_type}: '{path}'")
    
    try:
        resolved = resolve_data(path)
        shape = resolved.shape
        ndim = len(shape)
        
        print(f"   Shape: {shape}, dtype: {resolved.dtype}")
        
        # --- Apply slices if provided ---
        if slices:
            slice_tuple = _parse_slice_string(slices, shape)
            data = resolved[slice_tuple]
            print(f"   Applied slice: {slices} → shape {data.shape}")
        else:
            # For very large datasets, we need to be careful
            total_elements = np.prod(shape)
            if total_elements > max_size_final * 10:
                # Dataset is way too large to fetch entirely — require slices
                return {
                    "error": f"Data has {total_elements:,} elements — too large to visualize directly. "
                             f"Please provide 'slices' to select a smaller region.",
                    "shape": list(shape),
                    "suggestion": f"Try slices like '0:{min(100, shape[0])}' for 1D, "
                                  f"or '0, :, :' for a 2D slice of a 3D array."
                }
            data = resolved[:]
        
        # Convert to numpy if needed
        data = np.asarray(data)
        actual_ndim = data.ndim
        
        print(f"   Data to visualize: shape={data.shape}, ndim={actual_ndim}, dtype={data.dtype}")
        
        # Handle structured/compound dtypes — extract first field
        if data.dtype.names is not None:
            # Structured array with multiple fields
            first_field = data.dtype.names[0]
            print(f"   Structured dtype detected with fields: {data.dtype.names}")
            print(f"   Extracting field '{first_field}' for visualization")
            data = data[first_field]
            print(f"   After extraction: shape={data.shape}, dtype={data.dtype}")
        
        # NOTE: Visualization does NOT inject data into notebook namespace.
        # This is intentional — visualization is for viewing, not working with data.
        # To get data for manipulation, use get_slice or where_filter instead.
        
        # --- Handle ≥4D: Error with guidance ---
        if actual_ndim >= 4:
            dim_suggestion = ", ".join(["0"] * (actual_ndim - 3) + [":"] * 3)
            return {
                "error": f"Cannot visualize {actual_ndim}D data directly. "
                         f"Please provide slices to reduce to 3D or less.",
                "current_shape": list(data.shape),
                "suggestion": f"Use slices='{dim_suggestion}' to select a 3D subset, "
                              f"or add more indices to get 2D or 1D."
            }
        
        # --- Generate title ---
        plot_title = title or f"{'Variable' if is_local else 'Dataset'}: {path}"
        if slices:
            plot_title += f" [{slices}]"
        
        # --- Route to appropriate visualization ---
        downsampled = False
        zoom_factor = 1.0
        original_shape = data.shape
        fig: go.Figure
        viz_type: str
        
        if actual_ndim == 1:
            # 1D: Line plot
            if data.size > max_size_final:
                # Simple downsampling for 1D
                stride = int(np.ceil(data.size / max_size_final))
                data = data[::stride]
                downsampled = True
                print(f"   Downsampled 1D: stride={stride}, new size={data.size}")
            
            fig = _plot_1d(data, plot_title, path)
            viz_type = "line_plot"
        
        elif actual_ndim == 2:
            # 2D: Heatmap
            if data.size > max_size_final:
                # Use Zoom for 2D downsampling
                zoom_factor = (max_size_final / data.size) ** 0.5
                data = zoom(data, zoom_factor, order=1)
                downsampled = True
                print(f"   Downsampled 2D: factor={zoom_factor:.3f}, new shape={data.shape}")
            
            # Further downsample if aspect ratio is still extreme (for interactive performance)
            aspect_ratio = max(data.shape) / min(data.shape)
            if aspect_ratio > 100:
                print(f"   ⚠ Extreme aspect ratio ({data.shape[0]} × {data.shape[1]}), further downsampling...")
                # Aggressively downsample the long dimension to ~1000 pixels max
                max_long_dim = 1000
                if max(data.shape) > max_long_dim:
                    # Use striding instead of zoom to avoid rounding issues
                    if data.shape[0] < data.shape[1]:
                        # Wide: downsample columns by striding
                        stride = max(1, data.shape[1] // max_long_dim)
                        data = data[:, ::stride]
                    else:
                        # Tall: downsample rows by striding
                        stride = max(1, data.shape[0] // max_long_dim)
                        data = data[::stride, :]
                    
                    downsampled = True
                    print(f"   Further downsampled 2D: new shape={data.shape}")
                    print(f"   Suggestion: Use slices to visualize a specific region-of-interest")
            
            fig = _plot_2d(data, plot_title, colorscale_final)
            viz_type = "heatmap"
        
        elif actual_ndim == 3:
            # 3D: Volume rendering
            if data.size > max_size_final:
                data, zoom_factor = _downsample_3d(data, max_size_final)
                downsampled = True
                print(f"   Downsampled 3D: factor={zoom_factor:.3f}, new shape={data.shape}")
            
            fig = _plot_3d_volume(data, plot_title, colorscale_final, opacity_final)
            viz_type = "volume"
        else:
            # Fallback for unexpected dimensionality (should never reach here)
            return {
                "error": f"Unsupported dimensionality: {actual_ndim}D"
            }
        
        # --- Display the figure ---
        fig.show()
        
        # --- Build response ---
        result: Dict[str, Any] = {
            "status": "success",
            "visualization_type": viz_type,
            "path": path,
            "source": source_type,
            "original_shape": list(original_shape),
            "rendered_shape": list(data.shape),
            "colorscale": colorscale_final,
        }
        
        if actual_ndim == 3:
            result["opacity"] = opacity_final
        
        if downsampled:
            result["downsampled"] = True
            result["zoom_factor"] = round(zoom_factor, 3)
            result["note"] = (
                f"Data was downsampled for interactive performance. "
                f"Original: {list(original_shape)}, Rendered: {list(data.shape)}. "
                f"Increase 'max_size' for higher resolution."
            )
        
        if slices:
            result["slices_applied"] = slices
        
        return result
    
    except ValueError as e:
        print(f"   ✗ Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return {"error": f"Failed to visualize '{path}': {e}"}
