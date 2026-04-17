"""
Analysis tools for computing statistics on datasets.

Tools in this module:
- get_dataset_stats: Compute min, max, mean, std, and other statistics

Works with both:
- Server datasets (path starts with '@')
- Local variables (referenced by name)
"""

import logging
from typing import Dict, Any

logger = logging.getLogger('caterva2_agent')

from ._base import resolve_data, _to_json_safe


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Default statistics when none specified — the most commonly useful ones
DEFAULT_STATS = ["min", "max", "mean", "std"]

# All supported statistical operations
SUPPORTED_STATS = {"min", "max", "mean", "sum", "std", "var", "argmin", "argmax", "any", "all"}

# Supported reduction operations for collapse_dimensions
SUPPORTED_REDUCTIONS = {"min", "max", "mean", "sum", "std", "var", "prod"}


def _shape_product(shape: tuple[int, ...] | list[int]) -> int:
    """Compute number of elements from a shape tuple/list."""
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_stats",
            "description": (
                "Compute statistical summaries for a dataset or local variable. "
                "Returns multiple statistics in a single call (more efficient than separate calls). "
                "By default computes: min, max, mean, std. "
                "Use this when the user asks about data values, ranges, distributions, or wants to understand the data. "
                "Works with both server datasets (@path) and local variables (variable_name)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Server dataset path (e.g. '@public/examples/ds-1d.b2nd') "
                            "OR local variable name (e.g. 'my_data'). "
                            "Use '@' prefix for server datasets, plain name for local variables."
                        )
                    },
                    "stats": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["min", "max", "mean", "sum", "std", "var", "argmin", "argmax", "any", "all"]
                        },
                        "description": (
                            "Which statistics to compute. Defaults to ['min', 'max', 'mean', 'std'] if not specified. "
                            "Options: min, max, mean, sum, std (standard deviation), var (variance), "
                            "argmin (index of minimum), argmax (index of maximum), any (any True), all (all True)."
                        )
                    },
                    "axis": {
                        "type": "integer",
                        "description": (
                            "Axis along which to compute stats. "
                            "If not specified, computes over the flattened array (returns scalar). "
                            "For multi-dimensional arrays: axis=0 operates along rows, axis=1 along columns, etc."
                        )
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "collapse_dimensions",
            "description": (
                "Collapse a multi-dimensional dataset along one axis using an aggregation operation. "
                "This reduces N-dimensional data to (N-1)-dimensional data. "
                "CRITICAL for giant datasets: executes server-side on compressed Blosc2 data without downloading. "
                "Common use cases: "
                "- 3D tomography → 2D projection via max/mean/sum (medical imaging, microscopy) "
                "- 4D climate data → 3D spatial map via time-averaging "
                "- Point cloud density maps via spatial binning "
                "The result is registered in the notebook for visualization or further analysis. "
                "Works with both server datasets (@path) and local variables (variable_name)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Server dataset path (e.g. '@public/examples/ds-3d.b2nd') "
                            "OR local variable name (e.g. 'volume_data'). "
                            "Use '@' prefix for server datasets, plain name for local variables."
                        )
                    },
                    "axis": {
                        "type": "integer",
                        "description": (
                            "Axis along which to collapse the data (0-indexed). "
                            "For a 3D array with shape (100, 200, 300): "
                            "- axis=0 collapses the first dimension → result shape (200, 300) "
                            "- axis=1 collapses the second dimension → result shape (100, 300) "
                            "- axis=2 collapses the third dimension → result shape (100, 200)"
                        )
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["max", "mean", "sum", "min", "std", "var", "prod"],
                        "description": (
                            "Aggregation operation to apply: "
                            "- 'max': Maximum value along axis (max-intensity projection for tomographies) "
                            "- 'mean': Average value along axis (reduces noise, shows typical structure) "
                            "- 'sum': Sum along axis (useful for counting, density maps) "
                            "- 'min': Minimum value along axis "
                            "- 'std': Standard deviation along axis (shows variability) "
                            "- 'var': Variance along axis "
                            "- 'prod': Product along axis"
                        )
                    },
                    "variable_name": {
                        "type": "string",
                        "description": (
                            "Optional name for storing the result in the notebook namespace. "
                            "If not provided, auto-generates from path and operation (e.g. 'ds_3d_max_axis2'). "
                            "Use descriptive names like 'tomo_mip' or 'time_averaged_temp'."
                        )
                    }
                },
                "required": ["path", "axis", "operation"]
            }
        }
    },
]


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def get_dataset_stats(
    path: str,
    stats: list[str] | None = None,
    axis: int | None = None
) -> Dict[str, Any]:
    """
    Compute statistical summaries for a dataset or local variable.

    Groups multiple stats in one call for efficiency — avoids separate network
    round-trips for min, max, mean, etc. All stats share the same parameters
    (axis, keepdims) so they can be computed together.

    Args:
        path:  Server path (e.g. '@public/data.b2nd') or local variable name
        stats: Which statistics to compute. Defaults to ['min', 'max', 'mean', 'std'].
        axis:  Axis along which to compute. None = flattened array (scalar result).

    Returns:
        Dict with dataset metadata and computed statistics, or 'error' on failure.
    """
    # Validate and default the stats list
    stats_list: list[str]
    if stats is None:
        stats_list = DEFAULT_STATS
    else:
        stats_list = stats
    
    invalid_stats = set(stats_list) - SUPPORTED_STATS
    if invalid_stats:
        logger.warning(f"Unsupported statistics requested: {invalid_stats}")
        return {"error": f"Unsupported statistics: {invalid_stats}. Valid options: {SUPPORTED_STATS}"}

    source_type = "local variable" if not path.startswith('@') else "server dataset"
    logger.info(f"Computing stats for {source_type}: '{path}'")
    logger.debug(f"Stats: {stats_list}, axis: {axis}")

    try:
        resolved = resolve_data(path)
        data = resolved.data
        
        # Include basic metadata so the LLM has context
        result = {
            "name": path,
            "source": resolved.source,
            "shape": list(resolved.shape),
            "dtype": str(resolved.dtype),
            "axis": axis,
            "stats": {}
        }

        # Compute each requested statistic via backend-native array methods.
        for stat_name in stats_list:
            try:
                method = getattr(data, stat_name)
                raw_value = method(axis=axis)
            except AttributeError as e:
                return {
                    "error": (
                        f"Statistic '{stat_name}' is not available on backend "
                        f"'{resolved.backend}' for '{path}': {e}"
                    )
                }
            result["stats"][stat_name] = _to_json_safe(raw_value)

        return result

    except ValueError as e:
        # Variable not found or not array-like
        logger.warning(f"Stats validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to compute stats for '{path}': {e}")
        return {"error": f"Failed to compute stats for '{path}': {e}"}


def collapse_dimensions(
    path: str,
    axis: int,
    operation: str,
    variable_name: str | None = None
) -> Dict[str, Any]:
    """
    Collapse a multidimensional dataset along one axis using aggregation.
    
    This is THE key operation for exploring giant datasets — it executes
    server-side on Blosc2 compressed data, reducing dimensionality without
    downloading the full dataset.
    
    Common workflows:
    - 3D tomography → 2D max-intensity projection (medical imaging)
    - 4D climate data (time, lat, lon, alt) → 3D spatial average
    - Density maps via summing along spatial dimensions
    
    The result is automatically registered in the notebook namespace for
    visualization or further manipulation.
    
    Args:
        path: Server dataset path (e.g. '@public/examples/ds-3d.b2nd')
              OR local variable name (e.g. 'my_volume')
        axis: Which axis to collapse (0-indexed). For shape (100, 200, 300):
              axis=0 → result shape (200, 300)
              axis=1 → result shape (100, 300)
              axis=2 → result shape (100, 200)
        operation: Aggregation to apply: max, mean, sum, min, std, var, prod
        variable_name: Optional custom name for notebook storage. If None,
                      auto-generates like 'ds_3d_max_axis2'
    
    Returns:
        Dict with result metadata and storage info, or 'error' on failure.
    """
    # Validate operation
    if operation not in SUPPORTED_REDUCTIONS:
        return {
            "error": f"Unsupported operation: '{operation}'. "
                    f"Valid options: {sorted(SUPPORTED_REDUCTIONS)}"
        }
    
    source_type = "local variable" if not path.startswith('@') else "server dataset"
    logger.info(f"Collapsing {source_type}: '{path}'")
    logger.debug(f"Operation: {operation} along axis={axis}")
    
    try:
        from ._base import register_fetched_object
        
        resolved = resolve_data(path)
        data = resolved.data
        shape = resolved.shape
        ndim = len(shape)
        
        # Validate axis bounds
        if axis < 0 or axis >= ndim:
            return {
                "error": f"Invalid axis={axis} for {ndim}D array. "
                        f"Valid range: 0 to {ndim - 1}",
                "shape": list(shape)
            }
        
        logger.debug(f"Input shape: {shape}, ndim: {ndim}")
        
        # Calculate expected output size
        expected_shape = list(shape)
        expected_shape.pop(axis)
        expected_size = _shape_product(expected_shape) if expected_shape else 1
        
        # CRITICAL: Check if operation is feasible
        # For massive datasets, even the OUTPUT might be too large
        MAX_SAFE_OUTPUT = 100_000_000  # 100M elements (~400MB for float32)
        
        if expected_size > MAX_SAFE_OUTPUT:
            # Calculate reasonable stride for downsampling
            target_size = 2000  # Aim for 2000×2000 or equivalent
            current_max_dim = max(expected_shape)
            suggested_stride = max(1, current_max_dim // target_size)
            
            stride_example = ','.join(
                f'::{suggested_stride}' if i != axis else ':'
                for i in range(ndim)
            )
            
            return {
                "error": f"Output would be too large: {expected_size:,} elements "
                        f"({expected_size * 4 / 1e9:.2f} GB for float32)",
                "input_shape": list(shape),
                "output_shape": expected_shape,
                "axis": axis,
                "suggestion": f"For datasets this massive, first downsample via strided slicing:\n"
                             f"  1. Fetch downsampled data: get_slice('{path}', '{stride_example}', max_size=10000000)\n"
                             f"  2. Then collapse the smaller result\n"
                             f"This reduces {shape} → ~{tuple(max(1, s // suggested_stride) for s in shape)} "
                             f"before collapsing, making the operation feasible.",
                "note": "For truly massive datasets, consider pre-computed projections or tiled processing."
            }
        
        # Execute the reduction using backend-native methods.
        method = getattr(data, operation)
        result_data = method(axis=axis)

        result_shape = tuple(getattr(result_data, "shape", ()))
        result_size = int(getattr(result_data, "size", 1))
        result_dtype = str(getattr(result_data, "dtype", type(result_data).__name__))
        
        logger.debug(f"Result shape: {result_shape}, size: {result_size:,} elements")
        
        # Auto-generate variable name if not provided
        if variable_name is None:
            # Extract base name from path
            if path.startswith('@'):
                # '@public/dir/ds-3d.b2nd' → 'ds_3d'
                base_name = path.split('/')[-1].replace('.b2nd', '').replace('.b2frame', '')
                base_name = base_name.replace('-', '_')
            else:
                # 'my_data' → 'my_data'
                base_name = path.replace('-', '_')
            
            variable_name = f"{base_name}_{operation}_axis{axis}"
        
        # Sanitize variable name (remove special characters)
        variable_name = variable_name.replace('@', '').replace('/', '_').replace('.', '_')
        
        # Register in notebook namespace
        register_fetched_object(variable_name, result_data)
        
        logger.debug(f"✓ Stored as: '{variable_name}'")
        
        data_min = None
        data_max = None
        if hasattr(result_data, "min") and hasattr(result_data, "max"):
            data_min = _to_json_safe(result_data.min())
            data_max = _to_json_safe(result_data.max())

        # Build response
        result = {
            "status": "success",
            "operation": operation,
            "axis_collapsed": axis,
            "source_path": path,
            "source_shape": list(shape),
            "result_shape": list(result_shape),
            "result_size": int(result_size),
            "variable_name": variable_name,
            "dtype": result_dtype,
            "data_range": {
                "min": data_min,
                "max": data_max,
            },
            "note": (
                f"Result stored as '{variable_name}' in notebook. "
                f"Reduced {ndim}D → {len(result_shape)}D via {operation} along axis {axis}. "
                "Ready for immediate visualization."
            )
        }
        
        # Include a sample of the data if it's small enough
        if result_size <= 100:
            result["preview"] = _to_json_safe(result_data)
        
        return result
    
    except ValueError as e:
        logger.warning(f"Collapse validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to collapse '{path}': {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to collapse '{path}': {e}"}
