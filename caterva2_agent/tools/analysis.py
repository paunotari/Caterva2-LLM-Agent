"""
Analysis tools for computing statistics on datasets.

Tools in this module:
- get_dataset_stats: Compute min, max, mean, std, and other statistics

Works with both:
- Server datasets (path starts with '@')
- Local variables (referenced by name)
"""

from typing import Dict, Any
import numpy as np

from ._base import resolve_data, _to_json_safe


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Default statistics when none specified — the most commonly useful ones
DEFAULT_STATS = ["min", "max", "mean", "std"]

# All supported statistical operations
SUPPORTED_STATS = {"min", "max", "mean", "sum", "std", "var", "argmin", "argmax", "any", "all"}


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
        return {"error": f"Unsupported statistics: {invalid_stats}. Valid options: {SUPPORTED_STATS}"}

    source_type = "local variable" if not path.startswith('@') else "server dataset"
    print(f"→ Computing stats for {source_type}: '{path}'")
    print(f"   Stats: {stats_list}, axis: {axis}")

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

        # Compute each requested statistic
        # For local numpy arrays, we need to use numpy functions
        # For server datasets, we use the dataset methods
        for stat_name in stats_list:
            if resolved.is_local():
                # Use numpy functions for local arrays
                if stat_name == "argmin":
                    raw_value = np.argmin(data, axis=axis)
                elif stat_name == "argmax":
                    raw_value = np.argmax(data, axis=axis)
                else:
                    func = getattr(np, stat_name)
                    raw_value = func(data, axis=axis)
            else:
                # Use dataset methods for server data
                method = getattr(data, stat_name)
                raw_value = method(axis=axis)
            
            result["stats"][stat_name] = _to_json_safe(raw_value)

        return result

    except ValueError as e:
        # Variable not found or not array-like
        print(f"   ✗ {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return {"error": f"Failed to compute stats for '{path}': {e}"}
