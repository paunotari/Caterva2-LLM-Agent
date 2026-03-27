"""
Analysis tools for computing statistics on Caterva2 datasets.

Tools in this module:
- get_dataset_stats: Compute min, max, mean, std, and other statistics
"""

from typing import Dict, Any

from ._base import _get_dataset, _to_json_safe


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Default statistics when none specified — the most commonly useful ones
DEFAULT_STATS = ["min", "max", "mean", "std"]

# All supported statistical operations, mapped to Dataset method names
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
                "Compute statistical summaries for a dataset. "
                "Returns multiple statistics in a single call (more efficient than separate calls). "
                "By default computes: min, max, mean, std. "
                "Use this when the user asks about data values, ranges, distributions, or wants to understand the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path to the dataset including the root name. "
                            "Example: '@public/examples/ds-1d.b2nd'"
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
    Compute statistical summaries for a dataset.

    Groups multiple stats in one call for efficiency — avoids separate network
    round-trips for min, max, mean, etc. All stats share the same parameters
    (axis, keepdims) so they can be computed together.

    Args:
        path:  Full path to dataset (e.g. '@public/examples/ds-1d.b2nd')
        stats: Which statistics to compute. Defaults to ['min', 'max', 'mean', 'std'].
        axis:  Axis along which to compute. None = flattened array (scalar result).

    Returns:
        Dict with dataset metadata and computed statistics, or 'error' on failure.
    """
    # Validate and default the stats list
    if stats is None:
        stats = DEFAULT_STATS
    
    invalid_stats = set(stats) - SUPPORTED_STATS
    if invalid_stats:
        return {"error": f"Unsupported statistics: {invalid_stats}. Valid options: {SUPPORTED_STATS}"}

    print(f"→ Computing stats for dataset: '{path}'")
    print(f"   Stats: {stats}, axis: {axis}")
    print(f"   API: dataset.min(), .max(), etc.")

    try:
        dataset = _get_dataset(path)
        
        # Include basic metadata so the LLM has context
        result = {
            "path": path,
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "axis": axis,
            "stats": {}
        }

        # Compute each requested statistic
        for stat_name in stats:
            method = getattr(dataset, stat_name)
            # All stat methods accept axis parameter (None = flatten)
            raw_value = method(axis=axis)
            result["stats"][stat_name] = _to_json_safe(raw_value)

        return result

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return {"error": f"Failed to compute stats for '{path}': {e}"}
