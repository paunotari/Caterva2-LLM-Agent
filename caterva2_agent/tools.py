"""
Tool definitions and implementations for the Caterva2 agent.

This file contains:
1. Tool schemas (TOOLS)    — JSON descriptions the LLM reads to know what tools exist
2. Tool implementations    — Python functions that call the Caterva2 API
3. Tool dispatcher         — execute_tool() routes tool names to their implementations

Adding a new tool: touch 3 places in this file:
  1. Add a JSON schema entry to TOOLS
  2. Implement the Python function
  3. Register it in TOOL_MAP
"""

import json
from typing import Dict, Any

import caterva2 as cat2

from config import CATERVA2_URLBASE


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# These JSON schemas are sent to the LLM so it knows what tools it can call.
# Format follows the OpenAI function calling spec (also used by Groq).
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_roots",
            "description": (
                "Connect to the Caterva2 server and list all available roots. "
                "A root is a top-level data collection (like a folder). "
                "Only call this if the available roots are not already known from the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_datasets",
            "description": (
                "List datasets (arrays and files) within a given root or sub-path. "
                "Results are paginated: use 'limit' and 'offset' to navigate large listings. "
                "The response always includes 'total' (full count) and 'has_more' so you can "
                "tell the user if more results exist. To show the next page, call again with "
                "offset increased by limit. "
                "Only call this if the datasets for that path are not already known from the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "The root name or sub-path to list. "
                            "Root names always start with '@' (e.g. '@public'). "
                            "Use the exact root name returned by list_roots — never drop the '@' prefix. "
                            "Sub-paths use '/' as separator (e.g. '@public/examples')."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of datasets to return per page. Defaults to 50. Larger values are allowed if the user requests all datasets."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of datasets to skip before returning results. Defaults to 0."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_info",
            "description": (
                "Retrieve detailed metadata about a specific dataset. "
                "Returns shape, dtype, chunk layout, block layout, compression info, "
                "and modification time. "
                "Use this when the user asks about a dataset's structure or properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path to the dataset including the root name. "
                            "Root names always start with '@'. "
                            "Use the exact paths returned by list_datasets — never drop the '@' prefix. "
                            "Example: '@public/examples/ds-2d-fields.b2nd'"
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
    {
        "type": "function",
        "function": {
            "name": "get_slice",
            "description": (
                "Retrieve a slice of data values from a dataset. "
                "Use this when the user wants to see actual data, not just statistics. "
                "SAFETY: Limited to 10,000 elements maximum to avoid memory issues. "
                "For large datasets, use slicing to select a specific region of interest."
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
                    "slices": {
                        "type": "string",
                        "description": (
                            "Slice specification using Python syntax. "
                            "For 1D: '0:100' (first 100 elements), ':10' (first 10), '-5:' (last 5). "
                            "For 2D: '0:5, 0:3' (5 rows, 3 columns), ':, 0' (all rows, first column). "
                            "For 3D+: '0, :, 0:10' etc. Separate dimensions with commas. "
                            "Defaults to first elements up to the limit if not specified."
                        )
                    }
                },
                "required": ["path"]
            }
        }
    }
]





# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# These functions call the real Caterva2 API.
# Each one wraps its call in try/except and always returns a JSON-serializable dict.
# This ensures the agent always gets a structured result, even on failure.
# ---------------------------------------------------------------------------

_cat2_client: cat2.Client | None = None

def _get_client() -> cat2.Client:
    """
    Create (if needed) and return a Caterva2 Client for the configured subscriber URL.
    Only creates the client once and reuses it for all tool calls.
    """
    global _cat2_client
    if _cat2_client is None:
        _cat2_client = cat2.Client(CATERVA2_URLBASE)
    return _cat2_client


def _get_dataset(path: str) -> cat2.Dataset:
    """
    Retrieve a Dataset object from the Caterva2 server.

    Uses the direct-path approach: client.get('@root/path/to/file.b2nd').
    This works because list_datasets() already returns full paths including the root.

    Args:
        path: Full path to dataset (e.g. '@public/examples/ds-1d.b2nd')

    Returns:
        The Dataset object for manipulation (stats, slicing, etc.)
    """
    client = _get_client()
    return client.get(path)


def list_roots() -> Dict[str, Any]:
    """
    List all available roots (top-level collections) on the Caterva2 server.

    Returns:
        Dict with 'roots' key (list of root name strings), or 'error' on failure.
    """
    print("→ Listing available roots on the Caterva2 server")
    print("   API: client.get_roots()")
    try:
        client = _get_client()
        roots_dict = client.get_roots()
        # get_roots() returns {name: {"name": name}, ...} — we extract just the names
        return {"roots": sorted(roots_dict.keys())}
    except Exception as e:
        print("   ✗ Failed")
        return {"error": f"Failed to connect to Caterva2 server at {CATERVA2_URLBASE}: {e}"}


def list_datasets(path: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """
    List datasets within a root or sub-path on the Caterva2 server.

    get_list() always returns the full list from the server (no server-side pagination).
    We slice the result here (client-side pagination) to avoid flooding the LLM context window.
    The recommended page size is 50, but there is no enforced cap — the caller can request
    any limit, including a large value to retrieve all datasets at once.

    Args:
        path:   Root name or sub-path (e.g. '@public' or '@public/examples')
        limit:  Requested page size — recommended 50, no enforced maximum
        offset: Number of results to skip, for pagination (default 0)

    Returns:
        Dict with 'datasets' (one page), 'total', 'offset', and 'has_more',
        or 'error' on failure.
    """
    print(f"→ Listing datasets under path: '{path}' (offset={offset}, limit={limit})")
    print(f"   API: client.get_list('{path}')")
    try:
        client = _get_client()
        datasets = client.get_list(path)
        # Prefix results with the path so full paths are immediately usable by other tools
        full_paths = [f"{path}/{name}" for name in datasets]
        total = len(full_paths)
        page = full_paths[offset: offset + limit]
        return {
            "path": path,
            "datasets": page,
            "total": total,
            "offset": offset,
            "has_more": (offset + limit) < total,
        }
    except Exception as e:
        print("   ✗ Failed")
        return {"error": f"Failed to list datasets at path '{path}': {e}"}


def get_dataset_info(path: str) -> Dict[str, Any]:
    """
    Retrieve metadata for a specific dataset.

    Args:
        path: Full path including root (e.g. 'example/ds-2d-fields.b2nd')

    Returns:
        Dict with dataset properties (shape, dtype, chunks, etc.), or 'error' on failure.
    """
    print(f"→ Fetching metadata for dataset: '{path}'")
    print(f"   API: client.get_info('{path}')")
    try:
        client = _get_client()
        info = client.get_info(path)
        # info is already a dict — serialize any non-JSON-safe values to strings
        safe_info = {k: str(v) if not isinstance(v, (str, int, float, list, dict, bool, type(None))) else v
                     for k, v in info.items()}
        return {"path": path, "info": safe_info}
    except Exception as e:
        print("   ✗ Failed")
        return {"error": f"Failed to get info for dataset '{path}': {e}"}


# Default statistics when none specified — the most commonly useful ones
DEFAULT_STATS = ["min", "max", "mean", "std"]

# All supported statistical operations, mapped to Dataset method names
SUPPORTED_STATS = {"min", "max", "mean", "sum", "std", "var", "argmin", "argmax", "any", "all"}


def _to_json_safe(value) -> Any:
    """
    Convert numpy/blosc2 values to JSON-serializable Python types.
    
    Statistical methods return numpy scalars or arrays — these must be
    converted to native Python types for JSON serialization.
    """
    import numpy as np
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


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


# ---------------------------------------------------------------------------
# SLICE TOOL
# ---------------------------------------------------------------------------

# Hard limit to protect LLM context from huge data dumps
MAX_SLICE_ELEMENTS = 10_000


def _parse_slice_string(slice_str: str, shape: tuple) -> tuple:
    """
    Parse a Python-style slice string into a tuple of slice objects.
    
    Examples:
        "0:10"         → (slice(0, 10),)
        "0:5, 0:3"     → (slice(0, 5), slice(0, 3))
        ":, 0"         → (slice(None), 0)
        "0, :, 0:10"   → (0, slice(None), slice(0, 10))
    
    Args:
        slice_str: User-provided slice specification
        shape: Dataset shape (for validation and defaults)
    
    Returns:
        Tuple of slice objects and integers suitable for __getitem__
    
    Raises:
        ValueError: If slice syntax is invalid or doesn't match dimensions
    """
    parts = [p.strip() for p in slice_str.split(",")]
    
    if len(parts) > len(shape):
        raise ValueError(
            f"Too many dimensions in slice: got {len(parts)}, dataset has {len(shape)}"
        )
    
    result = []
    for part in parts:
        if part == "" or part == ":":
            # Full slice for this dimension
            result.append(slice(None))
        elif ":" in part:
            # Parse slice notation: start:stop or start:stop:step
            components = part.split(":")
            if len(components) == 2:
                start = int(components[0]) if components[0] else None
                stop = int(components[1]) if components[1] else None
                result.append(slice(start, stop))
            elif len(components) == 3:
                start = int(components[0]) if components[0] else None
                stop = int(components[1]) if components[1] else None
                step = int(components[2]) if components[2] else None
                result.append(slice(start, stop, step))
            else:
                raise ValueError(f"Invalid slice syntax: '{part}'")
        else:
            # Single index
            result.append(int(part))
    
    return tuple(result)


def _compute_slice_size(slices: tuple, shape: tuple) -> int:
    """
    Estimate the number of elements that will be returned by a slice.
    
    Uses shape to resolve None values in slices.
    """
    size = 1
    for i, s in enumerate(slices):
        if i >= len(shape):
            break
        dim_size = shape[i]
        
        if isinstance(s, int):
            # Single index — dimension is eliminated, contributes 1
            continue
        elif isinstance(s, slice):
            # Compute slice length
            start, stop, step = s.indices(dim_size)
            length = max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)
            size *= length
    
    # Remaining dimensions not covered by slices contribute their full size
    for i in range(len(slices), len(shape)):
        # Check if dimension was eliminated by an integer index
        if i < len(slices) and isinstance(slices[i], int):
            continue
        size *= shape[i]
    
    return size


def _default_slice_for_shape(shape: tuple, max_elements: int) -> tuple:
    """
    Generate a default slice that returns at most max_elements from the start.
    
    For 1D: slice(0, max_elements)
    For nD: Takes first elements from each dimension proportionally
    """
    if len(shape) == 1:
        return (slice(0, min(shape[0], max_elements)),)
    
    # For multi-dimensional, take a hypercube from the start
    # Calculate how many elements per dimension (geometric mean approach)
    import math
    ndim = len(shape)
    elements_per_dim = int(math.pow(max_elements, 1.0 / ndim))
    
    slices = []
    for dim_size in shape:
        slices.append(slice(0, min(dim_size, max(1, elements_per_dim))))
    
    return tuple(slices)


def _generate_preview(data, max_chars: int = 200) -> str:
    """
    Generate a truncated string preview of array data.
    
    Shows the structure without overwhelming the output.
    """
    full_str = str(data.tolist() if hasattr(data, 'tolist') else data)
    if len(full_str) <= max_chars:
        return full_str
    return full_str[:max_chars] + "..."


def _compute_summary(data) -> Dict[str, Any]:
    """
    Compute summary statistics for slice data.
    
    Pre-computes stats so the LLM can present a summary without
    showing all raw values. Also prepares for future viz tools.
    """
    import numpy as np
    
    arr = np.asarray(data)
    num_elements = arr.size
    
    summary = {
        "num_elements": num_elements,
        "preview": _generate_preview(arr),
    }
    
    # Only compute numeric stats for numeric dtypes
    if np.issubdtype(arr.dtype, np.number):
        summary["min"] = _to_json_safe(arr.min())
        summary["max"] = _to_json_safe(arr.max())
        summary["mean"] = _to_json_safe(arr.mean())
    
    return summary


# Threshold for auto-including full data vs summary-only
# Below this, include full data; above, LLM should use summary
SUMMARY_THRESHOLD = 100


def get_slice(path: str, slices: str | None = None) -> Dict[str, Any]:
    """
    Retrieve a slice of data from a dataset.
    
    Parses Python-style slice syntax and enforces a maximum element limit
    to protect the LLM context window from huge data dumps.
    
    Returns both raw data and a pre-computed summary. For large results
    (>100 elements), the LLM should present the summary and offer to
    show full data on request.
    
    Args:
        path: Full path to dataset (e.g. '@public/examples/ds-1d.b2nd')
        slices: Python slice syntax (e.g. '0:100', '0:5, 0:3')
    
    Returns:
        Dict with slice metadata, summary, and data values, or 'error' on failure.
    """
    print(f"→ Getting slice from dataset: '{path}'")
    print(f"   Requested slice: {slices or '(default)'}")
    
    try:
        dataset = _get_dataset(path)
        shape = dataset.shape
        
        # Parse or generate slice specification
        if slices is None:
            slice_tuple = _default_slice_for_shape(shape, MAX_SLICE_ELEMENTS)
            slice_str_used = str(slice_tuple)
        else:
            slice_tuple = _parse_slice_string(slices, shape)
            slice_str_used = slices
        
        # Estimate size and enforce limit
        estimated_size = _compute_slice_size(slice_tuple, shape)
        if estimated_size > MAX_SLICE_ELEMENTS:
            return {
                "error": f"Requested slice would return ~{estimated_size:,} elements, "
                         f"exceeding limit of {MAX_SLICE_ELEMENTS:,}. "
                         f"Please request a smaller slice.",
                "shape": list(shape),
                "requested_slice": slice_str_used
            }
        
        print(f"   Slice tuple: {slice_tuple}")
        print(f"   Estimated elements: {estimated_size}")
        
        # Fetch the data — __getitem__ returns numpy array
        data = dataset[slice_tuple]
        data_json = _to_json_safe(data)
        
        # Pre-compute summary for LLM presentation
        summary = _compute_summary(data)
        
        result = {
            "path": path,
            "dataset_shape": list(shape),
            "dtype": str(dataset.dtype),
            "slice": slice_str_used,
            "result_shape": list(data.shape) if hasattr(data, 'shape') else [],
            "summary": summary,
            "data": data_json
        }
        
        # Hint for the LLM on how to present results
        if summary["num_elements"] > SUMMARY_THRESHOLD:
            result["_hint"] = (
                f"Large result ({summary['num_elements']} elements). "
                "Present the summary to the user and offer to show full data if requested."
            )
        
        return result
    
    except ValueError as e:
        # Slice parsing errors
        print(f"   ✗ Invalid slice: {e}")
        return {"error": f"Invalid slice specification: {e}"}
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return {"error": f"Failed to get slice from '{path}': {e}"}


# ---------------------------------------------------------------------------
# TOOL DISPATCHER
# Maps tool names (as the LLM sends them) to their Python implementations.
# execute_tool() is the single entry point called by the agent loop.
# ---------------------------------------------------------------------------







TOOL_MAP = {
    "list_roots": list_roots,
    "list_datasets": list_datasets,
    "get_dataset_info": get_dataset_info,
    "get_dataset_stats": get_dataset_stats,
    "get_slice": get_slice,
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Execute a named tool with the given arguments.

    This is called by the agent loop whenever the LLM requests a tool call.
    Always returns a JSON string — the agent loop appends this to message history.

    Args:
        tool_name: Name of the tool (must match a key in TOOL_MAP)
        tool_args: Arguments parsed from the LLM's tool call

    Returns:
        JSON string with the tool result or an error message
    """
    tool_function = TOOL_MAP.get(tool_name)
    if tool_function is None:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})
    try:
        # Defensive: ensure tool_args is always a dict (covers None, "null", etc.)
        if not isinstance(tool_args, dict):
            tool_args = {}
        result = tool_function(**tool_args)
    except TypeError as e:
        # Catches wrong argument names or missing required args
        result = {"error": f"Invalid arguments for '{tool_name}': {e}"}
    return json.dumps(result)
