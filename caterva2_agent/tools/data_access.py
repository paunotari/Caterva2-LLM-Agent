"""
Data access tools for retrieving values from Caterva2 datasets.

Tools in this module:
- get_slice: Retrieve a portion of dataset values with safety limits

Future tools planned:
- where_filter: Conditional selection using where()
"""

from typing import Dict, Any

from ._base import _get_dataset, _to_json_safe


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Hard limit to protect LLM context from huge data dumps
MAX_SLICE_ELEMENTS = 10_000

# Threshold for auto-including full data vs summary-only
# Below this, include full data; above, LLM should use summary
SUMMARY_THRESHOLD = 100


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

DATA_ACCESS_TOOLS = [
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
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

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
