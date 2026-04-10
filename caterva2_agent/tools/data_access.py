"""
Data access tools for retrieving values from datasets.

Tools in this module:
- get_slice: Retrieve a portion of dataset values with safety limits
- where_filter: Conditional selection using where() — filter data based on conditions
- load_dataset: Load entire dataset into notebook (with safety checks)

Works with both:
- Server datasets (path starts with '@')
- Local variables (referenced by name)
"""

import logging
from typing import Dict, Any

logger = logging.getLogger('caterva2_agent')

from ._base import resolve_data, _to_json_safe, register_fetched_object


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Default size for auto-slice when get_slice is called without explicit slices.
# This is not a hard operation cap; explicit user slices may be much larger.
DEFAULT_AUTO_SLICE_ELEMENTS = 10_000

# Inline payload limit for LLM tool responses.
# Above this size, tools return summary metadata only (no full array values).
LLM_INLINE_DATA_MAX_ELEMENTS = 100

# Server-side operation guardrails (warnings and extreme safety stop).
SERVER_OP_WARNING_ELEMENTS = 100_000_000
SERVER_OP_HARD_LIMIT_ELEMENTS = 2_000_000_000

# Explicit materialization guardrails for load_dataset.
LOAD_DATASET_MAX_ELEMENTS = 100_000_000
LOAD_DATASET_MAX_BYTES = 100 * 1024 * 1024  # 100 MB (user-confirmed)

# Keep this name for compatibility with existing summary behavior/tests.
SUMMARY_THRESHOLD = LLM_INLINE_DATA_MAX_ELEMENTS


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

DATA_ACCESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_slice",
                "description": (
                    "Retrieve a slice of data values from a dataset or local variable. "
                    "Use this when the user wants to see actual data, not just statistics. "
                    "For large slices, returns metadata + summary by default instead of full values. "
                    "To materialize data into the notebook namespace, use load_dataset explicitly. "
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
    },
    {
        "type": "function",
        "function": {
            "name": "where_filter",
                "description": (
                    "Filter dataset or local variable values based on a condition (like SQL WHERE). "
                    "Returns value_if_true where condition is met, value_if_false otherwise. "
                    "Example use case: for elevation data, filter peaks above 3000m by returning "
                    "the actual elevation where > 3000, and 0 (or NaN) elsewhere. "
                    "This is useful for masking, thresholding, or highlighting specific data regions. "
                    "For large results, returns metadata + summary by default instead of full values. "
                    "To materialize full data into the notebook namespace, use load_dataset explicitly. "
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
                    "operator": {
                        "type": "string",
                        "enum": [">", ">=", "<", "<=", "==", "!="],
                        "description": (
                            "Comparison operator for the condition. "
                            "'>': greater than, '>=': greater or equal, "
                            "'<': less than, '<=': less or equal, "
                            "'==': equal to, '!=': not equal to."
                        )
                    },
                    "threshold": {
                        "type": "number",
                        "description": (
                            "The threshold value to compare against. "
                            "Example: 3000 for 'elevation > 3000'."
                        )
                    },
                    "value_if_true": {
                        "type": "number",
                        "description": (
                            "Value to return where condition is True. "
                            "Use special value 'data' (as string) to return the original data value. "
                            "Default: returns the original data value."
                        )
                    },
                    "value_if_false": {
                        "type": "number",
                        "description": (
                            "Value to return where condition is False. "
                            "Common choices: 0, NaN (use null), or a sentinel value. "
                            "Default: 0."
                        )
                    },
                    "slices": {
                        "type": "string",
                        "description": (
                            "Optional slice specification to limit the filter to a region. "
                            "Same syntax as get_slice. Highly recommended for large datasets."
                        )
                    }
                },
                "required": ["path", "operator", "threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
                "description": (
                    "Load an entire dataset into the notebook for manipulation. "
                    "Use this when you want to work with the complete dataset in memory. "
                    "SAFETY: This is explicit materialization and enforces strict size checks "
                    "(including a 100MB cap) before loading. "
                    "For large datasets, the tool suggests slice/filter/projection workflows. "
                    "The loaded data becomes available as a numpy array variable in the notebook. "
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
    
    # For multidimensional, take a hypercube from the start
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


def _estimate_nbytes(num_elements: int, dtype: Any) -> int | None:
    """Estimate payload size in bytes from element count and dtype."""
    import numpy as np

    try:
        itemsize = np.dtype(dtype).itemsize
    except (TypeError, ValueError):
        return None
    return int(num_elements) * int(itemsize)


def _operation_warning(estimated_elements: int, estimated_nbytes: int | None) -> str | None:
    """Return warning text for very large server-side operations."""
    if estimated_elements < SERVER_OP_WARNING_ELEMENTS:
        return None

    if estimated_nbytes is not None:
        estimated_mb = estimated_nbytes / (1024 * 1024)
        return (
            f"Large server-side operation requested (~{estimated_elements:,} elements, "
            f"~{estimated_mb:.1f} MB uncompressed). This may be slow or stress server resources. "
            "If needed, prefer slicing and/or server-side projection first."
        )

    return (
        f"Large server-side operation requested (~{estimated_elements:,} elements). "
        "This may be slow or stress server resources. If needed, prefer slicing "
        "and/or server-side projection first."
    )


def _materialize_to_numpy(value: Any) -> Any:
    """
    Materialize Caterva2/LazyExpr-like values to a NumPy array when possible.

    This helper lets where_filter try server-side expression execution first,
    then materialize only for summaries / optional inline data.
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "compute"):
        return np.asarray(value.compute())

    if hasattr(value, "__array__"):
        return np.asarray(value)

    if hasattr(value, "slice"):
        try:
            return np.asarray(value.slice(slice(None), as_blosc2=False))
        except Exception:
            pass

    try:
        return np.asarray(value[:])
    except Exception:
        return np.asarray(value)


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def get_slice(path: str, slices: str | None = None) -> Dict[str, Any]:
    """
    Retrieve a slice of data from a dataset or local variable.
    
    Parses Python-style slice syntax and keeps LLM context safe by returning
    summary-first payloads for large results.
    
    Returns both raw data and a pre-computed summary. For large results
    (>100 elements), the LLM should present the summary and offer to
    show full data on request.
    
    Args:
        path: Server dataset path (e.g. '@public/examples/ds-1d.b2nd')
              OR local variable name (e.g. 'my_data')
        slices: Python slice syntax (e.g. '0:100', '0:5, 0:3')
    
    Returns:
        Dict with slice metadata, summary, and data values, or 'error' on failure.
    """
    is_local = not path.startswith("@")
    source_type = "local variable" if is_local else "server dataset"
    
    logger.info(f"Getting slice from {source_type}: '{path}'")
    logger.debug(f"Requested slice: {slices or '(default)'}")
    
    try:
        resolved = resolve_data(path)
        shape = resolved.shape
        
        # Parse or generate slice specification
        if slices is None:
            slice_tuple = _default_slice_for_shape(shape, DEFAULT_AUTO_SLICE_ELEMENTS)
            slice_str_used = f"{slice_tuple} (auto-preview)"
        else:
            slice_tuple = _parse_slice_string(slices, shape)
            slice_str_used = slices
        
        # Estimate requested result size for safety metadata
        estimated_size = _compute_slice_size(slice_tuple, shape)
        estimated_nbytes = _estimate_nbytes(estimated_size, resolved.dtype)

        if estimated_size > SERVER_OP_HARD_LIMIT_ELEMENTS:
            return {
                "error": (
                    f"Requested slice is too large (~{estimated_size:,} elements). "
                    "This exceeds the tool's extreme safety limit for a single operation. "
                    "Use slicing and/or dimension reduction first."
                ),
                "shape": list(shape),
                "requested_slice": slice_str_used
            }

        warning = _operation_warning(estimated_size, estimated_nbytes)
        
        logger.debug(f"Slice tuple: {slice_tuple}")
        logger.debug(f"Estimated elements: {estimated_size}")
        
        # Fetch the data
        data = resolved[slice_tuple]
        
        # Pre-compute summary for LLM presentation
        summary = _compute_summary(data)
        
        result = {
            "path": path,
            "source": source_type,
            "dataset_shape": list(shape),
            "dtype": str(resolved.dtype),
            "slice": slice_str_used,
            "result_shape": list(data.shape) if hasattr(data, 'shape') else [],
            "estimated_elements": int(estimated_size),
            "summary": summary,
            "materialized_in_notebook": False
        }

        if estimated_nbytes is not None:
            result["estimated_size_mb"] = round(estimated_nbytes / (1024 * 1024), 2)

        if warning:
            result["warning"] = warning

        if summary["num_elements"] <= LLM_INLINE_DATA_MAX_ELEMENTS:
            result["data"] = _to_json_safe(data)
        else:
            result["_hint"] = (
                f"Large result ({summary['num_elements']} elements). "
                "Summary-only payload returned to protect LLM context size. "
                "Use load_dataset only if the user explicitly asks to materialize data."
            )
            result["note"] = (
                "Full data omitted from tool output due to context-size policy. "
                "Operation was executed, but only metadata and summary are returned."
            )
        
        return result
    
    except ValueError as e:
        # Slice parsing errors or variable not found
        logger.warning(f"Slice validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to get slice from '{path}': {e}")
        return {"error": f"Failed to get slice from '{path}': {e}"}


# ---------------------------------------------------------------------------
# SUPPORTED OPERATORS FOR where_filter
# Maps string operators to Python comparison functions
# ---------------------------------------------------------------------------

COMPARISON_OPERATORS = {
    ">": lambda data, threshold: data > threshold,
    ">=": lambda data, threshold: data >= threshold,
    "<": lambda data, threshold: data < threshold,
    "<=": lambda data, threshold: data <= threshold,
    "==": lambda data, threshold: data == threshold,
    "!=": lambda data, threshold: data != threshold,
}


def where_filter(
    path: str,
    operator: str,
    threshold: float,
    value_if_true: float | None = None,
    value_if_false: float | None = None,
    slices: str | None = None
) -> Dict[str, Any]:
    """
    Filter dataset or local variable values based on a condition.
    
    Applies a comparison (e.g., > 3000) to the data and returns:
    - value_if_true where the condition is met
    - value_if_false where the condition is not met
    
    Execution strategy:
    - Local variables: NumPy local path
    - Server datasets: attempt Caterva2 server-side expression + where path first;
      fallback to local NumPy if that capability is unavailable.
    
    Use cases:
    - Thresholding: elevation > 3000 → show peaks, mask valleys
    - Masking: set out-of-range values to NaN or 0
    - Binary classification: value > threshold → 1, else 0
    
    Args:
        path: Server dataset path (e.g. '@public/examples/ds-1d.b2nd')
              OR local variable name (e.g. 'my_data')
        operator: Comparison operator (>, >=, <, <=, ==, !=)
        threshold: Value to compare against
        value_if_true: Value where condition is True (default: original data)
        value_if_false: Value where condition is False (default: 0)
        slices: Optional slice to limit the region (recommended for large datasets)
    
    Returns:
        Dict with filtered data, condition info, and summary, or 'error' on failure.
    """
    # Validate operator
    if operator not in COMPARISON_OPERATORS:
        return {
            "error": f"Invalid operator: '{operator}'. "
                     f"Valid options: {list(COMPARISON_OPERATORS.keys())}"
        }
    
    is_local = not path.startswith("@")
    source_type = "local variable" if is_local else "server dataset"
    
    logger.info(f"Filtering {source_type}: '{path}'")
    logger.debug(f"Condition: data {operator} {threshold}")
    logger.debug(f"Values: if_true={value_if_true or 'data'}, if_false={value_if_false or 0}")
    if slices:
        logger.debug(f"Slice: {slices}")
    
    try:
        resolved = resolve_data(path)
        shape = resolved.shape
        
        # Determine slice to apply.
        # If no slice is provided, run on full dataset by default.
        if slices is None:
            slice_tuple = tuple(slice(None) for _ in shape)
            slice_str_used = "(full dataset)"
        else:
            slice_tuple = _parse_slice_string(slices, shape)
            slice_str_used = slices
        
        # Estimate requested region for safety metadata
        estimated_size = _compute_slice_size(slice_tuple, shape)
        estimated_nbytes = _estimate_nbytes(estimated_size, resolved.dtype)

        if estimated_size > SERVER_OP_HARD_LIMIT_ELEMENTS:
            return {
                "error": (
                    f"Requested filter region is too large (~{estimated_size:,} elements). "
                    "This exceeds the tool's extreme safety limit for a single operation. "
                    "Use slicing and/or server-side reduction first."
                ),
                "shape": list(shape),
                "requested_slice": slice_str_used
            }

        warning = _operation_warning(estimated_size, estimated_nbytes)
        
        logger.debug(f"Estimated elements: {estimated_size}")
        
        compare_fn = COMPARISON_OPERATORS[operator]
        import numpy as np
        v_false = 0 if value_if_false is None else value_if_false
        execution_mode = "local_numpy"

        # Server datasets: try server-side where first.
        if resolved.is_server():
            try:
                server_operand = resolved.data
                if slices is not None:
                    if not hasattr(server_operand, "slice"):
                        raise AttributeError("Dataset.slice() is unavailable")
                    server_operand = server_operand.slice(slice_tuple, as_blosc2=True)

                condition_obj = compare_fn(server_operand, threshold)
                if not hasattr(condition_obj, "where"):
                    raise AttributeError("Condition object has no where() method")

                v_true_server = server_operand if value_if_true is None else value_if_true
                result_obj = condition_obj.where(v_true_server, v_false)

                condition = np.asarray(_materialize_to_numpy(condition_obj), dtype=bool)
                result_data = np.asarray(_materialize_to_numpy(result_obj))
                execution_mode = "server_where"
            except Exception as e:
                logger.warning(
                    f"Server-side where unavailable for '{path}' ({type(e).__name__}: {e}); "
                    "falling back to local NumPy."
                )
                data_slice = resolved[slice_tuple]
                condition = compare_fn(data_slice, threshold)
                v_true_local = data_slice if value_if_true is None else value_if_true
                result_data = np.where(condition, v_true_local, v_false)
        else:
            data_slice = resolved[slice_tuple]
            condition = compare_fn(data_slice, threshold)
            v_true_local = data_slice if value_if_true is None else value_if_true
            result_data = np.where(condition, v_true_local, v_false)
        
        # Compute summary statistics
        summary = _compute_summary(result_data)
        
        # Count how many elements matched the condition
        num_matched = int(np.sum(condition))
        num_total = int(condition.size)
        match_percentage = (num_matched / num_total * 100) if num_total > 0 else 0
        
        result = {
            "path": path,
            "source": source_type,
            "dataset_shape": list(shape),
            "dtype": str(resolved.dtype),
            "condition": f"data {operator} {threshold}",
            "slice_applied": slice_str_used,
            "result_shape": list(result_data.shape),
            "estimated_elements": int(estimated_size),
            "value_if_true": "original_data" if value_if_true is None else value_if_true,
            "value_if_false": v_false,
            "execution_mode": execution_mode,
            "match_summary": {
                "matched": num_matched,
                "total": num_total,
                "percentage": round(match_percentage, 2)
            },
            "summary": summary,
            "materialized_in_notebook": False
        }

        if estimated_nbytes is not None:
            result["estimated_size_mb"] = round(estimated_nbytes / (1024 * 1024), 2)

        if warning:
            result["warning"] = warning

        if summary["num_elements"] <= LLM_INLINE_DATA_MAX_ELEMENTS:
            result["data"] = _to_json_safe(result_data)
        
        # Hint for LLM on how to present results
        if summary["num_elements"] > LLM_INLINE_DATA_MAX_ELEMENTS:
            result["_hint"] = (
                f"Large result ({summary['num_elements']} elements). "
                f"{num_matched} values ({match_percentage:.1f}%) matched the condition. "
                "Summary-only payload returned to protect LLM context size. "
                "Use load_dataset only if the user explicitly asks to materialize data."
            )
            result["note"] = (
                "Full filtered data omitted from tool output due to context-size policy. "
                "Operation was executed, but only metadata and summary are returned."
            )
        
        return result
    
    except ValueError as e:
        # Slice parsing errors or variable not found
        logger.warning(f"Slice validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to filter '{path}': {e}")
        return {"error": f"Failed to filter '{path}': {e}"}


# ---------------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> Dict[str, Any]:
    """
    Load an entire dataset into the notebook for manipulation.
    
    This is the explicit "I want all of this data" tool. It loads the complete
    dataset (after decompression if from Caterva2) into memory as a numpy array.
    
    Safety: This is explicit materialization. Enforces strict size checks including
    an explicit 100MB cap for notebook loading.
    
    Args:
        path: Server dataset path (e.g. '@public/examples/ds-1d.b2nd')
              OR local variable name (e.g. 'my_data')
    
    Returns:
        Dict with dataset metadata, summary, and data values, or 'error' on failure.
    """
    is_local = not path.startswith("@")
    source_type = "local variable" if is_local else "server dataset"
    
    logger.info(f"Loading full {source_type}: '{path}'")
    
    try:
        import numpy as np

        resolved = resolve_data(path)
        total_elements = int(np.prod(resolved.shape))
        estimated_nbytes = _estimate_nbytes(total_elements, resolved.dtype)

        if total_elements > LOAD_DATASET_MAX_ELEMENTS:
            return {
                "error": (
                    f"Dataset has {total_elements:,} elements, exceeding explicit load limit "
                    f"of {LOAD_DATASET_MAX_ELEMENTS:,} elements."
                ),
                "shape": list(resolved.shape),
                "dtype": str(resolved.dtype),
                "suggestion": (
                    "Use get_slice/where_filter/collapse_dimensions to work server-side, "
                    "or request a smaller slice before loading."
                )
            }

        if estimated_nbytes is not None and estimated_nbytes > LOAD_DATASET_MAX_BYTES:
            est_mb = estimated_nbytes / (1024 * 1024)
            return {
                "error": (
                    f"Estimated materialized size is ~{est_mb:.2f} MB, exceeding "
                    f"the 100 MB load limit."
                ),
                "shape": list(resolved.shape),
                "dtype": str(resolved.dtype),
                "estimated_size_mb": round(est_mb, 2),
                "suggestion": (
                    "Use get_slice/where_filter/collapse_dimensions for server-side operations, "
                    "then load a smaller result if needed."
                )
            }

        # Materialize explicitly
        data = resolved[:] if not resolved.is_local() else resolved.data
        data = np.asarray(data)

        # Final guard using actual in-memory size
        if data.nbytes > LOAD_DATASET_MAX_BYTES:
            actual_mb = data.nbytes / (1024 * 1024)
            return {
                "error": (
                    f"Materialized array size is {actual_mb:.2f} MB, exceeding "
                    f"the 100 MB load limit."
                ),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "size_mb": round(actual_mb, 2),
                "suggestion": (
                    "Use get_slice/where_filter/collapse_dimensions for server-side operations, "
                    "then load a smaller result if needed."
                )
            }

        # Register only for server datasets; local variables are already available.
        registered = False
        if not is_local:
            register_fetched_object(path, data)
            registered = True

        size_bytes = int(data.nbytes)
        size_mb = round(size_bytes / (1024 * 1024), 2)

        logger.debug(f"✓ Loaded: shape={list(data.shape)}, dtype={data.dtype}")
        logger.debug(f"Size: {size_mb} MB ({data.size:,} elements)")
        if registered:
            logger.debug("📦 Available in notebook for manipulation")
        
        # Compute summary statistics
        summary = _compute_summary(data)
        
        result = {
            "status": "success",
            "path": path,
            "source": "local" if is_local else "server",
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "num_elements": data.size,
            "summary": summary,
            "registered_in_notebook": registered
        }
        
        # Include full data only for very small payloads
        if data.size <= LLM_INLINE_DATA_MAX_ELEMENTS:
            result["data"] = _to_json_safe(data)
        else:
            result["note"] = (
                f"Data has {data.size:,} elements — returning summary only to protect LLM context. "
                f"The dataset is materialized in notebook memory via load_dataset."
            )
        
        return result
    
    except ValueError as e:
        # Size limit exceeded or variable not found
        logger.warning(f"Load validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to load '{path}': {e}")
        return {"error": f"Failed to load '{path}': {e}"}
