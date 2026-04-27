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
from datetime import datetime, timezone
from typing import Dict, Any

import blosc2

logger = logging.getLogger('caterva2_agent')

from ._base import (
    resolve_data,
    _to_json_safe,
    register_fetched_object,
    _get_client,
    get_client_auth_status,
    get_notebook_namespace,
    ResolvedData,
)


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
                    "For local variables, automatically registers the slice result in the notebook "
                    "namespace for use in follow-up operations. "
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
                    },
                    "persist_result": {
                        "type": "boolean",
                        "description": (
                            "Optional (server datasets only). Defaults to true. "
                            "When enabled and authenticated, saves the sliced result as a new "
                            "@personal dataset for follow-up server operations."
                        )
                    },
                    "save_path": {
                        "type": "string",
                        "description": (
                            "Optional destination path for persistence when persist_result=true. "
                            "Must start with '@personal/'. If omitted, auto-generates a path under "
                            "'@personal/slices/'."
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
                    "For authenticated server sessions, filtered results are auto-saved to "
                    "a generated '@personal/where_filter/...' dataset path for chaining. "
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
                    },
                    "compute": {
                        "type": "boolean",
                        "description": (
                            "Only used for server datasets when authenticated. "
                            "True (default): compute and materialize result on server during auto-save. "
                            "False: store lazy expression wrapper."
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
                    "The loaded data becomes available as a blosc2-backed variable in the notebook. "
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
    preview_obj = data

    # For large arrays, preview only a tiny corner instead of full materialization.
    if hasattr(data, "shape") and hasattr(data, "__getitem__"):
        shape = tuple(getattr(data, "shape", ()))
        if shape and _shape_product(shape) > 64:
            ndim = len(shape)
            per_dim = max(1, int(round(64 ** (1.0 / ndim))))
            preview_key = tuple(slice(0, min(int(dim), per_dim)) for dim in shape)
            try:
                preview_obj = data[preview_key]
            except Exception:
                preview_obj = data

    full_str = str(_to_json_safe(preview_obj))
    if len(full_str) <= max_chars:
        return full_str
    return full_str[:max_chars] + "..."


def _shape_product(shape: tuple[int, ...] | list[int]) -> int:
    """Compute number of elements from a shape tuple/list."""
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _compute_summary(data) -> Dict[str, Any]:
    """
    Compute summary statistics for slice data.
    
    Pre-computes stats so the LLM can present a summary without
    showing all raw values. Also prepares for future viz tools.
    """
    num_elements_attr = getattr(data, "size", None)
    if num_elements_attr is not None:
        num_elements = int(num_elements_attr)
    else:
        shape = tuple(getattr(data, "shape", ()))
        num_elements = _shape_product(shape) if shape else 1
    
    summary = {
        "num_elements": num_elements,
        "preview": _generate_preview(data),
    }
    
    # Compute numeric-like stats when backend supports these reductions.
    if all(hasattr(data, stat) for stat in ("min", "max", "mean")):
        try:
            summary["min"] = _to_json_safe(data.min())
            summary["max"] = _to_json_safe(data.max())
            summary["mean"] = _to_json_safe(data.mean())
        except (TypeError, ValueError):
            pass
    
    return summary


def _estimate_nbytes(num_elements: int, dtype: Any) -> int | None:
    """Estimate payload size in bytes from element count and dtype."""
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        try:
            probe = blosc2.zeros((1,), dtype=dtype)
            itemsize = probe.dtype.itemsize
        except Exception:
            return None
    try:
        itemsize_int = int(itemsize)
    except (TypeError, ValueError):
        return None
    return int(num_elements) * itemsize_int


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


def _materialize_backend_value(value: Any) -> Any:
    """
    Materialize Caterva2/LazyExpr-like values while keeping Blosc2-backed data.

    This helper avoids eager NumPy conversion and keeps internal computation
    in backends that support compressed operations.
    """
    if hasattr(value, "compute"):
        return value.compute()

    if hasattr(value, "slice"):
        try:
            return value.slice(slice(None), as_blosc2=True)
        except Exception:
            pass

    try:
        return value[:]
    except Exception:
        return value


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def get_slice(
    path: str,
    slices: str | None = None,
    persist_result: bool = True,
    save_path: str | None = None,
) -> Dict[str, Any]:
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
        persist_result: For authenticated server paths, store slice result in
            @personal for follow-up operations (default: True).
        save_path: Optional explicit @personal destination for persistence.
    
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
        
        # For blosc2-first architecture: normalize to blosc2 if needed
        # (Server datasets return numpy via resolved[...], local blosc2 stay blosc2)
        if not isinstance(data, blosc2.NDArray) and hasattr(data, "shape"):
            try:
                logger.debug(f"Normalizing fetched data to blosc2, original type: {type(data)}")
                data = blosc2.asarray(data)
            except Exception as e:
                # Keep original if conversion fails
                logger.debug(f"Could not normalize to blosc2: {e}, keeping as {type(data)}")
                pass
        
        persisted_result_path: str | None = None
        persistence_note: str | None = None

        if persist_result and resolved.is_server():
            auth_status = get_client_auth_status()
            if not auth_status.get("authenticated"):
                persistence_note = (
                    "Persistence skipped: session is not authenticated. "
                    "Use notebook login(...) to enable @personal persistence."
                )
            else:
                target_path = save_path.strip() if isinstance(save_path, str) else ""
                if not target_path:
                    source_name = path.rsplit("/", 1)[-1] if "/" in path else path
                    stem = source_name.removesuffix(".b2nd").removesuffix(".b2frame")
                    sanitized = "".join(
                        ch if ch.isalnum() or ch in ("-", "_") else "_"
                        for ch in stem
                    ) or "slice"
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    target_path = f"@personal/slices/{sanitized}_{timestamp}.b2nd"

                if not target_path.startswith("@personal/"):
                    return {
                        "error": "save_path must start with '@personal/' for get_slice persistence.",
                        "save_path": target_path,
                    }

                try:
                    payload_to_upload = data
                    if (
                        not isinstance(payload_to_upload, blosc2.NDArray)
                        and hasattr(payload_to_upload, "shape")
                    ):
                        try:
                            payload_to_upload = blosc2.asarray(payload_to_upload)
                        except Exception:
                            pass

                    client = _get_client()
                    if client is None:
                        raise RuntimeError("Caterva2 client is not available.")
                    persisted = client.upload(payload_to_upload, target_path)
                    persisted_result_path = (
                        getattr(persisted, "path", None)
                        or getattr(persisted, "name", None)
                        or target_path
                    )
                except Exception as e:
                    persistence_note = f"Failed to persist sliced result to '@personal': {e}"
        
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

        if resolved.is_server():
            result["stored_server_side"] = persisted_result_path is not None
            if persisted_result_path is not None:
                result["result_path"] = str(persisted_result_path)
                result["save_path"] = str(persisted_result_path)
                result["server_change_applied"] = True
                result["_next_step_hint"] = (
                    "Use result_path in follow-up server operations "
                    "(get_slice, where_filter, collapse_dimensions, visualize_dataset)."
                )
            else:
                result["server_change_applied"] = False
                if not persist_result:
                    result["persistence_note"] = (
                        "Persistence disabled for this call (persist_result=False)."
                    )
                elif persistence_note:
                    result["persistence_note"] = persistence_note

        # Auto-inject local results into notebook namespace for chaining
        injected_var_name: str | None = None
        if is_local:
            try:
                namespace = get_notebook_namespace()
                if namespace is not None:
                    # Generate a unique variable name for the sliced result
                    base_name = path.replace("_", "").replace("-", "").replace(".", "_")
                    base_name = f"sliced_{base_name}"
                    counter = 1
                    var_name = base_name
                    while var_name in namespace:
                        var_name = f"{base_name}_{counter}"
                        counter += 1
                    
                    # Inject into notebook namespace
                    namespace[var_name] = data
                    injected_var_name = var_name
                    register_fetched_object(var_name, data)
                    logger.info(f"Auto-injected local get_slice result as '{var_name}'")
            except Exception as e:
                logger.debug(f"Could not auto-inject local get_slice result: {e}")

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
        
        # Add injected variable name if applicable
        if injected_var_name is not None:
            result["injected_as_variable"] = injected_var_name
        
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


def _build_default_where_save_path(path: str) -> str:
    """Build a unique @personal path for persisted where_filter results."""
    source_name = path.rsplit("/", 1)[-1] if "/" in path else path
    stem = source_name.removesuffix(".b2nd").removesuffix(".b2frame")
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    if not sanitized:
        sanitized = "filtered"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"@personal/where_filter/{sanitized}_{timestamp}.b2nd"


def where_filter(
    path: str,
    operator: str,
    threshold: float,
    value_if_true: float | None = None,
    value_if_false: float | None = None,
    slices: str | None = None,
    compute: bool = True,
) -> Dict[str, Any]:
    """
    Filter dataset or local variable values based on a condition.
    
    Applies a comparison (e.g., > 3000) to the data and returns:
    - value_if_true where the condition is met
    - value_if_false where the condition is not met
    
    Execution strategy:
    - Prefer backend-native where execution for both server and local data.
    - Fallback to generic array backend only when native where is unavailable.
    
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
        compute: For server datasets with authenticated sessions, controls auto-save mode.
                 True (default): compute and materialize result during upload.
                 False: keep lazy expression wrapper.
    
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

        save_path_clean: str | None = None
        auto_save_requested = False
        if resolved.is_server():
            auth_status = get_client_auth_status()
            if auth_status.get("authenticated"):
                save_path_clean = _build_default_where_save_path(path)
                auto_save_requested = True
            else:
                logger.debug(
                    "Skipping @personal auto-save for where_filter('%s') because session is not authenticated.",
                    path,
                )
        
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
        v_false = 0 if value_if_false is None else value_if_false
        execution_mode = "blosc2_where"
        operand_for_counts: Any | None = None
        result_obj: Any | None = None
        persisted_result_path: str | None = None
        persistence_skipped_reason: str | None = None

        try:
            operand = resolved.data
            if slices is not None:
                if hasattr(operand, "slice"):
                    if resolved.is_server():
                        # Server datasets: use as_blosc2=True to materialize to blosc2
                        operand = operand.slice(slice_tuple, as_blosc2=True)
                    else:
                        # Local blosc2 arrays: use standard slice (blosc2 .slice() doesn't take as_blosc2)
                        operand = operand.slice(slice_tuple)
                else:
                    operand = operand[slice_tuple]

            operand_for_counts = operand
            condition_obj = compare_fn(operand, threshold)
            if not hasattr(condition_obj, "where"):
                raise AttributeError("Condition object has no where() method")

            v_true = operand if value_if_true is None else value_if_true
            result_obj = condition_obj.where(v_true, v_false)
            result_data = _materialize_backend_value(result_obj)
            if resolved.is_server():
                execution_mode = "server_where"
        except Exception as e:
            logger.warning(
                f"Native where unavailable for '{path}' ({type(e).__name__}: {e}); "
                "falling back to backend-normalized execution."
            )
            data_slice = resolved[slice_tuple]
            if not isinstance(data_slice, blosc2.NDArray) and hasattr(data_slice, "shape"):
                try:
                    data_slice = blosc2.asarray(data_slice)
                except Exception:
                    pass
            operand_for_counts = data_slice
            condition_data = compare_fn(data_slice, threshold)
            v_true = data_slice if value_if_true is None else value_if_true
            if hasattr(condition_data, "where"):
                result_data = _materialize_backend_value(condition_data.where(v_true, v_false))
                execution_mode = (
                    "blosc2_where_fallback"
                    if isinstance(data_slice, blosc2.NDArray)
                    else "backend_where_fallback"
                )
            else:
                import numpy as np
                result_data = np.where(condition_data, v_true, v_false)
                if hasattr(result_data, "shape"):
                    try:
                        result_data = blosc2.asarray(result_data)
                        execution_mode = "blosc2_from_numpy_fallback"
                    except Exception:
                        execution_mode = "numpy_where_fallback"
                else:
                    execution_mode = "numpy_where_fallback"

        if save_path_clean is not None:
            try:
                client = _get_client()
                if client is None:
                    raise RuntimeError("Caterva2 client is not available.")

                # For native server where + compute=False, keep lazy upload path.
                if execution_mode == "server_where" and result_obj is not None and not compute:
                    payload_to_upload = result_obj
                    if (
                        hasattr(payload_to_upload, "compute")
                        and not isinstance(payload_to_upload, blosc2.NDArray)
                    ):
                        persisted = client.upload(payload_to_upload, save_path_clean, compute=False)
                    else:
                        persisted = client.upload(payload_to_upload, save_path_clean)
                else:
                    payload_to_upload = result_data
                    if (
                        hasattr(payload_to_upload, "compute")
                        and not isinstance(payload_to_upload, blosc2.NDArray)
                    ):
                        payload_to_upload = payload_to_upload.compute()
                    if (
                        not isinstance(payload_to_upload, blosc2.NDArray)
                        and hasattr(payload_to_upload, "shape")
                    ):
                        try:
                            payload_to_upload = blosc2.asarray(payload_to_upload)
                        except Exception:
                            # Keep backend-provided representation if conversion fails.
                            pass
                    persisted = client.upload(payload_to_upload, save_path_clean)

                persisted_result_path = (
                    getattr(persisted, "path", None)
                    or getattr(persisted, "name", None)
                    or save_path_clean
                )
            except Exception as e:
                logger.error(
                    "Failed to persist where_filter result to '%s': %s",
                    save_path_clean,
                    e,
                )
                return {
                    "error": f"Failed to persist filtered result to '{save_path_clean}': {e}",
                    "path": path,
                    "save_path": save_path_clean,
                }
        
        result_dtype = str(getattr(result_data, "dtype", resolved.dtype))
        result_shape = list(getattr(result_data, "shape", ()))
        summary_source = result_data

        if execution_mode == "server_where" and persisted_result_path is not None and compute:
            try:
                persisted_resolved = resolve_data(str(persisted_result_path))
                summary_source = persisted_resolved.data
                result_dtype = str(persisted_resolved.dtype)
                result_shape = list(persisted_resolved.shape)
            except Exception as e:
                logger.warning(
                    "Failed to compute where_filter summary from persisted dataset '%s': %s",
                    persisted_result_path,
                    e,
                )

        # Compute summary statistics
        summary = _compute_summary(summary_source)
        
        # Count how many elements matched the condition.
        # Avoid materializing boolean lazy conditions directly: this can segfault
        # for some backend combinations (e.g. server slices + lazy bool eval).
        num_total = int(estimated_size)
        num_matched = 0
        try:
            if operand_for_counts is None:
                raise RuntimeError("No operand available for counting.")
            if hasattr(operand_for_counts, "shape") and hasattr(operand_for_counts, "__getitem__"):
                full_key = tuple(slice(None) for _ in getattr(operand_for_counts, "shape", ()))
                operand_values = operand_for_counts[full_key]
            else:
                operand_values = operand_for_counts
            condition_values = compare_fn(operand_values, threshold)
            num_total = int(getattr(condition_values, "size", estimated_size))
            if hasattr(condition_values, "sum"):
                num_matched = int(_to_json_safe(condition_values.sum()))
            else:
                try:
                    condition_values_b2 = blosc2.asarray(condition_values)
                    num_matched = int(_to_json_safe(condition_values_b2.sum()))
                except Exception:
                    import numpy as np
                    num_matched = int(np.sum(condition_values))
        except Exception as e:
            logger.warning(
                "Condition-count fallback for '%s' due to backend error: %s",
                path,
                e,
            )
            num_total = int(getattr(result_data, "size", estimated_size))
            if (
                value_if_true is not None
                and value_if_false is not None
                and value_if_true != value_if_false
            ):
                try:
                    if hasattr(result_data, "shape") and hasattr(result_data, "__getitem__"):
                        result_values = result_data[tuple(slice(None) for _ in getattr(result_data, "shape", ()))]
                    else:
                        result_values = result_data
                    comparison_values = result_values == value_if_true
                    if hasattr(comparison_values, "sum"):
                        num_matched = int(_to_json_safe(comparison_values.sum()))
                    else:
                        try:
                            comparison_b2 = blosc2.asarray(comparison_values)
                            num_matched = int(_to_json_safe(comparison_b2.sum()))
                        except Exception:
                            import numpy as np
                            num_matched = int(np.sum(comparison_values))
                except Exception:
                    num_matched = 0
        match_percentage = (num_matched / num_total * 100) if num_total > 0 else 0
        
        result = {
            "path": path,
            "source": source_type,
            "dataset_shape": list(shape),
            "dtype": result_dtype,
            "condition": f"data {operator} {threshold}",
            "slice_applied": slice_str_used,
            "result_shape": result_shape,
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

        if save_path_clean is not None:
            if persisted_result_path is not None:
                result["stored_server_side"] = True
                result["result_path"] = str(persisted_result_path)
                result["save_path"] = str(save_path_clean)
                result["compute_on_save"] = bool(compute)
                result["server_change_applied"] = True
                result["auto_saved_to_personal"] = bool(auto_save_requested)
                result["_next_step_hint"] = (
                    "Use result_path in follow-up server operations "
                    "(get_slice, collapse_dimensions, where_filter, visualize_dataset)."
                )
            else:
                result["stored_server_side"] = False
                result["auto_saved_to_personal"] = False
                if persistence_skipped_reason:
                    result["persistence_note"] = persistence_skipped_reason
        elif resolved.is_server():
            result["stored_server_side"] = False
            result["auto_saved_to_personal"] = False
            result["persistence_note"] = (
                "Not auto-saved to @personal because this session is not authenticated. "
                "Use notebook login(...) to enable default persistence."
            )

        if estimated_nbytes is not None:
            result["estimated_size_mb"] = round(estimated_nbytes / (1024 * 1024), 2)

        if warning:
            result["warning"] = warning

        # Auto-inject local results into notebook namespace (no size limit for local).
        # User already has the dataset locally; we assume they can handle the size.
        injected_var_name: str | None = None
        if is_local:
            try:
                namespace = get_notebook_namespace()
                if namespace is not None:
                    # Generate a unique variable name for the filtered result
                    base_name = path.replace("_", "").replace("-", "").replace(".", "_")
                    base_name = f"filtered_{base_name}"
                    counter = 1
                    var_name = base_name
                    while var_name in namespace:
                        var_name = f"{base_name}_{counter}"
                        counter += 1
                    
                    # Inject into notebook namespace
                    namespace[var_name] = result_data
                    injected_var_name = var_name
                    register_fetched_object(var_name, result_data)
                    logger.info(f"Auto-injected local where_filter result as '{var_name}'")
            except Exception as e:
                logger.debug(f"Could not auto-inject local where_filter result: {e}")

        if summary["num_elements"] <= LLM_INLINE_DATA_MAX_ELEMENTS:
            result["data"] = _to_json_safe(result_data)
        
        # Add injected variable name if applicable
        if injected_var_name is not None:
            result["injected_as_variable"] = injected_var_name
        
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
    dataset into notebook memory as a backend array object (Blosc2-first).
    
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
        resolved = resolve_data(path)
        total_elements = _shape_product(resolved.shape)
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

        # Materialize explicitly while keeping backend-native representation.
        if resolved.is_local():
            data = resolved.data
        else:
            # Keep the server dataset handle directly. Creating a full NDArray
            # via .slice(..., as_blosc2=True) can crash in backend reductions
            # (observed as kernel segfaults when computing summary stats).
            data = resolved.data

        # Final guard using actual in-memory size
        data_nbytes = getattr(data, "nbytes", None)
        if data_nbytes is None:
            data_nbytes = _estimate_nbytes(int(getattr(data, "size", total_elements)), getattr(data, "dtype", resolved.dtype))
        if data_nbytes is None:
            return {
                "error": "Could not estimate materialized size for loaded dataset.",
                "shape": list(getattr(data, "shape", resolved.shape)),
                "dtype": str(getattr(data, "dtype", resolved.dtype)),
            }
        if data_nbytes > LOAD_DATASET_MAX_BYTES:
            actual_mb = data_nbytes / (1024 * 1024)
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

        size_bytes = int(data_nbytes)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        data_size = int(getattr(data, "size", total_elements))

        logger.debug(f"✓ Loaded: shape={list(data.shape)}, dtype={data.dtype}")
        logger.debug(f"Size: {size_mb} MB ({data_size:,} elements)")
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
            "num_elements": data_size,
            "backend": "blosc2" if isinstance(data, blosc2.NDArray) else type(data).__name__,
            "summary": summary,
            "registered_in_notebook": registered
        }
        
        # Include full data only for very small payloads
        if data_size <= LLM_INLINE_DATA_MAX_ELEMENTS:
            full_slice = tuple(slice(None) for _ in data.shape) if hasattr(data, "shape") else slice(None)
            try:
                inline_data = data[full_slice]
            except Exception:
                inline_data = data
            result["data"] = _to_json_safe(inline_data)
        else:
            result["note"] = (
                f"Data has {data_size:,} elements — returning summary only to protect LLM context. "
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
