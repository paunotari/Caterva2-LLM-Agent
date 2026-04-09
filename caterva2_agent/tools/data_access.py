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

from ._base import resolve_data, _to_json_safe, register_fetched_object, fetch_and_register_data


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
                "Retrieve a slice of data values from a dataset or local variable. "
                "Use this when the user wants to see actual data, not just statistics. "
                "SAFETY: Limited to 10,000 elements maximum to avoid memory issues. "
                "For large datasets, use slicing to select a specific region of interest. "
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
                "SAFETY: Limited to 10,000 elements maximum — use slices for larger datasets. "
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
                "SAFETY: Checks dataset size before loading - rejects if too large. "
                "For large datasets (>10K elements), the tool will suggest using get_slice instead. "
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


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def get_slice(path: str, slices: str | None = None) -> Dict[str, Any]:
    """
    Retrieve a slice of data from a dataset or local variable.
    
    Parses Python-style slice syntax and enforces a maximum element limit
    to protect the LLM context window from huge data dumps.
    
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
        
        logger.debug(f"Slice tuple: {slice_tuple}")
        logger.debug(f"Estimated elements: {estimated_size}")
        
        # Fetch the data
        data = resolved[slice_tuple]
        data_json = _to_json_safe(data)
        
        # Generate variable name and register for notebook injection
        # Only register if this is from server (local vars are already in namespace)
        variable_name = None
        if not is_local:
            # Sanitize path to create a valid Python identifier
            # '@public/examples/ds-3d.b2nd' → 'ds_3d'
            base_name = path.split('/')[-1]  # Get filename
            base_name = base_name.replace('.b2nd', '').replace('.b2frame', '')
            base_name = base_name.replace('-', '_').replace('@', '').replace('.', '_')
            variable_name = base_name
            
            register_fetched_object(variable_name, data)
            logger.debug(f"✓ Registered as: '{variable_name}'")
        
        # Pre-compute summary for LLM presentation
        summary = _compute_summary(data)
        
        result = {
            "path": path,
            "source": source_type,
            "dataset_shape": list(shape),
            "dtype": str(resolved.dtype),
            "slice": slice_str_used,
            "result_shape": list(data.shape) if hasattr(data, 'shape') else [],
            "summary": summary,
            "data": data_json
        }
        
        # Include variable name if this was registered
        if variable_name:
            result["variable_name"] = variable_name
            result["note"] = f"Data stored as '{variable_name}' in notebook. Use this name to reference it in other tools."
        
        # Hint for the LLM on how to present results
        if summary["num_elements"] > SUMMARY_THRESHOLD:
            result["_hint"] = (
                f"Large result ({summary['num_elements']} elements). "
                "Present the summary to the user and offer to show full data if requested."
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
    
    This is implemented using NumPy's where function on the fetched slice.
    The tool is stateless — it fetches data fresh each call. If you previously
    used get_slice on a region, pass the same slice here to filter that region.
    
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
        
        # Determine slice to apply (for size safety)
        if slices is None:
            slice_tuple = _default_slice_for_shape(shape, MAX_SLICE_ELEMENTS)
            slice_str_used = "(default — first elements)"
        else:
            slice_tuple = _parse_slice_string(slices, shape)
            slice_str_used = slices
        
        # Estimate size and enforce limit
        estimated_size = _compute_slice_size(slice_tuple, shape)
        if estimated_size > MAX_SLICE_ELEMENTS:
            return {
                "error": f"Requested region would return ~{estimated_size:,} elements, "
                         f"exceeding limit of {MAX_SLICE_ELEMENTS:,}. "
                         f"Please specify a smaller slice.",
                "shape": list(shape),
                "requested_slice": slice_str_used
            }
        
        logger.debug(f"Estimated elements: {estimated_size}")
        
        # Step 1: Get the data slice as numpy array
        data_slice = resolved[slice_tuple]
        
        # Step 2: Build the boolean condition
        compare_fn = COMPARISON_OPERATORS[operator]
        condition = compare_fn(data_slice, threshold)
        
        # Step 3: Determine replacement values
        # If value_if_true is None, use the original data (passthrough)
        # If value_if_false is None, default to 0
        import numpy as np
        
        v_true = data_slice if value_if_true is None else value_if_true
        v_false = 0 if value_if_false is None else value_if_false
        
        # Step 4: Apply where condition
        # np.where(condition, x, y) returns x where True, y where False
        result_data = np.where(condition, v_true, v_false)
        
        # Register for notebook injection (user can access as a variable)
        # Use a descriptive path that includes the filter condition
        # Only register if this is from server (local vars are already in namespace)
        if not is_local:
            filter_path = f"{path}[{operator}{threshold}]"
            register_fetched_object(filter_path, result_data)
        
        # Convert to JSON-safe format
        result_json = _to_json_safe(result_data)
        
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
            "value_if_true": "original_data" if value_if_true is None else value_if_true,
            "value_if_false": v_false,
            "match_summary": {
                "matched": num_matched,
                "total": num_total,
                "percentage": round(match_percentage, 2)
            },
            "summary": summary,
            "data": result_json
        }
        
        # Hint for LLM on how to present results
        if summary["num_elements"] > SUMMARY_THRESHOLD:
            result["_hint"] = (
                f"Large result ({summary['num_elements']} elements). "
                f"{num_matched} values ({match_percentage:.1f}%) matched the condition. "
                "Present the summary to the user and offer to show full data if requested."
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
    
    Safety: Enforces the same 10K element limit as get_slice. For larger datasets,
    suggests using get_slice to fetch specific regions instead.
    
    Args:
        path: Server dataset path (e.g. '@public/examples/ds-1d.b2nd')
              OR local variable name (e.g. 'my_data')
    
    Returns:
        Dict with dataset metadata, summary, and data values, or 'error' on failure.
    """
    from ._base import fetch_and_register_data
    
    is_local = not path.startswith("@")
    source_type = "local variable" if is_local else "server dataset"
    
    logger.info(f"Loading full {source_type}: '{path}'")
    
    try:
        # Fetch the entire dataset with safety check
        data, metadata = fetch_and_register_data(
            path=path,
            slice_spec=None,  # Full dataset
            max_elements=MAX_SLICE_ELEMENTS
        )
        
        logger.debug(f"✓ Loaded: shape={metadata['shape']}, dtype={metadata['dtype']}")
        logger.debug(f"Size: {metadata['size_mb']} MB ({data.size:,} elements)")
        
        if metadata['registered']:
            logger.debug(f"📦 Available in notebook for manipulation")
        
        # Convert to JSON-safe format
        data_json = _to_json_safe(data)
        
        # Compute summary statistics
        summary = _compute_summary(data)
        
        result = {
            "status": "success",
            "path": path,
            "source": metadata['source'],
            "shape": metadata['shape'],
            "dtype": metadata['dtype'],
            "size_bytes": metadata['size_bytes'],
            "size_mb": metadata['size_mb'],
            "num_elements": data.size,
            "summary": summary,
            "registered_in_notebook": metadata['registered']
        }
        
        # Include full data if small enough for LLM context
        if data.size <= SUMMARY_THRESHOLD:
            result["data"] = data_json
        else:
            result["note"] = (
                f"Data has {data.size} elements — showing summary only. "
                f"Data is available in your notebook for direct manipulation."
            )
        
        return result
    
    except ValueError as e:
        # Size limit exceeded or variable not found
        logger.warning(f"Load validation error for '{path}': {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Failed to load '{path}': {e}")
        return {"error": f"Failed to load '{path}': {e}"}
