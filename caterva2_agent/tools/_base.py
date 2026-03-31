"""
Shared helpers for all tool implementations.

This module provides:
- Caterva2 client management (singleton pattern)
- Dataset retrieval helper
- JSON serialization utilities
- Object registry for notebook integration

These are shared across browsing, analysis, and data_access tools.
"""

from typing import Any

import caterva2 as cat2

from caterva2_agent.config import CATERVA2_URLBASE


# ---------------------------------------------------------------------------
# OBJECT REGISTRY FOR NOTEBOOK INTEGRATION
# ---------------------------------------------------------------------------
# When tools fetch data (get_slice, where_filter), they register results here.
# The notebook.py module reads this registry to inject variables into the
# user's namespace. This decouples tool execution from notebook integration.

# Format: {"@public/path/to/dataset.b2nd": <numpy array>, ...}
# Keys are full dataset paths for traceability.
_fetched_objects: dict[str, Any] = {}


def register_fetched_object(path: str, data: Any) -> None:
    """
    Register a fetched object for later injection into notebook namespace.
    
    Called by tools that retrieve data (get_slice, where_filter).
    The notebook module reads this registry after agent.run() completes.
    
    Args:
        path: Full dataset path (e.g. '@public/examples/ds-1d.b2nd')
        data: The fetched data (typically a numpy array)
    """
    _fetched_objects[path] = data


def get_fetched_objects() -> dict[str, Any]:
    """
    Get all fetched objects (for notebook injection).
    
    Returns:
        Dict mapping paths to fetched data
    """
    return _fetched_objects.copy()


def clear_fetched_objects() -> None:
    """Clear the fetched objects registry."""
    _fetched_objects.clear()


def pop_fetched_objects() -> dict[str, Any]:
    """
    Get and clear all fetched objects (atomic operation for injection).
    
    Returns:
        Dict mapping paths to fetched data
    """
    result = _fetched_objects.copy()
    _fetched_objects.clear()
    return result


# ---------------------------------------------------------------------------
# CATERVA2 CLIENT MANAGEMENT
# ---------------------------------------------------------------------------

_cat2_client: cat2.Client | None = None


def _get_client() -> cat2.Client:
    """
    Create (if needed) and return a Caterva2 Client for the configured subscriber URL.
    
    Uses singleton pattern — creates the client once and reuses it for all tool calls.
    This avoids repeated connection overhead.
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


# ---------------------------------------------------------------------------
# JSON SERIALIZATION UTILITIES
# ---------------------------------------------------------------------------

def _to_json_safe(value) -> Any:
    """
    Convert numpy/blosc2 values to JSON-serializable Python types.
    
    Statistical methods and slicing return numpy scalars or arrays — these must be
    converted to native Python types for JSON serialization.
    
    Args:
        value: Any value (numpy array, scalar, or native Python type)
    
    Returns:
        JSON-serializable equivalent
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
