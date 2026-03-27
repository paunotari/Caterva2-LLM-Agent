"""
Shared helpers for all tool implementations.

This module provides:
- Caterva2 client management (singleton pattern)
- Dataset retrieval helper
- JSON serialization utilities

These are shared across browsing, analysis, and data_access tools.
"""

from typing import Any

import caterva2 as cat2

from config import CATERVA2_URLBASE


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
