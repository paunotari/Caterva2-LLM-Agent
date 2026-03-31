"""
Shared helpers for all tool implementations.

This module provides:
- Caterva2 client management (singleton pattern)
- Unified data resolver (server datasets + local variables)
- JSON serialization utilities
- Object registry for notebook integration

These are shared across browsing, analysis, and data_access tools.
"""

from typing import Any, Protocol, runtime_checkable
import numpy as np

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
# NOTEBOOK NAMESPACE ACCESS
# ---------------------------------------------------------------------------
# For reading local variables when tools operate on user-created arrays.

_notebook_namespace: dict | None = None


def set_notebook_namespace(namespace: dict) -> None:
    """
    Set the notebook namespace for local variable access.
    
    Called by notebook.py before agent.run() to enable tools to read
    user-created variables.
    
    Args:
        namespace: The notebook's user_ns dict (from get_ipython().user_ns)
    """
    global _notebook_namespace
    _notebook_namespace = namespace


def get_notebook_namespace() -> dict | None:
    """Get the current notebook namespace, or None if not set."""
    return _notebook_namespace


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
# UNIFIED DATA RESOLVER
# ---------------------------------------------------------------------------
# Resolves a path_or_name to actual data, whether from server or local namespace.


@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array-like objects (numpy, blosc2, etc.)."""
    shape: tuple
    dtype: Any
    def __getitem__(self, key) -> Any: ...


class ResolvedData:
    """
    Container for resolved data with unified interface.
    
    Wraps both Caterva2 Dataset objects and local numpy arrays to provide
    a consistent interface for tools.
    """
    
    def __init__(self, data: Any, source: str, name: str):
        """
        Args:
            data: The actual data (Dataset or ndarray)
            source: 'server' or 'local'
            name: Original path or variable name
        """
        self.data = data
        self.source = source
        self.name = name
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def dtype(self) -> Any:
        return self.data.dtype
    
    def __getitem__(self, key) -> Any:
        return self.data[key]
    
    def is_local(self) -> bool:
        return self.source == 'local'
    
    def is_server(self) -> bool:
        return self.source == 'server'


def resolve_data(path_or_name: str) -> ResolvedData:
    """
    Resolve a path or variable name to actual data.
    
    This is the unified entry point for tools that can operate on both
    server datasets and local variables.
    
    Resolution rules:
    - If starts with '@' → server dataset (fetch from Caterva2)
    - Otherwise → local variable (read from notebook namespace)
    
    Args:
        path_or_name: Server path (e.g. '@public/data.b2nd') or local variable name
    
    Returns:
        ResolvedData wrapper with unified interface
    
    Raises:
        ValueError: If local variable not found or not array-like
    """
    if path_or_name.startswith('@'):
        # Server dataset
        dataset = _get_dataset(path_or_name)
        return ResolvedData(dataset, source='server', name=path_or_name)
    
    # Local variable
    namespace = get_notebook_namespace()
    if namespace is None:
        raise ValueError(
            f"Cannot access local variable '{path_or_name}': "
            "not running in notebook context"
        )
    
    if path_or_name not in namespace:
        raise ValueError(
            f"Variable '{path_or_name}' not found in notebook namespace"
        )
    
    value = namespace[path_or_name]
    
    # Validate it's array-like
    if not (hasattr(value, 'shape') and hasattr(value, 'dtype')):
        raise ValueError(
            f"Variable '{path_or_name}' is not array-like "
            f"(type: {type(value).__name__}). "
            "Only numpy arrays and similar objects are supported."
        )
    
    return ResolvedData(value, source='local', name=path_or_name)


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
