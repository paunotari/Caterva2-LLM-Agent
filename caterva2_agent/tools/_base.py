"""
Shared helpers for all tool implementations.

This module provides:
- Caterva2 client management (singleton pattern)
- Runtime authentication session management
- Unified data resolver (server datasets + local variables)
- JSON serialization utilities
- Object registry for notebook integration

These are shared across browsing, analysis, and data_access tools.
"""

import logging
from typing import Any, Protocol, runtime_checkable

import blosc2
import caterva2 as cat2
from caterva2 import Client

from caterva2_agent.config import CATERVA2_URLBASE

logger = logging.getLogger("caterva2_agent")

# ---------------------------------------------------------------------------
# OBJECT REGISTRY FOR NOTEBOOK INTEGRATION
# ---------------------------------------------------------------------------
# When tools explicitly materialize derived/full data (for example, load_dataset
# or collapse_dimensions), they register results here.
# The notebook.py module reads this registry to inject variables into the
# user's namespace. This decouples tool execution from notebook integration.

# Format: {"@public/path/to/dataset.b2nd": <array-like object>, ...}
# Keys are full dataset paths for traceability.
_fetched_objects: dict[str, Any] = {}


def register_fetched_object(path: str, data: Any) -> None:
    """
    Register a fetched object for later injection into notebook namespace.
    
    Called by tools that intentionally materialize data for notebook use.
    The notebook module reads this registry after agent.run() completes.
    
    Args:
        path: Full dataset path (e.g. '@public/examples/ds-1d.b2nd')
        data: The fetched data (typically a blosc2-backed array)
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
    
    Called by notebook.py before agent.run() to enable tools to read user-created variables.
    
    Args:
        namespace: The notebook's user_ns dict (from get_ipython().user_ns)
    """
    global _notebook_namespace
    _notebook_namespace = namespace


def get_notebook_namespace() -> dict | None:
    """Get the current notebook namespace, or None if not set."""
    return _notebook_namespace


# ---------------------------------------------------------------------------
# Caterva2 CLIENT MANAGEMENT
# ---------------------------------------------------------------------------

_cat2_client: cat2.Client | None = None
_cat2_client_urlbase: str = CATERVA2_URLBASE
_cat2_client_auth: tuple[str, str] | None = None
_cat2_auth_username: str | None = None


def _close_client(client: cat2.Client | None) -> None:
    """Best-effort close for a Caterva2 client instance."""
    if client is None:
        return

    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            # Closing should never break session state transitions.
            pass


def get_client_auth_status() -> dict[str, Any]:
    """
    Return current Caterva2 client authentication status for UI/reporting.

    Returns:
        Dict with:
        - urlbase: active Caterva2 server URL
        - authenticated: whether runtime auth is active
        - username: authenticated user or None
    """
    return {
        "urlbase": _cat2_client_urlbase,
        "authenticated": _cat2_client_auth is not None,
        "username": _cat2_auth_username,
    }


def set_client_auth(username: str, password: str, urlbase: str | None = None) -> dict[str, Any]:
    """
    Authenticate the Caterva2 client for the current runtime session.

    Authentication in Caterva2 is done at client construction time
    (`cat2.Client(urlbase, auth=(username, password))`). This function validates
    credentials with a lightweight request and, on success, swaps the shared
    singleton client to authenticated mode.

    Raises:
        ValueError: For invalid input or authentication failures.
    """
    global _cat2_client, _cat2_client_urlbase, _cat2_client_auth, _cat2_auth_username

    username_clean = username.strip()
    target_url = (urlbase or _cat2_client_urlbase or CATERVA2_URLBASE).strip()

    if not username_clean:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")
    if not target_url:
        raise ValueError("Caterva2 URL cannot be empty.")

    candidate = cat2.Client(target_url, auth=(username_clean, password))
    try:
        # Validate credentials and connectivity before mutating global state.
        candidate.get_roots()
    except Exception as e:
        _close_client(candidate)
        raise ValueError(f"Authentication failed for '{username_clean}': {e}") from e

    old_client = _cat2_client
    _cat2_client = candidate
    _cat2_client_urlbase = target_url
    _cat2_client_auth = (username_clean, password)
    _cat2_auth_username = username_clean

    if old_client is not candidate:
        _close_client(old_client)

    return {
        "status": "authenticated",
        "urlbase": _cat2_client_urlbase,
        "authenticated": True,
        "username": _cat2_auth_username,
    }


def clear_client_auth() -> dict[str, Any]:
    """
    Clear runtime Caterva2 authentication and return to anonymous mode.

    This does not change the current server URL; it only drops credentials and
    resets the shared client so the next call recreates an anonymous client.
    """
    global _cat2_client, _cat2_client_auth, _cat2_auth_username

    old_client = _cat2_client
    _cat2_client = None
    _cat2_client_auth = None
    _cat2_auth_username = None
    _close_client(old_client)

    return {
        "status": "anonymous",
        "urlbase": _cat2_client_urlbase,
        "authenticated": False,
        "username": None,
    }


def _get_client() -> Client | None:
    """
    Create (if needed) and return a Caterva2 Client for the configured subscriber URL.
    
    Uses singleton pattern — creates the client once and reuses it for all tool calls.
    This avoids repeated connection overhead.
    """
    global _cat2_client
    if _cat2_client is None:
        if _cat2_client_auth is None:
            _cat2_client = cat2.Client(_cat2_client_urlbase)
        else:
            _cat2_client = cat2.Client(_cat2_client_urlbase, auth=_cat2_client_auth)
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
    """Protocol for array-like objects (blosc2, numpy, etc.)."""
    shape: tuple[int, ...]
    dtype: Any
    def __getitem__(self, key) -> Any: ...


class ResolvedData:
    """
    Container for resolved data with unified interface.
    
    Wraps Caterva2 Dataset objects and local array-like values to provide
    a consistent interface for tools.
    """
    
    def __init__(
        self,
        data: Any,
        source: str,
        name: str,
        *,
        backend: str | None = None,
        normalized_from: str | None = None,
    ):
        """
        Args:
            data: The actual data (Dataset or array-like object)
            source: 'server' or 'local'
            name: Original path or variable name
            backend: Data backend identifier ('caterva2', 'blosc2', etc.)
            normalized_from: Original local type name if data was normalized
        """
        self.data = data
        self.source = source
        self.name = name
        self.backend = backend or _infer_backend(data, source)
        self.normalized_from = normalized_from
    
    @property
    def shape(self) -> tuple[int, ...]:
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

    def is_local_blosc2(self) -> bool:
        return self.is_local() and self.backend == "blosc2"


def _infer_backend(data: Any, source: str) -> str:
    """Infer the backend used by this resolved object."""
    if source == "server":
        return "caterva2"
    if isinstance(data, blosc2.NDArray):
        return "blosc2"
    return type(data).__name__.lower()


def _normalize_local_array(value: Any, variable_name: str) -> tuple[Any, str | None]:
    """
    Normalize local inputs while preserving lazy arrays for efficient chaining.

    Lazy arrays (blosc2.LazyArray) are kept as-is to enable composition
    without materialization. Only non-lazy inputs are converted to blosc2.

    Returns:
        (normalized_value, normalized_from_type_name_or_none)
    """
    # Keep NDArray and LazyArray as-is (both are blosc2-native)
    if isinstance(value, (blosc2.NDArray, blosc2.LazyArray)):
        return value, None

    source_type = type(value).__name__
    try:
        normalized = blosc2.asarray(value)
    except Exception as e:
        raise ValueError(
            f"Variable '{variable_name}' could not be converted to a blosc2 NDArray "
            f"from type '{source_type}': {e}"
        ) from e

    logger.debug(
        "Normalized local variable '%s' from %s to blosc2 NDArray (shape=%s, dtype=%s)",
        variable_name,
        source_type,
        getattr(normalized, "shape", "unknown"),
        getattr(normalized, "dtype", "unknown"),
    )
    return normalized, source_type


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
    
    # Check notebook namespace first (user may have modified variables)
    if path_or_name in namespace:
        value = namespace[path_or_name]
    # Also check fetched objects registry (from previous tool calls in same run)
    elif path_or_name in _fetched_objects:
        value = _fetched_objects[path_or_name]
    else:
        raise ValueError(
            f"Variable '{path_or_name}' not found in notebook namespace"
        )
    
    # Validate it's array-like
    if not (hasattr(value, 'shape') and hasattr(value, 'dtype')):
        raise ValueError(
            f"Variable '{path_or_name}' is not array-like "
            f"(type: {type(value).__name__}). "
            "Only NumPy, blosc2 NDArray, and similar array-like objects are supported."
        )

    normalized_value, normalized_from = _normalize_local_array(value, path_or_name)
    return ResolvedData(
        normalized_value,
        source='local',
        name=path_or_name,
        backend='blosc2',
        normalized_from=normalized_from,
    )


# ---------------------------------------------------------------------------
# SAFE DATA FETCHING WITH AUTO-INJECTION
# ---------------------------------------------------------------------------

def fetch_and_register_data(
    path: str,
    slice_spec: tuple | None = None,
    max_elements: int = 10_000
) -> tuple[Any, dict[str, Any]]:
    """
    Fetch data from a dataset and register it for notebook injection.
    
    This is the common path for all data-fetching tools (get_slice, where_filter,
    load_dataset). It enforces safety limits and handles both server datasets
    and local variables.
    
    Args:
        path: Server dataset path (e.g. '@public/data.b2nd') or local variable name
        slice_spec: Slice tuple (e.g. (slice(0, 100),)) or None for full dataset
        max_elements: Maximum number of elements to fetch
    
    Returns:
        Tuple of (data_array, metadata_dict)
        - data_array: The fetched array-like object (Blosc2-first)
        - metadata_dict: Info about the fetch (shape, size, etc.)
    
    Raises:
        ValueError: If data exceeds max_elements or other validation fails
    """
    # Resolve the data source
    resolved = resolve_data(path)
    is_local = resolved.source == 'local'
    
    # Determine what to fetch
    if slice_spec is None:
        # Full dataset - check size first
        total_elements = 1
        for dim in resolved.shape:
            total_elements *= int(dim)
        if total_elements > max_elements:
            raise ValueError(
                f"Dataset has {total_elements:,} elements, exceeding limit of {max_elements:,}. "
                f"Use slices to fetch a smaller region."
            )
        if is_local:
            data = resolved.data
        elif hasattr(resolved.data, "slice"):
            full_slice = tuple(slice(None) for _ in resolved.shape)
            data = resolved.data.slice(full_slice, as_blosc2=True)
        else:
            data = resolved[:]
    else:
        # Sliced region
        if not is_local and hasattr(resolved.data, "slice"):
            data = resolved.data.slice(slice_spec, as_blosc2=True)
        else:
            data = resolved[slice_spec]

    if not isinstance(data, blosc2.NDArray) and hasattr(data, "shape"):
        try:
            data = blosc2.asarray(data)
        except Exception:
            # Keep original representation if conversion is unavailable.
            pass
    
    # Calculate size for metadata
    data_size_bytes = getattr(data, "nbytes", None)
    if data_size_bytes is None:
        data_size_bytes = 0
        itemsize = getattr(getattr(data, "dtype", None), "itemsize", None)
        size = getattr(data, "size", None)
        if itemsize is not None and size is not None:
            data_size_bytes = int(itemsize) * int(size)
    data_size_mb = data_size_bytes / (1024 * 1024)
    
    # Register for notebook injection (only for server datasets)
    if not is_local:
        register_fetched_object(path, data)
    
    # Build metadata
    metadata = {
        "source": resolved.source,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "size_bytes": data_size_bytes,
        "size_mb": round(data_size_mb, 2),
        "registered": not is_local  # Local vars are already in namespace
    }
    
    return data, metadata


# ---------------------------------------------------------------------------
# JSON SERIALIZATION UTILITIES
# ---------------------------------------------------------------------------

def _to_json_safe(value) -> Any:
    """
    Convert numpy/blosc2 values to JSON-serializable Python types.
    
    Statistical methods and slicing may return NumPy scalars/arrays — these must
    be converted to native Python types for JSON serialization.
    
    Args:
        value: Any value (array, scalar, or native Python type)
    
    Returns:
        JSON-serializable equivalent
    """
    import numpy as np

    if isinstance(value, blosc2.NDArray):
        # NDArray slicing yields a NumPy array/scalar which we can serialize safely.
        return _to_json_safe(value[:])
    if hasattr(value, "__array__"):
        try:
            return _to_json_safe(np.asarray(value))
        except Exception:
            pass
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
