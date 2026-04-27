"""
Dataset management tools for server-side file operations.

Tools in this module:
- copy_dataset: Copy a dataset/file to another server path
- download_dataset: Get a direct download URL
- move_dataset: Move a dataset/file to another server path
- remove_dataset: Remove a dataset/file from the server
- upload_dataset: Upload local data (variable or array) to server
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import blosc2

from ._base import _get_dataset, _get_client, get_client_auth_status, resolve_data

logger = logging.getLogger("caterva2_agent")


# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

DATASET_MANAGEMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "copy_dataset",
            "description": (
                "Copy a server dataset/file from one path to another. "
                "This is a write operation and typically requires authentication "
                "(use notebook login first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Source server path (must start with '@').",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination server path (must start with '@').",
                    },
                },
                "required": ["path", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_dataset",
            "description": (
                "Get a direct download URL for a server dataset/file. "
                "This tool is URL-only to avoid exposing local filesystem paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Server path to download (must start with '@').",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_dataset",
            "description": (
                "Move a server dataset/file from one path to another. "
                "Safety policy: dry_run is true by default, and execution requires "
                "explicit user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Source server path (must start with '@').",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination server path (must start with '@').",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "When true (default), preview only. No server change occurs."
                        ),
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "Explicit confirmation flag for execution requests."
                        ),
                    },
                },
                "required": ["path", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_dataset",
            "description": (
                "Remove a server dataset/file permanently. "
                "Safety policy: dry_run is true by default, and execution requires "
                "explicit user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Server path to remove (must start with '@').",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "When true (default), preview only. No server change occurs."
                        ),
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "Explicit confirmation flag for execution requests."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_dataset",
            "description": (
                "Upload a local dataset (from notebook variable or array) to the Caterva2 server. "
                "The data is normalized to blosc2.NDArray format. "
                "This is a write operation requiring authentication (use notebook login first). "
                "Destination must be in @personal/ for user-scoped storage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Source data reference: either a notebook variable name (e.g. 'my_data') "
                            "or a local variable with {variable_name} syntax. "
                            "The variable must be array-like (NumPy, blosc2 NDArray, etc.)."
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "description": (
                            "Destination server path (must start with '@personal/'). "
                            "Example: '@personal/my_dataset.b2nd'"
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "When false (default), fail if destination exists. "
                            "When true, overwrite the destination."
                        ),
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# POLICY HELPERS
# ---------------------------------------------------------------------------

def _ensure_server_path(path: str, field_name: str = "path") -> str | None:
    """Validate that a path targets a server dataset/file."""
    if not isinstance(path, str) or not path.strip():
        return f"'{field_name}' must be a non-empty string."
    if not path.startswith("@"):
        return (
            f"'{field_name}' must be a server path starting with '@'. "
            "Local variables are not supported by this tool."
        )
    return None


def _ensure_personal_path(path: str, field_name: str = "destination") -> str | None:
    """Validate that a path targets @personal/ (user-scoped storage)."""
    error = _ensure_server_path(path, field_name)
    if error:
        return error
    if not path.startswith("@personal/"):
        return (
            f"'{field_name}' must start with '@personal/' for user-scoped uploads. "
            f"Got: {path}"
        )
    return None


def _auth_required_error(action: str) -> Dict[str, Any]:
    """Build a standard authentication-required payload."""
    status = get_client_auth_status()
    return {
        "error": (
            f"Authentication required to {action}. "
            "Run notebook `login()` and verify with `auth_status()`."
        ),
        "operation": action,
        "authenticated": False,
        "server": status.get("urlbase"),
    }


def _requires_auth_for_write(action: str) -> Dict[str, Any] | None:
    """Enforce auth policy for write/destructive operations."""
    status = get_client_auth_status()
    if status.get("authenticated"):
        return None
    return _auth_required_error(action)


def _error_with_auth_hint(action: str, error: Exception) -> Dict[str, Any]:
    """Format tool errors with auth guidance for permission-like failures."""
    message = str(error)
    message_lower = message.lower()
    if any(token in message_lower for token in ("401", "403", "unauthorized", "forbidden", "permission")):
        payload = _auth_required_error(action)
        payload["details"] = message
        return payload
    return {"error": f"Failed to {action}: {message}"}


def _dry_run_preview(operation: str, path: str, destination: str | None = None) -> Dict[str, Any]:
    """Return a standardized preview payload for risky operations."""
    result: Dict[str, Any] = {
        "status": "dry_run",
        "operation": operation,
        "path": path,
        "server_change_applied": False,
        "requires_confirmation": True,
        "next_step": "Await explicit user confirmation before executing this operation.",
    }
    if destination is not None:
        result["destination"] = destination
    if operation == "remove":
        result["warning"] = "Removal is irreversible."
    return result


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def copy_dataset(path: str, destination: str) -> Dict[str, Any]:
    """Copy a server dataset/file to another server path."""
    path_error = _ensure_server_path(path, "path")
    if path_error:
        return {"error": path_error}
    destination_error = _ensure_server_path(destination, "destination")
    if destination_error:
        return {"error": destination_error}
    if path == destination:
        return {"error": "Source and destination must be different paths."}

    auth_error = _requires_auth_for_write("copy datasets")
    if auth_error:
        return auth_error

    logger.info("Copying server dataset/file: %s -> %s", path, destination)
    try:
        file_obj = _get_dataset(path)
        copied_path = file_obj.copy(destination)
        return {
            "status": "success",
            "operation": "copy",
            "source_path": path,
            "destination_path": str(copied_path),
            "server_change_applied": True,
        }
    except Exception as e:
        logger.error("Failed to copy '%s' to '%s': %s", path, destination, e)
        return _error_with_auth_hint("copy dataset", e)


def download_dataset(path: str) -> Dict[str, Any]:
    """
    Return a direct download URL for a server dataset/file.
    """
    path_error = _ensure_server_path(path, "path")
    if path_error:
        return {"error": path_error}

    logger.info("Getting download URL for server dataset/file '%s'", path)
    try:
        file_obj = _get_dataset(path)
        download_url = file_obj.get_download_url()
        return {
            "status": "success",
            "operation": "download",
            "path": path,
            "mode": "url",
            "download_url": str(download_url),
            "server_change_applied": False,
            "local_download_performed": False,
            "_hint": (
                "Tool returns URL only. Use this link outside the agent/notebook "
                "to perform the actual file download."
            ),
        }
    except Exception as e:
        logger.error("Failed to download '%s': %s", path, e)
        return _error_with_auth_hint("download dataset", e)


def move_dataset(path: str, destination: str, dry_run: bool = True, confirm: bool = False) -> Dict[str, Any]:
    """Move a server dataset/file with dry-run + confirm safety policy."""
    path_error = _ensure_server_path(path, "path")
    if path_error:
        return {"error": path_error}
    destination_error = _ensure_server_path(destination, "destination")
    if destination_error:
        return {"error": destination_error}
    if path == destination:
        return {"error": "Source and destination must be different paths."}

    auth_error = _requires_auth_for_write("move datasets")
    if auth_error:
        return auth_error

    if dry_run:
        return _dry_run_preview("move", path, destination)
    if not confirm:
        return {
            "error": "Confirmation required before executing move.",
            "operation": "move",
            "path": path,
            "destination": destination,
            "requires_confirmation": True,
            "server_change_applied": False,
        }

    logger.info("Moving server dataset/file: %s -> %s", path, destination)
    try:
        file_obj = _get_dataset(path)
        moved_path = file_obj.move(destination)
        return {
            "status": "success",
            "operation": "move",
            "source_path": path,
            "destination_path": str(moved_path),
            "server_change_applied": True,
        }
    except Exception as e:
        logger.error("Failed to move '%s' to '%s': %s", path, destination, e)
        return _error_with_auth_hint("move dataset", e)


def remove_dataset(path: str, dry_run: bool = True, confirm: bool = False) -> Dict[str, Any]:
    """Remove a server dataset/file with dry-run + confirm safety policy."""
    path_error = _ensure_server_path(path, "path")
    if path_error:
        return {"error": path_error}

    auth_error = _requires_auth_for_write("remove datasets")
    if auth_error:
        return auth_error

    if dry_run:
        return _dry_run_preview("remove", path)
    if not confirm:
        return {
            "error": "Confirmation required before executing removal.",
            "operation": "remove",
            "path": path,
            "requires_confirmation": True,
            "server_change_applied": False,
            "warning": "Removal is irreversible.",
        }

    logger.info("Removing server dataset/file: %s", path)
    try:
        file_obj = _get_dataset(path)
        removed_path = file_obj.remove()
        return {
            "status": "success",
            "operation": "remove",
            "removed_path": str(removed_path),
            "server_change_applied": True,
            "warning": "Removal is irreversible.",
        }
    except Exception as e:
        logger.error("Failed to remove '%s': %s", path, e)
        return _error_with_auth_hint("remove dataset", e)


def upload_dataset(source: str, destination: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Upload local data (variable or array) to the Caterva2 server.
    
    Resolves the source (notebook variable or reference), normalizes to blosc2.NDArray,
    validates the destination path (@personal/ required), and uploads via client.upload().
    
    Args:
        source: Variable name or reference (e.g. 'my_data' or from {variable} syntax)
        destination: Server path (must start with '@personal/')
        overwrite: Allow overwriting existing destination (default: False)
    
    Returns:
        Dict with status, metadata, or error information
    """
    # Validate destination path
    dest_error = _ensure_personal_path(destination, "destination")
    if dest_error:
        return {"error": dest_error}
    
    # Check authentication
    auth_error = _requires_auth_for_write("upload datasets")
    if auth_error:
        return auth_error
    
    logger.info("Uploading data from '%s' to '%s'", source, destination)
    
    try:
        # Resolve the source data (handles both local variables and server paths)
        resolved = resolve_data(source)
        data = resolved.data
        
        # Ensure data is blosc2.NDArray for upload
        if not isinstance(data, blosc2.NDArray):
            try:
                data = blosc2.asarray(data)
            except Exception as e:
                logger.error("Failed to normalize data to blosc2: %s", e)
                return {
                    "error": f"Could not normalize source data to blosc2.NDArray: {e}",
                    "source": source,
                    "source_type": resolved.normalized_from or type(resolved.data).__name__,
                }
        
        # Get client and upload
        client = _get_client()
        if client is None:
            raise RuntimeError("Caterva2 client is not available.")
        
        # Upload the data
        result = client.upload(data, destination)
        
        # Extract the uploaded path
        uploaded_path = (
            getattr(result, "path", None)
            or getattr(result, "name", None)
            or destination
        )
        
        return {
            "status": "success",
            "operation": "upload",
            "source_name": source,
            "source_type": resolved.source,
            "data_shape": list(data.shape) if hasattr(data, "shape") else [],
            "data_dtype": str(data.dtype) if hasattr(data, "dtype") else "unknown",
            "destination_path": str(uploaded_path),
            "server_change_applied": True,
            "next_step": (
                f"The uploaded dataset is now available at {uploaded_path}. "
                "Use it in follow-up queries or share with other users."
            ),
        }
    
    except ValueError as e:
        # Variable resolution or validation errors
        logger.error("Failed to resolve source '%s': %s", source, e)
        return {
            "error": str(e),
            "source": source,
            "destination": destination,
        }
    
    except Exception as e:
        logger.error("Failed to upload '%s' to '%s': %s", source, destination, e)
        return _error_with_auth_hint("upload dataset", e)
