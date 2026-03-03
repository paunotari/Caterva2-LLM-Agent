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


# ---------------------------------------------------------------------------
# TOOL DISPATCHER
# Maps tool names (as the LLM sends them) to their Python implementations.
# execute_tool() is the single entry point called by the agent loop.
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "list_roots": list_roots,
    "list_datasets": list_datasets,
    "get_dataset_info": get_dataset_info,
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
