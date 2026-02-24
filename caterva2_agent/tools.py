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
                "Call this first when the user asks what data is available."
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
                "List all datasets (arrays and files) within a given root or sub-path. "
                "Returns paths relative to the given path. "
                "Use this to discover what datasets exist before fetching their details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "The root name or sub-path to list. "
                            "Example: 'example' or 'example/dir1'"
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
                            "Example: 'example/ds-2d-fields.b2nd'"
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

def _get_client() -> cat2.Client:
    """
    Create a Caterva2 Client for the configured subscriber URL.
    Separated into a helper so tools don't repeat the same initialization.
    """
    return cat2.Client(CATERVA2_URLBASE)


def list_roots() -> Dict[str, Any]:
    """
    List all available roots (top-level collections) on the Caterva2 server.

    Returns:
        Dict with 'roots' key (list of root name strings), or 'error' on failure.
    """
    try:
        client = _get_client()
        roots_dict = client.get_roots()
        # get_roots() returns {name: {"name": name}, ...} — we extract just the names
        return {"roots": sorted(roots_dict.keys())}
    except Exception as e:
        return {"error": f"Failed to connect to Caterva2 server at {CATERVA2_URLBASE}: {e}"}


def list_datasets(path: str) -> Dict[str, Any]:
    """
    List all datasets within a root or sub-path on the Caterva2 server.

    Args:
        path: Root name or sub-path (e.g. 'example' or 'example/dir1')

    Returns:
        Dict with 'datasets' key (list of path strings), or 'error' on failure.
    """
    try:
        client = _get_client()
        datasets = client.get_list(path)
        # Prefix results with the path so full paths are immediately usable
        full_paths = [f"{path}/{name}" for name in datasets]
        return {"path": path, "datasets": full_paths}
    except Exception as e:
        return {"error": f"Failed to list datasets at path '{path}': {e}"}


def get_dataset_info(path: str) -> Dict[str, Any]:
    """
    Retrieve metadata for a specific dataset.

    Args:
        path: Full path including root (e.g. 'example/ds-2d-fields.b2nd')

    Returns:
        Dict with dataset properties (shape, dtype, chunks, etc.), or 'error' on failure.
    """
    try:
        client = _get_client()
        info = client.get_info(path)
        # info is already a dict — serialize any non-JSON-safe values to strings
        safe_info = {k: str(v) if not isinstance(v, (str, int, float, list, dict, bool, type(None))) else v
                     for k, v in info.items()}
        return {"path": path, "info": safe_info}
    except Exception as e:
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
