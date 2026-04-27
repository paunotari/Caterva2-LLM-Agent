"""
Tool registry and dispatcher for the Caterva2 agent.

This module:
- Combines tool schemas from all category modules into TOOLS list
- Builds TOOL_MAP from all implementations
- Provides execute_tool() entry point for the agent loop

Adding a new tool:
1. Add schema to the appropriate category module
   (browsing.py, analysis.py, data_access.py, dataset_management.py, or visualization.py)
2. Implement the function in that module
3. Import and register it here in TOOL_MAP
"""

import json
from typing import Dict, Any

# Import schemas from category modules
from .browsing import BROWSING_TOOLS, list_roots, list_datasets, get_dataset_info
from .analysis import ANALYSIS_TOOLS, get_dataset_stats, collapse_dimensions
from .data_access import DATA_ACCESS_TOOLS, get_slice, where_filter, load_dataset
from .dataset_management import DATASET_MANAGEMENT_TOOLS, copy_dataset, download_dataset, move_dataset, remove_dataset, upload_dataset
from .visualization import VISUALIZATION_TOOLS, visualize_dataset, render_projection


# ---------------------------------------------------------------------------
# COMBINED TOOL SCHEMAS
# Sent to the LLM so it knows what tools exist.
# Order: browsing → analysis → data_access → management → visualization
# ---------------------------------------------------------------------------

TOOLS = (
    BROWSING_TOOLS
    + ANALYSIS_TOOLS
    + DATA_ACCESS_TOOLS
    + DATASET_MANAGEMENT_TOOLS
    + VISUALIZATION_TOOLS
)


# ---------------------------------------------------------------------------
# TOOL MAP
# Routes tool names to their implementations.
# ---------------------------------------------------------------------------

TOOL_MAP = {
    # Browsing tools
    "list_roots": list_roots,
    "list_datasets": list_datasets,
    "get_dataset_info": get_dataset_info,
    
    # Analysis tools
    "get_dataset_stats": get_dataset_stats,
    "collapse_dimensions": collapse_dimensions,
    
    # Data access tools
    "get_slice": get_slice,
    "where_filter": where_filter,
    "load_dataset": load_dataset,

    # Dataset management tools
    "copy_dataset": copy_dataset,
    "download_dataset": download_dataset,
    "move_dataset": move_dataset,
    "remove_dataset": remove_dataset,
    "upload_dataset": upload_dataset,
    
    # Visualization tools
    "visualize_dataset": visualize_dataset,
    "render_projection": render_projection,
}


# ---------------------------------------------------------------------------
# DISPATCHER
# Single entry point called by the agent loop.
# ---------------------------------------------------------------------------

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
