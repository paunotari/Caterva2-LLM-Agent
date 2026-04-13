"""
Tools package for the Caterva2 agent.

Provides tool schemas and implementations for LLM-powered dataset exploration.

Public API:
- TOOLS: List of tool schemas for the LLM
- TOOL_MAP: Dict mapping tool names to implementations
- execute_tool(): Dispatch function for tool execution

Tool categories:
- browsing: list_roots, list_datasets, get_dataset_info
- analysis: get_dataset_stats, collapse_dimensions
- data_access: get_slice, where_filter, load_dataset
- dataset_management: copy_dataset, download_dataset, move_dataset, remove_dataset
- visualization: visualize_dataset, render_projection
"""

from ._registry import TOOLS, TOOL_MAP, execute_tool

__all__ = ["TOOLS", "TOOL_MAP", "execute_tool"]
