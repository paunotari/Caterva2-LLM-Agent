"""
Caterva2 LLM Agent package.

An LLM-powered agent for exploring and analyzing Caterva2/Blosc2/HDF5 datasets
through natural language.

Main components:
- Agent: The core agent loop that handles conversation and tool execution
- tools: Tool schemas and implementations for dataset operations
- config: LLM and Caterva2 server configuration
- prompts: System prompts for agent behavior
"""

from caterva2_agent.agent import Agent

__version__ = "0.2.0"
__all__ = ["Agent"]
