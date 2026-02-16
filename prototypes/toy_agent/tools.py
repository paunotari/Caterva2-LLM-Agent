"""
Tool definitions and implementations.

This file contains:
1. Tool schemas (how we describe tools to the LLM)
2. Tool implementations (the actual Python functions)
3. A dispatcher to route tool calls to implementations
"""

import json
from typing import Dict, Any


# TOOL SCHEMAS
# These JSON schemas tell the LLM what tools are available and how to use them
# They follow the OpenAI function calling format, which Groq also uses

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic arithmetic operations. Supports addition, subtraction, multiplication, and division.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]


# TOOL IMPLEMENTATIONS
# These are the actual Python functions that execute when the LLM calls a tool

def calculator(operation: str, a: float, b: float) -> Dict[str, Any]:
    """
    Execute a calculator operation.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    
    Returns:
        Dict with 'result' key containing the answer, or 'error' if operation failed
    """
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero"}
            result = a / b
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {"result": result}
    
    except Exception as e:
        return {"error": str(e)}


# TOOL DISPATCHER
# This maps tool names to their implementations
# When the LLM requests a tool, we look it up here and execute it

TOOL_MAP = {
    "calculator": calculator
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute (e.g., 'calculator')
        tool_args: Dictionary of arguments to pass to the tool
    
    Returns:
        JSON string containing the tool's result
    """
    if tool_name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    # Get the function from our map and call it with unpacked arguments
    tool_function = TOOL_MAP[tool_name]
    result = tool_function(**tool_args)
    
    # Return as JSON string (this is what we'll send back to the LLM)
    return json.dumps(result)
