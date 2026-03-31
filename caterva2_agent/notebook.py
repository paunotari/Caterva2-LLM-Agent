"""
Jupyter notebook interface for the Caterva2 agent.

This module provides a cell-based interaction model where:
- Users call `chat("message")` to interact with the agent
- The agent injects fetched datasets into the notebook namespace
- Users can modify variables with their own code
- The agent can read modified variables on subsequent calls

Key functions:
- chat(message): Send a message to the agent, return response
- reset(): Clear agent memory (start fresh conversation)
- history(): Display conversation history

The agent and user share the same Python namespace — this enables
a professional workflow where users mix natural-language exploration
with custom data transformations.
"""

import re
from typing import Any

from IPython import get_ipython
from IPython.display import display, Markdown, HTML

from caterva2_agent.agent import Agent
from caterva2_agent.tools._base import pop_fetched_objects


# ---------------------------------------------------------------------------
# MODULE STATE
# ---------------------------------------------------------------------------

# The agent instance — created on first chat() call or explicitly via reset()
_agent: Agent | None = None

# Registry of objects the agent has fetched, keyed by variable name
# Format: {"temperature": <ndarray>, "gaia_slice": <ndarray>, ...}
_agent_objects: dict[str, Any] = {}

# Input length limit (same as main.py)
MAX_INPUT_CHARS = 5000


# ---------------------------------------------------------------------------
# NAMESPACE MANAGEMENT
# ---------------------------------------------------------------------------

def _get_notebook_namespace() -> dict | None:
    """
    Get the notebook's user namespace for variable injection.
    
    Returns None if not running in IPython/Jupyter.
    """
    ip = get_ipython()
    if ip is None:
        return None
    return ip.user_ns


def _sanitize_variable_name(name: str) -> str:
    """
    Convert a dataset path/name to a valid Python identifier.
    
    Examples:
        '@public/examples/ds-1d.b2nd' → 'ds_1d'
        'temperature_data.b2nd'       → 'temperature_data'
        '3d-array.b2nd'               → 'array_3d'  (can't start with digit)
    """
    # Extract the filename (last component of path)
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    
    # Remove extension
    if "." in name:
        name = name.rsplit(".", 1)[0]
    
    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    
    # Strip leading/trailing underscores
    name = name.strip("_")
    
    # Ensure it doesn't start with a digit
    if name and name[0].isdigit():
        name = "var_" + name
    
    # Fallback if empty
    if not name:
        name = "dataset"
    
    return name


def _unique_name(base_name: str, namespace: dict) -> str:
    """
    Generate a unique variable name that doesn't collide with existing names.
    
    If 'temperature' exists, returns 'temperature_1', 'temperature_2', etc.
    """
    if base_name not in namespace and base_name not in _agent_objects:
        return base_name
    
    counter = 1
    while True:
        candidate = f"{base_name}_{counter}"
        if candidate not in namespace and candidate not in _agent_objects:
            return candidate
        counter += 1


def inject_variable(name: str, value: Any) -> str:
    """
    Inject a variable into the notebook namespace.
    
    Also registers in _agent_objects so the agent can track what it has created.
    
    Args:
        name: Base name for the variable (will be sanitized and made unique)
        value: The value to inject (typically a numpy array)
    
    Returns:
        The actual variable name used (after sanitization and uniqueness)
    """
    namespace = _get_notebook_namespace()
    if namespace is None:
        # Not in Jupyter — just track internally
        sanitized = _sanitize_variable_name(name)
        _agent_objects[sanitized] = value
        return sanitized
    
    # Sanitize and ensure uniqueness
    sanitized = _sanitize_variable_name(name)
    final_name = _unique_name(sanitized, namespace)
    
    # Inject into notebook namespace
    namespace[final_name] = value
    
    # Track in agent registry
    _agent_objects[final_name] = value
    
    return final_name


def read_variable(name: str) -> Any | None:
    """
    Read a variable from the notebook namespace.
    
    Checks the notebook namespace first (user may have modified it),
    then falls back to the agent registry.
    
    Args:
        name: Variable name to read
    
    Returns:
        The variable value, or None if not found
    """
    namespace = _get_notebook_namespace()
    
    # Prefer notebook namespace (user may have modified the variable)
    if namespace is not None and name in namespace:
        return namespace[name]
    
    # Fallback to agent registry
    return _agent_objects.get(name)


def list_injected_variables() -> dict[str, str]:
    """
    List all variables the agent has injected.
    
    Returns:
        Dict mapping variable names to a brief description (type, shape if array)
    """
    result = {}
    namespace = _get_notebook_namespace() or {}
    
    for name in _agent_objects:
        # Get current value (may have been modified by user)
        value = namespace.get(name, _agent_objects[name])
        
        # Generate description
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            desc = f"array {value.shape} {value.dtype}"
        else:
            desc = type(value).__name__
        
        result[name] = desc
    
    return result


# ---------------------------------------------------------------------------
# AGENT INTERFACE
# ---------------------------------------------------------------------------

def _get_agent() -> Agent:
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def chat(message: str) -> str:
    """
    Send a message to the agent and return its response.
    
    This is the main interaction function. The agent will:
    1. Process your message using its tools
    2. Fetch any needed data from the Caterva2 server
    3. Inject fetched datasets into the notebook namespace
    4. Return a natural-language response
    
    After this call, any fetched datasets are available as variables
    in your notebook. The agent will tell you the variable names.
    
    Args:
        message: Your question or request in natural language
    
    Returns:
        The agent's response as a string
    
    Example:
        >>> response = chat("Show me the temperature dataset")
        >>> print(response)
        "I found the temperature dataset. It's a 1000x500 float32 array.
         The data is available as `temperature` in your notebook."
        >>> temperature.shape  # Now you can use it!
        (1000, 500)
    """
    message = message.strip()
    
    if not message:
        return "[No input provided]"
    
    if len(message) > MAX_INPUT_CHARS:
        return f"[Input too long: {len(message)} chars. Max is {MAX_INPUT_CHARS}]"
    
    if len(message) > 2000:
        print(f"[Note: Long input ({len(message)} chars) — may take longer]")
    
    agent = _get_agent()
    
    try:
        response = agent.run(message)
        
        # Inject any fetched data into the notebook namespace
        fetched = pop_fetched_objects()
        injected_names = []
        for path, data in fetched.items():
            var_name = inject_variable(path, data)
            injected_names.append(var_name)
        
        # Notify user about injected variables
        if injected_names:
            print(f"📦 Data available as: {', '.join(f'`{n}`' for n in injected_names)}")
        
        return response
    except Exception as e:
        error_msg = f"[Error: {type(e).__name__}: {e}]"
        print(error_msg)
        print("Try again or call reset() to clear the conversation.")
        # Auto-reset on unhandled exception (same as main.py)
        agent.reset()
        return error_msg


def reset() -> None:
    """
    Reset the agent's conversation memory.
    
    Clears:
    - Conversation history (agent forgets previous messages)
    - Token counter (resets to 0)
    
    Does NOT clear:
    - Injected variables (they remain in your namespace)
    - The _agent_objects registry
    
    Use this when you want to start a fresh conversation.
    """
    global _agent
    if _agent is not None:
        _agent.reset()
        print("🔄 Agent memory reset. Start a new conversation.")
    else:
        print("🔄 No active agent to reset.")


def history() -> None:
    """
    Display the conversation history.
    
    Shows all messages exchanged with the agent in the current session.
    Tool calls and results are summarized for readability.
    """
    agent = _get_agent()
    
    if len(agent.messages) <= 1:  # Only system prompt
        print("No conversation history yet. Use chat() to start.")
        return
    
    print("=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)
    
    for msg in agent.messages[1:]:  # Skip system prompt
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            print(f"\n👤 You: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tool_names = [tc.get("function", {}).get("name", "?") 
                              for tc in msg.get("tool_calls", [])]
                print(f"\n🤖 Agent: [Called tools: {', '.join(tool_names)}]")
            if content:
                # Truncate long responses for display
                if len(content) > 500:
                    content = content[:500] + "..."
                print(f"\n🤖 Agent: {content}")
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            # Don't show full tool results — too verbose
            print(f"   └─ Tool result from {tool_name}")
    
    print("\n" + "=" * 60)
    print(f"Token usage: {agent.total_tokens_used:,} / {agent.max_total_tokens:,}")
    print("=" * 60)


def variables() -> None:
    """
    Show all variables the agent has injected into the namespace.
    
    Useful to see what data is available for your own code.
    """
    injected = list_injected_variables()
    
    if not injected:
        print("No variables injected yet. Use chat() to fetch some data.")
        return
    
    print("📊 Agent-injected variables:")
    print("-" * 40)
    for name, desc in injected.items():
        print(f"  {name}: {desc}")
    print("-" * 40)
    print("You can use these in your own code.")


# ---------------------------------------------------------------------------
# DISPLAY HELPERS (for richer notebook output)
# ---------------------------------------------------------------------------

def _display_response(response: str) -> None:
    """
    Display an agent response with markdown rendering.
    
    Falls back to plain print if not in Jupyter.
    """
    try:
        display(Markdown(response))
    except Exception:
        print(response)


# ---------------------------------------------------------------------------
# CONVENIENCE ALIASES
# ---------------------------------------------------------------------------

# Shorter aliases for common operations
ask = chat  # Some users may prefer "ask" over "chat"
clear = reset  # Alias for those who expect "clear" to reset
