"""
Jupyter notebook interface for the Caterva2 agent.

This module provides a cell-based interaction model where:
- Users call `ask("message")` to interact with the agent
- The agent injects fetched datasets into the notebook namespace
- Users can modify variables with their own code
- The agent can read user variables via {variable} syntax

Key functions:
- ask(message): Send a question to the agent, display response
- reset(): Clear agent memory (start fresh conversation)
- variables(): List agent-injected variables

Variable Reference Syntax:
- Use {variable_name} to reference a local variable
- Example: ask("Compute stats on {my_data}")
- The agent will see the variable's metadata (shape, dtype, sample values)

The agent and user share the same Python namespace — this enables
a professional workflow where users mix natural-language exploration
with custom data transformations.
"""

import re
from typing import Any

from IPython import get_ipython
from IPython.display import display, Markdown

from caterva2_agent.agent import Agent
from caterva2_agent.tools._base import pop_fetched_objects, set_notebook_namespace


# ---------------------------------------------------------------------------
# MODULE STATE
# ---------------------------------------------------------------------------

# The agent instance — created on first ask() call or explicitly via reset()
_agent: Agent | None = None

# Registry of objects the agent has fetched, keyed by variable name
# Format: {"temp_data": <ndarray>, "gaia_slice": <ndarray>, ...}
_agent_objects: dict[str, Any] = {}

# Input length limit (same as main.py)
MAX_INPUT_CHARS = 5000

# Pattern for {variable} references in user messages
VARIABLE_REFERENCE_PATTERN = re.compile(r'\{(\w+)}')


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
    
    If 'temp_data' exists, returns 'temp_data_1', 'temp_data_2', etc.
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
# VARIABLE REFERENCE EXPANSION
# ---------------------------------------------------------------------------

def _describe_variable(name: str, value: Any) -> str:
    """
    Generate a description of a variable for the agent.
    
    Includes metadata that helps the agent understand and operate on the data.
    """
    parts = [f"Local variable '{name}'"]
    
    if hasattr(value, 'shape'):
        parts.append(f"shape={value.shape}")
    if hasattr(value, 'dtype'):
        parts.append(f"dtype={value.dtype}")
    if hasattr(value, '__len__'):
        parts.append(f"len={len(value)}")
    
    # Add sample statistics for numeric arrays
    if hasattr(value, 'min') and hasattr(value, 'max') and hasattr(value, 'mean'):
        try:
            parts.append(f"min={float(value.min()):.4g}")
            parts.append(f"max={float(value.max()):.4g}")
            parts.append(f"mean={float(value.mean()):.4g}")
        except (TypeError, ValueError):
            pass  # Non-numeric array
    
    return ", ".join(parts)


def _expand_variable_references(message: str, namespace: dict) -> tuple[str, list[str]]:
    """
    Expand {variable} references in a user's ask() message.
    
    Finds all {var} patterns, looks up each variable in the namespace,
    and builds context strings describing each referenced variable.
    
    Args:
        message: The user's ask() message with {variable} references
        namespace: The notebook namespace to look up variables
    
    Returns:
        Tuple of (processed_message, list_of_variable_descriptions)
        - processed_message: Original message with {var} → var (braces removed)
        - list_of_variable_descriptions: Metadata about each referenced variable
    
    Raises:
        ValueError: If a referenced variable doesn't exist or isn't array-like
    """
    referenced_vars = VARIABLE_REFERENCE_PATTERN.findall(message)
    
    if not referenced_vars:
        return message, []
    
    descriptions = []
    
    for var_name in referenced_vars:
        if var_name not in namespace:
            raise ValueError(
                f"Variable '{var_name}' not found. "
                f"Check the name and make sure it's defined in your notebook."
            )
        
        value = namespace[var_name]
        
        # Validate it's array-like (tools can only operate on arrays)
        if not (hasattr(value, 'shape') and hasattr(value, 'dtype')):
            raise ValueError(
                f"Variable '{var_name}' is not an array "
                f"(type: {type(value).__name__}). "
                f"Only numpy arrays and similar objects can be referenced."
            )
        
        descriptions.append(_describe_variable(var_name, value))
    
    # Remove braces from variable references in the message
    processed_message = VARIABLE_REFERENCE_PATTERN.sub(r'\1', message)
    
    return processed_message, descriptions


# ---------------------------------------------------------------------------
# AGENT INTERFACE
# ---------------------------------------------------------------------------

def _get_agent() -> Agent:
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        _agent = Agent()
    # Type checker: _agent is Agent here (never None after this point)
    assert _agent is not None
    return _agent


def ask(message: str) -> None:
    """
    Ask the agent a question about your datasets and display its response.
    
    This is the main interaction function. The agent will:
    1. Process your question using its tools
    2. Fetch any needed data from the Caterva2 server
    3. Inject fetched datasets into the notebook namespace
    4. Display the response with Markdown formatting
    
    Variable References:
        Use {variable_name} to reference a local variable.
        The agent can then operate on that variable (stats, slices, etc.)
    
    Args:
        message: Your question or request in natural language
    
    Example:
        1. ask("Show me the temperature dataset")
           # Agent fetches from server, injects as `temp_data`
        
        2. User transforms the data (in your notebook cell):
           my_temps = temp_data * 1.8 + 32
        
        3. ask("Compute stats on {my_temps}")
           # Agent operates on the user's local variable
    """
    message = message.strip()
    
    if not message:
        print("[No input provided]")
        return
    
    if len(message) > MAX_INPUT_CHARS:
        print(f"[Input too long: {len(message)} chars. Max is {MAX_INPUT_CHARS}]")
        return
    
    if len(message) > 2000:
        print(f"[Note: Long input ({len(message)} chars) — may take longer]")
    
    # Get namespace for variable expansion and tool access
    namespace = _get_notebook_namespace() or {}
    
    # Expand {variable} references and get descriptions
    try:
        processed_message, var_descriptions = _expand_variable_references(message, namespace)
    except ValueError as e:
        print(f"[Error: {e}]")
        return
    
    # If variables were referenced, prepend their descriptions to the message
    if var_descriptions:
        context = "Referenced variables:\n" + "\n".join(f"- {d}" for d in var_descriptions)
        processed_message = f"{context}\n\nUser request: {processed_message}"
    
    # Set namespace so tools can access local variables
    set_notebook_namespace(namespace)
    
    agent = _get_agent()
    
    try:
        response = agent.run(processed_message)
        
        # Inject any fetched data into the notebook namespace
        fetched = pop_fetched_objects()
        injected_names = []
        for path, data in fetched.items():
            var_name = inject_variable(path, data)
            injected_names.append(var_name)
        
        # Notify user about injected variables
        if injected_names:
            print(f"📦 Data available as: {', '.join(f'`{n}`' for n in injected_names)}")
        
        # Display response with Markdown formatting in Jupyter
        _display_response(response)
        
    except Exception as e:
        error_msg = f"[Error: {type(e).__name__}: {e}]"
        print(error_msg)
        print("Try again or call reset() to clear the conversation.")
        # Auto-reset on unhandled exception (same as main.py)
        agent.reset()


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


def variables() -> None:
    """
    Show all variables the agent has injected into the namespace.
    
    Useful to see what data is available for your own code.
    """
    injected = list_injected_variables()
    
    if not injected:
        _display_response("No variables injected yet. Use `ask()` to fetch some data.")
        return

    rows = "\n".join(f"| `{name}` | {desc} |" for name, desc in injected.items())
    md = (
        "### 📊 Agent-injected variables\n\n"
        "| Variable | Description |\n"
        "|---|---|\n"
        f"{rows}\n\n"
        "You can use these in your own code."
    )
    _display_response(md)


# ---------------------------------------------------------------------------
# DISPLAY HELPERS (for richer notebook output)
# ---------------------------------------------------------------------------

def _display_response(response: str) -> None:
    """
    Display an agent response with Markdown rendering.
    
    Falls back to plain print if not in Jupyter.
    """
    try:
        display(Markdown(response))
    except (RuntimeError, ValueError, AttributeError):
        # display() is unavailable or failed (not in Jupyter, or rendering error)
        print(response)


# ---------------------------------------------------------------------------
# CONVENIENCE ALIASES
# ---------------------------------------------------------------------------

# Shorter aliases for common operations
chat = ask  # Backwards compatibility alias for "chat"
clear = reset  # Alias for those who expect "clear" to reset
