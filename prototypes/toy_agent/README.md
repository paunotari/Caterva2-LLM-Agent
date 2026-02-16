# Basic Agent with Calculator Tool

A minimal agentic AI implementation using Groq's API. This demonstrates the fundamental agent loop pattern.

_Note: Use Python 3.12 or lower (.venv); newer versions like 3.14 are not supported by Groq's API._

## Setup

1. **Create and activate the virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1 # Windows PowerShell
```

2. **Install dependencies:**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

3. **Store your Groq API key in a .env file in the project root:**
```bash
GROQ_API_KEY=your-api-key-here
```

Get a free API key at: https://console.groq.com/

4. **Run the agent:**
```bash
python main.py
```

## Example Usage

```
You: What is 15 multiplied by 23?

[Agent Loop Iteration 1]
LLM requested 1 tool call(s)
  - Calling tool: calculator
    Arguments: {'operation': 'multiply', 'a': 15, 'b': 23}
    Result: {"result": 345}

Agent: 15 multiplied by 23 equals 345.
```

```
You: If I have $1000 and split it equally among 7 people, how much does each person get?

[Agent Loop Iteration 1]
LLM requested 1 tool call(s)
  - Calling tool: calculator
    Arguments: {'operation': 'divide', 'a': 1000, 'b': 7}
    Result: {"result": 142.85714285714286}

Agent: Each person would get approximately $142.86.
```

## Project Structure

```
basic-agent/
├── config.py      # API configuration and system prompt
├── tools.py       # Tool definitions and implementations
├── agent.py       # Core agent loop
├── main.py        # Entry point / CLI
└── requirements.txt
```

## How It Works

### The Agent Loop (agent.py)

This is the core concept. The agent follows this loop:

1. **User provides input** → Added to conversation history
2. **Call LLM** with conversation + tool definitions
3. **LLM responds** with either:
   - Final answer → Return to user (exit loop)
   - Tool call request → Go to step 4
4. **Execute tool(s)** → Add results to conversation
5. **Loop back to step 2** (LLM sees tool results and decides next action)

This continues until the LLM provides a final answer (no more tool calls).

### Message Flow Example

Here's what the conversation history looks like for "What is 5 + 3?":

```python
[
  # 1. System prompt (always first)
  {"role": "system", "content": "You are a helpful assistant..."},
  
  # 2. User asks question
  {"role": "user", "content": "What is 5 + 3?"},
  
  # 3. LLM decides to use calculator
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "function": {
          "name": "calculator",
          "arguments": '{"operation": "add", "a": 5, "b": 3}'
        }
      }
    ]
  },
  
  # 4. We execute the tool and add result
  {
    "role": "tool",
    "tool_call_id": "call_abc123",
    "name": "calculator",
    "content": '{"result": 8}'
  },
  
  # 5. LLM sees result and gives final answer
  {
    "role": "assistant",
    "content": "5 + 3 equals 8."
  }
]
```

### File Breakdown

#### config.py
- **Purpose**: Centralize configuration
- **Key elements**:
  - `client`: Groq API client instance
  - `MODEL_NAME`: Which model to use (we use the tool-use preview model)
  - `SYSTEM_PROMPT`: Instructions for the LLM's behavior

**Why separate this?** In real projects, you'll have multiple config values (API keys, model settings, timeouts, etc.). Keeping them in one place makes the code maintainable.

#### tools.py
- **Purpose**: Define and implement tools
- **Three main sections**:

1. **TOOLS list**: JSON schemas describing tools to the LLM
   - Follows OpenAI's function calling format
   - Includes name, description, parameter types
   - The LLM reads these to know what tools exist

2. **Tool implementations**: Actual Python functions
   - `calculator()`: Does the math
   - Returns structured results (dict with "result" or "error")

3. **execute_tool()**: Dispatcher function
   - Routes tool name → implementation
   - Handles errors if unknown tool requested
   - Returns JSON string (LLMs work with text)

**Why this structure?** It separates "what the LLM sees" (schemas) from "what actually runs" (implementations). You can modify implementations without changing the interface.

#### agent.py
- **Purpose**: Implements the agent loop
- **Key methods**:

1. **`__init__()`**: Sets up conversation history
   - Starts with system prompt
   - History persists across turns (the agent "remembers")

2. **`run(user_input)`**: The agent loop
   - Adds user message to history
   - Loop: Call LLM → Check for tool calls → Execute tools → Repeat
   - Returns when LLM gives final answer
   - `max_iterations` prevents infinite loops

3. **`reset()`**: Clears conversation history
   - Useful for starting fresh conversations

**Critical details**:
- Tool results get added to `messages` so the LLM sees them
- We append the assistant's response even when it contains tool calls
- The loop continues until `assistant_message.tool_calls` is None/empty

#### main.py
- **Purpose**: User interface / entry point
- **What it does**:
  - Creates agent instance
  - Provides interactive prompt
  - Handles commands (quit, reset)
  - Catches errors gracefully

**Why separate from agent.py?** You might want different interfaces (web API, GUI, batch processing). The `Agent` class is reusable across all of them.

## Key Concepts to Understand

### 1. Tool Schemas
The LLM doesn't "know" what tools exist. We tell it via JSON schemas:

```python
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Performs basic arithmetic...",  # LLM reads this
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": ["add", "subtract", ...]  # LLM must pick from these
        },
        ...
      }
    }
  }
}
```

**Good descriptions matter**: The LLM uses `description` fields to decide when to use the tool.

### 2. The tool_choice Parameter
```python
response = client.chat.completions.create(
    ...
    tool_choice="auto"  # Options: "auto", "none", {"type": "function", "function": {"name": "calculator"}}
)
```

- `"auto"`: LLM decides whether to use tools
- `"none"`: LLM never uses tools
- Specific function: Force the LLM to use that tool

### 3. Conversation State
Everything is in `self.messages`. The LLM is stateless - it only sees what's in this list.

**Implication**: Long conversations can hit token limits. Real agents need conversation management strategies:
- Summarization
- Sliding windows
- Pruning old messages

### 4. Tool Call IDs
```python
tool_call_id = tool_call.id
```

Each tool call gets a unique ID. When we return results, we must reference this ID:

```python
{
  "role": "tool",
  "tool_call_id": tool_call_id,  # Links result to original request
  "content": result
}
```

This lets the LLM handle multiple simultaneous tool calls correctly.

## Common Patterns You'll Use

### Adding More Tools
1. Add schema to `TOOLS` list in `tools.py`
2. Write implementation function
3. Add to `TOOL_MAP` dictionary

Example - add a "get_time" tool:

```python
# In TOOLS list:
{
  "type": "function",
  "function": {
    "name": "get_time",
    "description": "Get the current time",
    "parameters": {"type": "object", "properties": {}}
  }
}

# Implementation:
def get_time() -> Dict[str, Any]:
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

# In TOOL_MAP:
TOOL_MAP = {
    "calculator": calculator,
    "get_time": get_time
}
```

### Handling Tool Errors
Our calculator already does this:

```python
if b == 0:
    return {"error": "Cannot divide by zero"}
```

The LLM will see this error and can respond appropriately to the user.

### Multi-Step Reasoning
The LLM might use multiple tools in sequence:

```
User: "What's 5 + 3, then multiply that by 2?"

Iteration 1: Call calculator(add, 5, 3) → 8
Iteration 2: Call calculator(multiply, 8, 2) → 16
Final: "The result is 16"
```

The agent loop handles this automatically.

## Limitations & Next Steps

### Current Limitations
- No streaming (responses appear all at once)
- No error recovery (if tool fails, might confuse LLM)
- No conversation memory management (will hit token limits eventually)
- Simple CLI only

### Improvements for Real Projects
1. **Streaming**: Show LLM responses as they generate
2. **Async execution**: Run multiple tools concurrently
3. **Error handling**: Retry logic, fallbacks
4. **Logging**: Track all tool calls for debugging
5. **Rate limiting**: Respect API limits
6. **Conversation management**: Summarize or prune old messages
7. **Tool validation**: Validate LLM's tool arguments before execution

## Debugging Tips

1. **Print conversation history**:
```python
import json
print(json.dumps(self.messages, indent=2))
```

2. **Watch tool calls**: The agent already prints these during execution

3. **Check API responses**: Add logging in the agent loop

4. **Test tools independently**:

```python
from prototypes.tools import calculator

result = calculator("add", 5, 3)
print(result)  # Should print: {"result": 8}
```

## Questions to Explore

1. What happens if you remove the system prompt?
2. What if you set `max_iterations = 1`?
3. Can the LLM chain multiple tool calls together?
4. How does the LLM behave with poor tool descriptions?
5. What happens if a tool returns invalid JSON?

Experiment to build intuition!
