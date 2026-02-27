"""
Core agent loop for the Caterva2 dataset exploration agent.

The loop works as follows:
1. User message is appended to conversation history
2. LLM is called with the current history and available tools
3. If the LLM requests tool calls → execute them, append results, go to step 2
4. If the LLM gives a final answer (no tool calls) → return it to the user

This is the same pattern as the toy calculator agent, extended with:
- A Caterva2-specific system prompt
- Three dataset exploration tools instead of a calculator
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from config import client, MODEL_NAME, SYSTEM_PROMPT
from tools import TOOLS, execute_tool

import os
# Robust log path: project root if possible, else CWD (works in scripts and Jupyter)
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_PATH = os.path.join(PROJECT_ROOT, 'agent.log')
except NameError:
    LOG_PATH = os.path.join(os.getcwd(), 'agent.log')
_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

# Use a named logger for our code only — avoids capturing httpx/groq DEBUG noise
logger = logging.getLogger('caterva2_agent')
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
logger.propagate = False  # Don't bubble up to root logger

print(f"[Logging to: {LOG_PATH}]")  # Shows log location for all environments


class Agent:
    """
    A dataset exploration agent that uses Caterva2 tools to answer user questions
    about scientific datasets hosted on a Caterva2 subscriber.
    """

    def __init__(self):
        """Initialize agent with system prompt and empty conversation history."""
        # Conversation history: system prompt + all user/assistant/tool messages
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        # Safety guardrails — same as toy agent
        self.max_iterations = 10     # Prevent infinite tool-call loops
        self.total_tokens_used = 0
        self.max_total_tokens = 50000  # Cost safety limit per conversation
        # Context window management: keep only the last N messages plus system prompt
        # This is a simple heuristic; more sophisticated approaches could be implemented if needed (e.g., summarization, relevance scoring).
        self.max_history_messages = 20  # Tune as needed

    def _get_trimmed_history(self):
        """Return system prompt + last max_history_messages messages (for LLM context)."""
        if not self.messages:
            return []
        system = self.messages[0]
        recent = self.messages[1:][-self.max_history_messages:]
        return [system] + recent

    def run(self, user_input: str) -> str:
        """
        Process a user message and return the agent's final response.

        Runs the agent loop: appends user message → calls LLM → executes any
        tool calls → repeats until LLM gives a final answer.

        Args:
            user_input: The user's natural-language question or request

        Returns:
            The agent's final response as a string
        """
        # Enforce token budget before starting
        if self.total_tokens_used > self.max_total_tokens:
            return (
                f"[Token limit reached: {self.total_tokens_used} tokens used. "
                f"Please type 'reset' to start a new conversation.]"
            )

        # Add the user's message to conversation history
        self.messages.append({"role": "user", "content": user_input})

        iteration = 0
        logger.info(f"===== New Agent Run | User: {user_input} =====")
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"----- Iteration {iteration} -----")

            # Call the LLM with the current conversation and tool definitions
            # [PROVIDER: GroqCloud] — tool schema format and response structure
            # follow OpenAI's function calling spec
            trimmed_history = self._get_trimmed_history()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=trimmed_history,
                tools=TOOLS,
                tool_choice="auto",   # LLM decides whether to call a tool
                temperature=0.2,      # Low temperature: factual/precise for dataset queries
                max_tokens=1024
            )

            # Track token usage for cost and safety monitoring
            if hasattr(response, "usage"):
                tokens_this_call = response.usage.total_tokens
                self.total_tokens_used += tokens_this_call
                print(f"[Iteration {iteration}: {tokens_this_call} tokens | {self.total_tokens_used} total]")
                logger.info(f"Iteration {iteration}: {tokens_this_call} tokens | {self.total_tokens_used} total")

            assistant_message = response.choices[0].message

            # Log tool names only — raw Pydantic repr is unreadable
            if assistant_message.tool_calls:
                names = [tc.function.name for tc in assistant_message.tool_calls]
                logger.debug(f"LLM requested tool(s): {names}")
            else:
                logger.debug("LLM gave final answer (no tool calls)")

            # --- No tool calls: this is the final answer ---
            if not assistant_message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                logger.info("----- Agent Execution Complete -----\n")
                return "\n" + assistant_message.content or "[No response from LLM]"

            # --- Tool calls: execute each one and feed results back ---
            # Serialize tool_calls with .model_dump() before storing in history.
            # The raw Pydantic objects are not JSON-safe for the API.
            serialized_tool_calls = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in assistant_message.tool_calls
            ]
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": serialized_tool_calls
            })

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    logger.debug(f"Tool: {tool_name} | Args: {tool_args}")
                    tool_result = execute_tool(tool_name, tool_args)
                except json.JSONDecodeError as e:
                    tool_result = json.dumps({"error": f"Invalid JSON in tool arguments: {e}"})

                # Pretty-print tool result if JSON
                try:
                    parsed = json.loads(tool_result)
                    pretty = json.dumps(parsed, indent=2)
                    logger.debug(f"Tool result:\n{pretty}")
                except Exception:
                    logger.debug(f"Tool result: {tool_result}")

                # Append tool result — the LLM reads this in the next iteration
                # role="tool" with matching tool_call_id is required by the API
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })

            # Loop back: LLM will see the tool results and decide the next step

        logger.info("----- Agent Execution Complete (max iterations reached) -----\n")
        return "[Max iterations reached. Please try rephrasing your question.]"

    def reset(self):
        """Clear conversation history (keeping only the system prompt) and reset token counter."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.total_tokens_used = 0
        logger.info("Conversation and token counter reset")
        print("[Conversation and token counter reset]")
