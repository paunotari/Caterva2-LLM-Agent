"""
Core agent loop for the Caterva2 dataset exploration agent.

The loop works as follows:
1. User message is appended to conversation history
2. LLM is called with the current history and available tools
3. If the LLM requests tool calls → execute them, append results, go to step 2
4. If the LLM gives a final answer (no tool calls) → return it to the user

"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from typing import Any
import os

from caterva2_agent.config import client, MODEL_NAME, SYSTEM_PROMPT
from caterva2_agent.tools import TOOLS, execute_tool

# Robust log path: project root if possible, else CWD (works in scripts and Jupyter)
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)  # Create logs/ if it doesn't exist
    LOG_PATH = os.path.join(LOGS_DIR, 'agent.log')
except NameError:
    # Fallback for interactive environments (Jupyter, REPL)
    LOGS_DIR = os.path.join(os.getcwd(), 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOGS_DIR, 'agent.log')
_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

# Use a named logger for our code only — avoids capturing httpx/groq DEBUG noise
logger = logging.getLogger('caterva2_agent')
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
logger.propagate = True  # Let root logger handle console output for JupyterLab

def _run_tool(tool_call) -> tuple:
    """
    Execute a single tool call and return (tool_call_id, tool_name, tool_result).
    Module-level helper (not a method) because it needs no instance state —
    only logger and execute_tool, both available at module scope.
    Used by the agent loop to run tool calls in parallel via ThreadPoolExecutor.
    """
    t_name = tool_call.function.name
    try:
        t_args = json.loads(tool_call.function.arguments or "{}")
        logger.debug(f"Tool: {t_name} | Args: {t_args}")
        t_result = execute_tool(t_name, t_args)
    except json.JSONDecodeError as e:
        t_result = json.dumps({"error": f"Invalid JSON in tool arguments: {e}"})
    except Exception as e:
        # Catches any unexpected exception that escapes execute_tool()
        # Returns an error result rather than crashing the whole agent loop
        logger.error(f"Unexpected error executing tool '{t_name}': {e}")
        t_result = json.dumps({"error": f"Tool execution failed: {e}"})

    try:
        pretty = json.dumps(json.loads(t_result), indent=2)
        logger.debug(f"Tool result:\n{pretty}")
    except Exception:
        logger.debug(f"Tool result: {t_result}")

    return tool_call.id, t_name, t_result


class Agent:
    """
    A dataset exploration agent that uses Caterva2 tools to answer user questions
    about scientific datasets hosted on a Caterva2 subscriber.
    """

    def __init__(self):
        """Initialize agent with system prompt and empty conversation history."""
        # Conversation history: system prompt + all user/assistant/tool messages
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        # Safety guardrails
        self.max_iterations = 10     # Prevent infinite tool-call loops
        self.total_tokens_used = 0
        self.max_total_tokens = 50000  # Cost safety limit per conversation
        # Context window management: keep only the last N messages plus system prompt
        # This is a simple heuristic; more sophisticated approaches if needed (e.g., summarization, etc.).
        self.max_history_messages = 20  # Tune as needed

    @staticmethod
    def _call_llm_with_retry(**kwargs) -> Any | None:
        """Call the LLM with exponential backoff retry on transient errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        return None

    def _get_trimmed_history(self):
        """Return system prompt + last max_history_messages, heuristic, there's better sol like summarization"""
        if not self.messages:
            return []
        system = self.messages[0]
        recent = self.messages[1:][-self.max_history_messages:]
        return [system] + recent

    def run(self, user_input: str) -> str:
        """
        Process a user message and return the agent's final response.

        Runs the agent ReAct loop: appends user message → calls LLM → executes any
        tool calls → repeats until LLM gives a final answer. ReAct Pattern

        Args:
            user_input: The user's natural-language question or request

        Returns:
            The agent's final response as a string
            Other formats like visual plots may come from the execution of tools, but from the agent (LLM) itself - strings
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
            # [PROVIDER: GroqCloud] — tool schema format and response structure follows OpenAI's function calling spec
            trimmed_history = self._get_trimmed_history()
            # Resilience & Retry Logic
            response = self._call_llm_with_retry(
                model=MODEL_NAME,
                messages=trimmed_history,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
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

            # --- Tool calls: execute in parallel ---
            # All tool calls within a single LLM response are independent by design —
            # dependent calls always arrive in separate LLM turns.
            # ThreadPoolExecutor is used (not asyncio) because the Caterva2 library
            # is synchronous; threads are the right tool for parallel I/O-bound work.
            serialized_tool_calls = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in assistant_message.tool_calls
            ]
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": serialized_tool_calls
            })

            # Submit all tool calls at once — they run in parallel across threads
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_run_tool, tc) for tc in assistant_message.tool_calls]

            # Collect results in original order — futures[i] matches tool_calls[i]
            # Order matters: the API expects results in the same sequence as the requests
            for future in futures:
                tool_call_id, tool_name, tool_result = future.result()
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
