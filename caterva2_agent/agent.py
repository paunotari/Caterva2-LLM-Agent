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
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from typing import Any

from caterva2_agent.config import client, MODEL_NAME, SYSTEM_PROMPT
from caterva2_agent.tools import TOOLS, execute_tool

# Setup logging: project root logs/ directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, 'agent.log')

logger = logging.getLogger('caterva2_agent')
logger.setLevel(logging.DEBUG)
logger.propagate = False

# File handler with rotation (1 MB per file, keep 5 backups)
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(file_handler)

MAX_TOOL_RESULT_TEXT_CHARS = 4000
SENSITIVE_TOOL_FIELD_NAMES = {"image", "image_base64", "binary_blob"}
SENSITIVE_TOOL_FIELD_HINTS = ("base64", "data_uri", "binary")


def _is_sensitive_tool_field(field_name: str, value: Any) -> bool:
    """Detect payload fields that should not be sent back into the LLM context."""
    key = field_name.lower()
    if key in SENSITIVE_TOOL_FIELD_NAMES:
        return True
    if any(hint in key for hint in SENSITIVE_TOOL_FIELD_HINTS):
        return True
    return key == "image" and isinstance(value, str) and value.startswith("data:image/")


def _sanitize_tool_payload_for_llm(
    payload: Any,
    *,
    path: str = "",
    redactions: list[dict[str, Any]] | None = None,
) -> Any:
    """Strip oversized/binary fields from tool payloads before storing in chat history."""
    redactions = redactions if redactions is not None else []

    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            field_path = f"{path}.{key}" if path else key
            if _is_sensitive_tool_field(key, value) and isinstance(value, str):
                redactions.append(
                    {
                        "field": field_path,
                        "reason": "binary_or_data_uri",
                        "chars": len(value),
                    }
                )
                if key == "image":
                    sanitized["image_available_in_notebook"] = True
                # Omit sensitive payload fields entirely so the LLM does not try to
                # reference placeholder URLs (which can render as broken images).
                continue

            sanitized[key] = _sanitize_tool_payload_for_llm(
                value,
                path=field_path,
                redactions=redactions,
            )
        return sanitized

    if isinstance(payload, list):
        return [
            _sanitize_tool_payload_for_llm(
                item,
                path=f"{path}[{index}]",
                redactions=redactions,
            )
            for index, item in enumerate(payload)
        ]

    if isinstance(payload, str) and len(payload) > MAX_TOOL_RESULT_TEXT_CHARS:
        redactions.append(
            {
                "field": path or "<string>",
                "reason": "oversized_text",
                "chars": len(payload),
            }
        )
        overflow = len(payload) - MAX_TOOL_RESULT_TEXT_CHARS
        return (
            f"{payload[:MAX_TOOL_RESULT_TEXT_CHARS]}"
            f"... [truncated {overflow} chars for LLM context]"
        )

    return payload


def _sanitize_tool_result_for_llm(tool_name: str, tool_result: str) -> tuple[str, Any | None]:
    """
    Prepare tool output for LLM history while preserving the raw parsed result.

    Returns:
        (sanitized_json_for_llm, parsed_raw_payload_or_none)
    """
    try:
        parsed = json.loads(tool_result)
    except json.JSONDecodeError:
        return tool_result, None

    redactions: list[dict[str, Any]] = []
    sanitized = _sanitize_tool_payload_for_llm(parsed, redactions=redactions)

    if redactions and isinstance(sanitized, dict):
        sanitized["_llm_sanitization"] = {
            "tool": tool_name,
            "redacted_fields": redactions,
        }
        logger.debug(
            "Sanitized tool payload for LLM context: %s",
            json.dumps({"tool": tool_name, "redactions": redactions}),
        )

    return json.dumps(sanitized), parsed


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
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.total_tokens_used = 0
        self.max_total_tokens = 2_000_000  # Cost safety limit per conversation
        # Context window management: keep only the last N messages plus system prompt
        # This is a simple heuristic; more sophisticated approaches if needed (e.g., summarization, etc.).
        self.max_history_messages = 20  # Tune as needed
        # Raw tool outputs from the latest run; used by notebook integration for rich artifacts.
        self.last_tool_results: list[dict[str, Any]] = []

    def _token_usage_header(self) -> str:
        """Build a professional Markdown token usage summary for reply display."""
        return (
            "```\n"
            "Session Token Usage\n"
            f"├─ Input:  {self.prompt_tokens_used:,}\n"
            f"└─ Output: {self.completion_tokens_used:,}\n"
            "```\n\n"
            "---"
        )

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
                f"{self._token_usage_header()}\n\n"
                f"[Token limit reached: {self.total_tokens_used} tokens used. "
                f"Please type 'reset' to start a new conversation.]"
            )

        # Add the user's message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        self.last_tool_results = []

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
                usage = response.usage
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

                self.prompt_tokens_used += prompt_tokens
                self.completion_tokens_used += completion_tokens

                if prompt_tokens or completion_tokens:
                    tokens_this_call = prompt_tokens + completion_tokens
                    self.total_tokens_used = self.prompt_tokens_used + self.completion_tokens_used
                else:
                    # Fallback for providers/mocks that expose only total_tokens.
                    tokens_this_call = total_tokens
                    self.prompt_tokens_used += total_tokens
                    self.total_tokens_used += total_tokens

                logger.info(
                    "Iteration %s tokens | input=%s output=%s total=%s",
                    iteration,
                    prompt_tokens,
                    completion_tokens,
                    self.total_tokens_used,
                )

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
                final_answer = assistant_message.content or "[No response from LLM]"
                return f"{self._token_usage_header()}\n\n{final_answer}"

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
                sanitized_result, parsed_result = _sanitize_tool_result_for_llm(
                    tool_name, tool_result
                )
                self.last_tool_results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": parsed_result,
                    }
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": sanitized_result
                })

            # Loop back: LLM will see the tool results and decide the next step

        logger.info("----- Agent Execution Complete (max iterations reached) -----\n")
        return (
            f"{self._token_usage_header()}\n\n"
            "[Max iterations reached. Please try rephrasing your question.]"
        )

    def reset(self):
        """Clear conversation history (keeping only the system prompt) and reset token counter."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.total_tokens_used = 0
        self.last_tool_results = []
        logger.info("Conversation and token counter reset")
