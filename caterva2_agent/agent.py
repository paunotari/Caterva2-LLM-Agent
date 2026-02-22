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
from config import client, MODEL_NAME, SYSTEM_PROMPT
from tools import TOOLS, execute_tool


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
        while iteration < self.max_iterations:
            iteration += 1

            # Call the LLM with the current conversation and tool definitions
            # [PROVIDER: GroqCloud] — tool schema format and response structure
            # follow OpenAI's function calling spec
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",   # LLM decides whether to call a tool
                temperature=0.2,      # Low temperature: factual/precise for dataset queries
                max_tokens=1024
            )

            # Track token usage for cost and safety monitoring
            if hasattr(response, "usage"):
                tokens_this_call = response.usage.total_tokens
                self.total_tokens_used += tokens_this_call
                print(f"[Iteration {iteration}: {tokens_this_call} tokens | "
                      f"{self.total_tokens_used} total]")

            assistant_message = response.choices[0].message

            # Debug: show what the LLM decided to do this iteration
            print(f"[tool_calls: {assistant_message.tool_calls!r}]")

            # --- No tool calls: this is the final answer ---
            if not assistant_message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content or "[No response from LLM]"

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

            print(f"\n[Agent Loop — Iteration {iteration}]")
            print(f"  LLM requested {len(assistant_message.tool_calls)} tool call(s)")

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                print(f"  → Tool: {tool_name}")
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    print(f"    Args: {tool_args}")
                    tool_result = execute_tool(tool_name, tool_args)
                except json.JSONDecodeError as e:
                    tool_result = json.dumps({"error": f"Invalid JSON in tool arguments: {e}"})

                print(f"    Result: {tool_result}")

                # Append tool result — the LLM reads this in the next iteration
                # role="tool" with matching tool_call_id is required by the API
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })

            # Loop back: LLM will see the tool results and decide the next step

        return "[Max iterations reached. Please try rephrasing your question.]"

    def reset(self):
        """Clear conversation history (keeping only the system prompt) and reset token counter."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.total_tokens_used = 0
        print("[Conversation and token counter reset]")
