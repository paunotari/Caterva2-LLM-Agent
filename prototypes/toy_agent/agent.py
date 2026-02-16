"""
The core agent loop.

This file implements the agentic reasoning loop:
1. Send messages to LLM with tool definitions
2. Check if LLM wants to use a tool
3. If yes: execute tool, add result to conversation, go back to step 1
4. If no: return the LLM's response to the user

This is the heart of the agent - understanding this loop is critical.
"""

import json
# from typing import List, Dict, Any
from config import client, MODEL_NAME, SYSTEM_PROMPT
from tools import TOOLS, execute_tool


class Agent:
    """
    A basic agentic assistant that can use tools.
    """
    
    def __init__(self):
        """Initialize the agent with an empty conversation history."""
        # Conversation history includes system prompt and all messages
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.max_iterations = 10  # Prevent infinite loops
    
    def run(self, user_input: str) -> str:
        """
        Process a user input and return the agent's response.
        
        This implements the agent loop:
        - Add user message to history
        - Repeatedly call LLM until it gives a final answer (no more tool calls)
        - Return the final response
        
        Args:
            user_input: The user's question or request
            
        Returns:
            The agent's final response as a string
        """
        # Add the user's message to conversation history
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Agent loop - keep going until we get a final answer
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call the LLM with current conversation and available tools
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=TOOLS,  # This tells the LLM what tools it can use
                tool_choice="auto",  # LLM decides whether to use tools
                temperature=0.4, # Creativity grade: 0.0-deterministic, 1+ (up to 2)-randomness
                max_tokens=1024
            )
            
            # Get the assistant's response
                # .choices - response can contain multiple responses (usually we only ask for 1 - e.g. API call n=3)
                # .message - inside every choice object there's various objects, metadata, etc. (ask Claude for the structure of a response object, we only want the message object)
            assistant_message = response.choices[0].message

            # Check if the LLM wants to use any tools
            if not assistant_message.tool_calls:
                # No tool calls - this is the final answer
                # Add to history before returning
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content or "I don't have a response."

            # There are tool calls - add assistant message with tool_calls to history
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })

            # The LLM wants to use tools - execute them (show in output - for comprehension)
            print(f"\n[Agent Loop Iteration {iteration}]")
            print(f"LLM requested {len(assistant_message.tool_calls)} tool call(s)")

            # Process each tool call the LLM requested
            for tool_call in assistant_message.tool_calls:
                # Extract tool information
                tool_name = tool_call.function.name
                    # We convert the JSON arguments, as a dictionary
                tool_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id
                
                print(f"  - Calling tool: {tool_name}")
                print(f"    Arguments: {tool_args}")

                # Execute the tool - add error handling in case something goes wrong with the tool execution
                tool_result = execute_tool(tool_name, tool_args)
                print(f"    Result: {tool_result}")
                
                # Add tool result to conversation
                # This tells the LLM what happened when it called the tool
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })
            
            # Loop continues - LLM will see tool results and decide next step
        
        # If we hit max iterations, return a message
        return "I reached my maximum number of thinking steps. Please try rephrasing your question."
    
    def reset(self):
        """Clear conversation history (keeping only system prompt)."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
