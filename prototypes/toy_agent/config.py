"""
Configuration file for the Groq-based agent.
This centralizes API setup and model configuration.
"""

import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()  # Must be called BEFORE os.getenv()

# Initialize Groq client
# Set your API key as an environment variable: export GROQ_API_KEY='your-key-here'
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

client = Groq(api_key=api_key)

# Model configuration
# Groq supports several models;
MODEL_NAME = "openai/gpt-oss-120b"

# Agent system prompt
# This tells the LLM how to behave and when to use tools
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When the user asks a question that requires calculation, use the calculator tool.
For general questions, respond conversationally without using tools.

Be concise and clear in your responses.

CRITICAL RULES:
1. After providing your answer, STOP immediately. Do not continue elaborating.
2. If the user provides long text without a clear question, ask them: "What would you like me to help you with regarding this text?, unless its a short statement like: "hi" or similar, in which case respond conversationally without asking for clarification.
3. Do NOT analyze, summarize, or discuss text unless explicitly asked to do so.
4. Each response should be complete and final - wait for the user's next input before continuing.
5. Never continue a response across multiple turns without explicit user requests."""
