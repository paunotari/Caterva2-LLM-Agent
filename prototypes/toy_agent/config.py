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
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model configuration
# Groq supports several models;
MODEL_NAME = "openai/gpt-oss-120b"

# Agent system prompt
# This tells the LLM how to behave and when to use tools
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When the user asks a question that requires calculation, use the calculator tool.
For general questions, respond conversationally without using tools.

Be concise and clear in your responses."""
