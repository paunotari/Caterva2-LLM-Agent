"""
Configuration for the Caterva2 dataset exploration agent.

Centralizes:
- LLM client setup (Groq)
- Model selection
- Caterva2 subscriber URL

System prompt is defined in prompts.py and re-exported here for backwards compatibility.
"""

import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq  # [PROVIDER: GroqCloud]

from caterva2_agent.prompts import SYSTEM_PROMPT  # Re-export for backwards compatibility

# find_dotenv() searches upward from CWD until it finds .env
# This makes config.py work correctly whether Python is invoked from the
# project root (CLI), from caterva2_agent/ (direct), or from a Jupyter notebook.
load_dotenv(find_dotenv())

# --- LLM Setup ---

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

client = Groq(api_key=api_key)  # [PROVIDER: GroqCloud]

# Groq model to use. Change here to swap models without touching agent logic.
MODEL_NAME = "openai/gpt-oss-120b"

# --- Caterva2 Setup ---

# URL of the Caterva2 subscriber to connect to.
# Default: the public IronArray demo server — no local setup required.
# Override in .env: CATERVA2_URLBASE=http://localhost:8002
CATERVA2_URLBASE = os.getenv("CATERVA2_URLBASE", "https://cat2.cloud/demo")
