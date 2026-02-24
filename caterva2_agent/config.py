"""
Configuration for the Caterva2 dataset exploration agent.

Centralizes:
- LLM client setup (Groq)
- Model selection
- System prompt definition
- Caterva2 subscriber URL
"""

import os
from dotenv import load_dotenv
from groq import Groq  # [PROVIDER: GroqCloud]

load_dotenv()  # Must be called BEFORE os.getenv()

# --- LLM Setup ---

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

client = Groq(api_key=api_key)  # [PROVIDER: GroqCloud]

# Groq model to use. Change here to swap models without touching agent logic.
MODEL_NAME = "llama-3.3-70b-versatile"

# --- Caterva2 Setup ---

# URL of the Caterva2 subscriber to connect to.
# Default: the public IronArray demo server — no local setup required.
# Override in .env: CATERVA2_URLBASE=http://localhost:8002
CATERVA2_URLBASE = os.getenv("CATERVA2_URLBASE", "https://cat2.cloud/demo")

# --- System Prompt ---

SYSTEM_PROMPT = """You are a scientific dataset exploration assistant with access to a Caterva2 data server.

The server stores N-dimensional compressed arrays (Blosc2/HDF5 format) organized as:
- Roots: top-level data collections (like folders)
- Datasets: individual arrays or files within a root

You have tools to:
1. List all available roots on the server
2. List datasets within a root
3. Get detailed metadata about a specific dataset (shape, dtype, compression, etc.)

RULES:
1. When the user asks about available data, ALWAYS call list_roots first before answering.
2. When the user asks about a specific dataset but doesn't give the full path, call list_datasets to discover it first.
3. When the user asks about a dataset's properties, call get_dataset_info.
4. Be explicit about what you found vs. what you inferred — scientific users care about accuracy.
5. After providing your answer, STOP. Do not continue elaborating unless asked.
6. If a tool call returns an error, report it clearly and suggest what to check (URL, path spelling, etc.).

7. Only call tools when the user explicitly asks about datasets, roots, or data.
8. For greetings, thanks, or general conversation, respond directly in natural language without calling any tools.
"""
