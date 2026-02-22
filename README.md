# Caterva2 LLM Agent

A Python project for building LLM agents from scratch, using [Groq](https://console.groq.com/) as the LLM provider.

---

## Toy Agent (`prototypes/toy_agent/`)

A minimal calculator agent built to understand the fundamentals: the agent loop, tool schemas, and how tool results are fed back to the LLM. Completed — used as a reference implementation.

---

## Caterva2 Agent (`caterva2_agent/`) — 🚧 Under Development

A natural-language agent for exploring scientific datasets hosted on a [Caterva2](https://ironarray.io/caterva2-doc/) / [Blosc2](https://www.blosc.org/python-blosc2/python-blosc2.html) server. The goal is to let users browse, query, and understand remote N-dimensional compressed datasets through conversation.

Current capabilities:
- List available dataset roots on the server
- Browse datasets within a root
- Retrieve dataset metadata (shape, dtype, compression)

Coming next: data slicing, statistical summarization.

---

## Setup

Requires Python ≤ 3.12 and a free [Groq API key](https://console.groq.com/).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r caterva2_agent/requirements.txt
```

Create a `.env` file at the project root:

```
GROQ_API_KEY=your-key-here
CATERVA2_URLBASE=https://cat2.cloud/demo   # optional, this is the default
```

Then run:

```bash
cd caterva2_agent && python main.py
```
