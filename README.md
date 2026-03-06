# Caterva2 LLM Agent

> **A LLM Agent for Exploration and Manipulation of Blosc2/HDF5 Datasets in Caterva2**
>
> Plus a hands-on introduction to basic LLM agent architecture.

[![Python](https://img.shields.io/badge/Python-≤3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Caterva2](https://img.shields.io/badge/Caterva2-dataset%20server-5C7AEA)](https://github.com/ironArray/Caterva2)
[![Blosc2](https://img.shields.io/badge/Blosc2-compression-E85D4A)](https://github.com/Blosc/python-blosc2)
[![Groq](https://img.shields.io/badge/LLM-Groq%20compatible-orange)](https://groq.com/)

---

## Overview

This repository provides two Python-based LLM agents serving complementary purposes:

| Agent | Purpose | Start here if… |
|---|---|---|
| [**Caterva2 Agent**](caterva2_agent/) | Production-grade agent for browsing, querying, and visualizing Caterva2 Blosc2/HDF5 scientific datasets via natural language | You want to explore real scientific data |
| [**Toy Agent**](prototypes/toy_agent/) | Minimal calculator agent that teaches LLM Agents fundamentals | You're new to LLM agents and want to understand the product bases|

---

## Agents

### Caterva2 Agent

A LLM Agent for operating on and visualizing scientific datasets using natural language. Connect to your own Caterva2 server or the public demo server (dafault option) — directly from the command line or a Jupyter notebook.

**Key features:**
- Browse and query remote Caterva2 Blosc2/HDF5 datasets
- Visualize dataset contents as plots, charts
- Visualize multidimensional scientific renders — including tomographic imaging and N-dimensional array inspection
- Natural language interaction — no API knowledge required
- Extensible tool architecture for custom workflows

[Full documentation → `caterva2_agent/README.md`](caterva2_agent/README.md)

---

### Toy Agent

A minimal calculator agent designed as a hands-on introduction to LLM agent architecture and workflow. Clear, well-commented code that walks you through every basic layer of the agent loop.

**What you'll learn:**
- How the agent loop works
- Defining and registering tool schemas
- Handling tool execution and results

[Full documentation → `prototypes/toy_agent/README.md`](prototypes/toy_agent/README.md)

---

## Getting Started

### Prerequisites

- Use **Python 3.12** (recommended). Most modern tools (Poetry, Groq SDK, etc.) do NOT yet support Python 3.14. And Caterva2 API requires Python >= 3.11 
- [Poetry](https://python-poetry.org/) for dependency management
- An API key for your preferred LLM provider — if you don't have one, [Groq](https://console.groq.com/) offers a free tier to get started

#### How to install Python 3.12 (macOS/Linux):

Option 1: Homebrew (recommended for macOS users)
```bash
brew install python@3.12
```


```bash
# IMPORTANT: Add Python 3.12 to your PATH (for Homebrew installs)
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Option 2: pyenv (works on macOS and Linux)
```bash
pyenv install 3.12.2
```
```bash
pyenv global 3.12.2
```

After installing, make sure python3.12 is available in your PATH:
```bash
python3.12 --version
```


### Installation

1. **Install Poetry** (if you haven't already):

> **Important:** Always use `python3.12` (not just `python3`) to install Poetry, to avoid SSL and compatibility errors if your system default is Python 3.14.

```bash
curl -sSL https://install.python-poetry.org | python3.12 -
```

You can test that everything is set up by running:
```bash
poetry --version
```

2. **Clone the repository and install dependencies:**
```bash
git clone <https://github.com/paunotari/Caterva2-LLM-Agent>
```
```bash
# Change to the project directory (adjust path if you cloned elsewhere)
cd Caterva-LLM-Agent
```
```bash
poetry install --no-root
```
Poetry automatically creates an isolated virtual environment and installs all dependencies pinned to exact versions via `poetry.lock`.

> **Note:** If you see an error like "No file/folder found for package caterva2-llm-agent", it's because this project is not a Python package (just scripts). Use poetry install --no-root to avoid this, or set package-mode = false in pyproject.toml.

3. **Add your LLM provider API key** — create a `.env` file at the project root:
```bash
echo "GROQ_API_KEY=your-api-key-here" > .env
```

4. **Run an agent:**

You can interact with the Caterva2 Agent in two ways:

- **Jupyter Lab (Recommended for scientific exploration):**
  ```bash
  poetry run jupyter lab
  ```
- **Command Line Interface (CLI):**
  ```bash
  # Caterva2 agent (main agent)
  poetry run python caterva2_agent/main.py

  # Toy agent (learning reference)
  poetry run python prototypes/toy_agent/main.py
  ```
  Or simply run the main.py files from your favorite IDE (VSCode, PyCharm, etc.) — just make sure your IDE is using the Poetry virtual environment as the
  interpreter.

Or activate the Poetry shell once and run commands directly:
```bash
poetry shell
python caterva2_agent/main.py   # CLI
jupyter lab                     # Jupyter Lab (in the same shell)
```


---

## Project Structure

```
.
├── caterva2_agent/          # Production scientific dataset agent
│   └── README.md
├── prototypes/
│   └── toy_agent/           # Toy agent for learning the very basics behind agents
│       └── README.md
├── pyproject.toml           # Project dependencies (Poetry)
├── poetry.lock              # Pinned dependency versions (auto-generated)
└── README.md
```

---

## LLM Provider Support

Both agents are compatible with **Groq** (free tier available) and can be configured to use any supported LLM provider. See the individual agent READMEs for provider configuration details.

---

## Contributing

Contributions are welcome. Please open an issue or pull request for bug fixes, improvements, or new tool integrations.
