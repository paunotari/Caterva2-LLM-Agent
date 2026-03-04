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

- Python **≤ 3.12**
- An API key for your preferred LLM provider — if you don't have one, [Groq](https://console.groq.com/) offers a free tier to get started


Each agent directory contains its own `README.md` with complete setup and usage instructions.

---

## Project Structure

```
.
├── caterva2_agent/          # Production scientific dataset agent
│   └── README.md
├── prototypes/
│   └── toy_agent/           # Toy agent for learning the very basics behind agents
│       └── README.md
└── README.md
```

---

## LLM Provider Support

Both agents are compatible with **Groq** (free tier available) and can be configured to use any supported LLM provider. See the individual agent READMEs for provider configuration details.

---

## Contributing

Contributions are welcome. Please open an issue or pull request for bug fixes, improvements, or new tool integrations.
