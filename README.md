
# Caterva2Agent: A Companion for Scientific Dataset Exploration

*A deep dive into this LLM-powered agent for natural language exploration of Blosc2 NdArray datasets, how we architected it, and what makes it genuinely useful for scientific workflows.*

[![Python](https://img.shields.io/badge/Python-≤3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Caterva2](https://img.shields.io/badge/Caterva2-dataset%20server-5C7AEA)](https://github.com/ironArray/Caterva2)
[![Blosc2](https://img.shields.io/badge/Blosc2-compression-E85D4A)](https://github.com/Blosc/python-blosc2)
[![Groq](https://img.shields.io/badge/LLM-Groq%20compatible-orange)](https://groq.com/)


---

Scientific datasets are powerful, but exploring them can be tricky and time-consuming. You write boilerplate to list datasets, remember specific syntax, switch mental models between server-side APIs and local operations, and pray you don't accidentally load 50GB into RAM.

That's why we built the **Caterva2 Agent** — not as a replacement for the analyst, but as a companion that removes the busywork standing between your data and your insight. The core idea is simple: you ask for what you need in natural language, and the agent grounds every answer in real tool calls against the actual data. No hallucinated statistics. No made-up shapes. Every number, every slice, every plot comes from executing real operations on real datasets.

But here's what makes it genuinely useful for scientific work: the agent is designed to sit inside your existing Jupyter notebook exploration workflow, not replace it. You can start from the server, fetch a useful subset, work locally in your notebook with your own transformations, and then continue with the agent from that exact point, since it can see your variables and use them, and you can run your own code on the agent's responses. It's a loop: **agent-assisted exploration → manual analysis → agent-assisted exploration**, seamlessly switching between server-side operations and local ones as you go.


https://github.com/user-attachments/assets/62f65eb9-42b6-4c9a-ba59-5eb39e64476e

---

## How It Works: Architecture and Operations

Let's look at what the agent actually does and how it's structured. The architecture is built around three layers: the **LLM orchestrator** (the reasoning brain), the **tool layer** (the hands that manipulate data), and the **data bridge** (the connection between server, local memory, and your notebook).

<p align="center">
  <img 
    width="478" 
    height="577" 
    alt="caterva2_agent_workflow-3 (1)" 
    src="https://github.com/user-attachments/assets/2437c8c9-f593-4c7b-b2d2-c872fbee7c65" 
  />
</p>



### The Notebook Bridge: User-Agent Collaboration

The agent isn't a black box you send requests to and receive answers from. It lives inside your Jupyter notebook and shares the same Python namespace as you. This means the collaboration is bidirectional: you can see and operate on everything the agent does, and the agent can see and operate on everything you do.

When the agent fetches a dataset from the server, it doesn't just return a text summary — it injects the actual array into your notebook as a variable you can use immediately. Ask for a slice of `@public/examples/temperature.b2nd`, and `temperature` appears in your namespace. Run your own transformations (`temp_celsius = (temperature - 32) * 5/9`), then hand it back to the agent with `ask("Plot a histogram of {temp_celsius}")`. The agent reads your `temp_celsius` directly from the notebook's user namespace, sees its current shape and dtype, and continues from exactly where you left off. No data copying, no context switching, no "export to CSV and re-import."

The notebook interface provides the commands for this workflow: `ask(message)` sends questions and injects results, `variables()` lists what the agent has created, `clear_variables()` cleans up without losing conversation history, and `login()` / `logout()` manage server authentication. The `{variable}` syntax is the bridge — when you reference `{my_data}`, the agent expands it to include metadata (shape, dtype, statistics) in its prompt before deciding which tools to call.

Under the hood, this works through a simple registry pattern. When tools fetch server data, they register results in an internal `_fetched_objects` dictionary. After the agent finishes reasoning, `ask()` retrieves from this registry, sanitizes names to valid Python identifiers, and injects them into your notebook's user namespace. For local references, the agent looks up `{var_name}` in the notebook namespace, validates it's array-like, and prepends that context to your message. The loop is seamless: you work with the agent, then independently, then with the agent again — all within the same conversation and the same namespace.

### The Unified Data Model: One Interface, Two Sources

The most important architectural decision is the **unified resolver**. Every tool that operates on data uses the same resolution logic:

- **Path starts with `@`** → Server dataset. The agent creates a Caterva2 client (authenticated if you've called `login()`), fetches the dataset handle, and operates on it server-side.
- **Otherwise** → Local notebook variable. The agent reads from the notebook namespace, validates that the object is array-like (has `shape` and `dtype`), normalizes it to `blosc2.NDArray` when possible, and operates on it locally.

This means `get_dataset_stats("@public/data.b2nd")` and `get_dataset_stats("my_local_array")` use the **exact same code path**. The tool doesn't know or care where the data comes from — it sees a unified `ResolvedData` object with `.shape`, `.dtype`, and `[]` accessor. This is what enables the seamless switching between server and local operations in your workflow.

### The Five Tool Families

The agent exposes 15 tools organized into five categories. Each tool is defined by a JSON schema (description, parameters, types, enums) that the LLM uses to decide when and how to call it.

**Browsing tools** handle discovery. `list_roots` shows top-level collections (`@public`, `@personal`, `@shared`). `list_datasets("@public/examples")` paginates through items in a path. `get_dataset_info("@public/examples/ds-3d.b2nd")` returns full metadata: shape, dtype, chunk layout, compression codec, ratio, and timestamps. These tools answer "what do I have?" before you touch any data.

<img width="1351" height="753" alt="browsing_tool" src="https://github.com/user-attachments/assets/1887bda1-ec61-451e-a23e-632e9dafb7f5" />


**Analysis tools** compute on the data. `get_dataset_stats` calculates min, max, mean, std, var, argmin, argmax, any, and all — either globally or along a specific axis. `collapse_dimensions` is the heavy lifter for large datasets: it reduces N-D to (N-1)-D via aggregation (max, mean, sum, min, std, var, prod) executed server-side on compressed Blosc2 chunks without downloading the full array, or on local variables if desired. For a 3D tomography, `collapse_dimensions(path, axis=0, operation="max")` gives you a 2D max-intensity projection in seconds.

**Data access tools** retrieve values. `get_slice` parses Python-style slice strings (`"0:100"`, `"0:5, 0:3"`, `":, 0"`) and returns the data. For large slices (>100 elements), it returns a summary (shape, min, max, mean, preview) instead of dumping the full array into the conversation. You can persist sliced results to `@personal/slices/...` for follow-up operations. `where_filter` applies conditional selection (`elevation > 3000`) and returns `value_if_true` where the condition is met, and `value_if_false` elsewhere. It auto-saves to `@personal/where_filter/...` when authenticated with the Caterva2 server. `load_dataset`, does exactly what it sounds like: it loads datasets from the server into the notebook locally.

<img width="1349" height="629" alt="data_access_image_vlog" src="https://github.com/user-attachments/assets/6fd29d99-3c32-469b-ad78-3040487d95da" />

**Dataset management tools** handle server-side file operations. `copy_dataset`, `move_dataset`, and `remove_dataset` require authentication. Move and remove need explicit confirmation to the agent to execute — this prevents accidental data loss. `upload_dataset` takes a local variable and pushes it to `@personal/...` as a Blosc2 array. `download_dataset` returns a direct URL for external download.

**Visualization tools** render what you see. `visualize_dataset` auto-detects dimensionality and creates interactive Plotly plots: 1D line plots, 2D heatmaps with equal aspect ratio, 3D volume rendering with adjustable opacity. Large datasets are automatically downsampled. For giant data that exceeds interactive limits, `render_projection` first collapses dimensions, then downsamples to 2000×2000 pixels, and returns a base64-encoded PNG — ideal for multi-GB datasets where interactive rendering would be impossible.

<img width="1348" height="659" alt="visualization_image_vlog" src="https://github.com/user-attachments/assets/805e86a6-6ab1-4c3b-b9b7-c593da7fe2b3" />


### Chaining Operations: The Power of Persistence

The real power emerges when you chain these tools together. Because derived results can be persisted back to the server (under `@personal/`) or injected into your notebook namespace, subsequent operations can continue from exactly where you left off.

**Server-side chaining example:**

```python
# Filter a massive elevation dataset
ask("Filter @public/large/survey.b2nd where elevation > 3000")
# → Result persisted to @personal/where_filter/survey_elevation_20240115T120000.b2nd

# Collapse the filtered result without downloading it
ask("Collapse the previous result along axis 0 with mean")
# → Operates on the @personal path directly, returns @personal/collapsed/...

# Visualize the 2D projection
ask("Render a static projection of the collapsed data")
# → Server-side 2D PNG, never touches local RAM with the full dataset
```

**Local chaining example:**

```python
# Fetch a slice
ask("Get slice 0:100 from @public/examples/temperature.b2nd and load it locally")
# → Injects `temperature` into namespace

# Manual transformation
temp_celsius = (temperature - 32) * 5/9

# Continue with agent using the local variable
ask("Compute stats on {temp_celsius}")
# → Agent sees temp_celsius, operates locally, returns statistics
```

The agent remembers which operations produced server-side results and which produced local variables. Server results return `result_path` hints for follow-up server calls. Local results get unique variable names injected into your namespace. The loop is seamless: server → local → server → local, all within the same conversation.


---

## What This Looks Like in Practice

Let me walk you through a simple but realistic workflow:


https://github.com/user-attachments/assets/e2fdba69-bef6-479b-a8b5-03a6f0225515


Notice how, at some point, the user and the agent begin alternating responsibilities. The agent handles the boilerplate tasks (listing datasets, fetching data, retrieving metadata). The user handles the domain-specific reasoning, such as scaling the dataset values to address the dynamic-range visualization problem (`norm_kevlar = blosc2.clip(sliced_kevlar_tomo, 0, 5) * 13000`). Then the agent takes over again for the visualization. This is the collaborative loop in action.






---

## Getting Started

### Prerequisites

- Use **Python 3.12** (recommended). Most modern tools (Poetry, Groq SDK, etc.) do NOT yet support Python 3.14. And Caterva2 API requires Python >= 3.11 
- [Poetry](https://python-poetry.org/) for dependency management
- An API key for your preferred LLM provider — if you don't have one, [Groq](https://console.groq.com/) offers a free tier to get started



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
poetry install --all-groups
```
Poetry automatically creates an isolated virtual environment and installs all dependencies (including dev dependencies like pytest) pinned to exact versions via `poetry.lock`. The project is installed as an editable package so imports work correctly in tests and notebooks.

3. **Add your LLM provider API key** — create a `.env` file at the project root:
```bash
echo "GROQ_API_KEY=your-api-key-here" > .env
```

4. **Run the agent:**

Inside your project's folder:

- **Jupyter Lab:**
  
  ```bash
  poetry run jupyter lab
  ```
  Or activate the Poetry shell once and run commands directly:
  ```bash
  poetry shell
  jupyter lab                     # Jupyter Lab (in the same shell)
  ```


## How to use


This document lists all agent operations grouped by the 5 tool families in the project.

## 1) Browsing tools

| Tool | What it does | Parameters you can ask for | Simple example                                                        |
|---|---|---|-----------------------------------------------------------------------|
| `list_roots` | Lists top-level Caterva2 roots (collections). | *(none)* | `ask("List the available roots on the server")`                       |
| `list_datasets` | Lists datasets/files inside a root or sub-path, with pagination. | `path` (required), `limit` (optional), `offset` (optional) | `ask("List datasets in @public/examples with limit 20")` |
| `get_dataset_info` | Returns metadata for a server dataset or local variable (shape, dtype, chunks, etc.). | `path` (required) | `ask("Show metadata for @public/examples/ds-2d-fields.b2nd")`         |

## 2) Analysis tools

| Tool | What it does | Parameters you can ask for | Simple example                                                        |
|---|---|---|-----------------------------------------------------------------------|
| `get_dataset_stats` | Computes stats (min/max/mean/std/etc.) in one call. | `path` (required), `stats` (optional list), `axis` (optional) | `ask("Compute all the stats for @public/examples/ds-1d.b2nd")`        |
| `collapse_dimensions` | Reduces dimensionality by aggregating along one axis (e.g., 3D -> 2D). | `path` (required), `axis` (required), `operation` (required), `variable_name` (optional), `persist_result` (optional) | `ask("Collapse @public/examples/volume.b2nd along axis 2 using max")` |

## 3) Data access tools

| Tool | What it does | Parameters you can ask for | Simple example                                                                                    |
|---|---|---|---------------------------------------------------------------------------------------------------|
| `get_slice` | Fetches values from a region of a dataset/local variable. | `path` (required), `slices` (optional), `persist_result` (optional, server), `save_path` (optional, server) | `ask("Slice @public/examples/ds-2d.b2nd in 0:10,0:10")`                                           |
| `where_filter` | Applies a conditional mask/filter (`where`) to data. | `path` (required), `operator` (required), `threshold` (required), `value_if_true` (optional), `value_if_false` (optional), `slices` (optional), `compute` (optional) | `ask("Filter @public/examples/elevation.b2nd where values are > 3000 and set false values to 0")` |
| `load_dataset` | Materializes full dataset into notebook memory (with safety checks). | `path` (required) | `ask("Load @public/examples/ds-1d.b2nd locally in the notebook")`                                 |

## 4) Dataset management tools

| Tool | What it does | Parameters you can ask for | Simple example                                                               |
|---|---|---|------------------------------------------------------------------------------|
| `copy_dataset` | Copies a server dataset/file from one path to another. | `path` (required), `destination` (required) | `ask("Copy @public/a.b2nd to @personal/a_copy.b2nd")`                        |
| `download_dataset` | Returns a direct download URL for a server dataset/file. | `path` (required) | `ask("Give me the download URL for @public/examples/ds-1d.b2nd")`            |
| `move_dataset` | Moves a server dataset/file (dry-run + confirm safety flow). | `path` (required), `destination` (required), `dry_run` (optional), `confirm` (optional) | `ask("Move @personal/old.b2nd to the @shared root")`                         |
| `remove_dataset` | Deletes a server dataset/file (dry-run + confirm safety flow). | `path` (required), `dry_run` (optional), `confirm` (optional) | `ask("Delete @personal/tmp.b2nd")`                                           |
| `upload_dataset` | Uploads local variable/array to server (`@personal/`). | `source` (required), `destination` (required), `overwrite` (optional) | `ask("Upload my local variable {my_array} to @personal root in the server")` |

## 5) Visualization tools

| Tool | What it does | Parameters you can ask for | Simple example                                                         |
|---|---|---|------------------------------------------------------------------------|
| `visualize_dataset` | Interactive Plotly visualization (1D line, 2D heatmap, 3D volume). | `path` (required), `slices` (optional), `colorscale` (optional), `opacity` (optional), `max_size` (optional), `title` (optional) | `ask("Visualize @public/examples/kevlar-tomo.b2nd with opacity 0.25")` |
| `render_projection` | Static 2D PNG projection from higher-dimensional data. | `path` (required), `axis` (required), `operation` (required), `colormap` (optional), `title` (optional) | `ask("Render a max projection of my local {volume} along axis 0")`     |

And remember that you can always ask the agent to remember you what functions/tools can it use, and how to ask it: parameters, options...


---

## LLM Provider Support

Caterva2Agent is compatible with **Groq** (free tier available).

---

## Contributing

Contributions are welcome. Please open an issue or pull request for bug fixes, improvements, or new tool integrations.
