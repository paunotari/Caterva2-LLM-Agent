# Testing Guide

## Philosophy: Minimal but Critical

**Don't test everything** — test what matters for agent reliability and user experience.

### ✅ What to Test

- **Core behaviors** that would cause silent failures
- **Public contracts** (API response structures, pagination, error formats)
- **Error handling** (graceful degradation, not crashes)
- **Critical paths** that users depend on

### ❌ What NOT to Test

- **Implementation details** (internal helper functions)
- **External dependencies** (real Caterva2 API - use fakes/mocks instead)
- **Trivial pass-throughs** (pure delegation to external APIs with no logic)
- **Everything "just because"** (tests have maintenance cost)

---

## Running Tests

### Run all tests
```bash
poetry run pytest
```

### Run specific test file
```bash
poetry run pytest tests/unit/tools/test_list_datasets.py
```

### Run specific test
```bash
poetry run pytest tests/unit/tools/test_list_datasets.py::test_list_datasets_pagination_fields
```

### Run with verbose output
```bash
poetry run pytest -v
```

### Run with output from print statements
```bash
poetry run pytest -s
```

### Run in PyCharm
- Click the green ▶️ icon next to a test function, file, or folder
- View results in the hierarchical "Test Results" panel

---

## Test Structure

```
tests/
├── conftest.py              # Test bootstrap (fake config, fake caterva2 module)
├── README.md                # This file
└── unit/
    ├── agent/
    │   └── test_agent_core.py    # Agent loop behavior tests
    └── tools/
        ├── test_dispatch.py           # Tool dispatcher safety
        ├── test_list_roots.py         # list_roots() contract
        ├── test_list_datasets.py      # list_datasets() pagination
        └── test_get_dataset_info.py   # get_dataset_info() metadata
```

---

## File Organization Strategy

### One Test File Per Tool (Current Pattern)

**Pattern:**
```
tests/unit/tools/
├── test_dispatch.py           # Generic dispatcher (stays stable)
├── test_list_roots.py         # One tool = one file
├── test_list_datasets.py      
└── test_get_dataset_info.py   
```

**Why this works:**
- ✅ Scales well to 20+ tools without reorganization
- ✅ Clear ownership (tool name → test file name is predictable)
- ✅ Parallel execution (pytest can run files in parallel)
- ✅ Easy to navigate and locate tests

**When to reorganize:**
- When you have >10 tool test files in `tests/unit/tools/`
- When tools naturally group by capability (browsing vs data access vs computation)
- **Not before** — YAGNI (You Aren't Gonna Need It)

---

## Test Naming Convention

```python
def test_{function_name}_{what_is_tested}():
    # What this tests: [one sentence explaining WHAT is being verified]
    # Why important: [one sentence explaining WHY this matters for reliability]
    ...
```

**Examples:**
- `test_execute_tool_unknown_tool_returns_error_json`
- `test_list_datasets_pagination_fields`
- `test_get_dataset_info_returns_metadata_contract`

**Key principle:**
- Test name explains **WHAT** is tested
- Comment explains **WHY** it matters (context for future maintainers)

---

## Current Test Coverage

### Agent Loop (2 tests)
- ✅ Chat without tools (direct answers still work)
- ✅ Full tool execution cycle (request → execute → result → answer)

### Tool Dispatch Safety (2 tests)
- ✅ Unknown tool names return structured errors (no crashes)
- ✅ Invalid arguments are caught and reported clearly

### Tool Functionality (4 tests)
- ✅ `list_roots()` - root name format and sorting
- ✅ `list_datasets()` - pagination contract
- ✅ `get_dataset_info()` - metadata structure

**Total: 8 tests, all passing, <1 second runtime**

---

## Writing New Tests

### When to Add Tests

**Always test when adding:**
1. **New tools** - at minimum, test the return structure contract
2. **Pagination/complex logic** - prevent off-by-one errors
3. **Error paths** - ensure graceful failures, not crashes

**Example: Adding a new tool `fetch_slice()`**

Create `tests/unit/tools/test_fetch_slice.py`:

```python
"""Essential unit tests for data slicing contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _ensure_local_import_bootstrap() -> None:
    """Make direct file execution work (PyCharm 'Run file') as well as pytest runs."""
    project_root = Path(__file__).resolve().parents[3]
    agent_dir = project_root / "caterva2_agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    if "config" not in sys.modules:
        fake_config = types.ModuleType("config")
        fake_config.CATERVA2_URLBASE = "http://test-caterva2.local"
        fake_config.MODEL_NAME = "test-model"
        fake_config.SYSTEM_PROMPT = "test system prompt"
        fake_config.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )
        )
        sys.modules["config"] = fake_config

    if "caterva2" not in sys.modules:
        fake_caterva2 = types.ModuleType("caterva2")

        class _FakeClient:
            def __init__(self, _url: str):
                self.url = _url

        fake_caterva2.Client = _FakeClient
        sys.modules["caterva2"] = fake_caterva2


_ensure_local_import_bootstrap()
import tools


class _FakeClient:
    def fetch_slice(self, path: str, slice_spec: str):
        """Return fake slice data."""
        return [[1, 2, 3], [4, 5, 6]]


def test_fetch_slice_returns_data_array(monkeypatch) -> None:
    # What this tests: slice data is returned as a nested list structure.
    # Why important: users expect array-like data for plotting/analysis.
    monkeypatch.setattr(tools, "_get_client", lambda: _FakeClient())
    result = tools.fetch_slice("@public/example.b2nd", "0:2, 0:3")

    assert "data" in result
    assert len(result["data"]) == 2  # 2 rows
    assert len(result["data"][0]) == 3  # 3 columns
```

---

## Test Independence

All tests use **fakes**, not real network connections:

- `conftest.py` provides fake `config` and `caterva2` modules
- Individual tests use `monkeypatch` to inject specific behaviors
- **Why:** Fast, deterministic, no API keys or network required

**Never:**
- ❌ Make real HTTP requests in unit tests
- ❌ Require API keys or environment variables
- ❌ Depend on external services being available

---

## Debugging Failed Tests

### Read the test comments first
Every test has "What/Why" comments explaining its purpose.

### Run with verbose output
```bash
poetry run pytest -v -s
```

### Run just the failing test
```bash
poetry run pytest tests/unit/tools/test_list_datasets.py::test_list_datasets_pagination_fields -v
```

### Use PyCharm debugger
- Set breakpoints in test code
- Click "Debug" icon instead of "Run" icon

---

## Future Expansion

As the agent grows, testing will expand to:

### New test types
```
tests/
├── unit/           # Current: isolated logic tests (no network)
├── integration/    # Future: agent + real Caterva2 server (optional)
└── e2e/            # Future: full user scenarios (optional)
```

### Grouped tool tests (when >10 tools)
```
tests/unit/tools/
├── test_dispatch.py           # Stays stable
│
├── browsing/                  # Group by capability
│   ├── test_list_roots.py
│   ├── test_list_datasets.py
│   └── test_search_datasets.py
│
└── data_access/
    ├── test_get_dataset_info.py
    ├── test_fetch_slice.py
    └── test_download_dataset.py
```

**Don't reorganize prematurely** — current flat structure works well up to ~10 tools.

---

## Reference

- [pytest documentation](https://docs.pytest.org/)
- [Testing plan](~/.copilot/session-state/.../plan.md) (historical context)
- Agent architecture guide: `AGENTS.md` in project root
