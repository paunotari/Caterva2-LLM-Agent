"""Test bootstrap for importing caterva2_agent modules safely in unit tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# PATH SETUP
# Add project root to sys.path so caterva2_agent package is importable.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# FAKE EXTERNAL DEPENDENCIES
# Mock external modules before any caterva2_agent imports to avoid network/API deps.
# ---------------------------------------------------------------------------

# Fake caterva2 module — must be registered BEFORE importing any tools
fake_caterva2 = types.ModuleType("caterva2")


class _FakeClient:
    """Stub Client for testing without network."""
    def __init__(self, _url: str):
        self.url = _url


class _FakeDataset:
    """Stub Dataset class for type annotation in tools modules."""
    pass


fake_caterva2.Client = _FakeClient
fake_caterva2.Dataset = _FakeDataset
sys.modules["caterva2"] = fake_caterva2


# ---------------------------------------------------------------------------
# FAKE CONFIG MODULE
# Provide fake config values so tests don't need real API keys or .env file.
# This fake is used when importing from caterva2_agent.config
# ---------------------------------------------------------------------------

# Create a fake prompts module first (config depends on it)
fake_prompts = types.ModuleType("caterva2_agent.prompts")
fake_prompts.SYSTEM_PROMPT = "test system prompt"
sys.modules["caterva2_agent.prompts"] = fake_prompts

# Create fake config module
fake_config = types.ModuleType("caterva2_agent.config")
fake_config.CATERVA2_URLBASE = "http://test-caterva2.local"
fake_config.MODEL_NAME = "test-model"
fake_config.SYSTEM_PROMPT = "test system prompt"
fake_config.client = types.SimpleNamespace(
    chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kwargs: None)
    )
)
sys.modules["caterva2_agent.config"] = fake_config

# Also provide bare 'config' module for backwards compatibility with existing tests
sys.modules["config"] = fake_config


# ---------------------------------------------------------------------------
# IMPORT HELPERS
# After fakes are registered, tests can safely import from caterva2_agent.tools
# ---------------------------------------------------------------------------

def get_tools_module():
    """Import and return the tools package after fakes are set up."""
    from caterva2_agent import tools
    return tools

