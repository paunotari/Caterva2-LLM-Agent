"""Test bootstrap for importing caterva2_agent modules safely in unit tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path


# Make `agent.py` / `tools.py` importable as top-level modules for tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATERVA2_AGENT_DIR = PROJECT_ROOT / "caterva2_agent"
if str(CATERVA2_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CATERVA2_AGENT_DIR))


# Provide a lightweight fake `config` module so tests do not depend on real API keys.
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


# Provide a minimal fake `caterva2` module to avoid importing network dependencies in unit tests.
fake_caterva2 = types.ModuleType("caterva2")


class _FakeClient:
    def __init__(self, _url: str):
        self.url = _url


fake_caterva2.Client = _FakeClient
sys.modules["caterva2"] = fake_caterva2
