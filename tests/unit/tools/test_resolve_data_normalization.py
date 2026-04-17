"""Tests for Blosc2-first local data normalization in resolve_data()."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import blosc2
import numpy as np


def _ensure_local_import_bootstrap() -> None:
    """Make direct file execution work (PyCharm 'Run file') as well as pytest runs."""
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if "caterva2" not in sys.modules:
        fake_caterva2 = types.ModuleType("caterva2")

        class _FakeClient:
            def __init__(self, _url: str):
                self.url = _url

        class _FakeDataset:
            pass

        fake_caterva2.Client = _FakeClient
        fake_caterva2.Dataset = _FakeDataset
        sys.modules["caterva2"] = fake_caterva2

    if "caterva2_agent.prompts" not in sys.modules:
        fake_prompts = types.ModuleType("caterva2_agent.prompts")
        fake_prompts.SYSTEM_PROMPT = "test system prompt"
        sys.modules["caterva2_agent.prompts"] = fake_prompts

    if "caterva2_agent.config" not in sys.modules:
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
        sys.modules["config"] = fake_config


_ensure_local_import_bootstrap()
from caterva2_agent.tools._base import resolve_data, set_notebook_namespace


def test_resolve_data_normalizes_local_numpy_to_blosc2() -> None:
    local_numpy = np.arange(12, dtype=np.float32).reshape(3, 4)
    set_notebook_namespace({"local_numpy": local_numpy})

    resolved = resolve_data("local_numpy")

    assert resolved.is_local()
    assert resolved.backend == "blosc2"
    assert resolved.normalized_from == "ndarray"
    assert isinstance(resolved.data, blosc2.NDArray)
    assert tuple(resolved.shape) == (3, 4)
    assert str(resolved.dtype) == "float32"


def test_resolve_data_keeps_local_blosc2_ndarray() -> None:
    local_blosc = blosc2.asarray(np.arange(8, dtype=np.int64))
    set_notebook_namespace({"local_blosc": local_blosc})

    resolved = resolve_data("local_blosc")

    assert resolved.is_local()
    assert resolved.backend == "blosc2"
    assert resolved.normalized_from is None
    assert resolved.data is local_blosc
