"""Essential unit tests for upload_dataset tool."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import blosc2


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
from caterva2_agent.tools import dataset_management


class _FakeUploadResult:
    """Simulates Caterva2 client.upload() result."""
    
    def __init__(self, path: str):
        self.path = path
        self.name = path


def test_upload_dataset_requires_auth(monkeypatch) -> None:
    # What this tests: write ops are blocked without authenticated session.
    # Why important: privileged operations must not run anonymously.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": False, "urlbase": "http://test", "username": None},
    )

    result = dataset_management.upload_dataset("my_data", "@personal/result.b2nd")

    assert "error" in result
    assert "Authentication required" in result["error"]


def test_upload_dataset_validates_destination_is_personal(monkeypatch) -> None:
    # What this tests: uploads must target @personal/ (user-scoped, not public/shared).
    # Why important: prevents accidental leaks or overwrites of shared datasets.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )

    result = dataset_management.upload_dataset("my_data", "@public/result.b2nd")

    assert "error" in result
    assert "@personal/" in result["error"]


def test_upload_dataset_rejects_non_server_paths(monkeypatch) -> None:
    # What this tests: destination must be a server path (start with @).
    # Why important: prevents confusion; destination must be on server.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )

    result = dataset_management.upload_dataset("my_data", "/local/path.b2nd")

    assert "error" in result
    assert "@" in result["error"]


def test_upload_dataset_with_local_numpy_array(monkeypatch) -> None:
    # What this tests: local NumPy arrays can be uploaded after normalization to blosc2.
    # Why important: enables numpy workflows (user creates array in notebook, uploads it).
    fake_upload_result = _FakeUploadResult("@personal/uploaded.b2nd")
    fake_client = types.SimpleNamespace(upload=lambda data, path: fake_upload_result)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    # Mock resolve_data to return a local NumPy array
    fake_resolved = types.SimpleNamespace(
        data=np.array([1, 2, 3, 4, 5]),
        source='local',
        normalized_from='ndarray',
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("my_numpy_array", "@personal/result.b2nd")

    assert result["status"] == "success"
    assert result["operation"] == "upload"
    assert result["source_type"] == "local"
    assert result["server_change_applied"] is True
    assert result["destination_path"] == "@personal/uploaded.b2nd"


def test_upload_dataset_with_blosc2_array(monkeypatch) -> None:
    # What this tests: blosc2 NDArrays can be uploaded directly (no conversion needed).
    # Why important: blosc2-first architecture; native format avoids re-encoding.
    fake_upload_result = _FakeUploadResult("@personal/blosc2_uploaded.b2nd")
    fake_client = types.SimpleNamespace(upload=lambda data, path: fake_upload_result)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    # Create an actual blosc2 NDArray
    blosc2_array = blosc2.asarray(np.array([10, 20, 30]))
    
    # Mock resolve_data to return a blosc2 array
    fake_resolved = types.SimpleNamespace(
        data=blosc2_array,
        source='local',
        normalized_from=None,  # Already blosc2, no conversion
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("my_blosc2_array", "@personal/result.b2nd")

    assert result["status"] == "success"
    assert result["operation"] == "upload"
    assert result["server_change_applied"] is True


def test_upload_dataset_reports_data_metadata(monkeypatch) -> None:
    # What this tests: result includes shape, dtype of uploaded data.
    # Why important: LLM feedback; user confirmation of what was uploaded.
    fake_upload_result = _FakeUploadResult("@personal/metadata.b2nd")
    fake_client = types.SimpleNamespace(upload=lambda data, path: fake_upload_result)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    fake_resolved = types.SimpleNamespace(
        data=blosc2.asarray(test_data),
        source='local',
        normalized_from='ndarray',
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("data_2d", "@personal/result.b2nd")

    assert result["data_shape"] == [2, 3]
    assert "float32" in result["data_dtype"]


def test_upload_dataset_handles_resolution_error(monkeypatch) -> None:
    # What this tests: tool gracefully handles variable resolution failures.
    # Why important: user-friendly error for missing/invalid variable reference.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    
    # Mock resolve_data to raise ValueError (variable not found)
    def fake_resolve(_):
        raise ValueError("Variable 'nonexistent' not found in notebook namespace")
    
    monkeypatch.setattr(dataset_management, "resolve_data", fake_resolve)

    result = dataset_management.upload_dataset("nonexistent", "@personal/result.b2nd")

    assert "error" in result
    assert "not found" in result["error"]
    assert result.get("source") == "nonexistent"


def test_upload_dataset_handles_normalization_error(monkeypatch) -> None:
    # What this tests: tool handles blosc2 normalization failures gracefully.
    # Why important: some exotic types may not convert to blosc2; give clear feedback.
    fake_client = types.SimpleNamespace(upload=lambda data, path: None)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    # Mock resolve_data to return data that can't convert to blosc2
    class BadArrayLike:
        shape = (5,)
        dtype = float
    
    fake_resolved = types.SimpleNamespace(
        data=BadArrayLike(),
        source='local',
        normalized_from='BadArrayLike',
    )
    
    # Mock blosc2.asarray to raise during attempted normalization
    original_asarray = blosc2.asarray
    
    def failing_asarray(data):
        if isinstance(data, BadArrayLike):
            raise ValueError("Cannot normalize BadArrayLike")
        return original_asarray(data)
    
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)
    monkeypatch.setattr(blosc2, "asarray", failing_asarray)

    result = dataset_management.upload_dataset("bad_data", "@personal/result.b2nd")

    assert "error" in result
    assert "normalize" in result["error"].lower()


def test_upload_dataset_handles_server_error(monkeypatch) -> None:
    # What this tests: tool handles server-side upload failures with auth hints.
    # Why important: user gets actionable feedback for permission/auth issues.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    
    # Mock client.upload() to raise a permission error
    def failing_upload(data, path):
        raise Exception("403 Forbidden: insufficient permissions")
    
    fake_client = types.SimpleNamespace(upload=failing_upload)
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    test_data = np.array([1, 2, 3])
    fake_resolved = types.SimpleNamespace(
        data=blosc2.asarray(test_data),
        source='local',
        normalized_from='ndarray',
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("data", "@personal/result.b2nd")

    assert "error" in result
    # Should include auth hint for permission failures
    assert "403" in str(result).lower() or "forbidden" in str(result).lower()


def test_upload_dataset_next_step_hint(monkeypatch) -> None:
    # What this tests: successful upload includes guidance for next operations.
    # Why important: agent knows how to chain operations (use uploaded path in follow-ups).
    fake_upload_result = _FakeUploadResult("@personal/chained.b2nd")
    fake_client = types.SimpleNamespace(upload=lambda data, path: fake_upload_result)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client)
    
    test_data = np.array([1, 2, 3])
    fake_resolved = types.SimpleNamespace(
        data=blosc2.asarray(test_data),
        source='local',
        normalized_from='ndarray',
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("data", "@personal/result.b2nd")

    assert "next_step" in result
    assert "upstream" not in result["next_step"].lower()  # Not upstream, downstream use


def test_upload_dataset_destination_path_extraction(monkeypatch) -> None:
    # What this tests: destination path is correctly extracted from upload result.
    # Why important: accurate path reporting for follow-up operations.
    
    # Test case 1: result.path attribute
    fake_upload_result_with_path = types.SimpleNamespace(path="@personal/result1.b2nd")
    fake_client_1 = types.SimpleNamespace(upload=lambda data, path: fake_upload_result_with_path)
    
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_client", lambda: fake_client_1)
    
    test_data = np.array([1, 2, 3])
    fake_resolved = types.SimpleNamespace(
        data=blosc2.asarray(test_data),
        source='local',
        normalized_from='ndarray',
    )
    monkeypatch.setattr(dataset_management, "resolve_data", lambda _: fake_resolved)

    result = dataset_management.upload_dataset("data", "@personal/result1.b2nd")
    assert result["destination_path"] == "@personal/result1.b2nd"


if __name__ == "__main__":
    # Allow direct execution for debugging
    import pytest
    pytest.main([__file__, "-v"])
