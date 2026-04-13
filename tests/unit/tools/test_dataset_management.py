"""Essential unit tests for dataset management tools."""

from __future__ import annotations

import sys
import types
from pathlib import Path


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


class _FakeFile:
    """Simple file/dataset double with management methods."""

    def __init__(self):
        self.copy_calls: list[str] = []
        self.move_calls: list[str] = []
        self.remove_calls = 0
        self.download_calls: list[str | None] = []

    def copy(self, dst: str):
        self.copy_calls.append(dst)
        return dst

    def move(self, dst: str):
        self.move_calls.append(dst)
        return dst

    def remove(self):
        self.remove_calls += 1
        return "@personal/data/removed.b2nd"

    def get_download_url(self):
        return "https://cat2.example/api/fetch/personal/data.b2nd"

    def download(self, localpath=None):
        self.download_calls.append(localpath)
        return localpath or "personal/data.b2nd"


def test_copy_dataset_requires_auth(monkeypatch) -> None:
    # What this tests: write ops are blocked without authenticated session.
    # Why important: privileged operations must not run anonymously.
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": False, "urlbase": "http://test", "username": None},
    )

    result = dataset_management.copy_dataset("@public/a.b2nd", "@public/b.b2nd")

    assert "error" in result
    assert "Authentication required" in result["error"]


def test_copy_dataset_executes_when_authenticated(monkeypatch) -> None:
    # What this tests: copy path works with auth + valid server paths.
    # Why important: core non-destructive management operation.
    fake_file = _FakeFile()
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_dataset", lambda _path: fake_file)

    result = dataset_management.copy_dataset("@personal/a.b2nd", "@personal/b.b2nd")

    assert result["status"] == "success"
    assert result["operation"] == "copy"
    assert fake_file.copy_calls == ["@personal/b.b2nd"]


def test_download_dataset_defaults_to_url_mode(monkeypatch) -> None:
    # What this tests: preferred download path returns URL, not local write.
    # Why important: avoids unnecessary local-path exposure to the model.
    fake_file = _FakeFile()
    monkeypatch.setattr(dataset_management, "_get_dataset", lambda _path: fake_file)

    result = dataset_management.download_dataset("@public/data.b2nd")

    assert result["status"] == "success"
    assert result["mode"] == "url"
    assert "download_url" in result
    assert fake_file.download_calls == []


def test_move_dataset_is_dry_run_by_default(monkeypatch) -> None:
    # What this tests: move does preview by default.
    # Why important: reduces accidental destructive operations.
    fake_file = _FakeFile()
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_dataset", lambda _path: fake_file)

    result = dataset_management.move_dataset("@personal/a.b2nd", "@personal/b.b2nd")

    assert result["status"] == "dry_run"
    assert result["operation"] == "move"
    assert result["requires_confirmation"] is True
    assert "confirm=true" not in result["next_step"]
    assert fake_file.move_calls == []


def test_move_dataset_requires_confirm_for_execution(monkeypatch) -> None:
    # What this tests: move execution path enforces explicit confirm.
    # Why important: confirms safety policy layer is active.
    fake_file = _FakeFile()
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_dataset", lambda _path: fake_file)

    result = dataset_management.move_dataset(
        "@personal/a.b2nd",
        "@personal/b.b2nd",
        dry_run=False,
        confirm=False,
    )

    assert "error" in result
    assert "Confirmation required" in result["error"]
    assert result["requires_confirmation"] is True
    assert fake_file.move_calls == []


def test_remove_dataset_executes_only_with_confirm(monkeypatch) -> None:
    # What this tests: remove requires dry_run off + confirm true.
    # Why important: irreversible operations need strong guardrails.
    fake_file = _FakeFile()
    monkeypatch.setattr(
        dataset_management,
        "get_client_auth_status",
        lambda: {"authenticated": True, "urlbase": "http://test", "username": "alice"},
    )
    monkeypatch.setattr(dataset_management, "_get_dataset", lambda _path: fake_file)

    preview = dataset_management.remove_dataset("@personal/a.b2nd")
    execute = dataset_management.remove_dataset(
        "@personal/a.b2nd",
        dry_run=False,
        confirm=True,
    )

    assert preview["status"] == "dry_run"
    assert execute["status"] == "success"
    assert fake_file.remove_calls == 1
