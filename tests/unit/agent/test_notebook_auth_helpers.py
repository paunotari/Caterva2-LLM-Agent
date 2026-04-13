"""Unit tests for notebook-level login/logout/auth_status helpers."""

from __future__ import annotations

import sys
import types

if "IPython" not in sys.modules:
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None
    sys.modules["IPython"] = fake_ipython

if "IPython.display" not in sys.modules:
    fake_display = types.ModuleType("IPython.display")
    fake_display.display = lambda *_args, **_kwargs: None
    fake_display.Markdown = lambda value: value
    sys.modules["IPython.display"] = fake_display

from caterva2_agent import notebook


def test_login_interactive_prompts_hidden_password_and_calls_login(monkeypatch) -> None:
    # What this tests: interactive helper reads username + hidden password then delegates.
    # Why important: keeps passwords out of notebook cells while reusing login flow.
    captured: dict[str, tuple[str, str]] = {}

    monkeypatch.setattr("builtins.input", lambda _prompt: "alice")
    monkeypatch.setattr(notebook, "getpass", lambda _prompt: "secret")
    monkeypatch.setattr(
        notebook,
        "login",
        lambda username, password: captured.update({"credentials": (username, password)}),
    )

    notebook.login_interactive()

    assert captured["credentials"] == ("alice", "secret")


def test_login_interactive_uses_provided_username_without_prompt(monkeypatch) -> None:
    # What this tests: optional username bypasses input prompt.
    # Why important: allows scripted notebook flows with hidden password only.
    captured: dict[str, tuple[str, str]] = {}
    input_called = {"value": False}

    def _unexpected_input(_prompt: str) -> str:
        input_called["value"] = True
        return "should-not-be-used"

    monkeypatch.setattr("builtins.input", _unexpected_input)
    monkeypatch.setattr(notebook, "getpass", lambda _prompt: "secret")
    monkeypatch.setattr(
        notebook,
        "login",
        lambda username, password: captured.update({"credentials": (username, password)}),
    )

    notebook.login_interactive("alice")

    assert input_called["value"] is False
    assert captured["credentials"] == ("alice", "secret")


def test_login_success_calls_set_client_auth_and_displays_status(monkeypatch) -> None:
    # What this tests: notebook login wires to runtime auth state manager.
    # Why important: this is the user-facing entry point for authentication.
    captured: dict[str, tuple[str, str]] = {}
    outputs: list[str] = []

    def _fake_set_client_auth(username: str, password: str) -> dict[str, str | bool]:
        captured["credentials"] = (username, password)
        return {
            "status": "authenticated",
            "urlbase": "http://server.local",
            "authenticated": True,
            "username": username,
        }

    monkeypatch.setattr(notebook, "set_client_auth", _fake_set_client_auth)
    monkeypatch.setattr(notebook, "_display_response", outputs.append)

    notebook.login(" alice ", "secret")

    assert captured["credentials"] == ("alice", "secret")
    assert "Authenticated on `http://server.local` as `alice`" in outputs[0]


def test_login_empty_password_prints_clear_error(capsys) -> None:
    # What this tests: invalid input is rejected before touching client state.
    # Why important: avoids accidental auth calls with malformed arguments.
    notebook.login("alice", "")
    captured = capsys.readouterr()
    assert "password cannot be empty" in captured.out


def test_auth_status_formats_authenticated_state(monkeypatch) -> None:
    # What this tests: status message includes server and username in auth mode.
    # Why important: users need a quick, trustworthy session check before writes.
    outputs: list[str] = []
    monkeypatch.setattr(
        notebook,
        "get_client_auth_status",
        lambda: {
            "urlbase": "http://server.local",
            "authenticated": True,
            "username": "alice",
        },
    )
    monkeypatch.setattr(notebook, "_display_response", outputs.append)

    notebook.auth_status()

    assert "Mode: `authenticated`" in outputs[0]
    assert "User: `alice`" in outputs[0]


def test_logout_calls_clear_client_auth_and_displays_status(monkeypatch) -> None:
    # What this tests: notebook logout delegates and confirms anonymous mode.
    # Why important: explicit de-auth should be visible and predictable.
    outputs: list[str] = []
    monkeypatch.setattr(
        notebook,
        "clear_client_auth",
        lambda: {
            "status": "anonymous",
            "urlbase": "http://server.local",
            "authenticated": False,
            "username": None,
        },
    )
    monkeypatch.setattr(notebook, "_display_response", outputs.append)

    notebook.logout()

    assert "Logged out" in outputs[0]
    assert "`http://server.local`" in outputs[0]
