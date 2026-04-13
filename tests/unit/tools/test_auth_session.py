"""Unit tests for Caterva2 runtime authentication lifecycle."""

from __future__ import annotations

import pytest

from caterva2_agent.tools import _base as base


class _AuthCapableClient:
    """Client double that supports auth and close semantics."""

    instances: list["_AuthCapableClient"] = []

    def __init__(self, url: str, auth: tuple[str, str] | None = None):
        self.url = url
        self.auth = auth
        self.closed = False
        self.__class__.instances.append(self)

    def get_roots(self) -> dict[str, dict[str, str]]:
        if self.auth is not None and self.auth[1] != "ok-password":
            raise RuntimeError("401 Unauthorized")
        return {"@public": {"name": "@public"}}

    def close(self) -> None:
        self.closed = True


def _reset_client_state() -> None:
    """Reset module-level auth/client globals for isolated tests."""
    base._cat2_client = None
    base._cat2_client_urlbase = base.CATERVA2_URLBASE
    base._cat2_client_auth = None
    base._cat2_auth_username = None
    _AuthCapableClient.instances.clear()


def test_set_client_auth_success_replaces_anonymous_client(monkeypatch) -> None:
    # What this tests: login swaps shared client to authenticated mode.
    # Why important: all tools must start using auth without restarting kernel.
    monkeypatch.setattr(base.cat2, "Client", _AuthCapableClient)
    _reset_client_state()

    anonymous_client = base._get_client()
    status = base.set_client_auth("alice", "ok-password")

    assert status["authenticated"] is True
    assert status["username"] == "alice"
    assert base._get_client().auth == ("alice", "ok-password")
    assert anonymous_client.closed is True


def test_set_client_auth_failure_preserves_existing_client(monkeypatch) -> None:
    # What this tests: failed login does not destroy a valid active session.
    # Why important: avoids accidental logouts during credential retries.
    monkeypatch.setattr(base.cat2, "Client", _AuthCapableClient)
    _reset_client_state()

    base.set_client_auth("alice", "ok-password")
    active_client = base._get_client()

    with pytest.raises(ValueError, match="Authentication failed"):
        base.set_client_auth("bob", "wrong-password")

    assert base._get_client() is active_client
    status = base.get_client_auth_status()
    assert status["authenticated"] is True
    assert status["username"] == "alice"


def test_clear_client_auth_returns_to_anonymous_mode(monkeypatch) -> None:
    # What this tests: logout clears runtime auth and recreates anonymous client.
    # Why important: users need explicit account switching during notebook sessions.
    monkeypatch.setattr(base.cat2, "Client", _AuthCapableClient)
    _reset_client_state()

    base.set_client_auth("alice", "ok-password")
    auth_client = base._get_client()
    status = base.clear_client_auth()
    anon_client = base._get_client()

    assert status["authenticated"] is False
    assert status["username"] is None
    assert auth_client.closed is True
    assert anon_client.auth is None
