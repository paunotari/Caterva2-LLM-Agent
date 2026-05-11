"""Unit tests for notebook display helpers."""

from __future__ import annotations

import importlib
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


def test_display_response_strips_markdown_image_tags(monkeypatch) -> None:
    # What this tests: assistant markdown image tags are removed before rendering.
    # Why important: visualization tools already render figures; duplicate markdown
    # images can appear as broken placeholders in notebook output.
    importlib.reload(notebook)

    rendered: list[str] = []
    monkeypatch.setattr(notebook, "display", lambda value: rendered.append(value))
    monkeypatch.setattr(notebook, "Markdown", lambda value: value)

    notebook._display_response(
        "Projection ready.\n\n![projection](data:image/png;base64,AAAA)\n\nDetails below."
    )

    assert len(rendered) == 1
    assert "![projection]" not in rendered[0]
    assert "Projection ready." in rendered[0]
    assert "Details below." in rendered[0]
