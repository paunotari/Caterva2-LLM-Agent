"""Unit tests for clear_variables() notebook helper."""

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


def test_clear_variables_with_empty_registry() -> None:
    # What this tests: clear_variables() gracefully handles no variables.
    # Why important: user should get clear feedback, not an error.
    notebook._agent_objects.clear()
    outputs: list[str] = []
    
    notebook._display_response = outputs.append
    notebook.clear_variables()
    
    assert len(outputs) == 1
    assert "No variables to clear" in outputs[0]


def test_clear_variables_removes_from_registry() -> None:
    # What this tests: clear_variables() clears the _agent_objects tracking dict.
    # Why important: registry must be in sync with namespace.
    notebook._agent_objects.clear()
    notebook._agent_objects["data1"] = "value1"
    notebook._agent_objects["data2"] = "value2"
    
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    assert len(notebook._agent_objects) == 2
    notebook.clear_variables()
    
    assert len(notebook._agent_objects) == 0
    assert "Cleared 2 variables" in outputs[0]


def test_clear_variables_removes_from_namespace(monkeypatch) -> None:
    # What this tests: clear_variables() deletes variables from notebook namespace.
    # Why important: variables must be actually removed for memory cleanup.
    notebook._agent_objects.clear()
    
    # Create a fake namespace
    fake_namespace = {"data1": "value1", "data2": "value2", "user_var": "keep"}
    notebook._agent_objects["data1"] = "value1"
    notebook._agent_objects["data2"] = "value2"
    
    monkeypatch.setattr(notebook, "_get_notebook_namespace", lambda: fake_namespace)
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    notebook.clear_variables()
    
    # Agent-tracked variables should be gone
    assert "data1" not in fake_namespace
    assert "data2" not in fake_namespace
    
    # Non-tracked variables should remain
    assert "user_var" in fake_namespace
    assert fake_namespace["user_var"] == "keep"
    
    # Registry should be cleared
    assert len(notebook._agent_objects) == 0


def test_clear_variables_reports_count_singular(monkeypatch) -> None:
    # What this tests: clear_variables() uses correct grammar for singular.
    # Why important: user-facing text should read naturally.
    notebook._agent_objects.clear()
    notebook._agent_objects["single"] = "value"
    
    fake_namespace = {"single": "value"}
    monkeypatch.setattr(notebook, "_get_notebook_namespace", lambda: fake_namespace)
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    notebook.clear_variables()
    
    assert "Cleared 1 variable" in outputs[0]  # Not "variables"


def test_clear_variables_reports_count_plural(monkeypatch) -> None:
    # What this tests: clear_variables() uses correct grammar for plural.
    # Why important: user-facing text should read naturally.
    notebook._agent_objects.clear()
    notebook._agent_objects["data1"] = "value1"
    notebook._agent_objects["data2"] = "value2"
    notebook._agent_objects["data3"] = "value3"
    
    fake_namespace = {"data1": "value1", "data2": "value2", "data3": "value3"}
    monkeypatch.setattr(notebook, "_get_notebook_namespace", lambda: fake_namespace)
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    notebook.clear_variables()
    
    assert "Cleared 3 variables" in outputs[0]


def test_clear_variables_handles_missing_namespace(monkeypatch) -> None:
    # What this tests: clear_variables() handles None namespace gracefully.
    # Why important: not all contexts have a notebook namespace (e.g., testing).
    notebook._agent_objects.clear()
    notebook._agent_objects["data1"] = "value1"
    
    monkeypatch.setattr(notebook, "_get_notebook_namespace", lambda: None)
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    notebook.clear_variables()
    
    # Should still clear registry and report success
    assert len(notebook._agent_objects) == 0
    assert "Cleared 1 variable" in outputs[0]


def test_clear_variables_partial_namespace_overlap(monkeypatch) -> None:
    # What this tests: handles case where tracking registry differs from namespace.
    # Why important: real world: user may have deleted variables manually.
    notebook._agent_objects.clear()
    notebook._agent_objects["data1"] = "value1"
    notebook._agent_objects["data2"] = "value2"
    notebook._agent_objects["data3"] = "value3"
    
    # Only data1 and data3 in namespace (data2 was deleted by user)
    fake_namespace = {"data1": "value1", "data3": "value3"}
    monkeypatch.setattr(notebook, "_get_notebook_namespace", lambda: fake_namespace)
    outputs: list[str] = []
    notebook._display_response = outputs.append
    
    notebook.clear_variables()
    
    # Should delete what exists and clear registry
    assert "data1" not in fake_namespace
    assert "data3" not in fake_namespace
    assert len(notebook._agent_objects) == 0
    assert "Cleared 3 variables" in outputs[0]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
