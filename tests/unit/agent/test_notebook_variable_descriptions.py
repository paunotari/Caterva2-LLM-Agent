"""Unit tests for notebook variable description behavior."""

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


def test_describe_variable_skips_lazy_stats() -> None:
    # What this tests: lazy-like objects are described without computing reductions.
    # Why important: reductions on lazy server-backed data can trigger brittle fetches.
    class FakeLazyExpr:
        __module__ = "blosc2.lazyexpr"

        shape = (10, 20)
        dtype = "float32"

        def __len__(self):
            return 10

        def min(self):
            raise AssertionError("min() should not be called for lazy objects")

        def max(self):
            raise AssertionError("max() should not be called for lazy objects")

        def mean(self):
            raise AssertionError("mean() should not be called for lazy objects")

    desc = notebook._describe_variable("norm_kevlar", FakeLazyExpr())

    assert "shape=(10, 20)" in desc
    assert "dtype=float32" in desc
    assert "stats=deferred(lazy)" in desc


def test_describe_variable_handles_stats_failure_gracefully() -> None:
    # What this tests: unexpected reduction failures do not crash description generation.
    # Why important: ask() must stay robust even if array backends throw at runtime.
    class FailingStatsArray:
        shape = (5, 5)
        dtype = "float64"

        def __len__(self):
            return 5

        def min(self):
            raise RuntimeError("backend unavailable")

        def max(self):
            raise RuntimeError("backend unavailable")

        def mean(self):
            raise RuntimeError("backend unavailable")

    desc = notebook._describe_variable("arr", FailingStatsArray())
    assert "stats=unavailable" in desc
