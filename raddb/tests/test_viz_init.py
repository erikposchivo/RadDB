"""Tests for :mod:`raddb.viz` — the plotting subpackage.

``raddb/viz/__init__.py`` is a docstring and nothing else: importing ``raddb.viz`` must
stay cheap and must not drag in the optional interactive stack.  ``interactive`` needs
ipyleaflet/ipywidgets, which are a ``viz`` extra, so pulling them in eagerly would make
``import raddb`` fail on a minimal install.
"""

from __future__ import annotations

import sys

import pytest

import raddb.viz


def test_the_subpackage_imports():
    """A bare ``import raddb.viz`` succeeds and is a package."""
    assert raddb.viz.__doc__
    assert hasattr(raddb.viz, "__path__"), "raddb.viz must be a package, not a module"


def test_plot_is_importable():
    """``raddb.viz.plot`` holds the four plots plus the quicklook."""
    from raddb.viz import plot

    for name in ("plot_ppi", "plot_rhi", "plot_cappi", "plot_vcs", "plot_aoi_quicklook"):
        assert callable(getattr(plot, name))


def test_interactive_is_importable():
    """``raddb.viz.interactive`` holds the ipyleaflet AOI selector."""
    pytest.importorskip("ipyleaflet")
    from raddb.viz import interactive

    assert hasattr(interactive, "AOISelector")


def test_importing_the_subpackage_does_not_pull_in_ipyleaflet():
    """``interactive`` is optional; ``raddb.viz`` must not require it.

    Checked structurally rather than by watching ``sys.modules``, because another test in
    the session may already have imported ipyleaflet.
    """
    import ast
    from pathlib import Path

    src = Path(raddb.viz.__file__).read_text(encoding="utf-8")
    imported = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imported == [], "raddb/viz/__init__.py must stay import-free"


def test_lonboard_is_not_imported_eagerly():
    """Lonboard caches a broken pyproj context if it imports before ``raddb._proj``."""
    import ast
    from pathlib import Path

    plot_src = Path(sys.modules["raddb.viz.plot"].__file__).read_text(encoding="utf-8")
    top_level = [n for n in ast.parse(plot_src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {getattr(n, "module", None) for n in top_level}
    names |= {alias.name for n in top_level if isinstance(n, ast.Import) for alias in n.names}
    assert "lonboard" not in names, "lonboard must only be imported lazily, inside a function"
