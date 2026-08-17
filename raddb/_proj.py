"""PROJ data-directory sanity check, run once when ``raddb`` is imported.

Importing this module must happen *before* anything imports pyproj (directly
or through geopandas / cartopy); ``raddb/__init__.py`` imports it first.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

__all__ = ["PROJ_DATA", "fix_foreign_proj_data"]


def fix_foreign_proj_data() -> str | None:
    """Stop a foreign ``PROJ_DATA``/``PROJ_LIB`` from breaking every CRS lookup.

    A ``PROJ_DATA`` (or legacy ``PROJ_LIB``) inherited from *another*
    environment — a conda base env, a system PROJ — passes pyproj's
    "does proj.db exist?" test but ships a database of the wrong PROJ version,
    so every projection raises::

        CRSError: Invalid projection: EPSG:2056:
        (Internal Proj Error: proj_create: no database context specified)

    which disables ``RadDB.extent`` / ``crs`` / ``to_geopandas`` / ``crop_*`` /
    ``extract_cross_section`` and the LUT ``x_*``/``y_*`` columns.  Point PROJ
    at the running interpreter's own ``share/proj`` instead; if this prefix has
    none, drop the foreign variables and let pyproj use the data it bundles.

    Nothing happens when the variables are unset (pyproj resolves its own data
    correctly) or already point inside ``sys.prefix``.

    Returns
    -------
    str or None
        The PROJ data directory that was set, or ``None`` if nothing was changed
        (or the foreign variables were merely removed).
    """
    current = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if not current:
        return None

    prefix = Path(sys.prefix).resolve()
    # RuntimeError as well as OSError: pathlib re-raises a symlink loop as RuntimeError.
    with contextlib.suppress(OSError, RuntimeError):
        if Path(current).resolve().is_relative_to(prefix):
            return None  # this environment's own PROJ data — leave it alone

    own = prefix / "share" / "proj"
    if (own / "proj.db").is_file():
        new = str(own)
        os.environ["PROJ_DATA"] = new
        os.environ["PROJ_LIB"] = new
    else:
        new = None
        os.environ.pop("PROJ_DATA", None)
        os.environ.pop("PROJ_LIB", None)

    # pyproj reads the environment once, at import time; if something already
    # imported it (geopandas, cartopy, ...) the variables alone are too late.
    pyproj = sys.modules.get("pyproj")
    if pyproj is not None and new is not None:
        with contextlib.suppress(Exception):
            pyproj.datadir.set_data_dir(new)
    return new


PROJ_DATA = fix_foreign_proj_data()


def _demo() -> None:
    """Self-check: a foreign PROJ_DATA is replaced, an in-prefix one is kept."""
    import tempfile

    saved = {k: os.environ.get(k) for k in ("PROJ_DATA", "PROJ_LIB")}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["PROJ_DATA"] = tmp  # foreign, and without a proj.db
            os.environ.pop("PROJ_LIB", None)
            fix_foreign_proj_data()
            assert os.environ.get("PROJ_DATA") != tmp, "foreign PROJ_DATA kept"

        own = Path(sys.prefix) / "share" / "proj"
        if (own / "proj.db").is_file():
            os.environ["PROJ_DATA"] = str(own)
            assert fix_foreign_proj_data() is None, "own PROJ_DATA should be left alone"
            assert os.environ["PROJ_DATA"] == str(own)

        os.environ.pop("PROJ_DATA", None)
        os.environ.pop("PROJ_LIB", None)
        assert fix_foreign_proj_data() is None, "unset PROJ_DATA should stay unset"
        assert "PROJ_DATA" not in os.environ

        import pyproj

        assert pyproj.CRS.from_epsg(4326).name  # a CRS lookup actually works
        print("PROJ data OK:", pyproj.datadir.get_data_dir())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    _demo()
