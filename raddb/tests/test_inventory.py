"""
raddb/tests/test_inventory.py
------------------------------
Tests for ``RadDB.inventory()`` — the on-disk data overview (archive side and
DataTree-input side).

Run with:
    pytest raddb/tests/test_inventory.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from raddb import RadDB
from raddb.tests.test_fixes import _make_datatree

RADAR = "A"
VOL_TIMES = [pd.Timestamp("2024-08-01 12:00:00"), pd.Timestamp("2024-08-02 06:30:00")]


@pytest.fixture
def archive(tmp_path):
    """A two-volume, one-radar archive."""
    db = RadDB(archive_dir=str(tmp_path / "archive"), crs=2056)
    db.archive(datatree={str(t): _make_datatree(vol_time=t) for t in VOL_TIMES}, radar=RADAR)
    return db


@pytest.fixture
def datatree_dir(tmp_path):
    """A directory of saved DataTree files, not archived."""
    d = tmp_path / "datatrees"
    d.mkdir()
    for t in VOL_TIMES:
        _make_datatree(vol_time=t).to_netcdf(d / f"{RADAR}_{t:%Y%m%d_%H%M%S}.nc")
    return d


class TestInventoryArchive:
    def test_lists_radar_volumes_and_time_range(self, archive, capsys):
        assert archive.inventory() is None  # prints, returns nothing
        out = capsys.readouterr().out
        assert "archived data" in out
        assert f"\n  {RADAR} " in out          # the per-radar row
        assert "2024-08-01 12:00:00 .. 2024-08-02 06:30:00" in out
        assert "volumes" in out

    def test_detailed_adds_lut_columns_and_days(self, archive, capsys):
        archive.inventory(detailed=True)
        out = capsys.readouterr().out
        assert "LUT:" in out and "sweeps" in out
        assert "DBZH" in out                     # per-volume moment columns
        assert "2024-08-01" in out and "2024-08-02" in out
        assert "volume(s)" in out

    def test_empty_archive_dir(self, tmp_path, capsys):
        RadDB(archive_dir=str(tmp_path), crs=2056).inventory()
        assert "nothing archived here yet" in capsys.readouterr().out

    def test_without_archive_dir_raises(self):
        with pytest.raises(ValueError):
            RadDB().inventory()


class TestInventoryDataTrees:
    def test_lists_files_radar_and_time_range(self, datatree_dir, capsys):
        RadDB().inventory(datatree_dir=str(datatree_dir))
        out = capsys.readouterr().out
        assert "not archived yet" in out
        assert "files     : 2" in out
        assert "2024-08-01 12:00:00 .. 2024-08-02 06:30:00" in out

    def test_four_letter_radar_not_warned(self, tmp_path, capsys):
        """A NEXRAD-style name is archivable under gate_id v2 — no warning."""
        d = tmp_path / "nexrad"
        d.mkdir()
        _make_datatree(vol_time=VOL_TIMES[0]).to_netcdf(d / "KTLX_20240801_120000.nc")
        RadDB().inventory(datatree_dir=str(d), detailed=True)
        out = capsys.readouterr().out
        assert "KTLX" in out
        assert "not a usable radar name" not in out

    def test_warns_on_unusable_radar_name(self, tmp_path, capsys):
        d = tmp_path / "odd"
        d.mkdir()
        _make_datatree(vol_time=VOL_TIMES[0]).to_netcdf(d / "OVERLONG_20240801_120000.nc")
        RadDB().inventory(datatree_dir=str(d), detailed=True)
        out = capsys.readouterr().out
        assert "OVERLONG" in out
        assert "not a usable radar name" in out

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RadDB().inventory(datatree_dir=str(tmp_path / "nope"))

    def test_empty_directory(self, tmp_path, capsys):
        RadDB().inventory(datatree_dir=str(tmp_path))
        assert "no .zarr / .nc" in capsys.readouterr().out
