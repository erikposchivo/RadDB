"""
raddb/tests/test_datatree_io.py
-------------------------------
Tests for the public DataTree-file workflow: discovery
(``find_datatree_files``), loading (``open_any_datatree``), and end-to-end
archiving from saved DataTree files (``RadDB.archive(datatree_dir=...)``).

All tests use synthetic DataTrees written to tmp_path — no real radar
files required.  NetCDF/Zarr-dependent tests skip when the backend is
not installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from raddb.tests.test_fixes import _make_datatree  # noqa: E402  (has lat/lon for LUT gen)

RADAR = "A"


def _write_nc_volumes(directory: Path, minutes: tuple[int, ...] = (0, 5, 10)) -> list[Path]:
    """Write synthetic volumes as NetCDF, named with a parseable timestamp."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in minutes:
        vol_time = pd.Timestamp(f"2024-01-01 12:{m:02d}:00")
        dt = _make_datatree(n_sweeps=2, vol_time=vol_time)
        p = directory / f"vol_{vol_time:%Y%m%d_%H%M%S}.nc"
        dt.to_netcdf(p)
        paths.append(p)
    return paths


# ===========================================================================
# open_any_datatree
# ===========================================================================

class TestOpenAnyDatatree:

    def test_netcdf_roundtrip(self, tmp_path):
        pytest.importorskip("netCDF4")
        from raddb.io_core import open_any_datatree

        dt = _make_datatree(n_sweeps=2)
        p = tmp_path / "vol_20240101_120000.nc"
        dt.to_netcdf(p)

        loaded = open_any_datatree(p)
        assert sorted(g.lstrip("/") for g in loaded.groups if "sweep" in g) == [
            "sweep_1", "sweep_2",
        ]
        np.testing.assert_allclose(
            loaded["sweep_1"].to_dataset()["DBZH"].values,
            dt["sweep_1"].to_dataset()["DBZH"].values,
        )

    def test_zarr_roundtrip(self, tmp_path):
        pytest.importorskip("zarr")
        from raddb.io_core import open_any_datatree

        dt = _make_datatree(n_sweeps=2)
        p = tmp_path / "vol_20240101_120000.zarr"
        dt.to_zarr(p)

        loaded = open_any_datatree(p)
        np.testing.assert_allclose(
            loaded["sweep_1"].to_dataset()["DBZH"].values,
            dt["sweep_1"].to_dataset()["DBZH"].values,
        )

    def test_missing_path_raises(self, tmp_path):
        from raddb.io_core import open_any_datatree

        with pytest.raises(FileNotFoundError):
            open_any_datatree(tmp_path / "nope.nc")


# ===========================================================================
# find_datatree_files
# ===========================================================================

class TestFindDatatreeFiles:

    def test_finds_nc_and_zarr_leaves(self, tmp_path):
        from raddb.discovery import find_datatree_files

        (tmp_path / "vol_20240101_000000.nc").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "vol_20240101_000500.nc").touch()
        store = tmp_path / "vol_20240101_001000.zarr"
        store.mkdir()
        # a .nc INSIDE the zarr store must not be matched (store is a leaf)
        (store / "inner_20240101_002000.nc").touch()
        (tmp_path / "notes.txt").touch()

        found = find_datatree_files(tmp_path)
        names = [p.name for p in found]
        assert names == [
            "vol_20240101_000000.nc",
            "vol_20240101_000500.nc",
            "vol_20240101_001000.zarr",
        ]

    def test_non_recursive(self, tmp_path):
        from raddb.discovery import find_datatree_files

        (tmp_path / "vol_20240101_000000.nc").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "vol_20240101_000500.nc").touch()

        found = find_datatree_files(tmp_path, recursive=False)
        assert [p.name for p in found] == ["vol_20240101_000000.nc"]

    def test_time_filter(self, tmp_path):
        from raddb.discovery import find_datatree_files

        (tmp_path / "vol_20240101_000000.nc").touch()
        (tmp_path / "vol_20240101_010000.nc").touch()
        (tmp_path / "vol_20240101_020000.nc").touch()
        (tmp_path / "no_timestamp_here.nc").touch()

        found = find_datatree_files(
            tmp_path,
            start_time="2024-01-01 00:30",
            end_time="2024-01-01 01:30",
        )
        # in-range file kept + unparseable file kept (strict_time=False, last)
        assert [p.name for p in found] == [
            "vol_20240101_010000.nc",
            "no_timestamp_here.nc",
        ]

        found_strict = find_datatree_files(
            tmp_path,
            start_time="2024-01-01 00:30",
            end_time="2024-01-01 01:30",
            strict_time=True,
        )
        assert [p.name for p in found_strict] == ["vol_20240101_010000.nc"]

    def test_missing_directory_raises(self, tmp_path):
        from raddb.discovery import find_datatree_files

        with pytest.raises(FileNotFoundError):
            find_datatree_files(tmp_path / "nope")


class TestParseDatatreeFileTime:

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("vol_20240101_000000.nc", "2024-01-01 00:00:00"),
            ("A_20240715T235959.nc", "2024-07-15 23:59:59"),
            ("radar-20240101-1230.zarr", "2024-01-01 12:30:00"),
            ("x_202401011230.nc", "2024-01-01 12:30:00"),
            ("no_time.nc", None),
        ],
    )
    def test_patterns(self, name, expected):
        from raddb.discovery import _parse_datatree_file_time

        ts = _parse_datatree_file_time(Path(name))
        if expected is None:
            assert ts is None
        else:
            assert ts == pd.Timestamp(expected, tz="UTC")


# ===========================================================================
# RadDB.archive(datatree_dir=...) (end to end)
# ===========================================================================

class TestArchiveFromDatatrees:

    def test_end_to_end(self, tmp_path):
        pytest.importorskip("netCDF4")
        from raddb.main import RadDB

        src = tmp_path / "input"
        out = tmp_path / "archive"
        _write_nc_volumes(src)

        db = RadDB(archive_dir=str(out))
        res = db.archive(datatree_dir=src, radar=RADAR)
        assert (res["n_archived"], res["n_failed"]) == (3, 0)

        # LUT auto-generated
        assert (out / RADAR / "LUT" / f"{RADAR}_LUT.parquet").exists()
        # one POL parquet per volume
        pol_files = list((out / RADAR).rglob("*_POL.parquet"))
        assert len(pol_files) == 3

        # loading works and the filter kept only DBZH > 0
        rdf = db.open(
            radars=RADAR,
            time_period=("2024-01-01 00:00", "2024-01-01 23:59"),
        )
        assert len(rdf) > 0
        assert (rdf.data["DBZH"] > 0.0).all()

    def test_resume_skips_archived(self, tmp_path):
        pytest.importorskip("netCDF4")
        from raddb.main import RadDB

        src = tmp_path / "input"
        out = tmp_path / "archive"
        _write_nc_volumes(src)

        db = RadDB(archive_dir=str(out))
        assert db.archive(datatree_dir=src, radar=RADAR)["n_archived"] == 3
        # second run: everything checkpointed
        assert db.archive(datatree_dir=src, radar=RADAR)["n_archived"] == 0

    def test_time_period_subset(self, tmp_path):
        pytest.importorskip("netCDF4")
        from raddb.main import RadDB

        src = tmp_path / "input"
        out = tmp_path / "archive"
        _write_nc_volumes(src, minutes=(0, 5, 10))

        db = RadDB(archive_dir=str(out))
        res = db.archive(
            datatree_dir=src, radar=RADAR,
            time_period=("2024-01-01 12:04", "2024-01-01 12:11"),
        )
        assert res["n_archived"] == 2  # 12:05 and 12:10 only

    def test_unrecognized_radar_skipped(self, tmp_path):
        pytest.importorskip("netCDF4")
        from raddb.main import RadDB

        src = tmp_path / "input"
        out = tmp_path / "archive"
        _write_nc_volumes(src)  # files are named vol_* -> radar "vol" (not A-Z)

        db = RadDB(archive_dir=str(out))
        res = db.archive(datatree_dir=src)  # radar=None -> infer per file; "vol" skipped
        assert res["n_archived"] == 0

    def test_both_sources_raises(self, tmp_path):
        from raddb.main import RadDB

        db = RadDB(archive_dir=str(tmp_path))
        with pytest.raises(ValueError, match="exactly one"):
            db.archive(datatree_dir=tmp_path, datatree=object())

    def test_empty_source_returns_zero(self, tmp_path):
        from raddb.main import RadDB

        src = tmp_path / "empty"
        src.mkdir()
        db = RadDB(archive_dir=str(tmp_path / "out"))
        assert db.archive(datatree_dir=src, radar=RADAR)["n_archived"] == 0
