"""
raddb/tests/test_fixes.py
--------------------------
Tests for:
1. load_datatree — multi-volume reconstruction (duplicate gate handling)
2. sweep column present in parquet_to_dataframe(merge_lut=True)
3. generate_lut projection_epsg / projection_crs parameter

Run with:
    pytest raddb/tests/test_fixes.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from raddb.lut import generate_lut_from_datatree

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

RADAR = "A"
N_AZ = 12
N_RNG = 24


def _make_datatree(
    n_az: int = N_AZ,
    n_rng: int = N_RNG,
    dbzh_min: float = 1.0,
    dbzh_max: float = 30.0,
    n_sweeps: int = 2,
    vol_time: pd.Timestamp | None = None,
) -> xr.DataTree:
    """Build a minimal DataTree with all-positive DBZH (no clear-sky filtering)."""
    if vol_time is None:
        vol_time = pd.Timestamp("2024-08-01 12:00:00")

    az = np.linspace(0, 360 - 360 / n_az, n_az)
    rng_vals = np.linspace(1000, 20_000, n_rng)
    time_vals = np.array([vol_time] * n_az, dtype="datetime64[ns]")

    dict_ds = {}
    for sweep_idx in range(1, n_sweeps + 1):
        rng_gen = np.random.default_rng(seed=42 + sweep_idx)
        dbzh = rng_gen.uniform(dbzh_min, dbzh_max, (n_az, n_rng)).astype(np.float32)
        ds = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], dbzh),
                "ZDR": (["azimuth", "range"], np.ones((n_az, n_rng), np.float32)),
                "RHOHV": (["azimuth", "range"], np.full((n_az, n_rng), 0.95, np.float32)),
                "PHIDP": (["azimuth", "range"], np.zeros((n_az, n_rng), np.float32)),
                "time": (["azimuth"], time_vals),
            },
            coords={
                "azimuth": az,
                "range": rng_vals,
                "elevation": (["azimuth"], np.full(n_az, 0.5 * sweep_idx)),
                "elevation_angle": 0.5 * sweep_idx,
                "latitude": 46.0,
                "longitude": 7.0,
                "altitude": 1000.0,
            },
        )
        ds.attrs["sweep_number"] = sweep_idx
        dict_ds[f"sweep_{sweep_idx}"] = ds

    return xr.DataTree.from_dict(dict_ds)


# ===========================================================================
# 1. load_datatree — single volume round-trip
# ===========================================================================

class TestLoadDatatreeSingleVolume:
    """Archive one volume, generate LUT, reconstruct DataTree."""

    def test_single_volume_round_trip(self, tmp_path):
        from raddb.lut import generate_lut_from_datatree
        from raddb.io_core import datatree_to_parquet, parquet_to_datatree

        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))
        base = str(tmp_path)

        # Generate LUT
        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=base, projection_epsg=2056)

        # Archive volume
        datatree_to_parquet(dt, radar=RADAR, base_output_path=base)

        # Reconstruct
        dt_loaded = parquet_to_datatree(
            radar=RADAR, base_path=base,
            start_time="2024-08-01 00:00", end_time="2024-08-02 00:00",
            label_column="DBZH",
        )

        from raddb.helper import list_sweep_names
        sweeps = list_sweep_names(dt_loaded)
        assert len(sweeps) >= 1, "Reconstructed DataTree should have at least one sweep"
        for name in sweeps:
            ds = dt_loaded[name].to_dataset()
            assert "DBZH" in ds, f"Missing DBZH in {name}"


# ===========================================================================
# 2. load_datatree — multi-volume round-trip (the bug fix)
# ===========================================================================

class TestLoadDatatreeMultiVolume:
    """Archive two volumes, then reconstruct — must not crash on duplicates."""

    def test_multi_volume_keeps_latest(self, tmp_path):
        from raddb.lut import generate_lut_from_datatree
        from raddb.io_core import datatree_to_parquet, parquet_to_datatree

        base = str(tmp_path)
        dt1 = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))
        dt2 = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:05:00"))

        # Generate LUT from first volume
        generate_lut_from_datatree(dt1, radar=RADAR, output_base_path=base, projection_epsg=2056)

        # Archive both volumes
        datatree_to_parquet(dt1, radar=RADAR, base_output_path=base)
        datatree_to_parquet(dt2, radar=RADAR, base_output_path=base)

        # Reconstruct from both — this should NOT crash
        dt_loaded = parquet_to_datatree(
            radar=RADAR, base_path=base,
            start_time="2024-08-01 00:00", end_time="2024-08-02 00:00",
            label_column="DBZH",
        )

        from raddb.helper import list_sweep_names
        sweeps = list_sweep_names(dt_loaded)
        assert len(sweeps) >= 1
        for name in sweeps:
            ds = dt_loaded[name].to_dataset()
            assert "DBZH" in ds


# ===========================================================================
# 3. sweep column present in parquet_to_dataframe(merge_lut=True)
# ===========================================================================

class TestSweepColumnInDataFrame:
    """Verify sweep column appears when merge_lut=True."""

    def test_sweep_present_after_lut_merge(self, tmp_path):
        from raddb.lut import generate_lut_from_datatree
        from raddb.io_core import datatree_to_parquet, parquet_to_dataframe

        base = str(tmp_path)
        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))

        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=base, projection_epsg=2056)
        datatree_to_parquet(dt, radar=RADAR, base_output_path=base)

        df = parquet_to_dataframe(
            radar=RADAR, base_path=base,
            start_time="2024-08-01 00:00", end_time="2024-08-02 00:00",
            merge_lut=True,
        )

        assert not df.is_empty(), "DataFrame should not be empty"
        assert "sweep" in df.columns, "sweep column must be present after LUT merge"
        assert "azimuth" in df.columns
        assert "range" in df.columns

        # sweep should be adjacent to azimuth in column order
        cols = list(df.columns)
        sweep_idx = cols.index("sweep")
        azimuth_idx = cols.index("azimuth")
        assert abs(sweep_idx - azimuth_idx) == 1, (
            f"sweep (pos {sweep_idx}) should be adjacent to azimuth (pos {azimuth_idx})"
        )

    def test_sweep_present_in_multi_radar(self, tmp_path):
        """Verify sweep column appears after a multi-radar open + geometry merge."""
        from raddb.lut import generate_lut_from_datatree
        from raddb.io_core import datatree_to_parquet
        from raddb.main import RadDB

        base = str(tmp_path)
        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))

        # Set up two radars
        for r in ["A", "D"]:
            generate_lut_from_datatree(dt, radar=r, output_base_path=base, projection_epsg=2056)
            datatree_to_parquet(dt, radar=r, base_output_path=base)

        db = RadDB(archive_dir=base, crs=2056)
        rdf = db.open(
            radars=["A", "D"],
            time_period=("2024-08-01 00:00", "2024-08-02 00:00"),
        )
        df_multi = rdf.to_pandas(with_geometry=True)

        assert not df_multi.empty
        assert "sweep" in df_multi.columns, "sweep must be present after geometry merge"
        assert "radar" in df_multi.columns
        assert set(df_multi["radar"].unique()) == {"A", "D"}


# ===========================================================================
# 4. generate_lut with projection_epsg
# ===========================================================================

class TestGenerateLutProjection:
    """Verify projection columns are added and saved when projection_epsg/crs is set."""

    # CH1903+ / LV95 as a proj4 string (works even without the PROJ database)
    LV95_PROJ4 = (
        "+proj=somerc +lat_0=46.9524056 +lon_0=7.4395833 "
        "+k_0=1 +x_0=2600000 +y_0=1200000 "
        "+ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 "
        "+units=m +no_defs"
    )

    def _make_crs(self):
        """Create a pyproj CRS that works regardless of PROJ DB availability."""
        import pyproj
        return pyproj.CRS.from_proj4(self.LV95_PROJ4)

    def test_projection_columns_in_saved_lut(self, tmp_path):
        pyproj = pytest.importorskip("pyproj")

        from raddb.lut import generate_lut_from_datatree, load_radar_lut

        base = str(tmp_path)
        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))
        crs = self._make_crs()

        generate_lut_from_datatree(
            dt, radar=RADAR, output_base_path=base,
            projection_crs=crs,
        )

        lut = load_radar_lut(RADAR, base)
        # Column suffix is "custom" when EPSG cannot be detected from proj4
        proj_x_cols = [c for c in lut.columns if c.startswith("x_")]
        proj_y_cols = [c for c in lut.columns if c.startswith("y_")]
        assert len(proj_x_cols) == 1, f"Expected one x_ projection column, got {proj_x_cols}"
        assert len(proj_y_cols) == 1, f"Expected one y_ projection column, got {proj_y_cols}"
        assert lut[proj_x_cols[0]].is_not_null().any(), "Projected x values should not all be NaN"

    def test_archiving_without_a_crs_is_refused(self, tmp_path):
        """A CRS is mandatory: a wrong or absent one silently breaks every AOI."""
        with pytest.raises(ValueError, match="requires a CRS"):
            generate_lut_from_datatree(
                _make_datatree(), radar=RADAR, output_base_path=str(tmp_path)
            )

    def test_refusal_names_a_usable_crs(self, tmp_path):
        """The message must tell the user what to pass, not just complain."""
        with pytest.raises(ValueError, match=r"RadDB\(crs=32632\)"):
            generate_lut_from_datatree(
                _make_datatree(), radar=RADAR, output_base_path=str(tmp_path)
            )

    def test_a_crs_invalid_at_the_site_is_refused(self, tmp_path):
        """EPSG:2056 outside Switzerland distorts distance ~20%."""
        from raddb.tests.test_fixes import _make_datatree as mk
        dt = mk()
        for name in list(dt.children):
            ds = dt[name].to_dataset().assign_coords(latitude=35.33, longitude=-97.28)
            dt[name] = xr.DataTree(ds)
        with pytest.raises(ValueError, match="distorts distance"):
            generate_lut_from_datatree(
                dt, radar=RADAR, output_base_path=str(tmp_path), projection_epsg=2056
            )

    def test_api_archive_with_projection(self, tmp_path):
        """archive() auto-generates a LUT with projected columns when crs is set."""
        pyproj = pytest.importorskip("pyproj")

        from raddb.main import RadDB
        from raddb.lut import load_radar_lut

        crs = self._make_crs()
        db = RadDB(archive_dir=str(tmp_path), crs=crs)
        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))

        db.archive(datatree=dt, radar=RADAR)

        lut = load_radar_lut(RADAR, str(tmp_path))
        proj_x_cols = [c for c in lut.columns if c.startswith("x_")]
        proj_y_cols = [c for c in lut.columns if c.startswith("y_")]
        assert len(proj_x_cols) == 1
        assert len(proj_y_cols) == 1


# ===========================================================================
# 4. datatree_to_dataframe — dimension coordinates that carry no index
# ===========================================================================


class TestUnindexedDimensionCoordinate:
    """Raw NEXRAD Level II sweeps arrive with ``range`` as a coordinate that has
    no index.  ``to_dataframe`` then indexes that dimension by position and emits
    the true values as a column of the same name, which ``reset_index`` refuses
    to insert.  The flattener rebuilds the index first.
    """

    @staticmethod
    def _drop_range_index(dt: xr.DataTree) -> xr.DataTree:
        """Reproduce the shape xradar hands back for a raw Level II volume."""
        for name in dt.children:
            dt[name].dataset = dt[name].to_dataset().drop_indexes("range")
        return dt

    def test_flatten_keeps_true_range_values(self):
        from raddb.io_core import datatree_to_dataframe

        expected = datatree_to_dataframe(_make_datatree())
        got = datatree_to_dataframe(self._drop_range_index(_make_datatree()))

        assert got.shape == expected.shape
        # Metres, not the positions 0..n_rng-1 that the broken shape would give.
        assert sorted(got["range"].unique().to_list()) == sorted(expected["range"].unique().to_list())
        assert got["range"].min() == pytest.approx(1000.0)

    def test_archive_accepts_it(self, tmp_path):
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        dt = self._drop_range_index(_make_datatree())
        result = RadDB(archive_dir=str(tmp_path), crs=2056).archive(datatree=dt, radar=RADAR)

        assert result["n_archived"] == 1 and result["n_failed"] == 0


# ===========================================================================
# 6. A volume with nothing to archive is skipped, not a crash
# ===========================================================================


class TestEmptyVolumeIsSkipped:
    """A clear-air volume used to crash the batch instead of being skipped.

    ``_save_polar_parquet`` builds the output path out of the volume's own
    time.  When every gate fails the filter the frame is empty, ``.min()`` is
    ``None``, ``pd.to_datetime(None)`` is ``NaT``, and ``pd.NaT.month`` is
    *nan* — a float — so ``f"{...:02d}"`` raised ``Unknown format code 'd' for
    object of type 'float'``.  Two real Rad4Alp volumes hit this.
    """

    @staticmethod
    def _blank_dbzh(dt: xr.DataTree) -> xr.DataTree:
        """Null out DBZH everywhere, so the default ``DBZH > 0`` keeps nothing."""
        for name in dt.children:
            ds = dt[name].to_dataset()
            ds["DBZH"] = ds["DBZH"].where(False)  # all-NaN, same shape/dtype
            dt[name].dataset = ds
        return dt

    @staticmethod
    def _blank_time(dt: xr.DataTree) -> xr.DataTree:
        """Make every ray's time NaT while leaving DBZH intact."""
        for name in dt.children:
            ds = dt[name].to_dataset()
            ds["time"] = xr.full_like(ds["time"], np.datetime64("NaT"))
            dt[name].dataset = ds
        return dt

    def test_no_gates_survive_the_filter(self, tmp_path):
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        dt = self._blank_dbzh(_make_datatree())
        res = RadDB(archive_dir=str(tmp_path), crs=2056).archive(datatree=dt, radar=RADAR)

        assert (res["n_archived"], res["n_failed"], res["n_skipped"]) == (0, 0, 1)
        assert not list((tmp_path / RADAR).rglob("*_POL.parquet"))

    def test_all_nat_time_is_skipped(self, tmp_path):
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        dt = self._blank_time(_make_datatree())
        res = RadDB(archive_dir=str(tmp_path), crs=2056).archive(datatree=dt, radar=RADAR)

        assert (res["n_archived"], res["n_failed"], res["n_skipped"]) == (0, 0, 1)
        assert not list((tmp_path / RADAR).rglob("*_POL.parquet"))

    def test_counts_sum_to_the_volumes_attempted(self, tmp_path):
        """One good volume + one empty: 1 archived, 0 failed, 1 skipped."""
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        good = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))
        empty = self._blank_dbzh(_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:05:00")))
        res = RadDB(archive_dir=str(tmp_path), crs=2056).archive(
            datatree=[good, empty], radar=RADAR
        )

        assert (res["n_archived"], res["n_failed"], res["n_skipped"]) == (1, 0, 1)
        assert res["n_archived"] + res["n_failed"] + res["n_skipped"] == 2
        assert len(list((tmp_path / RADAR).rglob("*_POL.parquet"))) == 1

    def test_the_archive_stays_readable(self, tmp_path):
        """A skipped volume must not poison the rest of the archive."""
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        db = RadDB(archive_dir=str(tmp_path), crs=2056)
        db.archive(datatree=_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00")),
                   radar=RADAR)
        db.archive(datatree=self._blank_dbzh(_make_datatree(
            vol_time=pd.Timestamp("2024-08-01 12:05:00"))), radar=RADAR)

        rdf = RadDB(archive_dir=str(tmp_path)).open(radars=RADAR)
        assert len(rdf) > 0
        assert rdf.radars() == [RADAR]

    def test_save_polar_parquet_returns_none_directly(self):
        """The guard itself, without going through archive()."""
        import polars as pl

        from raddb.io_core import _save_polar_parquet

        empty = pl.DataFrame({"gate_id": [], "time": []})
        assert _save_polar_parquet(empty, RADAR, "/nonexistent") is None

    def test_disk_path_counts_and_checkpoints_a_skip(self, tmp_path):
        """``datatree_dir=`` counts a skip separately and does not retry it."""
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        src = tmp_path / "trees"
        src.mkdir()
        _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00")).to_netcdf(
            src / f"{RADAR}_20240801_120000.nc"
        )
        self._blank_dbzh(_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:05:00"))).to_netcdf(
            src / f"{RADAR}_20240801_120500.nc"
        )
        arch = tmp_path / "arch"

        res = RadDB(archive_dir=str(arch), crs=2056).archive(datatree_dir=str(src), radar=RADAR)
        assert (res["n_archived"], res["n_failed"], res["n_skipped"]) == (1, 0, 1)
        assert len(list((arch / RADAR).rglob("*_POL.parquet"))) == 1

        # The skip is checkpointed, so a resume re-attempts nothing.
        again = RadDB(archive_dir=str(arch), crs=2056).archive(datatree_dir=str(src), radar=RADAR)
        assert (again["n_archived"], again["n_failed"], again["n_skipped"]) == (0, 0, 0)

    def test_multi_radar_path_counts_a_skip(self, tmp_path):
        """The ``{radar: [volumes]}`` form keeps the three counts separate too."""
        pytest.importorskip("pyproj")
        from raddb.main import RadDB

        volumes = {
            RADAR: [
                _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00")),
                self._blank_dbzh(_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:05:00"))),
            ],
            "D": [_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))],
        }
        res = RadDB(archive_dir=str(tmp_path), crs=2056).archive(datatree=volumes)

        assert (res["n_archived"], res["n_failed"], res["n_skipped"]) == (2, 0, 1)
        assert res["n_archived"] + res["n_failed"] + res["n_skipped"] == 3
        assert sorted(res["radars"]) == ["A", "D"]
