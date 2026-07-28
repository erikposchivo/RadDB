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
        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=base)

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
        generate_lut_from_datatree(dt1, radar=RADAR, output_base_path=base)

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

        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=base)
        datatree_to_parquet(dt, radar=RADAR, base_output_path=base)

        df = parquet_to_dataframe(
            radar=RADAR, base_path=base,
            start_time="2024-08-01 00:00", end_time="2024-08-02 00:00",
            merge_lut=True,
        )

        assert not df.empty, "DataFrame should not be empty"
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
            generate_lut_from_datatree(dt, radar=r, output_base_path=base)
            datatree_to_parquet(dt, radar=r, base_output_path=base)

        db = RadDB(archive_dir=base)
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

    def test_lut_without_projection_has_no_extra_cols(self, tmp_path):
        from raddb.lut import generate_lut_from_datatree, load_radar_lut

        base = str(tmp_path)
        dt = _make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00"))

        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=base)

        lut = load_radar_lut(RADAR, base)
        proj_cols = [c for c in lut.columns if c.startswith("x_") or c.startswith("y_")]
        assert len(proj_cols) == 0, f"No projection columns expected, got {proj_cols}"

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
