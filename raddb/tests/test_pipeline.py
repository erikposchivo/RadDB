"""
raddb/tests/test_pipeline.py
----------------------------
Tests for the sequential archiving pipeline (raddb.pipeline) and the
sequential MCH batch pipeline (mch_pipeline.process_mch_volumes).

All tests use synthetic DataTrees — no real METRANET files are required.
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
    dbzh_min: float = -5.0,
    dbzh_max: float = 30.0,
    n_sweeps: int = 1,
    vol_time: pd.Timestamp | None = None,
) -> xr.DataTree:
    """Minimal but valid DataTree with DBZH spanning clear-sky and rain."""
    if vol_time is None:
        vol_time = pd.Timestamp("2024-08-01 12:00:00")

    az = np.linspace(0, 360 - 360 / n_az, n_az)
    rng_vals = np.linspace(1000, 20_000, n_rng)
    time_vals = np.array([vol_time] * n_az, dtype="datetime64[ns]")

    dict_ds = {}
    for sweep_idx in range(1, n_sweeps + 1):
        dbzh = np.random.uniform(dbzh_min, dbzh_max, (n_az, n_rng)).astype(np.float32)
        ds = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], dbzh),
                "ZDR":  (["azimuth", "range"], np.ones((n_az, n_rng), np.float32)),
                "RHOHV":(["azimuth", "range"], np.full((n_az, n_rng), 0.95, np.float32)),
                "PHIDP":(["azimuth", "range"], np.zeros((n_az, n_rng), np.float32)),
                "time": (["azimuth"], time_vals),
            },
            coords={
                "azimuth":         az,
                "range":           rng_vals,
                "elevation":       (["azimuth"], np.full(n_az, 0.5 * sweep_idx)),
                "elevation_angle": 0.5 * sweep_idx,
            },
        )
        ds.attrs["sweep_number"] = sweep_idx
        dict_ds[f"sweep_{sweep_idx}"] = ds

    return xr.DataTree.from_dict(dict_ds)


@pytest.fixture
def tiny_datatree():
    return _make_datatree()


@pytest.fixture
def base_path(tmp_path):
    return str(tmp_path)


# ===========================================================================
# SEQUENTIAL ARCHIVING
# ===========================================================================

class TestSequentialArchive:
    """archive_multiple_volumes correctness tests."""

    def test_single_volume_success(self, tiny_datatree, base_path):
        from raddb.pipeline import archive_multiple_volumes

        results = archive_multiple_volumes(
            {"vol_001": tiny_datatree},
            radar=RADAR,
            base_output_path=base_path,
            verbose=False,
        )

        assert len(results) == 1
        r = results[0]
        assert r["success"] is True, f"Expected success, got error: {r.get('error')}"
        assert r["error"] is None
        assert r["n_gates"] > 0

        df = pd.read_parquet(r["polar_path"])
        assert "gate_id" in df.columns
        assert "DBZH" in df.columns
        assert (df["DBZH"] > 0.0).all(), "Clear-sky gates should have been removed"

    def test_multiple_volumes(self, tiny_datatree, base_path):
        from raddb.pipeline import archive_multiple_volumes

        volumes = {
            f"vol_{i:03d}": _make_datatree(
                vol_time=pd.Timestamp(f"2024-08-01 12:0{i}:00")
            )
            for i in range(4)
        }
        results = archive_multiple_volumes(
            volumes, radar=RADAR, base_output_path=base_path, verbose=False
        )

        assert len(results) == 4
        assert all(r["success"] for r in results)

    def test_bad_datatree_captured_as_failure(self, base_path):
        from raddb.pipeline import archive_multiple_volumes

        bad_ds = xr.Dataset({"DBZH": (["azimuth", "range"], np.ones((5, 5)))})
        bad_dt = xr.DataTree.from_dict({"sweep_1": bad_ds})

        results = archive_multiple_volumes(
            {"bad_vol": bad_dt},
            radar=RADAR,
            base_output_path=base_path,
            verbose=False,
        )

        assert len(results) == 1
        r = results[0]
        assert r["success"] is False
        assert r["error"] is not None


# ===========================================================================
# MULTI-RADAR ARCHIVING
# ===========================================================================

class TestMultiRadarArchive:
    """archive_volumes_multi_radar tests."""

    def test_multi_radar_sequential(self, base_path):
        from raddb.pipeline import archive_volumes_multi_radar

        volumes_by_radar = {
            "A": {"vol_001": _make_datatree(vol_time=pd.Timestamp("2024-08-01 17:00:00"))},
            "D": {"vol_001": _make_datatree(vol_time=pd.Timestamp("2024-08-01 17:00:00"))},
        }
        all_results = archive_volumes_multi_radar(
            volumes_by_radar,
            base_output_path=base_path,
            verbose=False,
        )

        assert set(all_results.keys()) == {"A", "D"}
        for radar_key, res_list in all_results.items():
            assert len(res_list) == 1
            assert res_list[0]["success"] is True


# ===========================================================================
# TIMER AGGREGATION
# ===========================================================================

class TestTimerAggregation:
    """StageTimer records accumulate across sequential archiving."""

    def test_timer_accumulates_across_run(self, base_path):
        from raddb.pipeline import archive_multiple_volumes
        from raddb.helper import StageTimer

        timer = StageTimer()
        volumes = {
            f"vol_{i:03d}": _make_datatree(
                vol_time=pd.Timestamp(f"2024-08-01 19:0{i}:00")
            )
            for i in range(3)
        }

        archive_multiple_volumes(
            volumes, radar=RADAR, base_output_path=base_path,
            timer=timer, verbose=False,
        )

        df = timer.to_dataframe()
        assert len(df) >= 3
        assert "archive_volume" in df["stage"].values
