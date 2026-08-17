"""
raddb/tests/test_azimuth_grid.py
--------------------------------
The nominal azimuth grid: the LUT stores a radar's *scan strategy*, and every
volume's rays are snapped onto it before their ``gate_id`` is built.

This exists because the LUT used to freeze the measured azimuths of whichever
volume was archived first.  An antenna reports where it actually pointed, which
drifts a few hundredths of a degree between rotations, and ``gate_id`` resolves
0.1° — so a drifting ray changed bin and its gates matched no LUT row.  Measured
on real data: **6%** of gates lost per volume on Rad4Alp, **35%** on WSR-88D,
silently, on every volume after the first.

Synthetic throughout; the drift is injected to match what the real files show.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from raddb.lut import (  # noqa: E402
    AZIMUTH_SCALE,
    AZIMUTH_STEPS,
    azimuth_grid_tolerance,
    nominal_azimuth_grid,
    snap_azimuths_to_grid,
    load_azimuth_grids,
)
from raddb.main import RadDB  # noqa: E402
from raddb.tests.test_fixes import _make_datatree  # noqa: E402

# Real drift, measured from the files themselves: Rad4Alp reports every ray
# ~0.033 deg past its nominal angle with a ~0.007 deg spread; WSR-88D is
# zero-mean with a spread up to ~0.045 deg.
MCH_BIAS, MCH_SPREAD = 0.0327, 0.0069
NEXRAD_SPREAD = 0.045


def _jitter(az, rng, bias=0.0, spread=MCH_SPREAD):
    return (np.asarray(az, float) + bias + rng.normal(0, spread, len(az))) % 360.0


def _retime(dt, when, rng, bias=0.0, spread=MCH_SPREAD):
    """Copy a DataTree at a new time, with the antenna pointing slightly differently."""
    out = {}
    for name, node in dt.children.items():
        ds = node.to_dataset()
        n = ds.sizes["azimuth"]
        ds = ds.assign_coords(azimuth=_jitter(ds["azimuth"].values, rng, bias, spread))
        ds["time"] = ("azimuth", np.array([when] * n, dtype="datetime64[ns]"))
        ds.attrs.update(node.attrs)
        out[name] = ds
    return xr.DataTree.from_dict(out)


# ===========================================================================
# nominal_azimuth_grid
# ===========================================================================

class TestNominalGrid:

    @pytest.mark.parametrize("n_rays,step_tenths", [(360, 10), (720, 5), (180, 20)])
    def test_spacing_follows_the_ray_count(self, n_rays, step_tenths):
        """One rule, no per-network constant: the grid comes from n_rays."""
        grid = nominal_azimuth_grid(np.arange(n_rays) * (360 / n_rays) + 0.25)
        assert grid.size == n_rays
        assert np.all(np.diff(grid) == step_tenths)

    def test_recovers_the_grid_from_jittered_rays(self):
        rng = np.random.default_rng(0)
        nominal = np.arange(360) + 0.5
        grid = nominal_azimuth_grid(_jitter(nominal, rng, MCH_BIAS))
        assert np.array_equal(grid, np.round(nominal * AZIMUTH_SCALE).astype(np.int64))

    def test_is_stable_across_volumes(self):
        """The whole point: different volumes must derive the *same* grid."""
        rng = np.random.default_rng(1)
        nominal = np.arange(360) + 0.5
        grids = [nominal_azimuth_grid(_jitter(nominal, rng, MCH_BIAS)) for _ in range(25)]
        assert all(np.array_equal(g, grids[0]) for g in grids)

    def test_super_resolution_grid_stays_uniform(self):
        """720 rays put every centre on x.x5; banker's rounding would alternate 0.4/0.6."""
        grid = nominal_azimuth_grid(np.arange(720) * 0.5 + 0.25)
        assert np.all(np.diff(grid) == 5)

    def test_offset_is_circular(self):
        """Rays straddling 0 deg must not drag the offset to the middle of the step."""
        rng = np.random.default_rng(2)
        nominal = np.arange(360) * 1.0          # rays centred on 0, 1, 2, ...
        grid = nominal_azimuth_grid(_jitter(nominal, rng, 0.0, 0.02))
        assert np.array_equal(grid, np.arange(360) * 10)

    def test_rejects_spacing_finer_than_the_resolution(self):
        with pytest.raises(ValueError, match="finer than"):
            nominal_azimuth_grid(np.arange(7200) * 0.05)

    def test_rejects_empty_sweep(self):
        with pytest.raises(ValueError, match="no rays"):
            nominal_azimuth_grid([])

    def test_rejects_a_sector_scan(self):
        """90 rays over 90 deg would silently get a 4 deg grid and collapse."""
        with pytest.raises(ValueError, match="full rotation"):
            nominal_azimuth_grid(np.arange(90, 180, 1.0))

    def test_rejects_a_sweep_with_a_large_gap(self):
        az = np.concatenate([np.arange(0, 120, 1.0), np.arange(240, 360, 1.0)])
        with pytest.raises(ValueError, match="full rotation"):
            nominal_azimuth_grid(az)

    @pytest.mark.parametrize("dropped", [[7], [7, 8], [0, 359], [3, 100, 250]])
    def test_a_rotation_with_holes_keeps_the_full_grid(self, dropped):
        """718 of 720 is a rotation with holes, not a 0.5014 deg scan strategy."""
        nominal = np.arange(720) * 0.5 + 0.25
        recorded = np.delete(nominal, dropped)
        grid = nominal_azimuth_grid(recorded)
        assert grid.size == 720                     # the missing rays keep their slots
        assert np.all(np.diff(grid) == 5)
        assert np.array_equal(grid, nominal_azimuth_grid(nominal))

    def test_holes_survive_antenna_drift(self):
        """The real case: WSR-88D drift plus two dropped rays.

        The grid is the same rotation, but not necessarily the same integers: a
        720-ray grid is centred on ``x.x5``, exactly the 0.1° rounding boundary,
        so the ~0.002° the two ray sets differ by can tip the whole grid one
        tenth either way.  That is a tenth of a degree against a half-spacing
        tolerance of 0.25°, so every ray still snaps to its own point.
        """
        rng = np.random.default_rng(7)
        nominal = np.arange(720) * 0.5 + 0.25
        recorded = np.delete(_jitter(nominal, rng, 0.0, NEXRAD_SPREAD), [11, 12])

        grid = nominal_azimuth_grid(recorded)
        assert grid.size == 720
        assert np.all(np.diff(grid) == 5)

        shift = (grid - nominal_azimuth_grid(nominal) + AZIMUTH_STEPS // 2) % AZIMUTH_STEPS
        assert np.all(np.abs(shift - AZIMUTH_STEPS // 2) <= 1)      # at most one tenth

        _, dist = snap_azimuths_to_grid(recorded, grid)
        assert dist.max() <= azimuth_grid_tolerance(grid)

    def test_too_many_holes_is_not_a_rotation(self):
        """Past the coverage floor it is indistinguishable from a sector scan."""
        nominal = np.arange(360) + 0.5
        recorded = np.delete(nominal, np.arange(0, 100))     # 260 of 360
        with pytest.raises(ValueError, match="full rotation"):
            nominal_azimuth_grid(recorded)


# ===========================================================================
# snap_azimuths_to_grid
# ===========================================================================

class TestSnapping:

    @pytest.fixture
    def grid(self):
        return nominal_azimuth_grid(np.arange(360) + 0.5)   # 0.5, 1.5, ... 359.5

    def test_snaps_to_the_nearest_point(self, grid):
        snapped, dist = snap_azimuths_to_grid([0.49, 0.51, 1.44, 1.56], grid)
        assert list(snapped) == [5, 5, 15, 15]
        assert np.allclose(dist, [0.1, 0.1, 0.6, 0.6])

    @pytest.mark.parametrize("az,expected", [
        (359.97, 3595),   # 0.47 from 359.5, 0.53 from 0.5 -> stays below the seam
        (0.02, 5),        # 0.48 from 0.5, 0.52 from 359.5 -> stays above it
        (359.60, 3595),
        (0.60, 5),
    ])
    def test_seam_is_measured_the_short_way(self, grid, az, expected):
        """Distance across 0/360 must go the short way round, not through 180."""
        snapped, _ = snap_azimuths_to_grid([az], grid)
        assert snapped[0] == expected

    def test_a_ray_below_360_can_snap_to_a_grid_point_at_zero(self):
        """With rays centred on 0, 1, 2 ..., 359.7 deg belongs to 0.0, not 359.0."""
        grid = nominal_azimuth_grid(np.arange(360) * 1.0)
        assert grid[0] == 0
        snapped, dist = snap_azimuths_to_grid([359.7, 359.4, 0.3], grid)
        assert list(snapped) == [0, 3590, 0]
        assert np.allclose(dist, [3.0, 4.0, 3.0])

    def test_full_precision_decides_the_match(self, grid):
        """Rounding to 0.1 deg first would make 0.02 deg an ambiguous tie."""
        snapped, _ = snap_azimuths_to_grid([0.02], grid)
        assert snapped[0] == 5          # not 3595

    def test_is_a_bijection_under_real_drift(self, grid):
        rng = np.random.default_rng(3)
        snapped, dist = snap_azimuths_to_grid(
            _jitter(np.arange(360) + 0.5, rng, MCH_BIAS), grid
        )
        assert np.unique(snapped).size == 360        # no two rays collapse
        assert dist.max() <= azimuth_grid_tolerance(grid)

    def test_survives_nexrad_scale_drift(self):
        grid = nominal_azimuth_grid(np.arange(720) * 0.5 + 0.25)
        rng = np.random.default_rng(4)
        snapped, dist = snap_azimuths_to_grid(
            _jitter(np.arange(720) * 0.5 + 0.25, rng, 0.0, NEXRAD_SPREAD), grid
        )
        assert np.unique(snapped).size == 720
        assert dist.max() <= azimuth_grid_tolerance(grid)

    def test_output_is_inside_one_turn(self, grid):
        snapped, _ = snap_azimuths_to_grid([0.0, 180.0, 359.999, 360.0], grid)
        assert np.all((snapped >= 0) & (snapped < AZIMUTH_STEPS))

    def test_rejects_empty_grid(self):
        with pytest.raises(ValueError, match="empty azimuth grid"):
            snap_azimuths_to_grid([1.0], [])


# ===========================================================================
# End to end: the bug this was written for
# ===========================================================================

class TestVolumesJoinTheirLut:

    @pytest.fixture
    def archive(self, tmp_path):
        """One LUT-defining volume plus four later ones with drifting azimuths."""
        db = RadDB(archive_dir=str(tmp_path / "a"), crs=2056)
        base = _make_datatree(n_az=360, n_rng=40, n_sweeps=3)
        db.archive(datatree=base, radar="A")
        rng = np.random.default_rng(7)
        for k in range(1, 5):
            when = pd.Timestamp("2024-08-01 12:00:00") + pd.Timedelta(minutes=5 * k)
            db.archive(datatree=_retime(base, when, rng, MCH_BIAS), radar="A")
        return tmp_path / "a"

    def test_every_volume_joins_completely(self, archive):
        lut = pl.read_parquet(archive / "A" / "LUT" / "A_LUT.parquet", columns=["gate_id"])
        pols = sorted((archive / "A").rglob("*_POL.parquet"))
        assert len(pols) == 5
        for f in pols:
            pol = pl.read_parquet(f, columns=["gate_id"])
            matched = pol.join(lut, on="gate_id", how="semi").height
            assert matched == pol.height, f"{f.name}: {matched}/{pol.height} joined"

    def test_the_same_ray_keeps_its_gate_id(self, archive):
        """Across volumes a gate must keep one identity, or nothing can be compared."""
        pols = sorted((archive / "A").rglob("*_POL.parquet"))
        sets = [set(pl.read_parquet(f, columns=["gate_id"])["gate_id"].to_list()) for f in pols]
        assert all(s == sets[0] for s in sets)

    def test_grid_is_recovered_from_the_lut(self, archive):
        """The info YAML no longer restates it; the LUT parquet is the source."""
        grids = load_azimuth_grids("A", archive)
        assert grids is not None and set(grids) == {1, 2, 3}
        for g in grids.values():
            assert g.size == 360 and np.all(np.diff(g) == 10)

        info = yaml.safe_load((archive / "A" / "LUT" / "A_info.yaml").read_text())
        assert "azimuths" not in info["sweeps"][1]

    def test_no_lut_means_no_grid(self, tmp_path):
        """Nothing to snap onto — the caller must keep the measured azimuths."""
        assert load_azimuth_grids("A", tmp_path) is None

    def test_lut_azimuths_are_the_grid(self, archive):
        """The LUT holds nominal angles now, not one volume's measurements."""
        lut = pl.read_parquet(archive / "A" / "LUT" / "A_LUT.parquet", columns=["sweep", "azimuth"])
        az = np.unique(lut.filter(pl.col("sweep") == 1)["azimuth"].to_numpy())
        assert np.allclose(az * AZIMUTH_SCALE, np.round(az * AZIMUTH_SCALE))

    def test_plots_and_crops_see_every_gate(self, archive):
        """The loss was invisible because it only showed up in LUT joins."""
        rdf = RadDB(archive_dir=str(archive)).open(radars="A")
        assert rdf.to_geopandas().shape[0] == rdf.data.height


class TestScanStrategyGuardrails:

    @pytest.fixture
    def db(self, tmp_path):
        d = RadDB(archive_dir=str(tmp_path / "a"), crs=2056)
        d.archive(datatree=_make_datatree(n_az=360, n_rng=20, n_sweeps=2), radar="A")
        return d

    def test_refuses_a_different_ray_count(self, db):
        with pytest.raises(ValueError, match="different scan strategy"):
            db.archive(
                datatree=_make_datatree(n_az=720, n_rng=20, n_sweeps=2,
                                        vol_time=pd.Timestamp("2024-08-01 13:00")),
                radar="A",
            )

    def test_refuses_an_unknown_sweep(self, db):
        with pytest.raises(ValueError, match="no sweep"):
            db.archive(
                datatree=_make_datatree(n_az=360, n_rng=20, n_sweeps=4,
                                        vol_time=pd.Timestamp("2024-08-01 14:00")),
                radar="A",
            )

    def test_accepts_a_volume_that_dropped_rays(self, db):
        """A volume short of a ray or two is a rotation with holes, not a new strategy."""
        rng = np.random.default_rng(21)
        base = _retime(_make_datatree(n_az=360, n_rng=20, n_sweeps=2),
                       pd.Timestamp("2024-08-01 18:00"), rng, MCH_BIAS)
        holed = xr.DataTree.from_dict({
            name: node.to_dataset().isel(azimuth=np.delete(np.arange(360), [5, 6, 200]))
            for name, node in base.children.items()
        })
        res = db.archive(datatree=holed, radar="A")
        assert (res["n_archived"], res["n_failed"]) == (1, 0)

    def test_a_lut_built_from_a_holed_volume_still_holds_every_ray(self, tmp_path):
        """The LUT is the rotation, not one volume: the dropped rays keep their rows,
        so a later complete volume joins 100%."""
        complete = _make_datatree(n_az=360, n_rng=20, n_sweeps=2)
        holed = xr.DataTree.from_dict({
            name: node.to_dataset().isel(azimuth=np.delete(np.arange(360), [5, 6, 200]))
            for name, node in complete.children.items()
        })

        db = RadDB(archive_dir=str(tmp_path / "a"), crs=2056)
        db.archive(datatree=holed, radar="A")                     # LUT from the holed volume
        lut = db.get_lut("A")
        assert lut.filter(pl.col("sweep") == 1)["azimuth"].n_unique() == 360

        res = db.archive(
            datatree=_retime(complete, pd.Timestamp("2024-08-01 19:00"),
                             np.random.default_rng(22), MCH_BIAS),
            radar="A",
        )
        assert (res["n_archived"], res["n_failed"]) == (1, 0)

        data = db.open(radars="A")
        lut_ids = set(lut["gate_id"].to_list())
        assert all(g in lut_ids for g in data.data["gate_id"].to_list())

    def test_accepts_ordinary_drift(self, db):
        rng = np.random.default_rng(11)
        base = _make_datatree(n_az=360, n_rng=20, n_sweeps=2)
        res = db.archive(
            datatree=_retime(base, pd.Timestamp("2024-08-01 15:00"), rng, MCH_BIAS),
            radar="A",
        )
        assert (res["n_archived"], res["n_failed"]) == (1, 0)

    def test_batch_reports_the_refusal_instead_of_aborting(self, db, tmp_path):
        """One incompatible volume must not take the whole batch down."""
        good = _retime(_make_datatree(n_az=360, n_rng=20, n_sweeps=2),
                       pd.Timestamp("2024-08-01 16:00"), np.random.default_rng(12), MCH_BIAS)
        bad = _make_datatree(n_az=720, n_rng=20, n_sweeps=2,
                             vol_time=pd.Timestamp("2024-08-01 17:00"))
        res = db.archive(datatree={"good": good, "bad": bad}, radar="A")
        assert res["n_archived"] == 1 and res["n_failed"] == 1

    def test_warns_when_the_lut_has_no_grid(self, tmp_path, caplog):
        """A pre-grid archive keeps working, but says that gates may not join."""
        from raddb.io_core import _build_polar_dataframe

        df = pl.DataFrame({
            "sweep": [1, 1], "azimuth": [0.53, 1.53], "range": [1000.0, 1000.0],
            "DBZH": [10.0, 20.0], "time": [pd.Timestamp("2024-01-01")] * 2,
        })
        with caplog.at_level("WARNING"):
            _build_polar_dataframe(df, "A", "DBZH", 0.0, ">", azimuth_grids=None)
        assert "no nominal azimuth grid" in caplog.text
