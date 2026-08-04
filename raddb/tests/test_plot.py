"""
raddb/tests/test_plot.py
------------------------
Tests for the four gate-accurate plots — ``plot_ppi``, ``plot_rhi``,
``plot_cappi`` and ``plot_vcs`` — plus the LUT geometry they read.

Covers:

1. each plot returns a matplotlib artist and honours ``ax=`` (subplot composition)
2. filtered / ``sel``-ed / cropped frames plot exactly the gates they still hold
3. one exact geometry path; beamwidth resolution and its warning
4. DataTree input — geometry computed from its own coords, no archive
5. GeoDataFrame input and its error contract
6. CAPPI slice invariants — every drawn gate really spans the altitude, chords
   stay inside their range bin, overlap resolution leaves no double coverage
7. coordinate frames, volume selection, and the error paths
8. ``aoi.py`` now derives gate footprints from the ``h_plane`` lattice

All tests are synthetic (``tmp_path``); no real radar files needed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely

from raddb.main import RadDB
from raddb.lut import cappi_chords, ensure_gate_planes, load_plane_nodes, lut_file_path
from raddb.tests.test_fixes import _make_datatree

RADAR = "L"
N_AZ, N_RNG, N_SWEEPS = 72, 60, 6
VOL_TIMES = [pd.Timestamp("2024-08-01 12:00:00"), pd.Timestamp("2024-08-02 06:30:00")]


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A one-radar, one-volume archive with the full 5-file LUT."""
    base = tmp_path_factory.mktemp("plot_archive")
    db = RadDB(archive_dir=str(base), crs=2056)
    db.archive(datatree={RADAR: [_make_datatree(N_AZ, N_RNG, n_sweeps=N_SWEEPS)]})
    return base


@pytest.fixture(scope="module")
def rdf(archive):
    return RadDB(archive_dir=str(archive), crs=2056).open(radars=RADAR)


@pytest.fixture(scope="module")
def site(archive):
    """Radar site in EPSG:2056."""
    from raddb.aoi import _reproject_to_aoi
    info = RadDB(archive_dir=str(archive), crs=2056).get_radar_info(RADAR)
    p = _reproject_to_aoi(shapely.Point(info["longitude"], info["latitude"]), 4326, 2056)
    return (p.x, p.y)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _n_polys(artist):
    return len(artist.get_paths())


# ===========================================================================
# 1. Each plot draws one plot, returns an artist, and composes via ax=
# ===========================================================================

class TestBasicRendering:
    def test_ppi_returns_polycollection(self, rdf):
        from matplotlib.collections import PolyCollection
        p = rdf.plot_ppi(sweep=1)
        assert isinstance(p, PolyCollection)
        assert _n_polys(p) > 0

    def test_rhi_returns_polycollection(self, rdf):
        assert _n_polys(rdf.plot_rhi(azimuth=0)) > 0

    def test_cappi_returns_polycollection(self, rdf):
        assert _n_polys(rdf.plot_cappi(altitude=1200)) > 0

    def test_vcr_returns_polycollection(self, rdf, site):
        line = ((site[0] - 12_000, site[1] - 12_000), (site[0] + 12_000, site[1] + 12_000))
        assert _n_polys(rdf.plot_vcs(line=line)) > 0

    @pytest.mark.parametrize("call", [
        lambda r, ax: r.plot_ppi(sweep=1, ax=ax),
        lambda r, ax: r.plot_rhi(azimuth=0, ax=ax),
        lambda r, ax: r.plot_cappi(altitude=1200, ax=ax),
    ])
    def test_draws_into_the_supplied_axes(self, rdf, call):
        fig, ax = plt.subplots()
        p = call(rdf, ax)
        assert p.axes is ax
        assert len(ax.collections) == 1

    def test_composes_a_multi_panel_figure(self, rdf):
        """One plot per Axes — the user builds the panel, not the plot function."""
        fig, axes = plt.subplots(2, 2)
        for ax, var in zip(axes.ravel(), ["DBZH", "ZDR", "RHOHV", "PHIDP"]):
            rdf.plot_ppi(sweep=1, variable=var, ax=ax)
        assert all(len(ax.collections) == 1 for ax in axes.ravel())

    def test_save_writes_a_file(self, rdf, tmp_path):
        out = tmp_path / "ppi.png"
        rdf.plot_ppi(sweep=1, save=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_title_and_colorbar_are_optional(self, rdf):
        p = rdf.plot_ppi(sweep=1, add_colorbar=False, title="custom")
        assert p.axes.get_title() == "custom"


# ===========================================================================
# 2. Partial frames — a crop plots exactly the gates it still holds
# ===========================================================================

class TestPartialFrames:
    def test_filtered_frame_draws_fewer_gates(self, rdf):
        sub = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})
        assert 0 < len(sub) < len(rdf)
        assert _n_polys(sub.plot_ppi(sweep=1)) < _n_polys(rdf.plot_ppi(sweep=1))

    def test_polygon_count_equals_surviving_gate_count(self, rdf):
        sub = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})
        n_sweep1 = (
            sub.data.select("gate_id")
            .join(RadDB(archive_dir=str(sub.archive_dir), crs=2056).get_lut(RADAR)
                  .filter(pl.col("sweep") == 1).select("gate_id"),
                  on="gate_id", how="semi")
            .height
        )
        assert _n_polys(sub.plot_ppi(sweep=1)) == n_sweep1

    def test_cropped_frame_plots(self, rdf, site):
        crop = rdf.crop_around_point(site, distance=8_000)
        assert 0 < len(crop) < len(rdf)
        assert _n_polys(crop.plot_ppi(sweep=1)) > 0
        assert _n_polys(crop.plot_cappi(altitude=1100)) > 0

    def test_sel_frame_plots(self, rdf):
        sel = rdf.sel(range=slice(2_000, 12_000))
        assert 0 < len(sel) < len(rdf)
        assert _n_polys(sel.plot_ppi(sweep=1)) > 0

    def test_empty_selection_raises(self, rdf):
        empty = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 1e9})
        with pytest.raises(ValueError):
            empty.plot_ppi(sweep=1)
# ===========================================================================
# 3. Geometry source — one exact path, chosen by input type
# ===========================================================================

class TestGeometrySource:
    """There is no approximate mode: every plot draws the exact frustum."""

    @pytest.mark.parametrize("fn", ["plot_ppi", "plot_rhi", "plot_cappi"])
    def test_no_plot_takes_a_beamwidth(self, fn):
        """Beamwidth belongs to LUT generation, not to plotting."""
        import inspect
        import raddb.viz.plot as vp
        assert "beamwidth_deg" not in inspect.signature(getattr(vp, fn)).parameters

    def test_datatree_beamwidth_comes_from_the_file(self):
        """No archive to bake it in, so it is inferred as LUT generation does."""
        from raddb.viz.plot import _beamwidth
        from raddb.lut import DEFAULT_BEAMWIDTH_DEG

        dt = _make_datatree(N_AZ, N_RNG, n_sweeps=N_SWEEPS)
        src = type("S", (), {"kind": "datatree", "dtree": dt})()
        assert _beamwidth(src) == DEFAULT_BEAMWIDTH_DEG

        dt.attrs["radar_beam_width_h"] = 0.5
        assert _beamwidth(src) == 0.5


# ===========================================================================
# 3b. DataTree input — geometry from its own coordinates, no archive
# ===========================================================================

class TestDataTreeInput:
    @pytest.fixture(scope="class")
    def dtree(self):
        return _make_datatree(N_AZ, N_RNG, n_sweeps=N_SWEEPS)

    def test_ppi_exact_from_a_raw_datatree(self, dtree):
        from raddb.viz.plot import plot_ppi
        assert _n_polys(plot_ppi(dtree, sweep=1, variable="DBZH")) == N_AZ * N_RNG

    def test_cappi_uses_the_files_beamwidth(self, dtree):
        """A wider declared beam reaches the slice altitude over more bins."""
        from raddb.viz.plot import plot_cappi
        import copy
        narrow = copy.deepcopy(dtree); narrow.attrs["radar_beam_width_h"] = 0.5
        wide = copy.deepcopy(dtree); wide.attrs["radar_beam_width_h"] = 2.0
        assert (_n_polys(plot_cappi(wide, altitude=1200, variable="DBZH"))
                > _n_polys(plot_cappi(narrow, altitude=1200, variable="DBZH")))

    def test_no_archive_is_needed(self, dtree):
        """The whole point: a DataTree is self-describing."""
        from raddb.viz.plot import plot_ppi
        assert _n_polys(plot_ppi(dtree, sweep=1, variable="DBZH", archive_dir=None)) > 0

    def test_datatree_ppi_matches_the_lut_geometry(self, dtree, archive):
        """Computed corners must equal the ones generation stored."""
        from raddb.viz.plot import plot_ppi
        from raddb.lut import gate_corner_table

        dt_verts = np.array([q.vertices[:4]
                             for q in plot_ppi(dtree, sweep=1, variable="DBZH").get_paths()])
        tbl = gate_corner_table(RADAR, archive, kind="h_plane", sweep=1)
        lut_verts = np.stack([
            np.stack([tbl[f"x_{k}"].to_numpy(), tbl[f"y_{k}"].to_numpy()], axis=1)
            for k in range(1, 5)], axis=1)
        # The lattices are stored as float32, so ~1e-7 relative — a few cm at
        # 200 km range. Anything tighter would be testing parquet, not geometry.
        assert np.abs(np.sort(dt_verts, axis=0) - np.sort(lut_verts, axis=0)).max() < 5e-2

    def test_rhi_from_a_datatree(self, dtree):
        from raddb.viz.plot import plot_rhi
        assert _n_polys(plot_rhi(dtree, azimuth=0, variable="DBZH")) == N_RNG * N_SWEEPS

    def test_cappi_from_a_datatree(self, dtree):
        from raddb.viz.plot import plot_cappi
        assert _n_polys(plot_cappi(dtree, altitude=1200, variable="DBZH")) > 0

    def test_datatree_cappi_matches_the_lut_path(self, dtree, rdf):
        from raddb.viz.plot import plot_cappi
        assert (_n_polys(plot_cappi(dtree, altitude=1200, variable="DBZH"))
                == _n_polys(rdf.plot_cappi(altitude=1200)))

    def test_unknown_variable_raises(self, dtree):
        from raddb.viz.plot import plot_ppi
        with pytest.raises(KeyError):
            plot_ppi(dtree, sweep=1, variable="NOPE")

    def test_missing_sweep_raises(self, dtree):
        from raddb.viz.plot import plot_ppi
        with pytest.raises(ValueError, match="sweep"):
            plot_ppi(dtree, sweep=99, variable="DBZH")


# ===========================================================================
# 3c. GeoDataFrame input
# ===========================================================================

class TestGeoDataFrameInput:
    @pytest.fixture(scope="class")
    def gdf(self, archive):
        return RadDB(archive_dir=str(archive), crs=2056).open(radars=RADAR).to_geopandas()

    def test_goes_through_the_lut(self, gdf, archive):
        from raddb.viz.plot import plot_ppi
        assert _n_polys(plot_ppi(gdf, sweep=1, archive_dir=archive)) > 0

    def test_without_an_archive_raises(self, gdf):
        """A gdf now always needs the LUT: geometry comes from there."""
        from raddb.viz.plot import plot_ppi
        with pytest.raises(ValueError, match="archive"):
            plot_ppi(gdf, sweep=1)

    def test_rhi_from_a_gdf(self, gdf, archive):
        from raddb.viz.plot import plot_rhi
        assert _n_polys(plot_rhi(gdf, azimuth=0, archive_dir=archive)) > 0

    def test_cappi_from_a_gdf_works_like_a_frame(self, gdf, archive, rdf):
        """Geometry comes from the LUT, so a gdf is just a frame here."""
        from raddb.viz.plot import plot_cappi
        assert (_n_polys(plot_cappi(gdf, altitude=1200, archive_dir=archive))
                == _n_polys(rdf.plot_cappi(altitude=1200)))

    def test_geometry_column_is_not_required_in_exact_mode(self, gdf, archive):
        """A gdf is just a frame there; its geometry is ignored."""
        from raddb.viz.plot import plot_ppi
        plain = gdf.drop(columns=gdf.geometry.name)
        assert _n_polys(plot_ppi(plain, sweep=1, archive_dir=archive)) > 0


# ===========================================================================
# 4. CAPPI geometry
# ===========================================================================

class TestCappiChords:
    def test_every_reported_bin_spans_the_altitude(self, archive):
        z0 = 1200.0
        chords = cappi_chords(RADAR, archive, z0)
        assert not chords.is_empty()

        nodes = load_plane_nodes(RADAR, archive, "v_plane")
        nodes = nodes.filter(pl.col("az_idx") == pl.col("az_idx").min())
        for (sw,), sub in chords.group_by(["sweep"], maintain_order=True):
            bot = nodes.filter((pl.col("sweep") == sw) & (pl.col("el_level") == -1)).sort("rng_idx")
            top = nodes.filter((pl.col("sweep") == sw) & (pl.col("el_level") == 1)).sort("rng_idx")
            zb, zt = bot["z_asl"].to_numpy(), top["z_asl"].to_numpy()
            j = sub["rng_idx"].to_numpy()
            lo = np.minimum.reduce([zb[j], zb[j + 1], zt[j], zt[j + 1]])
            hi = np.maximum.reduce([zb[j], zb[j + 1], zt[j], zt[j + 1]])
            assert ((lo - 1e-3 <= z0) & (z0 <= hi + 1e-3)).all()

    def test_chords_stay_inside_their_range_bin(self, archive):
        chords = cappi_chords(RADAR, archive, 1200.0)
        nodes = load_plane_nodes(RADAR, archive, "v_plane")
        nodes = nodes.filter(pl.col("az_idx") == pl.col("az_idx").min())
        for (sw,), sub in chords.group_by(["sweep"], maintain_order=True):
            bot = nodes.filter((pl.col("sweep") == sw) & (pl.col("el_level") == -1)).sort("rng_idx")
            top = nodes.filter((pl.col("sweep") == sw) & (pl.col("el_level") == 1)).sort("rng_idx")
            db_, dt_ = bot["d"].to_numpy(), top["d"].to_numpy()
            j = sub["rng_idx"].to_numpy()
            lo = np.minimum.reduce([db_[j], db_[j + 1], dt_[j], dt_[j + 1]])
            hi = np.maximum.reduce([db_[j], db_[j + 1], dt_[j], dt_[j + 1]])
            assert (sub["d_near"].to_numpy() >= lo - 1e-2).all()
            assert (sub["d_far"].to_numpy() <= hi + 1e-2).all()

    def test_d_near_is_below_d_far(self, archive):
        chords = cappi_chords(RADAR, archive, 1200.0)
        assert (chords["d_near"].to_numpy() <= chords["d_far"].to_numpy()).all()

    def test_each_sweep_contributes_a_contiguous_band(self, archive):
        """Beam thickness far exceeds the rise per bin, so bands are contiguous."""
        for (_sw,), sub in cappi_chords(RADAR, archive, 1200.0).group_by(["sweep"]):
            j = np.sort(sub["rng_idx"].to_numpy())
            assert np.array_equal(j, np.arange(j.min(), j.max() + 1))

    def test_above_every_beam_is_empty(self, archive):
        assert cappi_chords(RADAR, archive, 50_000.0).is_empty()

    def test_asl_and_relative_altitudes_agree(self, archive):
        info = RadDB(archive_dir=str(archive), crs=2056).get_radar_info(RADAR)
        a = cappi_chords(RADAR, archive, 1200.0, height="asl")
        b = cappi_chords(RADAR, archive, 1200.0 - info["altitude"], height="rel")
        assert a.height == b.height

    def test_bad_height_reference_raises(self, archive):
        with pytest.raises(ValueError):
            cappi_chords(RADAR, archive, 1200.0, height="furlongs")


class TestCappiRendering:
    def test_overlap_nearest_draws_fewer_gates_than_all(self, rdf):
        assert (_n_polys(rdf.plot_cappi(altitude=1200, overlap="nearest"))
                < _n_polys(rdf.plot_cappi(altitude=1200, overlap="all")))

    def test_overlap_nearest_leaves_no_double_coverage(self, archive):
        """The resolved chords must partition the ground-distance axis."""
        from raddb.viz.plot import _resolve_chord_overlap
        resolved = _resolve_chord_overlap(cappi_chords(RADAR, archive, 1200.0))
        iv = np.sort(np.stack([resolved["d_near"].to_numpy(),
                               resolved["d_far"].to_numpy()], axis=1), axis=0)
        assert (iv[1:, 0] >= iv[:-1, 1] - 1e-3).all()

    def test_fill_lowest_extends_the_far_field(self, rdf):
        assert (_n_polys(rdf.plot_cappi(altitude=1200, fill_lowest=True))
                >= _n_polys(rdf.plot_cappi(altitude=1200, fill_lowest=False)))

    def test_altitude_above_every_beam_raises(self, rdf):
        with pytest.raises(ValueError, match="reaches"):
            rdf.plot_cappi(altitude=99_999.0)

    def test_higher_slice_draws_fewer_gates(self, rdf):
        """Fewer beams reach higher, so the slice shrinks."""
        assert (_n_polys(rdf.plot_cappi(altitude=1400))
                < _n_polys(rdf.plot_cappi(altitude=1100)))

    def test_bad_overlap_raises(self, rdf):
        with pytest.raises(ValueError):
            rdf.plot_cappi(altitude=1200, overlap="sometimes")

    def test_slice_polygons_sit_inside_the_full_footprints(self, rdf, archive):
        """The constant-z cut trims gates along the beam; it never grows them."""
        from raddb.lut import gate_corner_table
        p = rdf.plot_cappi(altitude=1200, overlap="all")
        drawn = np.array([path.vertices[:4] for path in p.get_paths()])
        tbl = gate_corner_table(RADAR, archive, kind="h_plane")
        full = np.stack([np.stack([tbl[f"x_{k}"].to_numpy(), tbl[f"y_{k}"].to_numpy()], axis=1)
                         for k in range(1, 5)], axis=1)
        assert shapely.area(shapely.polygons(drawn)).sum() <= \
            shapely.area(shapely.polygons(full)).sum()


# ===========================================================================
# 5. Coordinates, volume selection, errors
# ===========================================================================

class TestCoordinateFrames:
    @pytest.mark.parametrize("coords", ["xy", "cartesian", "lonlat", "geo",
                                        "projected", "swiss", "lv95", 2056])
    def test_accepted_frames(self, rdf, coords):
        assert _n_polys(rdf.plot_ppi(sweep=1, coords=coords)) > 0

    def test_lonlat_axes_are_in_degrees(self, rdf):
        p = rdf.plot_ppi(sweep=1, coords="lonlat")
        assert -180 <= p.axes.get_xlim()[0] <= 180
        assert -90 <= p.axes.get_ylim()[0] <= 90

    def test_projected_axes_are_lv95_metres(self, rdf):
        p = rdf.plot_ppi(sweep=1, coords=2056)
        assert 2.4e6 < p.axes.get_xlim()[0] < 2.9e6

    def test_xy_is_centred_on_the_radar(self, rdf):
        p = rdf.plot_ppi(sweep=1, coords="xy")
        assert p.axes.get_xlim()[0] < 0 < p.axes.get_xlim()[1]

    def test_unknown_frame_raises(self, rdf):
        with pytest.raises(ValueError, match="coords"):
            rdf.plot_ppi(sweep=1, coords="banana")

    def test_projected_resolves_the_archives_own_crs(self, archive):
        """Reading needs no CRS: the archive records the one it was written with."""
        plain = RadDB(archive_dir=str(archive)).open(radars=RADAR)
        assert plain._crs is None                      # nothing was declared
        assert plain.crs().to_epsg() == 2056           # but the archive knows
        assert _n_polys(plain.plot_ppi(sweep=1, coords="projected")) > 0

    def test_projected_raises_when_nothing_declares_a_crs(self, rdf):
        """A bare frame with no archive has nothing to resolve from."""
        from raddb.viz.plot import plot_ppi
        with pytest.raises((ValueError, KeyError)):
            plot_ppi(rdf.data, sweep=1, coords="projected", archive_dir=None)


class TestVolumeSelection:
    @pytest.fixture(scope="class")
    def multi(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("multi_vol")
        db = RadDB(archive_dir=str(base), crs=2056)
        db.archive(datatree={str(t): _make_datatree(24, 20, n_sweeps=2, vol_time=t)
                             for t in VOL_TIMES}, radar=RADAR)
        return db.open(radars=RADAR)

    def test_several_volumes_without_timestep_raises(self, multi):
        with pytest.raises(ValueError, match="volumes"):
            multi.plot_ppi(sweep=1)

    def test_timestep_picks_the_nearest_volume(self, multi):
        assert _n_polys(multi.plot_ppi(sweep=1, timestep=VOL_TIMES[0])) > 0

    def test_time_window_narrows_to_one_volume(self, multi):
        assert _n_polys(multi.plot_ppi(
            sweep=1, start_time="2024-08-01", end_time="2024-08-01 23:59")) > 0

    def test_window_that_excludes_everything_raises(self, multi):
        with pytest.raises(ValueError):
            multi.plot_ppi(sweep=1, start_time="1999-01-01", end_time="1999-12-31")


class TestErrors:
    def test_unknown_variable_raises(self, rdf):
        with pytest.raises(KeyError):
            rdf.plot_ppi(sweep=1, variable="NOT_A_VAR")

    def test_missing_sweep_raises(self, rdf):
        with pytest.raises(ValueError):
            rdf.plot_ppi(sweep=99)

    def test_rhi_beyond_az_tol_raises(self, rdf):
        with pytest.raises(ValueError, match="no sweep has a ray within"):
            rdf.plot_rhi(azimuth=2.5, az_tol=0.1)

    def test_vcr_without_a_section_raises(self, rdf):
        with pytest.raises(ValueError, match="cross-section"):
            rdf.plot_vcs()

    def test_bare_frame_without_archive_dir_raises(self, rdf):
        from raddb.viz.plot import plot_ppi
        with pytest.raises(ValueError, match="archive"):
            plot_ppi(rdf.data, sweep=1)

    def test_bare_frame_with_archive_dir_works(self, rdf):
        from raddb.viz.plot import plot_ppi
        assert _n_polys(plot_ppi(rdf.data, sweep=1, archive_dir=rdf.archive_dir)) > 0

    def test_multi_radar_frame_without_radar_raises(self, tmp_path):
        db = RadDB(archive_dir=str(tmp_path), crs=2056)
        db.archive(datatree={"A": [_make_datatree(24, 20, n_sweeps=2)],
                             "D": [_make_datatree(24, 20, n_sweeps=2)]})
        with pytest.raises(ValueError, match="radars"):
            db.open().plot_ppi(sweep=1)


class TestRhi:
    def test_height_reference_shifts_the_axis(self, rdf, archive):
        alt = RadDB(archive_dir=str(archive), crs=2056).get_radar_info(RADAR)["altitude"]
        asl = rdf.plot_rhi(azimuth=0, height="asl").axes.get_ylim()[0]
        rel = rdf.plot_rhi(azimuth=0, height="rel").axes.get_ylim()[0]
        assert asl - rel == pytest.approx(alt, abs=1.0)

    def test_bad_height_raises(self, rdf):
        with pytest.raises(ValueError):
            rdf.plot_rhi(azimuth=0, height="furlongs")

    def test_draws_one_ray_across_every_sweep(self, rdf):
        assert _n_polys(rdf.plot_rhi(azimuth=0)) == N_RNG * N_SWEEPS

    def test_picks_a_ray_per_sweep_when_azimuths_jitter(self, tmp_path):
        """Real antenna azimuths differ slightly between sweeps.

        Matching one azimuth *value* across the whole LUT would then select a
        single sweep and collapse the RHI — every sweep needs its own nearest ray.
        """
        import xarray as xr

        dt = _make_datatree(n_az=36, n_rng=20, n_sweeps=4)
        jittered = {}
        for i, (name, node) in enumerate(dt.children.items()):
            ds = node.to_dataset()
            # Offset each sweep's azimuths, as a real antenna does.
            ds = ds.assign_coords(azimuth=ds["azimuth"].values + 0.13 * i)
            jittered[name] = ds
        dt = xr.DataTree.from_dict(jittered)

        db = RadDB(archive_dir=str(tmp_path), crs=2056)
        db.archive(datatree={RADAR: [dt]})
        r = db.open(radars=RADAR)

        lut = db.get_lut(RADAR)
        # Each sweep has its own 36 azimuths, so no value is shared between them.
        assert lut["azimuth"].n_unique() == 36 * 4, "fixture should have per-sweep jitter"

        p = r.plot_rhi(azimuth=90.0, az_tol=1.0)
        # One ray per sweep, every range bin: 20 x 4.  Selecting a single azimuth
        # *value* across the whole LUT would yield 20 — one sweep only.
        assert _n_polys(p) == 20 * 4


class TestVcr:
    def test_accepts_a_linestring(self, rdf, site):
        line = shapely.LineString([(site[0] - 12_000, site[1] - 12_000),
                                   (site[0] + 12_000, site[1] + 12_000)])
        assert _n_polys(rdf.plot_vcs(line=line)) > 0

    def test_accepts_a_precut_raddb(self, rdf, site):
        cs = rdf.extract_cross_section((site[0] - 12_000, site[1] - 12_000),
                                       (site[0] + 12_000, site[1] + 12_000))
        assert _n_polys(cs.plot_vcs()) == len(cs)

    def test_deprecated_alias_still_works(self, rdf, site):
        cs = rdf.extract_cross_section((site[0] - 12_000, site[1] - 12_000),
                                       (site[0] + 12_000, site[1] + 12_000))
        with pytest.deprecated_call():
            assert _n_polys(cs.plot_cross_section()) > 0

    def test_datatree_is_refused(self):
        from raddb.viz.plot import plot_vcs
        with pytest.raises(TypeError, match="Archive the volume first"):
            plot_vcs(_make_datatree(24, 20, n_sweeps=2), line=((0, 0), (1, 1)))

    def test_line_and_precut_frame_is_ambiguous(self, rdf, site):
        cs = rdf.extract_cross_section((site[0] - 12_000, site[1] - 12_000),
                                       (site[0] + 12_000, site[1] + 12_000))
        with pytest.raises(ValueError, match="ambiguous"):
            cs.plot_vcs(line=((site[0], site[1]), (site[0] + 5_000, site[1])))

    def test_area_cropped_frame_has_no_section(self, rdf, site):
        """An AOI crop selects an area, not a line."""
        crop = rdf.crop_around_point(site, distance=10_000)
        with pytest.raises(ValueError, match="no 'cs_polygon'"):
            crop.plot_vcs()

    def test_geojson_line_honours_its_own_crs(self, rdf, site, tmp_path, archive):
        """A lon/lat GeoJSON must not be read as LV95 metres."""
        import json
        from raddb.aoi import _to_pyproj_crs
        import pyproj

        tf = pyproj.Transformer.from_crs(_to_pyproj_crs(2056), _to_pyproj_crs(4326),
                                         always_xy=True)
        a = tf.transform(site[0] - 12_000, site[1] - 12_000)
        b = tf.transform(site[0] + 12_000, site[1] + 12_000)
        path = tmp_path / "section.geojson"
        path.write_text(json.dumps({"type": "LineString", "coordinates": [list(a), list(b)]}))

        from_file = _n_polys(rdf.plot_vcs(line=str(path)))
        from_lv95 = _n_polys(rdf.plot_vcs(line=((site[0] - 12_000, site[1] - 12_000),
                                                (site[0] + 12_000, site[1] + 12_000))))
        assert from_file > 0
        assert abs(from_file - from_lv95) <= 0.02 * from_lv95

    def test_precut_frame_survives_a_pandas_or_gdf_round_trip(self, rdf, site, archive):
        """cs_polygon holds shapely objects; they must WKB-encode into polars."""
        from raddb.viz.plot import plot_vcs
        cs = rdf.extract_cross_section((site[0] - 12_000, site[1] - 12_000),
                                       (site[0] + 12_000, site[1] + 12_000))
        n = _n_polys(cs.plot_vcs())
        assert _n_polys(plot_vcs(cs.to_pandas(), archive_dir=archive)) == n
        assert _n_polys(plot_vcs(cs.to_geopandas(), archive_dir=archive)) == n

    @pytest.mark.parametrize("as_frame", ["polars", "pandas", "geopandas"])
    def test_line_works_from_a_bare_frame_with_an_archive(self, rdf, site, archive, as_frame):
        """A bare frame has gate_id and the archive has the geometry, so the
        section is cuttable — plot_vcs must not be stricter than plot_ppi."""
        from raddb.viz.plot import plot_vcs
        data = {"polars": rdf.data,
                "pandas": rdf.to_pandas(),
                "geopandas": rdf.to_geopandas()}[as_frame]
        line = ((site[0] - 12_000, site[1] - 12_000), (site[0] + 12_000, site[1] + 12_000))
        assert (_n_polys(plot_vcs(data, line=line, archive_dir=archive))
                == _n_polys(rdf.plot_vcs(line=line)))

    def test_line_from_a_bare_frame_without_an_archive_raises(self, rdf, site):
        from raddb.viz.plot import plot_vcs
        with pytest.raises(ValueError, match="archive"):
            plot_vcs(rdf.data, line=((site[0], site[1]), (site[0] + 5_000, site[1])))

    def test_shapefile_line(self, rdf, site, tmp_path):
        import shapefile
        w = shapefile.Writer(str(tmp_path / "sec")); w.field("id", "N")
        w.line([[[site[0] - 12_000, site[1] - 12_000], [site[0] + 12_000, site[1] + 12_000]]])
        w.record(1); w.close()
        assert _n_polys(rdf.plot_vcs(line=str(tmp_path / "sec.shp"))) > 0


# ===========================================================================
# 6. LUT plumbing the plots depend on
# ===========================================================================

class TestPlaneBackfill:
    def test_missing_lattices_are_rebuilt_on_read(self, archive, tmp_path):
        """A pre-geometry archive (2 files) backfills instead of failing."""
        lut_dir = tmp_path / RADAR / "LUT"
        lut_dir.mkdir(parents=True)
        for kind in ("lut", "info"):
            src = lut_file_path(RADAR, kind, archive)
            (lut_dir / src.name).write_bytes(src.read_bytes())

        assert not lut_file_path(RADAR, "h_plane", tmp_path).exists()
        assert ensure_gate_planes(RADAR, tmp_path) is True
        for kind in ("h_plane", "v_plane", "corners"):
            assert lut_file_path(RADAR, kind, tmp_path).exists()
        assert ensure_gate_planes(RADAR, tmp_path) is False

    def test_backfill_recovers_the_projection_from_the_lut(self, archive, tmp_path):
        """Old info.yaml files have no crs block; the LUT columns still say EPSG."""
        lut_dir = tmp_path / RADAR / "LUT"
        lut_dir.mkdir(parents=True)
        for kind in ("lut", "info"):
            src = lut_file_path(RADAR, kind, archive)
            (lut_dir / src.name).write_bytes(src.read_bytes())
        ensure_gate_planes(RADAR, tmp_path)
        cols = load_plane_nodes(RADAR, tmp_path, "h_plane").columns
        assert "x_2056" in cols and "y_2056" in cols


class TestAoiGeometryUnification:
    def test_footprints_come_from_the_h_plane_lattice(self, archive):
        """aoi.py and the plots must draw the same gate."""
        from raddb.aoi import _lut_cs_table, _gate_footprints

        cs_t = _lut_cs_table(archive, [RADAR]).to_pandas().head(300)
        foot = _gate_footprints(cs_t, np.tan(np.deg2rad(0.5)), base_path=archive, epsg=2056)

        hp = RadDB(archive_dir=str(archive), crs=2056).get_h_plane(RADAR, per_gate=True)
        aligned = pl.DataFrame({"gate_id": cs_t["gate_id"].to_numpy()}).join(
            hp, on="gate_id", how="left", maintain_order="left")
        ref = shapely.polygons(np.stack([
            np.stack([aligned[f"x_2056_{k}"].to_numpy(),
                      aligned[f"y_2056_{k}"].to_numpy()], axis=1)
            for k in range(1, 5)], axis=1).astype(np.float64))

        assert np.allclose(shapely.get_coordinates(foot), shapely.get_coordinates(ref))

    def test_planar_fallback_still_available(self, archive):
        """Archives without the lattices keep working on the old approximation."""
        from raddb.aoi import _lut_cs_table, _gate_footprints

        cs_t = _lut_cs_table(archive, [RADAR]).to_pandas().head(50)
        planar = _gate_footprints(cs_t, np.tan(np.deg2rad(0.5)), base_path=None, epsg=None)
        assert shapely.is_valid(planar).all()

    def test_crops_are_unaffected(self, rdf, site):
        """Crops resolve on centroids, so unifying footprints must not move them."""
        assert len(rdf.crop_around_point(site, distance=8_000)) == 9504

    def test_cross_section_height_follows_the_v_plane(self, rdf, site, archive):
        cs = rdf.extract_cross_section((site[0] - 12_000, site[1] - 12_000),
                                       (site[0] + 12_000, site[1] + 12_000))
        from raddb.main import _decode_geometry
        pdf = _decode_geometry(cs.data.to_pandas()).head(200)
        heights = np.array([p.bounds[3] - p.bounds[1] for p in pdf["cs_polygon"]])

        vp = RadDB(archive_dir=str(archive), crs=2056).get_v_plane(RADAR, per_gate=True)
        va = pl.DataFrame({"gate_id": pdf["gate_id"].to_numpy()}).join(
            vp, on="gate_id", how="left", maintain_order="left")
        thick = 0.5 * (np.abs(va["z_asl_4"].to_numpy() - va["z_asl_1"].to_numpy())
                       + np.abs(va["z_asl_3"].to_numpy() - va["z_asl_2"].to_numpy()))
        assert np.corrcoef(heights, thick)[0, 1] > 0.9


# ===========================================================================
# 9. Opt-in polar coordinates on the converters
# ===========================================================================

class TestPolarCoordColumns:
    POLAR = ("range", "azimuth", "elevation_angle")

    def test_absent_by_default(self, rdf):
        for c in self.POLAR:
            assert c not in rdf.to_pandas(with_geometry=True).columns
            assert c not in rdf.to_geopandas().columns

    def test_present_on_request(self, rdf):
        pdf = rdf.to_pandas(with_polar_coords=True)
        gdf = rdf.to_geopandas(with_polar_coords=True)
        for c in self.POLAR:
            assert c in pdf.columns and c in gdf.columns

    def test_values_match_the_lut(self, rdf, archive):
        pdf = rdf.to_pandas(with_polar_coords=True)
        lut = RadDB(archive_dir=str(archive), crs=2056).get_lut(RADAR)
        ref = pl.DataFrame({"gate_id": pdf["gate_id"].to_numpy()}).join(
            lut.select(["gate_id", *self.POLAR]), on="gate_id",
            how="left", maintain_order="left")
        for c in self.POLAR:
            assert np.allclose(pdf[c].to_numpy(), ref[c].to_numpy())

    def test_with_polar_coords_implies_geometry(self, rdf):
        """Polar columns come from the LUT, so the join happens either way."""
        assert "latitude" in rdf.to_pandas(with_polar_coords=True).columns

    def test_row_count_is_unchanged(self, rdf):
        assert len(rdf.to_pandas(with_polar_coords=True)) == len(rdf)


# ===========================================================================
# 10. Axis tick labels (km from metres)
# ===========================================================================

class TestKmTickLabels:
    """Labels must be unique *and* equal to the value they sit on."""

    @staticmethod
    def _labels(lo, hi, offset=0.0):
        from raddb.viz.plot import _KmFormatter
        fig, ax = plt.subplots()
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(_KmFormatter(offset))
        fig.canvas.draw()
        locs = np.asarray(ax.yaxis.get_majorticklocs(), dtype=float)
        labs = [t.get_text() for t in ax.get_yticklabels()]
        plt.close(fig)
        return [(v, l) for v, l in zip(locs, labs) if l and lo <= v <= hi]

    @pytest.mark.parametrize("lo,hi,offset", [
        (1400, 5900, 0.0),        # the reported case: 500 m steps
        (0, 20000, 0.0),          # matplotlib picks 2.5 km steps
        (0, 13000, 0.0),
        (0, 500, 0.0),
        (0, 200, 0.0),            # 25 m steps -> 3 decimals
        (0, 60, 0.0),
        (0, 250000, 0.0),
        (2_600_000, 2_760_000, 2e6),   # LV95 easting
        (1_050_000, 1_150_000, 1e6),   # LV95 northing
    ])
    def test_labels_are_unique_and_exact(self, lo, hi, offset):
        pairs = self._labels(lo, hi, offset)
        labels = [l for _, l in pairs]
        assert len(labels) == len(set(labels)), f"repeated labels: {labels}"
        for v, l in pairs:
            km = (v - offset) / 1e3
            assert abs(float(l) - km) < 1e-6 * max(1.0, abs(km)), \
                f"label {l!r} does not equal {km}"

    def test_the_reported_regression(self):
        """A 1.4-5.9 km section used to read 1,2,2,2,3,4,4,4,5,6,6."""
        labels = [l for _, l in self._labels(1400, 5900)]
        assert labels == ["1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5"]

    def test_real_plots_have_no_duplicate_ticks(self, rdf, site):
        """Every plot, both axes."""
        cases = [
            rdf.plot_ppi(sweep=1),
            rdf.plot_ppi(sweep=1, coords="swiss"),
            rdf.plot_rhi(azimuth=0),
            rdf.plot_cappi(altitude=1200),
            rdf.plot_vcs(line=((site[0] - 12_000, site[1] - 12_000),
                               (site[0] + 12_000, site[1] + 12_000))),
        ]
        for p in cases:
            p.axes.figure.canvas.draw()
            for axis in (p.axes.xaxis, p.axes.yaxis):
                labs = [t.get_text() for t in axis.get_ticklabels()
                        if t.get_text() and t.get_visible()]
                assert len(labs) == len(set(labs)), f"repeated ticks: {labs}"
