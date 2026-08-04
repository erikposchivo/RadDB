"""
raddb/tests/test_lut_planes.py
------------------------------
Tests for the five-file LUT directory: the gate-centroid LUT plus the
horizontal-face, vertical-face and 3-D corner node lattices, and the extended
info YAML.

Key invariants covered:

1. all five files are written by ``archive()`` / ``generate_lut_from_datatree``
2. **the frustum property** — a gate's far face is strictly larger than its near
   face (the beam widens with range)
3. each gate's centroid lies inside its own horizontal footprint, and between the
   bottom and top elevation levels
4. node sharing — the lattice is ``(n_az+1) x (n_rng+1)`` per level
5. ``z_asl - z_rel == site altitude`` everywhere
6. **file-size budgets** — the geometry files must stay compact

All tests use synthetic DataTrees in ``tmp_path``; no real radar files needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import shapely
import yaml

from raddb.lut import (
    DEFAULT_BEAMWIDTH_DEG,
    LUT_FILES,
    gate_corner_table,
    generate_lut_from_datatree,
    lut_file_path,
)
from raddb.main import RadDB
from raddb.tests.test_fixes import RADAR, _make_datatree

# The synthetic volume: 12 azimuths x 24 ranges x 2 sweeps (see test_fixes).
N_AZ, N_RNG, N_SWEEPS = 12, 24, 2
N_GATES = N_AZ * N_RNG * N_SWEEPS


@pytest.fixture
def lut_dir(tmp_path):
    """Generate a full 5-file LUT directory and return its path."""
    generate_lut_from_datatree(
        _make_datatree(), radar=RADAR, output_base_path=str(tmp_path),
        projection_epsg=2056,
    )
    return tmp_path


@pytest.fixture
def base(tmp_path):
    """The archive base path with a generated LUT (for the accessors)."""
    generate_lut_from_datatree(
        _make_datatree(), radar=RADAR, output_base_path=str(tmp_path),
        projection_epsg=2056,
    )
    return str(tmp_path)


# A realistically-sampled volume: 1 deg azimuth spacing, like a real radar.
#
# Needed by two groups of tests:
#
# * geometry — a gate footprint is a straight-sided quad, so its outer chord cuts
#   inside the true arc by the sagitta. The centroid falls outside its own
#   footprint beyond r ~ dR*cos(h)/(1-cos(h)) where h is the azimuth half-spacing.
#   At the 30 deg spacing of the small fixture that is only ~11.7 km; at 1 deg it
#   is ~5400 km, i.e. never. Real radars sample at 1 deg.
# * file sizes — bytes-per-gate is meaningless on a 576-gate file, where the
#   fixed parquet footer dominates.
REAL_N_AZ, REAL_N_RNG, REAL_N_SWEEPS = 360, 200, 2
REAL_N_GATES = REAL_N_AZ * REAL_N_RNG * REAL_N_SWEEPS


@pytest.fixture(scope="module")
def real_base(tmp_path_factory):
    """A realistically-sampled LUT (1 deg azimuths), built once per module."""
    d = tmp_path_factory.mktemp("realistic")
    generate_lut_from_datatree(
        _make_datatree(n_az=REAL_N_AZ, n_rng=REAL_N_RNG, n_sweeps=REAL_N_SWEEPS),
        radar=RADAR, output_base_path=str(d), projection_epsg=2056,
    )
    return str(d)


@pytest.fixture(scope="module")
def real_base_plain(tmp_path_factory):
    """As :func:`real_base` but with no projected coordinate columns."""
    d = tmp_path_factory.mktemp("realistic_plain")
    generate_lut_from_datatree(
        _make_datatree(n_az=REAL_N_AZ, n_rng=REAL_N_RNG, n_sweeps=REAL_N_SWEEPS),
        radar=RADAR, output_base_path=str(d),
        projection_epsg=2056,
    )
    return str(d)


def _face_area(t: pl.DataFrame, ks) -> np.ndarray:
    """Planar polygon area of a 4-corner face in 3-D (Newell's method)."""
    pts = np.stack([
        np.stack([t[f"x_{k}"].to_numpy(), t[f"y_{k}"].to_numpy(),
                  t[f"z_rel_{k}"].to_numpy()], axis=1)
        for k in ks
    ], axis=1)
    n = np.zeros((pts.shape[0], 3))
    for i in range(4):
        n += np.cross(pts[:, i], pts[:, (i + 1) % 4])
    return 0.5 * np.linalg.norm(n, axis=1)


class TestAllFilesWritten:
    def test_generate_writes_five_files(self, lut_dir):
        d = lut_dir / RADAR / "LUT"
        for kind, tmpl in LUT_FILES.items():
            f = d / tmpl.format(radar=RADAR)
            assert f.exists(), f"{kind} missing: {f.name}"
            assert f.stat().st_size > 0

    def test_archive_writes_five_files(self, tmp_path):
        db = RadDB(archive_dir=str(tmp_path), crs=2056)
        db.archive(datatree=_make_datatree(), radar=RADAR)
        d = tmp_path / RADAR / "LUT"
        assert sorted(f.name for f in d.iterdir()) == sorted(
            t.format(radar=RADAR) for t in LUT_FILES.values()
        )

    def test_missing_planes_are_backfilled(self, lut_dir):
        """An archive predating the lattices regenerates them, keeping its LUT."""
        d = lut_dir / RADAR / "LUT"
        lut_file = d / LUT_FILES["lut"].format(radar=RADAR)
        stamp = lut_file.stat().st_mtime_ns
        for kind in ("h_plane", "v_plane", "corners"):
            (d / LUT_FILES[kind].format(radar=RADAR)).unlink()

        generate_lut_from_datatree(
            _make_datatree(), radar=RADAR, output_base_path=str(lut_dir),
            projection_epsg=2056,
        )
        for kind in ("h_plane", "v_plane", "corners"):
            assert (d / LUT_FILES[kind].format(radar=RADAR)).exists()
        # the centroid LUT was not rewritten
        assert lut_file.stat().st_mtime_ns == stamp


class TestInfoYaml:
    def test_extended_keys(self, lut_dir):
        info = yaml.safe_load(
            (lut_dir / RADAR / "LUT" / f"{RADAR}_info.yaml").read_text()
        )
        for key in ("radar", "network", "latitude", "longitude", "altitude",
                    "crs", "ke", "beamwidth_deg", "n_sweeps", "n_gates", "sweeps"):
            assert key in info, f"missing info key {key!r}"
        assert info["ke"] == pytest.approx(4.0 / 3.0)
        assert info["beamwidth_deg"] == DEFAULT_BEAMWIDTH_DEG
        assert info["n_gates"] == N_GATES
        assert info["n_sweeps"] == N_SWEEPS
        assert info["crs"]["epsg"] == 2056
        assert info["crs"]["columns"] == ["x_2056", "y_2056"]

    def test_per_sweep_keys(self, lut_dir):
        info = yaml.safe_load(
            (lut_dir / RADAR / "LUT" / f"{RADAR}_info.yaml").read_text()
        )
        s = info["sweeps"][1]
        for key in ("n_azimuths", "n_ranges", "n_gates", "elevation",
                    "range_resolution", "range_start", "dR"):
            assert key in s, f"missing per-sweep key {key!r}"
        assert s["n_gates"] == s["n_azimuths"] * s["n_ranges"]
        # both are rounded to mm in the YAML, so allow half a mm of slack
        assert s["dR"] == pytest.approx(s["range_resolution"] / 2.0, abs=1e-3)

    def test_crs_block_records_what_was_used(self, real_base):
        """A CRS is mandatory, so the block is always populated."""
        info = yaml.safe_load(
            (Path(real_base) / RADAR / "LUT" / f"{RADAR}_info.yaml").read_text())
        assert info["crs"]["epsg"] == 2056
        assert info["crs"]["columns"] == ["x_2056", "y_2056"]



class TestLatticeShape:
    def test_h_plane_is_one_node_grid_per_sweep(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        nodes = db.get_h_plane(RADAR)
        assert nodes.height == N_SWEEPS * (N_AZ + 1) * (N_RNG + 1)
        assert "el_level" not in nodes.columns          # centre level only

    def test_corners_has_two_elevation_levels(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        nodes = db.get_corners(RADAR)
        assert sorted(nodes["el_level"].unique().to_list()) == [-1, 1]
        assert nodes.height == 2 * N_SWEEPS * (N_AZ + 1) * (N_RNG + 1)

    def test_sweep_filter(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        one = db.get_h_plane(RADAR, sweep=1)
        assert one["sweep"].unique().to_list() == [1]
        assert one.height == (N_AZ + 1) * (N_RNG + 1)

    def test_projected_columns_present(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        assert {"x_2056", "y_2056"} <= set(db.get_h_plane(RADAR).columns)


class TestPerGateCorners:
    def test_h_plane_has_four_corners(self, base):
        t = RadDB(archive_dir=base, crs=2056).get_h_plane(RADAR, per_gate=True)
        assert t.height == N_GATES
        for k in range(1, 5):
            assert f"x_{k}" in t.columns and f"y_{k}" in t.columns
        assert "x_5" not in t.columns

    def test_corners_has_eight(self, base):
        t = RadDB(archive_dir=base, crs=2056).get_corners(RADAR, per_gate=True)
        assert t.height == N_GATES
        for k in range(1, 9):
            assert {f"x_{k}", f"y_{k}", f"z_rel_{k}"} <= set(t.columns)
        assert "x_9" not in t.columns

    def test_eight_corners_are_distinct(self, base):
        """8 distinct corners, except the degenerate innermost range bin."""
        db = RadDB(archive_dir=base, crs=2056)
        t = db.get_corners(RADAR, per_gate=True)
        pts = np.stack([
            np.stack([t[f"x_{k}"].to_numpy(), t[f"y_{k}"].to_numpy(),
                      t[f"z_rel_{k}"].to_numpy()], axis=1)
            for k in range(1, 9)
        ], axis=1)
        n_distinct = np.array([
            len({tuple(np.round(p, 3)) for p in pts[i]}) for i in range(pts.shape[0])
        ])
        # the first range bin's near face collapses onto the radar -> 5 distinct
        assert set(np.unique(n_distinct)) <= {5, 8}
        assert (n_distinct == 8).mean() > 0.9

    def test_gate_ids_match_the_lut(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        lut_ids = set(db.get_lut(RADAR)["gate_id"].to_list())
        assert set(db.get_corners(RADAR, per_gate=True)["gate_id"].to_list()) == lut_ids


class TestFrustumProperty:
    """The beam widens with range: the far face must exceed the near face."""

    def test_far_face_is_larger_than_near_face(self, base):
        t = RadDB(archive_dir=base, crs=2056).get_corners(RADAR, per_gate=True)
        near = _face_area(t, [1, 2, 3, 4])
        far = _face_area(t, [5, 6, 7, 8])
        assert np.all(far > near), (
            f"{int((far <= near).sum())} gate(s) have a far face no larger than "
            "the near face"
        )

    def test_ratio_is_physically_sane(self, base):
        """Excluding the degenerate innermost bin, the ratio stays bounded."""
        t = RadDB(archive_dir=base, crs=2056).get_corners(RADAR, per_gate=True)
        near = _face_area(t, [1, 2, 3, 4])
        far = _face_area(t, [5, 6, 7, 8])
        ok = near > 1.0                      # drop the r~0 near face
        ratio = far[ok] / near[ok]
        assert ratio.min() > 1.0
        assert ratio.max() < 100.0

    def test_faces_are_valid_polygons(self, base):
        t = RadDB(archive_dir=base, crs=2056).get_h_plane(RADAR, per_gate=True)
        ring = np.stack([
            np.stack([t[f"x_{k}"].to_numpy(), t[f"y_{k}"].to_numpy()], axis=1)
            for k in (1, 2, 3, 4, 1)
        ], axis=1)
        assert shapely.is_valid(shapely.polygons(ring)).all()


class TestCentroidContainment:
    def test_centroid_inside_its_own_footprint(self, real_base):
        """Needs realistic (1 deg) azimuth sampling — see ``real_base``."""
        db = RadDB(archive_dir=real_base, crs=2056)
        t = db.get_h_plane(RADAR, per_gate=True).sort("gate_id")
        lut = db.get_lut(RADAR).sort("gate_id")
        ring = np.stack([
            np.stack([t[f"x_{k}"].to_numpy(), t[f"y_{k}"].to_numpy()], axis=1)
            for k in (1, 2, 3, 4, 1)
        ], axis=1)
        polys = shapely.polygons(ring)
        pts = shapely.points(
            np.stack([lut["x"].to_numpy(), lut["y"].to_numpy()], axis=1)
        )
        assert shapely.covers(polys, pts).all()

    def test_centroid_between_the_elevation_levels(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        t = db.get_corners(RADAR, per_gate=True).sort("gate_id")
        lut = db.get_lut(RADAR).sort("gate_id")
        zc = lut["z"].to_numpy()
        zs = np.stack([t[f"z_rel_{k}"].to_numpy() for k in range(1, 9)], axis=1)
        # 1 cm tolerance: on a negative-elevation sweep z(r) has a turning point,
        # so a centre can sit ~mm outside the bracket of its own corners.
        assert (zc >= zs.min(axis=1) - 0.01).all()
        assert (zc <= zs.max(axis=1) + 0.01).all()


class TestVPlane:
    def test_altitude_references_differ_by_site_altitude(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        site_alt = db.get_radar_info(RADAR)["altitude"]
        nodes = db.get_v_plane(RADAR)
        d = nodes["z_asl"].to_numpy() - nodes["z_rel"].to_numpy()
        assert np.allclose(d, site_alt, atol=1e-3)

    def test_ground_distance_is_monotonic_in_range(self, base):
        nodes = RadDB(archive_dir=base, crs=2056).get_v_plane(RADAR, sweep=1)
        sub = nodes.filter(pl.col("el_level") == 1).sort(["az_idx", "rng_idx"])
        d = sub.filter(pl.col("az_idx") == 0)["d"].to_numpy()
        assert np.all(np.diff(d) > 0)

    def test_per_gate_has_four_corners(self, base):
        t = RadDB(archive_dir=base, crs=2056).get_v_plane(RADAR, per_gate=True)
        assert t.height == N_GATES
        for k in range(1, 5):
            assert {f"d_{k}", f"z_asl_{k}", f"z_rel_{k}"} <= set(t.columns)

    def test_azimuth_selection_picks_one_ray(self, base):
        db = RadDB(archive_dir=base, crs=2056)
        t = db.get_v_plane(RADAR, azimuth=0.0, per_gate=True)
        assert 0 < t.height < N_GATES
        assert t.height == N_RNG * N_SWEEPS

    def test_azimuth_without_per_gate_raises(self, base):
        with pytest.raises(ValueError):
            RadDB(archive_dir=base, crs=2056).get_v_plane(RADAR, azimuth=90.0)


class TestGeoParquetExport:
    def test_export_embeds_crs_and_is_ccw(self, base, tmp_path):
        gpd = pytest.importorskip("geopandas")
        out = tmp_path / "h_plane.parquet"
        RadDB(archive_dir=base, crs=2056).export_h_plane_geoparquet(RADAR, out)
        g = gpd.read_parquet(out)
        assert len(g) == N_GATES
        assert g.crs is not None and g.crs.to_epsg() == 2056
        assert g.geometry.is_valid.all()
        assert shapely.is_ccw(shapely.get_exterior_ring(g.geometry.values)).all()

    def test_export_falls_back_to_wgs84(self, base, tmp_path):
        gpd = pytest.importorskip("geopandas")
        out = tmp_path / "h_plane_4326.parquet"
        RadDB(archive_dir=base, crs=2056).export_h_plane_geoparquet(RADAR, out, epsg=9999)
        g = gpd.read_parquet(out)
        assert g.crs.to_epsg() == 4326
        assert g.geometry.is_valid.all()


class TestFileSizeBudget:
    """The geometry files must stay compact — the whole point of lattices.

    Budgets are bytes per *gate* on disk, measured on the realistically-sampled
    fixture (144 k gates) so the fixed parquet footer does not dominate.  For
    reference, radar L (1.72 M gates, 20 sweeps) measures
    7.4 B/gate for h_plane unprojected, 13.6 projected, 14.8 for corners and
    0.2 for v_plane — ~49 MB of geometry against a 79 MB centroid LUT.

    ``v_plane`` is startlingly small because ground distance and altitude do not
    depend on azimuth, so parquet run-length-encodes it almost completely away.
    """

    BUDGET_PROJECTED = {"h_plane": 18.0, "v_plane": 2.0, "corners": 20.0}
    BUDGET_PLAIN = {"h_plane": 10.0, "v_plane": 2.0, "corners": 20.0}

    def _sizes(self, base):
        return {
            kind: lut_file_path(RADAR, kind, base).stat().st_size / REAL_N_GATES
            for kind in ("h_plane", "v_plane", "corners")
        }

    def test_projected_budget(self, real_base):
        for kind, bpg in self._sizes(real_base).items():
            assert bpg <= self.BUDGET_PROJECTED[kind], (
                f"{kind} is {bpg:.1f} B/gate, over the "
                f"{self.BUDGET_PROJECTED[kind]} B/gate budget"
            )


    def test_geometry_stays_smaller_than_the_centroid_lut(self, real_base):
        """All three geometry files together must not dwarf the LUT itself."""
        lut = lut_file_path(RADAR, "lut", real_base).stat().st_size
        geom = sum(
            lut_file_path(RADAR, k, real_base).stat().st_size
            for k in ("h_plane", "v_plane", "corners")
        )
        assert geom < lut, f"geometry {geom/1e6:.1f} MB vs LUT {lut/1e6:.1f} MB"

    def test_lattice_beats_per_gate_materialisation(self, real_base):
        """The stored lattice must be smaller than expanding every gate's corners."""
        db = RadDB(archive_dir=real_base, crs=2056)
        stored = lut_file_path(RADAR, "corners", real_base).stat().st_size
        per_gate = db.get_corners(RADAR, per_gate=True)
        # 8 corners x 3 coords x 4 bytes, the floor for a per-gate layout
        naive = per_gate.height * 8 * 3 * 4
        assert stored < naive


class TestBeamwidth:
    def test_beamwidth_parameter_widens_the_gate(self, tmp_path):
        heights = {}
        for bw in (1.0, 2.0):
            d = tmp_path / f"bw{bw}"
            generate_lut_from_datatree(
                _make_datatree(), radar=RADAR, output_base_path=str(d),
                beamwidth_deg=bw,
                projection_epsg=2056,
            )
            t = gate_corner_table(RADAR, str(d), kind="corners", sweep=1)
            zs = np.stack([t[f"z_rel_{k}"].to_numpy() for k in range(1, 9)], axis=1)
            heights[bw] = float(np.mean(zs.max(axis=1) - zs.min(axis=1)))
        assert heights[2.0] > heights[1.0] * 1.5

    def test_beamwidth_recorded_in_yaml(self, tmp_path):
        generate_lut_from_datatree(
            _make_datatree(), radar=RADAR, output_base_path=str(tmp_path),
            beamwidth_deg=1.5,
            projection_epsg=2056,
        )
        info = yaml.safe_load(
            (tmp_path / RADAR / "LUT" / f"{RADAR}_info.yaml").read_text()
        )
        assert info["beamwidth_deg"] == 1.5
