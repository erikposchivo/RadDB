"""
raddb/tests/test_polars_backend.py
----------------------------------
Tests for the polars backend contract:

1. the LUT loads as polars, and ``gate_id`` decoding inverts the encoding
2. crops **select** rows and never widen them with LUT columns
3. ``to_geoarrow`` emits GeoArrow-tagged point and wedge-polygon geometry
4. the dynamic values and the LUT stay separate tables

Run with:
    pytest raddb/tests/test_polars_backend.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from raddb.tests.test_fixes import RADAR, _make_datatree  # noqa: E402


@pytest.fixture
def archive(tmp_path):
    """A tiny single-volume archive with a LUT, in polars-backed RadDB form."""
    from raddb.main import RadDB

    db = RadDB(archive_dir=str(tmp_path), crs=2056)
    db.archive(datatree=_make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00")),
               radar=RADAR)
    return db


class TestLutIsPolars:
    def test_load_radar_lut_returns_polars(self, archive):
        lut = archive.get_lut(RADAR)
        assert isinstance(lut, pl.DataFrame)
        assert "gate_id" in lut.columns and lut.height > 0

    def test_lut_coordinates_stay_float64(self, archive):
        """Gate positions are float64 — precision is a hard requirement."""
        lut = archive.get_lut(RADAR)
        for col in ("latitude", "longitude", "altitude", "x", "y", "z"):
            if col in lut.columns:
                assert lut.schema[col] == pl.Float64, f"{col} lost float64"

    def test_decode_gate_ids_inverts_encode(self, archive):
        from raddb.lut import decode_gate_ids

        lut = archive.get_lut(RADAR)
        sweeps, azimuths, ranges = decode_gate_ids(lut["gate_id"].to_numpy())
        assert np.array_equal(sweeps, lut["sweep"].to_numpy().astype(np.int64))
        # gate_id stores azimuth rounded to 0.1 deg and range as integer metres.
        assert np.allclose(azimuths, np.round(lut["azimuth"].to_numpy() * 10) / 10)
        assert np.allclose(ranges, lut["range"].to_numpy().astype(np.int64))


class TestCropsSelectNotWiden:
    def test_crop_keeps_the_same_columns(self, archive):
        rdf = archive.open()
        geo = rdf.to_pandas(with_geometry=True)
        cx, cy = float(geo["x_2056"].median()), float(geo["y_2056"].median())
        crop = rdf.crop_by_bbox(bounds=(cx - 5000, cy - 5000, cx + 5000, cy + 5000))

        assert 0 < len(crop) < len(rdf), "crop should select a strict, non-empty subset"
        assert crop.columns() == rdf.columns(), (
            "crop widened the frame with LUT columns: "
            f"{set(crop.columns()) - set(rdf.columns())}"
        )

    def test_crop_matches_an_isin_reference(self, archive):
        """The semi-join must select exactly what the old isin() filter did."""
        import shapely

        from raddb.aoi import _lut_centroids, _reproject_to_aoi, _resolve_aoi_centroids

        rdf = archive.open()
        geo = rdf.to_pandas(with_geometry=True)
        cx, cy = float(geo["x_2056"].median()), float(geo["y_2056"].median())
        bounds = (cx - 5000, cy - 5000, cx + 5000, cy + 5000)

        crop = rdf.crop_by_bbox(bounds=bounds)
        cen = _resolve_aoi_centroids(
            _lut_centroids(archive.archive_dir, rdf.radars()),
            _reproject_to_aoi(shapely.box(*bounds), 2056, 2056),
        )
        expected = set(np.intersect1d(
            rdf.data["gate_id"].to_numpy(), cen["gate_id"].to_numpy()
        ))
        assert set(crop.data["gate_id"].to_numpy()) == expected

    def test_empty_crop_does_not_raise(self, archive):
        """A crop that matches nothing still converts (regression: pl.concat([]))."""
        rdf = archive.open()
        far = rdf.crop_by_bbox(bounds=(9e6, 9e6, 9e6 + 1000, 9e6 + 1000))
        assert len(far) == 0
        assert far.to_pandas(with_geometry=True).empty


class TestToGeoArrow:
    def test_point_geometry_is_tagged(self, archive):
        tab = archive.open().to_geoarrow(geometry="point")
        meta = tab.schema.field("geometry").metadata
        assert meta[b"ARROW:extension:name"] == b"geoarrow.point"
        assert b"EPSG:4326" in meta[b"ARROW:extension:metadata"]

    def test_polygon_rings_are_closed_wedges(self, archive):
        tab = archive.open().to_geoarrow(geometry="polygon")
        assert tab.schema.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"

        rings = tab.column("geometry").to_pylist()
        assert rings and all(r is not None for r in rings), "every gate should be placed"
        for r in rings:
            assert len(r) == 1, "a gate is a single ring"
            assert len(r[0]) == 5, "4 corners + closing vertex"
            assert r[0][0] == r[0][-1], "ring must be closed"

    def test_polygon_centroid_sits_on_the_gate(self, archive):
        """The wedge must surround the LUT centroid it was built from."""
        shapely = pytest.importorskip("shapely")

        rdf = archive.open()
        tab = rdf.to_geoarrow(geometry="polygon")
        polys = shapely.polygons(
            np.array([r[0] for r in tab.column("geometry").to_pylist()])
        )
        assert shapely.is_valid(polys).all()

        lut = archive.get_lut(RADAR)
        ref = (
            pl.DataFrame({"gate_id": tab.column("gate_id").to_numpy()})
            .join(lut.select("gate_id", "latitude", "longitude"), on="gate_id", how="left")
        )
        cen = shapely.centroid(polys)
        dx = (shapely.get_x(cen) - ref["longitude"].to_numpy()) * 111_320 * np.cos(np.radians(46.0))
        dy = (shapely.get_y(cen) - ref["latitude"].to_numpy()) * 111_320
        gate_len = 20_000 / 24  # _make_datatree: 24 gates over ~20 km
        assert np.hypot(dx, dy).max() < gate_len

    def test_max_rows_guardrail(self, archive):
        rdf = archive.open()
        with pytest.raises(ValueError, match="max_rows"):
            rdf.to_geoarrow(max_rows=1)
        assert rdf.to_geoarrow(max_rows=None).num_rows == len(rdf)

    def test_polygon_needs_a_single_radar(self, archive):
        """Corner arrays are per radar, so a multi-radar frame must be rejected."""
        rdf = archive.open()
        mixed = rdf._derive(
            pl.concat([rdf.data, rdf.data.with_columns(pl.lit("B").alias("radar"))])
        )
        with pytest.raises(ValueError, match="single radar"):
            mixed.to_geoarrow(geometry="polygon", max_rows=None)


def test_lut_is_not_carried_by_the_dynamic_frame(archive):
    """open() returns dynamic values only — no geometry columns."""
    rdf = archive.open()
    lut_only = {"latitude", "longitude", "altitude", "x", "y", "z", "x_2056", "y_2056",
                "azimuth", "range", "elevation_angle"}
    assert not lut_only & set(rdf.columns()), (
        f"LUT columns leaked into the dynamic frame: {lut_only & set(rdf.columns())}"
    )
    # ... and are reachable on request.
    assert {"latitude", "longitude"} <= set(rdf.to_pandas(with_geometry=True).columns)


class TestFilterOnLutColumns:
    """Vertical subsetting must survive the crops no longer carrying LUT columns."""

    def test_altitude_filter_without_a_crop(self, archive):
        rdf = archive.open()
        assert "altitude" not in rdf.columns(), "altitude is LUT data, not dynamic"

        alt = rdf.to_pandas(with_geometry=True)["altitude"]
        cut = float(alt.median())
        band = rdf.filter({"var": "altitude", "logic": ">", "threshold": cut})

        assert 0 < len(band) < len(rdf)
        assert band.columns() == rdf.columns(), "the borrowed LUT column leaked"
        assert band.to_pandas(with_geometry=True)["altitude"].min() > cut

    def test_altitude_band_after_a_crop(self, archive):
        rdf = archive.open()
        geo = rdf.to_pandas(with_geometry=True)
        cx, cy = float(geo["x_2056"].median()), float(geo["y_2056"].median())
        aoi = rdf.crop_by_bbox(bounds=(cx - 5000, cy - 5000, cx + 5000, cy + 5000))

        lo, hi = np.percentile(geo["altitude"], [25, 75])
        band = aoi.filter([{"var": "altitude", "logic": ">", "threshold": float(lo)},
                           {"var": "altitude", "logic": "<", "threshold": float(hi)}])
        assert len(band) <= len(aoi)
        assert band.columns() == rdf.columns()

    def test_unknown_filter_column_raises(self, archive):
        with pytest.raises(KeyError, match="cannot filter on"):
            archive.open().filter({"var": "nope", "logic": ">", "threshold": 0})
