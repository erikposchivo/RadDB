"""
raddb/tests/test_crs.py
-----------------------
The CRS contract: a projection must be declared at archive time and must be
valid where the radar actually is.

This exists because a hardcoded EPSG:2056 silently mis-selected US gates by
**17%** — a "50 km" crop reached only ~46 km — while looking entirely normal.
Nothing here may be inferred, defaulted or guessed.

Synthetic throughout; the fixture radar is relocated to test non-Swiss sites.
"""
from __future__ import annotations

import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr

from raddb.main import RadDB
from raddb.lut import (
    CRS_REFUSE_PCT, crs_distance_error, generate_lut_from_datatree,
    suggest_crs, validate_crs_for_site,
)
from raddb.tests.test_fixes import RADAR, _make_datatree

CH = (7.0, 46.0)          # the fixture's own site
US = (-97.2775, 35.3331)  # KTLX, Oklahoma


def relocate(dt, lon, lat):
    """Move a synthetic volume to another place on Earth."""
    out = {}
    for name, node in dt.children.items():
        ds = node.to_dataset().assign_coords(latitude=lat, longitude=lon)
        ds.attrs.update(node.attrs)
        out[name] = ds
    return xr.DataTree.from_dict(out)


class TestSuggestion:
    def test_suggests_the_utm_zone(self):
        assert suggest_crs(*CH) == 32632      # zone 32N
        assert suggest_crs(*US) == 32614      # zone 14N

    def test_southern_hemisphere_gets_a_south_zone(self):
        assert suggest_crs(151.2, -33.9) == 32756   # Sydney, zone 56S


class TestMeasuredValidation:
    """Validity is measured, because declared metadata is not enough."""

    @pytest.mark.parametrize("crs,site,ok", [
        (2056, CH, True),      # LV95 at home
        (32632, CH, True),     # UTM 32N at home
        (32614, US, True),     # UTM 14N at KTLX
        (2056, US, False),     # the bug: LV95 in Oklahoma
        (3857, CH, False),     # Web Mercator: claims the world, distorts hugely
        (3857, US, False),
    ])
    def test_accepts_and_refuses_by_measurement(self, crs, site, ok):
        if ok:
            assert validate_crs_for_site(crs, *site) < CRS_REFUSE_PCT
        else:
            with pytest.raises(ValueError, match="distorts distance"):
                validate_crs_for_site(crs, *site)

    def test_area_of_use_alone_would_not_catch_web_mercator(self):
        """EPSG:3857 declares the whole world, so bounds checks pass it."""
        au = pyproj.CRS.from_epsg(3857).area_of_use
        assert au.west <= CH[0] <= au.east and au.south <= CH[1] <= au.north
        assert crs_distance_error(3857, *CH) > 10.0

    def test_geographic_crs_is_refused(self):
        with pytest.raises(ValueError, match="geographic"):
            validate_crs_for_site(4326, *CH)

    def test_refusal_names_a_replacement(self):
        with pytest.raises(ValueError, match="32614"):
            validate_crs_for_site(2056, *US)


class TestArchiveRequiresACrs:
    def test_no_crs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires a CRS"):
            RadDB(archive_dir=str(tmp_path)).archive(datatree={RADAR: [_make_datatree()]})

    def test_bad_crs_aborts_and_writes_nothing(self, tmp_path):
        """A rejected CRS must not leave POL files behind with no usable LUT."""
        dt = relocate(_make_datatree(), *US)
        with pytest.raises(ValueError, match="distorts distance"):
            RadDB(archive_dir=str(tmp_path), crs=2056).archive(datatree={RADAR: [dt]})
        assert not list(tmp_path.rglob("*POL.parquet"))

    def test_correct_crs_archives(self, tmp_path):
        dt = relocate(_make_datatree(), *US)
        RadDB(archive_dir=str(tmp_path), crs=32614).archive(datatree={RADAR: [dt]})
        assert list(tmp_path.rglob("*POL.parquet"))
        info = RadDB(archive_dir=str(tmp_path)).get_radar_info(RADAR)
        assert info["crs"]["epsg"] == 32614


class TestAoiUsesTheArchiveCrs:
    @pytest.fixture(scope="class")
    def us_archive(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("us")
        dt = relocate(_make_datatree(n_az=72, n_rng=60, n_sweeps=3), *US)
        RadDB(archive_dir=str(base), crs=32614).archive(datatree={RADAR: [dt]})
        return base

    def test_aoi_epsg_comes_from_the_archive(self, us_archive):
        from raddb.aoi import aoi_epsg
        assert aoi_epsg(us_archive, RADAR) == 32614

    def test_crop_radius_is_true_metres(self, us_archive):
        """The bug in one assertion: a 10 km crop must select a 10 km radius."""
        from raddb.aoi import _lut_centroids, _resolve_gate_ids, _reproject_to_aoi, aoi_epsg

        db = RadDB(archive_dir=str(us_archive))
        lut = db.get_lut(RADAR)
        geod = pyproj.Geod(ellps="WGS84")
        lon = lut["longitude"].to_numpy(); lat = lut["latitude"].to_numpy()
        _, _, d = geod.inv(np.full(lon.size, US[0]), np.full(lat.size, US[1]), lon, lat)

        epsg = aoi_epsg(us_archive, RADAR)
        centroids = _lut_centroids(us_archive, [RADAR])
        pt = _reproject_to_aoi(shapely.Point(*US), 4326, epsg)
        for radius in (10_000, 15_000):
            truth = int((d <= radius).sum())
            got = len(_resolve_gate_ids(centroids, pt.buffer(radius)))
            assert abs(got - truth) <= 0.01 * truth, (
                f"{radius/1000:.0f} km crop selected {got} gates, truth {truth}"
            )

    def test_cross_section_distance_is_true_metres(self, us_archive):
        """A section line outside Switzerland must measure real ground distance."""
        rdf = RadDB(archive_dir=str(us_archive)).open(radars=RADAR)
        p1 = (US[0] - 0.2, US[1])
        p2 = (US[0] + 0.2, US[1])
        truth = pyproj.Geod(ellps="WGS84").inv(p1[0], p1[1], p2[0], p2[1])[2]

        cs = rdf.extract_cross_section(p1=p1, p2=p2, crs=4326)
        assert cs.data.height > 0, "the section selected no gates"
        # The far end of the section, read off the gate footprints: `d_center`
        # alone would sit half a gate short and eat most of the tolerance.
        polygons = cs.to_pandas()["cs_polygon"].to_numpy()
        span = float(shapely.bounds(polygons)[:, 2].max())
        # UTM 14N at KTLX is accurate to ~0.03%; a hardcoded LV95 would be ~20% out.
        assert abs(span - truth) <= 0.005 * truth, (
            f"section spans {span:,.0f} m, true geodesic {truth:,.0f} m"
        )

    def test_quicklook_context_lands_in_the_archive_frame(self, us_archive):
        """A caller's context geometry must not be reprojected to LV95."""
        from raddb.aoi import _resolve_context, _reproject_to_aoi, aoi_epsg

        epsg = aoi_epsg(us_archive, RADAR)
        gpd = pytest.importorskip("geopandas")
        box = shapely.box(US[0] - 1, US[1] - 1, US[0] + 1, US[1] + 1)   # WGS-84
        gdf = gpd.GeoDataFrame(geometry=[box], crs="EPSG:4326")

        got = _resolve_context(gdf, aoi_epsg=epsg)
        want = _reproject_to_aoi(box, 4326, epsg)
        assert got.distance(want) < 1.0

        # ... and it must sit on top of the radar, not thousands of km away.
        site = _reproject_to_aoi(shapely.Point(*US), 4326, epsg)
        assert got.contains(site)

    def test_quicklook_context_defaults_to_the_aoi_frame(self, us_archive):
        """A bare shapely geometry is taken as already being in the AOI frame."""
        from raddb.aoi import _resolve_context, _reproject_to_aoi, aoi_epsg

        epsg = aoi_epsg(us_archive, RADAR)
        site = _reproject_to_aoi(shapely.Point(*US), 4326, epsg)
        geom = site.buffer(50_000)
        assert _resolve_context(geom, aoi_epsg=epsg).equals(geom)

    @pytest.mark.parametrize("kind", ["crop", "section"])
    def test_quicklook_is_framed_on_the_archive(self, us_archive, kind):
        """Both quicklook call sites must pass the AOI frame, not default to LV95."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from raddb.aoi import _reproject_to_aoi, aoi_epsg

        rdf = RadDB(archive_dir=str(us_archive)).open(radars=RADAR)
        if kind == "crop":
            rdf.crop_around_point(US, distance=20_000, crs=4326, quicklook=True)
        else:
            rdf.extract_cross_section(
                p1=(US[0] - 0.2, US[1]), p2=(US[0] + 0.2, US[1]),
                crs=4326, quicklook=True,
            )
        ax = plt.gcf().axes[0]
        site = _reproject_to_aoi(shapely.Point(*US), 4326, aoi_epsg(us_archive, RADAR))
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        plt.close("all")
        # Axes hold metres (labelled in km by _KmFormatter); a Swiss-framed view
        # would put the site millions of metres off-axis.
        assert x0 <= site.x <= x1, f"{kind}: site outside x-range {(x0, x1)}"
        assert y0 <= site.y <= y1, f"{kind}: site outside y-range {(y0, y1)}"

    def test_aoi_crs_override_is_validated(self, us_archive):
        rdf = RadDB(archive_dir=str(us_archive)).open(radars=RADAR)
        with pytest.raises(ValueError, match="distorts distance"):
            rdf.crop_around_point(US, distance=10_000, crs=4326, aoi_crs=2056)

    def test_mixed_crs_radars_refuse_a_shared_aoi(self, tmp_path):
        """No silent reprojection: the user must name the common frame."""
        from raddb.aoi import aoi_epsg_for

        base = tmp_path
        RadDB(archive_dir=str(base), crs=2056).archive(
            datatree={"A": [_make_datatree(n_az=24, n_rng=20, n_sweeps=2)]})
        RadDB(archive_dir=str(base), crs=32614).archive(
            datatree={"D": [relocate(_make_datatree(n_az=24, n_rng=20, n_sweeps=2), *US)]})
        assert aoi_epsg_for(base, ["A"]) == 2056
        assert aoi_epsg_for(base, ["D"]) == 32614
        with pytest.raises(ValueError, match="different CRSs"):
            aoi_epsg_for(base, ["A", "D"])


class TestQuicklookFollowsTheFrame:
    """The AOI quicklook draws in the archive's CRS, not always in LV95."""

    @pytest.fixture(scope="class")
    def archives(self, tmp_path_factory):
        ch = tmp_path_factory.mktemp("ql_ch")
        us = tmp_path_factory.mktemp("ql_us")
        RadDB(archive_dir=str(ch), crs=2056).archive(
            datatree={RADAR: [_make_datatree(n_az=36, n_rng=30, n_sweeps=2)]})
        RadDB(archive_dir=str(us), crs=32614).archive(
            datatree={RADAR: [relocate(_make_datatree(n_az=36, n_rng=30, n_sweeps=2), *US)]})
        return ch, us

    @pytest.mark.parametrize("which,epsg,site", [(0, 2056, CH), (1, 32614, US)])
    def test_crop_is_in_view(self, archives, which, epsg, site):
        """A hardcoded Swiss y-band used to push a US AOI off-screen entirely."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from raddb.aoi import _reproject_to_aoi

        base = archives[which]
        rdf = RadDB(archive_dir=str(base)).open(radars=RADAR)
        pt = _reproject_to_aoi(shapely.Point(*site), 4326, epsg)
        rdf.crop_around_point((pt.x, pt.y), distance=8_000, quicklook=True)
        ax = plt.gcf().axes[0]
        assert ax.get_xlim()[0] <= pt.x <= ax.get_xlim()[1]
        assert ax.get_ylim()[0] <= pt.y <= ax.get_ylim()[1]
        plt.close("all")

    def test_quicklook_runs_without_a_declared_crs(self, archives):
        """Reading needs no CRS, so neither does the quicklook."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from raddb.aoi import _reproject_to_aoi

        rdf = RadDB(archive_dir=str(archives[1])).open(radars=RADAR)
        pt = _reproject_to_aoi(shapely.Point(*US), 4326, 32614)
        rdf.crop_around_point((pt.x, pt.y), distance=8_000, quicklook=True)
        plt.close("all")
