"""
raddb/tests/test_sel.py
-----------------------
Tests for ``RadDB.sel()`` — xarray-style label selection.

Covers:

1. dynamic-column selection (slice / scalar / list), inclusive slice bounds
2. **static (LUT) column** selection — the borrowed column must be dropped
   again, so the result still carries dynamic values only
3. the LUT staying synchronised with the data after a selection
4. immutability — ``sel`` never mutates the receiver

All tests use synthetic DataTrees in ``tmp_path``; no real radar files needed.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from raddb.main import RadDB
from raddb.tests.test_fixes import RADAR, _make_datatree

VOL_TIMES = [pd.Timestamp("2024-08-01 12:00:00"), pd.Timestamp("2024-08-02 06:30:00")]


@pytest.fixture
def rdf(tmp_path):
    """Data-carrying RadDB from a tiny two-volume, one-radar archive."""
    db = RadDB(archive_dir=str(tmp_path), crs=2056)
    db.archive(datatree={str(t): _make_datatree(vol_time=t) for t in VOL_TIMES},
               radar=RADAR)
    return db.open(radars=RADAR)


class TestSelDynamic:
    def test_slice_is_inclusive_on_both_ends(self, rdf):
        out = rdf.sel(DBZH=slice(5, 15))
        vals = out.data["DBZH"].to_numpy()
        assert len(out) > 0
        assert vals.min() >= 5.0 and vals.max() <= 15.0

    def test_open_ended_slices_match_filter(self, rdf):
        assert len(rdf.sel(DBZH=slice(10, None))) == len(
            rdf.filter({"var": "DBZH", "logic": ">=", "threshold": 10})
        )
        assert len(rdf.sel(DBZH=slice(None, 10))) == len(
            rdf.filter({"var": "DBZH", "logic": "<=", "threshold": 10})
        )

    def test_columns_are_unchanged(self, rdf):
        assert rdf.sel(DBZH=slice(0, 10)).columns() == rdf.columns()

    def test_no_args_is_a_noop(self, rdf):
        assert len(rdf.sel()) == len(rdf)

    def test_keywords_are_anded(self, rdf):
        both = rdf.sel(DBZH=slice(10, None), ZDR=slice(None, 5))
        chained = rdf.sel(DBZH=slice(10, None)).sel(ZDR=slice(None, 5))
        assert len(both) == len(chained)


class TestSelTime:
    def test_partial_day_string_selects_the_whole_day(self, rdf):
        out = rdf.sel(time="2024-08-01")
        assert 0 < len(out) < len(rdf)

    def test_partial_month_string(self, rdf):
        assert len(rdf.sel(time="2024-08")) == len(rdf)

    def test_non_matching_period_is_empty(self, rdf):
        assert len(rdf.sel(time="1999-01")) == 0

    def test_time_slice(self, rdf):
        out = rdf.sel(time=slice("2024-08-01", "2024-08-01"))
        assert 0 < len(out) < len(rdf)


class TestSelStaticLutColumns:
    """Selection on LUT columns must borrow, evaluate, then drop."""

    def test_range_selection_does_not_leak_the_column(self, rdf):
        out = rdf.sel(range=slice(2_000, 10_000))
        assert 0 < len(out) < len(rdf)
        assert out.columns() == rdf.columns()
        assert "range" not in out.columns()

    def test_range_selection_matches_the_lut(self, rdf, tmp_path):
        lut = RadDB(archive_dir=str(tmp_path)).get_lut(RADAR)
        want = set(
            lut.filter((pl.col("range") >= 2_000) & (pl.col("range") <= 10_000))["gate_id"]
            .to_list()
        )
        got = set(rdf.sel(range=slice(2_000, 10_000)).data["gate_id"].to_list())
        assert got == set(rdf.data["gate_id"].to_list()) & want

    def test_sweep_scalar(self, rdf):
        out = rdf.sel(sweep=1)
        assert 0 < len(out) < len(rdf)
        assert "sweep" not in out.columns()

    def test_lat_lon_aliases(self, rdf):
        ge = rdf.geographic_extent()
        out = rdf.sel(lon=slice(ge[0], ge[1]), lat=slice(ge[2], ge[3]))
        assert len(out) == len(rdf)          # full extent keeps everything
        assert out.columns() == rdf.columns()

    def test_mixed_static_and_dynamic(self, rdf):
        out = rdf.sel(DBZH=slice(10, None), range=slice(2_000, 10_000), sweep=1)
        assert out.columns() == rdf.columns()
        assert len(out) <= len(rdf)


class TestSelRadars:
    def test_radars_list_keeps_present_radar(self, rdf):
        assert len(rdf.sel(radars=[RADAR])) == len(rdf)

    def test_radars_list_excluding_present_radar_is_empty(self, rdf):
        other = "W" if RADAR != "W" else "L"
        assert len(rdf.sel(radars=[other])) == 0

    def test_multi_radar_selection(self, tmp_path):
        db = RadDB(archive_dir=str(tmp_path), crs=2056)
        db.archive(datatree={"A": [_make_datatree(vol_time=VOL_TIMES[0])],
                             "D": [_make_datatree(vol_time=VOL_TIMES[0])]})
        both = db.open()
        assert sorted(both.radars()) == ["A", "D"]
        only_a = both.sel(radars=["A"])
        assert only_a.radars() == ["A"]
        assert 0 < len(only_a) < len(both)


class TestSelKeepsLutSynchronised:
    def test_geometry_shrinks_with_the_data(self, rdf):
        out = rdf.sel(range=slice(2_000, 10_000))
        assert len(out._gate_geometry()) < len(rdf._gate_geometry())

    def test_geometry_gate_ids_match_data_gate_ids(self, rdf):
        out = rdf.sel(range=slice(2_000, 10_000), DBZH=slice(10, None))
        geo = out._gate_geometry()
        assert set(geo["gate_id"].to_list()) == set(out.data["gate_id"].to_list())

    def test_with_geometry_converter_still_works(self, rdf):
        out = rdf.sel(range=slice(2_000, 10_000))
        pdf = out.to_pandas(with_geometry=True)
        assert len(pdf) == len(out)
        assert "latitude" in pdf.columns


class TestSelImmutability:
    def test_receiver_is_untouched(self, rdf):
        before_len, before_cols = len(rdf), rdf.columns()
        rdf.sel(DBZH=slice(0, 1), range=slice(2_000, 3_000), sweep=1)
        assert len(rdf) == before_len
        assert rdf.columns() == before_cols

    def test_returns_a_new_object_with_same_config(self, rdf):
        out = rdf.sel(DBZH=slice(0, 10))
        assert out is not rdf
        assert isinstance(out, RadDB)
        assert out.crs() == rdf.crs()
        assert str(out.archive_dir) == str(rdf.archive_dir)


class TestSelErrors:
    def test_unknown_column_raises_keyerror(self, rdf):
        with pytest.raises(KeyError):
            rdf.sel(NOT_A_COLUMN=1)

    def test_step_in_slice_raises_valueerror(self, rdf):
        with pytest.raises(ValueError):
            rdf.sel(DBZH=slice(0, 10, 2))
