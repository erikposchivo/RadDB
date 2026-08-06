"""
raddb/tests/test_radar_code.py
------------------------------
Tests for the base-36 radar code that forms the leading field of a ``gate_id``
(encoding v2), the name normalisation it rests on, and the v1 archive guard.

Synthetic throughout — the archive tests build DataTrees in ``tmp_path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from raddb.helper import (  # noqa: E402
    RADAR_ALPHABET,
    RADAR_CODE_LEN,
    is_valid_radar_name,
    normalize_radar_name,
)
from raddb.lut import (  # noqa: E402
    GATE_ID_RADAR_BASE,
    LEGACY_RADAR_TO_IDX,
    MAX_RADAR_CODE,
    decode_gate_ids,
    decode_gate_radars,
    decode_radar_code,
    encode_gate_ids,
    encode_radar_code,
)
from raddb.tests.test_fixes import _make_datatree  # noqa: E402


# ===========================================================================
# normalize_radar_name
# ===========================================================================

class TestNormalizeRadarName:

    @pytest.mark.parametrize("raw,expected", [
        ("A", "A"),
        ("a", "A"),
        (" L ", "L"),
        ("MLA", "A"),          # MeteoSwiss spelling
        ("mlw", "W"),
        ("KTLX", "KTLX"),      # NEXRAD survives whole
        ("koun", "KOUN"),
        ("000A", "A"),         # zero padding is not part of the name
        ("0A", "A"),
        ("0", "0"),            # ... but a radar may be named "0"
        ("ZZZZ", "ZZZZ"),
    ])
    def test_canonical_forms(self, raw, expected):
        assert normalize_radar_name(raw) == expected

    def test_multi_letter_names_are_not_truncated(self):
        """The v1 bug: every name collapsed to its last character."""
        assert normalize_radar_name("KTLX") != "X"
        assert normalize_radar_name("KOUN") != "N"
        # Two sites sharing a final letter must stay distinct, or one would
        # overwrite the other's archive.
        assert normalize_radar_name("KTLX") != normalize_radar_name("KABX")

    def test_ml_rule_only_applies_at_three_characters(self):
        assert normalize_radar_name("MLA") == "A"
        assert normalize_radar_name("MLAB") == "MLAB"   # a real 4-char name

    @pytest.mark.parametrize("bad", [
        "", "   ", "chlem", "ABCDE", "A-B", "vol.", "A B", "MLABC", "é",
    ])
    def test_rejects_unusable_names(self, bad):
        assert not is_valid_radar_name(bad)
        with pytest.raises(ValueError, match="not usable"):
            normalize_radar_name(bad)

    def test_rejects_non_string(self):
        assert not is_valid_radar_name(7)
        with pytest.raises(ValueError, match="must be a string"):
            normalize_radar_name(7)


# ===========================================================================
# encode_radar_code / decode_radar_code
# ===========================================================================

class TestRadarCode:

    def test_known_values(self):
        assert encode_radar_code("A") == 10
        assert encode_radar_code("L") == 21
        assert encode_radar_code("KTLX") == 971_493
        assert encode_radar_code("0") == 0
        assert encode_radar_code("ZZZZ") == MAX_RADAR_CODE

    def test_capacity_is_36_pow_4(self):
        assert MAX_RADAR_CODE == 36 ** RADAR_CODE_LEN - 1 == 1_679_615

    def test_every_code_fits_int64(self):
        """The largest gate_id must not overflow the int64 column."""
        largest = MAX_RADAR_CODE * GATE_ID_RADAR_BASE + (GATE_ID_RADAR_BASE - 1)
        assert largest < np.iinfo(np.int64).max
        assert np.int64(largest) == largest

    def test_five_characters_do_not_fit(self):
        """Documents why RADAR_CODE_LEN is 4: 36**5 blows the int64 budget."""
        budget = (np.iinfo(np.int64).max - (GATE_ID_RADAR_BASE - 1)) // GATE_ID_RADAR_BASE
        assert 36 ** 4 - 1 <= budget < 36 ** 5 - 1

    def test_decode_is_injective(self):
        """Distinct codes never name the same radar."""
        names = {decode_radar_code(c) for c in range(MAX_RADAR_CODE + 1)}
        assert len(names) == MAX_RADAR_CODE + 1 == 1_679_616

    def test_name_round_trip_is_total_over_canonical_names(self):
        """Every name that is its own canonical form survives encode -> decode."""
        names = {decode_radar_code(c) for c in range(MAX_RADAR_CODE + 1)}
        canonical = {n for n in names if normalize_radar_name(n) == n}
        assert len(canonical) == 1_679_580  # all but the 36 ML? aliases
        assert all(decode_radar_code(encode_radar_code(n)) == n for n in canonical)

    def test_only_ml_aliases_break_the_code_round_trip(self):
        """encode(decode(c)) == c except where a name is an alias for another."""
        broken = [c for c in range(MAX_RADAR_CODE + 1)
                  if encode_radar_code(decode_radar_code(c)) != c]
        assert len(broken) == 36
        assert all(decode_radar_code(c).startswith("ML") for c in broken)
        assert all(len(decode_radar_code(c)) == 3 for c in broken)

    def test_zero_padding_is_transparent(self):
        assert encode_radar_code("A") == encode_radar_code("000A") == encode_radar_code("0A")

    def test_alphabet_positions_define_the_values(self):
        for i, char in enumerate(RADAR_ALPHABET):
            assert encode_radar_code(char.rjust(RADAR_CODE_LEN, "0")) == i

    @pytest.mark.parametrize("code", [-1, MAX_RADAR_CODE + 1, 10 ** 9])
    def test_decode_rejects_out_of_range(self, code):
        with pytest.raises(ValueError, match="names no radar"):
            decode_radar_code(code)

    def test_more_than_26_radars_are_distinct(self):
        """The point of the change: no 26-radar ceiling."""
        names = [f"K{a}{b}" for a in "ABCDE" for b in "ABCDEFGHIJ"]  # 50 sites
        codes = [encode_radar_code(n) for n in names]
        assert len(set(codes)) == len(names) == 50
        assert sorted(decode_radar_code(c) for c in codes) == sorted(names)


# ===========================================================================
# gate_id encoding
# ===========================================================================

class TestGateIdEncoding:

    def test_radar_field_is_the_code(self):
        gid = encode_gate_ids("KTLX", 3, np.array([91.4]), np.array([12_500.0]))
        assert gid[0] // GATE_ID_RADAR_BASE == encode_radar_code("KTLX")

    def test_low_fields_are_independent_of_the_radar(self):
        """Only the leading field differs between radars — what migration relies on."""
        az, rng = np.array([91.4, 270.0]), np.array([12_500.0, 240_000.0])
        a = encode_gate_ids("A", 3, az, rng)
        k = encode_gate_ids("KTLX", 3, az, rng)
        delta = (encode_radar_code("KTLX") - encode_radar_code("A")) * GATE_ID_RADAR_BASE
        assert np.array_equal(k - a, np.full(2, delta))

    def test_decode_round_trip(self):
        az, rng, sweeps = np.array([0.0, 91.4, 359.9]), np.array([0.0, 12_500.0, 999_999.0]), 7
        gid = encode_gate_ids("KTLX", sweeps, az, rng)
        got_sweeps, got_az, got_rng = decode_gate_ids(gid)
        assert np.array_equal(got_sweeps, np.full(3, sweeps))
        assert np.allclose(got_az, az)
        assert np.allclose(got_rng, rng)
        assert decode_gate_radars(gid) == ["KTLX"]

    def test_decode_radars_spans_several(self):
        az, rng = np.array([10.0]), np.array([1000.0])
        gid = np.concatenate([
            encode_gate_ids(r, 1, az, rng) for r in ("L", "KTLX", "A")
        ])
        assert decode_gate_radars(gid) == ["A", "KTLX", "L"]

    def test_decode_radars_empty(self):
        assert decode_gate_radars(np.array([], dtype=np.int64)) == []

    def test_decode_radars_skips_unknown_code(self, caplog):
        bogus = np.array([(MAX_RADAR_CODE + 5) * GATE_ID_RADAR_BASE], dtype=np.int64)
        with caplog.at_level("WARNING"):
            assert decode_gate_radars(bogus) == []
        assert "names no radar" in caplog.text

    def test_ml_name_encodes_as_its_letter(self):
        az, rng = np.array([10.0]), np.array([1000.0])
        assert np.array_equal(
            encode_gate_ids("MLA", 1, az, rng), encode_gate_ids("A", 1, az, rng)
        )

    def test_unusable_name_raises(self):
        with pytest.raises(ValueError, match="not usable"):
            encode_gate_ids("OVERLONG", 1, np.array([10.0]), np.array([1000.0]))


# ===========================================================================
# Archive round-trip and the v1 guard
# ===========================================================================

def _archive(tmp_path, radar):
    from raddb.main import RadDB

    db = RadDB(archive_dir=str(tmp_path / "archive"), crs=2056)
    db.archive(datatree=_make_datatree(n_sweeps=2, vol_time=pd.Timestamp("2024-01-01 12:00:00")),
               radar=radar)
    return db


class TestArchiveWithLongNames:

    def test_four_letter_radar_round_trips(self, tmp_path):
        db = _archive(tmp_path, "KTLX")
        assert db.list_radars() == ["KTLX"]

        rdf = db.open(radars="KTLX")
        assert rdf.data.height > 0
        assert rdf.radars() == ["KTLX"]

        gids = rdf.data["gate_id"].to_numpy()
        assert set(gids // GATE_ID_RADAR_BASE) == {encode_radar_code("KTLX")}
        assert decode_gate_radars(gids) == ["KTLX"]

    def test_lut_and_pol_gate_ids_join(self, tmp_path):
        db = _archive(tmp_path, "KTLX")
        lut = db.get_lut("KTLX")
        pol = db.open(radars="KTLX").data
        matched = pol.join(lut.select("gate_id"), on="gate_id", how="semi")
        assert matched.height == pol.height

    def test_info_yaml_records_no_version(self, tmp_path):
        """Only v2 is ever written, so the version key was dropped entirely."""
        db = _archive(tmp_path, "KTLX")
        assert "gate_id_version" not in db.get_radar_info("KTLX")

    def test_sel_by_radar_uses_the_code(self, tmp_path):
        db = _archive(tmp_path, "KTLX")
        rdf = db.open(radars="KTLX")
        assert rdf.sel(radar="KTLX").data.height == rdf.data.height
        assert rdf.sel(radar="A").data.height == 0

    def test_two_radars_stay_distinct(self, tmp_path):
        from raddb.main import RadDB

        db = RadDB(archive_dir=str(tmp_path / "archive"), crs=2056)
        for radar in ("KTLX", "KOUN"):
            db.archive(
                datatree=_make_datatree(n_sweeps=2, vol_time=pd.Timestamp("2024-01-01 12:00:00")),
                radar=radar,
            )
        assert db.list_radars() == ["KOUN", "KTLX"]

        both = db.open(radars=["KTLX", "KOUN"])
        assert sorted(both.radars()) == ["KOUN", "KTLX"]
        codes = set(both.data["gate_id"].to_numpy() // GATE_ID_RADAR_BASE)
        assert codes == {encode_radar_code("KTLX"), encode_radar_code("KOUN")}


class TestGateIdMigration:
    """The v1 -> v2 migration, now that ``info.yaml`` records no version.

    Nothing detects the encoding any more, so the tool is an unconditional
    offset that the caller must vouch for.  These tests pin that contract,
    including the part that is genuinely worse than before: running it twice
    corrupts the archive, and only ``--assume-v1`` stands between the two.
    """

    def _downgrade_to_v1(self, tmp_path, radar):
        """Rewrite an archive back to the v1 encoding, as if written long ago."""
        lut_dir = tmp_path / "archive" / radar / "LUT"
        delta = (LEGACY_RADAR_TO_IDX[radar] - encode_radar_code(radar)) * GATE_ID_RADAR_BASE
        for f in [lut_dir / f"{radar}_LUT.parquet",
                  *sorted((tmp_path / "archive" / radar).rglob("*_POL.parquet"))]:
            pl.read_parquet(f).with_columns(
                (pl.col("gate_id") + delta).alias("gate_id")
            ).write_parquet(f)
        return delta

    def test_v1_archive_is_read_without_complaint(self, tmp_path):
        """The version guard is gone: a v1 archive now loads silently.

        Its ids decode to the wrong radar — that is the cost of dropping the
        key, and it is pinned here so the trade-off stays visible.
        """
        db = _archive(tmp_path, "L")
        self._downgrade_to_v1(tmp_path, "L")

        assert "gate_id_version" not in db.get_radar_info("L")
        assert decode_gate_radars(db.open(radars="L").data["gate_id"].to_numpy()) == ["B"]

    def test_migration_restores_the_archive(self, tmp_path):
        from raddb.tools.migrate_gate_id_v2 import migrate_radar

        db = _archive(tmp_path, "L")
        before = db.open(radars="L").data["gate_id"].to_numpy().copy()
        self._downgrade_to_v1(tmp_path, "L")

        dry = migrate_radar(tmp_path / "archive", "L", dry_run=True)
        assert dry["status"] == "would migrate" and dry["files"] >= 2
        assert dry["rows"] == 0  # a dry run writes nothing

        res = migrate_radar(tmp_path / "archive", "L")
        assert res["status"] == "migrated" and res["rows"] > 0

        after = db.open(radars="L").data["gate_id"].to_numpy()
        assert np.array_equal(np.sort(after), np.sort(before))
        assert decode_gate_radars(after) == ["L"]

    def test_migration_is_no_longer_idempotent(self, tmp_path):
        """Without a recorded version there is nothing to short-circuit on."""
        from raddb.tools.migrate_gate_id_v2 import migrate_radar

        db = _archive(tmp_path, "L")
        self._downgrade_to_v1(tmp_path, "L")
        migrate_radar(tmp_path / "archive", "L")
        again = migrate_radar(tmp_path / "archive", "L")

        assert again["status"] == "migrated"       # it runs again, blindly
        assert decode_gate_radars(db.open(radars="L").data["gate_id"].to_numpy()) != ["L"]

    def test_cli_refuses_to_write_without_assume_v1(self, tmp_path):
        """The only guard left against a double migration."""
        from raddb.tools.migrate_gate_id_v2 import main

        db = _archive(tmp_path, "L")
        before = db.open(radars="L").data["gate_id"].to_numpy().copy()

        assert main([str(tmp_path / "archive")]) == 2
        assert np.array_equal(db.open(radars="L").data["gate_id"].to_numpy(), before)

        assert main([str(tmp_path / "archive"), "--dry-run"]) == 0
        assert np.array_equal(db.open(radars="L").data["gate_id"].to_numpy(), before)
