"""Migrate an archive from the v1 ``gate_id`` encoding to v2, in place.

v1 numbered radars ``A=0 … Z=25``; v2 uses the base-36 value of the (zero-padded,
4-character) radar name, so ``"L"`` moved from 11 to 21.  Only the leading radar
field changes — ``sweep``/``azimuth``/``range`` occupy the low 12 digits and are
untouched — so the whole migration is one integer offset per radar::

    gate_id += (encode_radar_code(radar) - LEGACY_RADAR_TO_IDX[radar]) * 10**12

``gate_id`` is stored in exactly two kinds of file, verified against the archives
on disk: the centroid LUT ``{radar}/LUT/{radar}_LUT.parquet`` and every volume
``{radar}/**/{radar}_*_POL.parquet``.  The three geometry lattices
(``h_plane`` / ``v_plane`` / ``corners``) are node lattices addressed by
``(sweep, az_idx, rng_idx)`` and carry no ``gate_id``, so they need no rewrite.

No geometry is recomputed and nothing is re-ingested, which matters because the
source volumes of an archive are often no longer around.

Usage
-----
::

    python -m raddb.tools.migrate_gate_id_v2 <archive_dir> --dry-run
    python -m raddb.tools.migrate_gate_id_v2 <archive_dir>
    python -m raddb.tools.migrate_gate_id_v2 <archive_dir> --radar L --radar W
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

from raddb.helper import is_valid_radar_name, normalize_radar_name
from raddb.lut import (
    GATE_ID_RADAR_BASE,
    GATE_ID_VERSION,
    LEGACY_RADAR_TO_IDX,
    encode_radar_code,
)


def _archive_radars(archive_dir: Path) -> list[str]:
    """Radar directories in *archive_dir*, whether or not they are migrated yet."""
    return sorted(
        p.name
        for p in archive_dir.iterdir()
        if p.is_dir() and is_valid_radar_name(p.name) and (p / "LUT").is_dir()
    )


def _info_path(archive_dir: Path, radar: str) -> Path:
    return archive_dir / radar / "LUT" / f"{radar}_info.yaml"


def _gate_id_files(archive_dir: Path, radar: str) -> list[Path]:
    """Every parquet under *radar* that holds a ``gate_id`` column."""
    lut = archive_dir / radar / "LUT" / f"{radar}_LUT.parquet"
    files = [lut] if lut.exists() else []
    files += sorted((archive_dir / radar).rglob("*_POL.parquet"))
    return files


def _shift_gate_ids(path: Path, delta: int) -> int:
    """Rewrite ``gate_id`` in *path* by *delta*. Returns the row count.

    Written to a sibling temp file and moved into place, so an interrupted run
    leaves the original parquet intact rather than a half-written one.
    """
    df = pl.read_parquet(path)
    df = df.with_columns((pl.col("gate_id") + delta).alias("gate_id"))
    tmp = path.with_suffix(path.suffix + ".migrating")
    df.write_parquet(tmp)
    tmp.replace(path)
    return df.height


def migrate_radar(archive_dir: Path, radar: str, dry_run: bool = False) -> dict:
    """Migrate one radar. Returns a summary dict; a migrated radar is skipped."""
    info_path = _info_path(archive_dir, radar)
    if not info_path.exists():
        return {"radar": radar, "status": "no info.yaml", "files": 0, "rows": 0}

    info = yaml.safe_load(info_path.read_text()) or {}
    version = int(info.get("gate_id_version", 1))
    if version == GATE_ID_VERSION:
        return {"radar": radar, "status": "already v2", "files": 0, "rows": 0}
    if version != 1:
        return {"radar": radar, "status": f"unknown v{version} — left alone",
                "files": 0, "rows": 0}

    name = normalize_radar_name(info.get("radar") or radar)
    legacy = LEGACY_RADAR_TO_IDX.get(name)
    if legacy is None:
        # v1 could only ever encode A-Z, so a v1 archive naming anything else is
        # inconsistent and guessing an offset would corrupt it.
        return {"radar": radar, "status": f"{name!r} is not a v1 (A-Z) radar — skipped",
                "files": 0, "rows": 0}

    delta = (encode_radar_code(name) - legacy) * GATE_ID_RADAR_BASE
    files = _gate_id_files(archive_dir, radar)

    rows = 0
    if not dry_run:
        for f in files:
            rows += _shift_gate_ids(f, delta)
        info["gate_id_version"] = GATE_ID_VERSION
        with open(info_path, "w") as fh:
            yaml.dump(info, fh, default_flow_style=False, sort_keys=False)

    return {
        "radar": radar,
        "status": "would migrate" if dry_run else "migrated",
        "files": len(files),
        "rows": rows,
        "delta": delta,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive_dir", type=Path, help="RadDB archive base directory")
    ap.add_argument("--radar", action="append", default=None,
                    help="restrict to this radar (repeatable); default is all")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args(argv)

    archive_dir: Path = args.archive_dir
    if not archive_dir.is_dir():
        print(f"error: {archive_dir} is not a directory", file=sys.stderr)
        return 2

    radars = [normalize_radar_name(r) for r in args.radar] if args.radar \
        else _archive_radars(archive_dir)
    if not radars:
        print(f"no radar directories found in {archive_dir}")
        return 0

    print(f"{'gate_id v1 -> v2':<24}{archive_dir}")
    print(f"{'radar':<8}{'files':>8}{'rows':>14}   status")
    print("-" * 62)
    for radar in radars:
        res = migrate_radar(archive_dir, radar, dry_run=args.dry_run)
        print(f"{res['radar']:<8}{res['files']:>8}{res['rows']:>14,}   {res['status']}")
    print("-" * 62)
    if args.dry_run:
        print("dry run — nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
