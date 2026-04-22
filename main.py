"""
RadDB end-to-end archiving entry point.

Runs the full pipeline on a server: METRANET ingestion -> xarray DataTree
-> Parquet archive, one volume at a time. Suitable for year-long ranges
across all Swiss radars (A, D, L, P, W).

Design notes:
  - Each volume is processed and archived immediately, then its DataTree
    is dropped, so RAM stays bounded regardless of how many volumes are
    processed.
  - A checkpoint file (plain text, one ``{radar}:{stem}`` per line) is
    appended after each successful archive. Re-running the command with
    the same ``--base-path`` resumes where it left off.
  - The range is walked day by day (UTC) so that a crash at day N only
    costs the volumes of day N.

Typical usage on the server:

    python main.py \
        --start "2023-08-01 00:00" \
        --end   "2023-08-31 23:59" \
        --radar all \
        --base-path         /ltenas8/users/giacobbi/raddb \
        --raw-data-dir      /ltenas8/data/RADAR \
        --static-vis-dir    /ltenas8/data/Rad4Alp_LUTs/static_vis \
        --qpegrid-to-rad-dir /ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import raddb
from mch_pipeline import (
    _group_files_by_volume,
    _parse_volume_time,
    find_files_with_fallback,
    generate_mch_lut,
    process_mch_volume,
)

SWISS_RADARS = ["A", "D", "L", "P", "W"]
CHECKPOINT_NAME = "_archive_checkpoint.txt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full RadDB archiving pipeline end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True,
                   help='Start time, e.g. "2024-01-01 00:00".')
    p.add_argument("--end", required=True,
                   help='End time (inclusive), e.g. "2024-12-31 23:59".')
    p.add_argument("--radar", default="all",
                   help='Radar letter(s): "A", "A,D", or "all".')
    p.add_argument("--network", default="MCH_LTE",
                   help="radar_api network identifier.")
    p.add_argument("--base-path", required=True,
                   help="Output directory for the RadDB archive.")
    p.add_argument("--raw-data-dir", required=True,
                   help="Root directory of the raw METRANET files.")
    p.add_argument("--static-vis-dir", required=True,
                   help="Directory holding static visibility LUTs.")
    p.add_argument("--qpegrid-to-rad-dir", required=True,
                   help="Directory holding qpegrid -> radar LUTs.")
    p.add_argument("--filter-feature", default="DBZH",
                   help="Column used for clear-sky gate removal.")
    p.add_argument("--filter-threshold", type=float, default=0.0,
                   help="Threshold for the filter.")
    p.add_argument("--filter-logic", default=">",
                   help="Comparison operator (>, >=, <, <=, ==, !=).")
    p.add_argument("--no-hzt", action="store_true",
                   help="Disable HZT (freezing-level) ingestion.")
    p.add_argument("--no-hym", action="store_true",
                   help="Disable HYM hydrometeor-class ingestion.")
    p.add_argument("--no-pyart-hc", action="store_true",
                   help="Disable pyart semi-supervised hydroclass.")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose per-volume logging.")
    return p.parse_args()


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _resolve_radars(arg: str) -> list[str]:
    if arg.lower() == "all":
        return list(SWISS_RADARS)
    radars = [r.strip().upper() for r in arg.split(",") if r.strip()]
    unknown = [r for r in radars if r not in SWISS_RADARS]
    if unknown:
        raise SystemExit(f"Unknown radar(s): {unknown}. Valid: {SWISS_RADARS}")
    return radars


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (day_start, day_end) pairs covering [start, end] inclusively."""
    day = start.normalize()
    last = end.normalize()
    while day <= last:
        day_start = max(day, start)
        day_end = min(day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), end)
        yield day_start, day_end
        day = day + pd.Timedelta(days=1)


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r") as f:
        return {line.strip() for line in f if line.strip()}


def _append_checkpoint(path: Path, entry: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry + "\n")


def _lut_exists(base_path: Path, radar: str) -> bool:
    return (base_path / radar / "LUT" / f"{radar}_LUT.parquet").exists()


def _process_day_for_radar(
    *,
    radar: str,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    args: argparse.Namespace,
    db: raddb.RadDB,
    base_path: Path,
    checkpoint_path: Path,
    checkpoint_seen: set[str],
) -> tuple[int, int]:
    """Process every volume for one (radar, day). Returns (n_ok, n_fail)."""
    try:
        pol_files = find_files_with_fallback(
            network=args.network,
            radar=radar,
            start_time=day_start.to_pydatetime(),
            end_time=day_end.to_pydatetime(),
            product="POL",
            raw_data_dir=args.raw_data_dir,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"  [{radar} {day_start:%Y-%m-%d}] find_files failed: {e}")
        return (0, 0)

    if not pol_files:
        return (0, 0)

    volumes = _group_files_by_volume(pol_files)
    n_ok = n_fail = 0

    iterator = sorted(volumes.items())
    pbar = tqdm(
        iterator,
        desc=f"{radar} {day_start:%Y-%m-%d}",
        unit="vol",
        leave=False,
    )
    for stem, sweep_paths in pbar:
        ckpt_key = f"{radar}:{stem}"
        if ckpt_key in checkpoint_seen:
            continue

        try:
            volume_time = _parse_volume_time(stem)
            vol_dt = process_mch_volume(
                sweep_filepaths=sweep_paths,
                network=args.network,
                radar=radar,
                volume_time=volume_time,
                raw_data_dir=args.raw_data_dir,
                static_vis_dir=args.static_vis_dir,
                qpegrid_to_rad_dir=args.qpegrid_to_rad_dir,
                hzt_enabled=not args.no_hzt,
                hym_enabled=not args.no_hym,
                compute_pyart_hc=not args.no_pyart_hc,
                verbose=args.verbose,
                volume_label=stem,
            )

            db.archive_volume(
                dt=vol_dt,
                radar=radar,
                filter_feature=args.filter_feature,
                filter_threshold=args.filter_threshold,
                filter_logic=args.filter_logic,
                volume_label=stem,
            )

            if not _lut_exists(base_path, radar):
                try:
                    generate_mch_lut(
                        radar=radar,
                        network=args.network,
                        sample_volume_filepaths=sweep_paths,
                        output_base_path=str(base_path),
                        qpegrid_to_rad_dir=args.qpegrid_to_rad_dir,
                    )
                except Exception as e:
                    print(f"  [{radar}] LUT generation failed: {e}")

            _append_checkpoint(checkpoint_path, ckpt_key)
            checkpoint_seen.add(ckpt_key)
            n_ok += 1

            del vol_dt

        except Exception as e:
            n_fail += 1
            print(f"  [{radar}] FAIL {stem}: {e}")
            if args.verbose:
                traceback.print_exc()

    return (n_ok, n_fail)


def main() -> int:
    args = _parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    radars = _resolve_radars(args.radar)

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = base_path / CHECKPOINT_NAME
    checkpoint_seen = _load_checkpoint(checkpoint_path)

    db = raddb.RadDB(base_path=str(base_path))

    print("=" * 70)
    print("RadDB archiving run")
    print(f"  range     : {start} -> {end}")
    print(f"  radars    : {radars}")
    print(f"  base_path : {base_path}")
    print(f"  resuming  : {len(checkpoint_seen)} volume(s) already archived")
    print("=" * 70)

    pipeline_start_time = time.time()
    totals = {r: [0, 0] for r in radars}

    days = list(_iter_days(start, end))
    outer = tqdm(days, desc="days", unit="day")
    for day_start, day_end in outer:
        for radar in radars:
            n_ok, n_fail = _process_day_for_radar(
                radar=radar,
                day_start=day_start,
                day_end=day_end,
                args=args,
                db=db,
                base_path=base_path,
                checkpoint_path=checkpoint_path,
                checkpoint_seen=checkpoint_seen,
            )
            totals[radar][0] += n_ok
            totals[radar][1] += n_fail

    print("\n" + "=" * 70)
    print("Run complete")
    for radar, (n_ok, n_fail) in totals.items():
        print(f"  {radar}: {n_ok} archived, {n_fail} failed")
    print(f"  checkpoint: {checkpoint_path}")
    elapsed_time = time.time() - pipeline_start_time
    print(f"  elapsed time: {_format_elapsed_time(elapsed_time)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
