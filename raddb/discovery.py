"""
raddb/discovery.py
------------------
File discovery for RadDB — both sides of the archive:

- **Raw data scanning** (input side): walk a METRANET directory tree
  (``{root}/{NETWORK}/{yyyy}/{mm}/{dd}/{hh}/ML{R}/...``) and summarise which
  radars, time periods, and products are available for archiving.  This is
  what powers :meth:`raddb.RadDB.show_available_data`.
- **Archived POL search** (output side): locate ``*_POL.parquet`` files in a
  time range inside an existing RadDB archive.

Everything here is pure filesystem + pandas — no pyart / radar_api
dependency — so discovery works in any environment.
"""
from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path

import pandas as pd

from raddb.helper import ensure_utc


# ============================================================================
# METRANET filename helpers  (shared with raddb.mch_pipeline)
# ============================================================================

def _parse_volume_time(stem: str) -> datetime.datetime:
    """Parse the timestamp from a METRANET filename stem.

    Works for any ``XXXYYJJJHHMM...`` stem (3-char prefix + 2-digit year +
    day-of-year + hour + minute), e.g. ``MLA2419423300U`` or
    ``HZT2124010000L``.  Returns 1970-01-01 when the stem cannot be parsed.
    """
    try:
        y, j, h, m = (
            int(stem[3:5]),
            int(stem[5:8]),
            int(stem[8:10]),
            int(stem[10:12]),
        )
        return datetime.datetime(2000 + y, 1, 1) + datetime.timedelta(
            days=j - 1, hours=h, minutes=m
        )
    except Exception:
        return datetime.datetime(1970, 1, 1)


def _group_files_by_volume(paths: list[str]) -> dict:
    """Group sweep files by volume (based on filename stem)."""
    vols = defaultdict(list)
    for p in paths:
        stem = Path(p).stem
        vols[stem].append(p)
    return dict(vols)


# ============================================================================
# Raw METRANET archive scanning  (input side)
# ============================================================================

def _raw_network_roots(raw_data_dir: str | Path) -> list[Path]:
    """Return network root dirs (e.g. ``.../RADAR/MCH``) under *raw_data_dir*.

    Accepts either the directory that contains network folders
    (``.../RADAR``) or a network folder itself (``.../RADAR/MCH``) — detected
    by whether 4-digit year subdirectories are present.
    """
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    def _has_year_dirs(d: Path) -> bool:
        return any(
            s.is_dir() and s.name.isdigit() and len(s.name) == 4
            for s in d.iterdir()
        )

    if _has_year_dirs(root):
        return [root]
    nets = [d for d in root.iterdir() if d.is_dir() and _has_year_dirs(d)]
    if not nets:
        raise FileNotFoundError(
            f"No METRANET layout found under {root} "
            "(expected {root}/[NETWORK]/yyyy/mm/dd/hh/ML*/...)"
        )
    return sorted(nets)


def scan_raw_archive(
    raw_data_dir: str | Path,
    radars: list[str] | None = None,
) -> pd.DataFrame:
    """Scan a raw METRANET tree and summarise available data per (radar, day).

    Parameters
    ----------
    raw_data_dir : str or Path
        Root of the raw archive — either ``.../RADAR`` (containing network
        dirs like ``MCH``) or a network dir itself.
    radars : list of str, optional
        Restrict the scan to these radar letters (e.g. ``["A", "L"]``).

    Returns
    -------
    pd.DataFrame
        One row per (network, radar, date) with columns:
        ``network, radar, date, n_volumes, first_volume, last_volume,
        n_sweep_files, has_hym, has_hzt``.
    """
    want = {r.upper() for r in radars} if radars else None
    rows = []

    for net_root in _raw_network_roots(raw_data_dir):
        network = net_root.name
        for year_dir in sorted(d for d in net_root.iterdir() if d.is_dir()):
            for month_dir in sorted(d for d in year_dir.iterdir() if d.is_dir()):
                for day_dir in sorted(d for d in month_dir.iterdir() if d.is_dir()):
                    # Collect per-radar info across the hour dirs of this day
                    day_pol: dict[str, dict] = {}
                    day_hym: set[str] = set()
                    day_hzt = False
                    for hour_dir in sorted(d for d in day_dir.iterdir() if d.is_dir()):
                        for prod_dir in hour_dir.iterdir():
                            if not prod_dir.is_dir():
                                continue
                            name = prod_dir.name.upper()
                            if name == "HZT":
                                day_hzt = True
                            elif name.startswith("YM") and len(name) == 3:
                                day_hym.add(name[-1])
                            elif name.startswith("ML") and len(name) == 3:
                                radar = name[-1]
                                if want and radar not in want:
                                    continue
                                files = [f for f in prod_dir.iterdir() if f.is_file()]
                                if not files:
                                    continue
                                info = day_pol.setdefault(
                                    radar, {"stems": set(), "n_files": 0}
                                )
                                info["stems"].update(f.stem for f in files)
                                info["n_files"] += len(files)

                    for radar, info in sorted(day_pol.items()):
                        times = sorted(_parse_volume_time(s) for s in info["stems"])
                        rows.append({
                            "network": network,
                            "radar": radar,
                            "date": f"{year_dir.name}-{month_dir.name}-{day_dir.name}",
                            "n_volumes": len(info["stems"]),
                            "first_volume": times[0],
                            "last_volume": times[-1],
                            "n_sweep_files": info["n_files"],
                            "has_hym": radar in day_hym,
                            "has_hzt": day_hzt,
                        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["network", "radar", "date"]).reset_index(drop=True)
    return df


def summarize_raw_archive(df_days: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a :func:`scan_raw_archive` result to one row per radar."""
    if df_days.empty:
        return pd.DataFrame()
    g = df_days.groupby(["network", "radar"])
    out = pd.DataFrame({
        "days": g["date"].nunique(),
        "volumes": g["n_volumes"].sum(),
        "first_volume": g["first_volume"].min(),
        "last_volume": g["last_volume"].max(),
        "hym": g["has_hym"].all(),
        "hzt": g["has_hzt"].all(),
    }).reset_index()
    return out


def print_available_data(
    raw_data_dir: str | Path,
    radars: list[str] | None = None,
    detail: bool = False,
) -> pd.DataFrame:
    """Print a human-readable summary of available raw data; return the scan.

    Parameters
    ----------
    raw_data_dir : str or Path
        Raw METRANET root (see :func:`scan_raw_archive`).
    radars : list of str, optional
        Restrict to these radars.
    detail : bool
        Also print the per-day table (volumes per radar per day).

    Returns
    -------
    pd.DataFrame
        The per-day scan DataFrame (as from :func:`scan_raw_archive`),
        for programmatic use.
    """
    df_days = scan_raw_archive(raw_data_dir, radars=radars)

    print("=" * 78)
    print(f"  RadDB — available raw data   (root: {raw_data_dir})")
    print("=" * 78)
    if df_days.empty:
        print("  No METRANET volumes found.")
        print("=" * 78)
        return df_days

    summary = summarize_raw_archive(df_days)
    hdr = f"  {'net':<5} {'radar':<6} {'period':<28} {'days':>5} {'volumes':>8}  products"
    print(hdr)
    print("-" * 78)
    for _, r in summary.iterrows():
        period = f"{r['first_volume']:%Y-%m-%d %H:%M} -> {r['last_volume']:%Y-%m-%d %H:%M}"
        products = "POL" + ("+HYM" if r["hym"] else "") + ("+HZT" if r["hzt"] else "")
        print(
            f"  {r['network']:<5} {r['radar']:<6} {period:<28} "
            f"{r['days']:>5} {r['volumes']:>8,}  {products}"
        )
    print("-" * 78)
    print(f"  total: {summary['volumes'].sum():,} volumes across "
          f"{df_days['date'].nunique()} day(s), {len(summary)} radar(s)")

    if detail:
        print("-" * 78)
        print("  per-day detail:")
        for _, r in df_days.iterrows():
            print(
                f"    {r['radar']}  {r['date']}  "
                f"{r['n_volumes']:>4} volumes  "
                f"({r['first_volume']:%H:%M} -> {r['last_volume']:%H:%M})"
                f"{'  +HYM' if r['has_hym'] else ''}"
                f"{'  +HZT' if r['has_hzt'] else ''}"
            )
    print("=" * 78)
    return df_days


# ============================================================================
# Archived POL parquet search  (output side)
# ============================================================================

def _find_polar_files_in_range(
    radar_path: Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> list[Path]:
    """Return POLAR parquet files within the given time range, sorted by timestamp."""
    polar_files = sorted(radar_path.rglob("*_POL.parquet"))
    if not polar_files:
        return []

    start_dt = ensure_utc(start_time) if start_time else None
    end_dt = ensure_utc(end_time) if end_time else None

    valid = []
    for f in polar_files:
        # Filename: {radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet
        stem = f.stem.replace("_POL", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            ts = pd.to_datetime(parts[-2] + "_" + parts[-1], format="%Y%m%d_%H%M%S")
        except Exception:
            continue
        ts = ensure_utc(ts)
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        valid.append((ts, f))

    valid.sort(key=lambda x: x[0])
    return [f for _, f in valid]
