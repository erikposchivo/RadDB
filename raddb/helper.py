"""
raddb/helper.py
---------------
Shared utilities and configuration for RadDB.
"""
import re
import time as _time
import contextlib as _contextlib
import datetime as _dt
from pathlib import Path

import pandas as pd
import xarray as xr

# --- DataTree Helpers ---
def list_sweep_names(dt: xr.DataTree) -> list[str]:
    """Return sorted sweep group names from a DataTree."""
    pat = re.compile(r"^sweep_\d+$")
    return sorted(s.lstrip("/") for s in dt.groups if pat.match(s.lstrip("/")))

# --- Parquet Helpers ---
def ensure_utc(dt_input):
    """
    Ensure a datetime-like input is timezone-aware and in UTC.
    """
    if dt_input is None:
        return None

    ts = pd.to_datetime(dt_input)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def read_parquet_files(
    base_path: str | Path,
    pattern: str = "**/*POL.parquet",
    columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    files = sorted(Path(base_path).rglob(pattern))
    if not files:
        if verbose: print(f"[RadDB] No files found matching '{pattern}' in {base_path}")
        return pd.DataFrame()
    if verbose: print(f"[RadDB] Found {len(files)} file(s) — loading...")
    return pd.concat([pd.read_parquet(f, columns=columns, engine="pyarrow") for f in files], ignore_index=True)

def check_dataframe(df: pd.DataFrame) -> None:
    print("-" * 50)
    print(f"Shape:   {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("-" * 50)
    print(f"Missing values:\n{df.isnull().sum()}")
    print("-" * 50)
    print(df.head())
    print("-" * 50)

# --- Radar Name Normalization ---
def normalize_radar_name(radar: str) -> str:
    """
    Normalize radar name to use only the single letter identifier.

    Handles both formats:
    - "MLA" -> "A"
    - "A" -> "A"
    - "MLW" -> "W"

    Parameters
    ----------
    radar : str
        Radar name (can be "ML*" format or just the letter)

    Returns
    -------
    str
        Normalized single-letter radar name (uppercase)

    Examples
    --------
    >>> normalize_radar_name("MLA")
    'A'
    >>> normalize_radar_name("A")
    'A'
    >>> normalize_radar_name("MLW")
    'W'
    """
    radar_upper = radar.upper().strip()

    # If it starts with "ML", extract the last character
    if radar_upper.startswith("ML") and len(radar_upper) > 2:
        return radar_upper[-1]

    # Otherwise, return the last character (or the string if it's already a single char)
    return radar_upper[-1] if len(radar_upper) > 0 else radar_upper


# --- Filter Logic Registry ---
FILTER_LOGICS: dict[str, callable] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


def resolve_filter_logic(logic: str):
    """Return the comparison function for *logic*.

    Raises
    ------
    ValueError
        If ``logic`` is not one of the supported operators.
    """
    fn = FILTER_LOGICS.get(logic)
    if fn is None:
        raise ValueError(
            f"Unknown logic '{logic}'. Choose from: {list(FILTER_LOGICS)}"
        )
    return fn


def filter_df(
    df: pd.DataFrame,
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> pd.DataFrame:
    """Filter a DataFrame, keeping rows where ``feature [logic] threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature : str
        Column name to filter on (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with non-matching rows dropped (index reset).

    Raises
    ------
    KeyError
        If ``feature`` is not a column of ``df``.
    ValueError
        If ``logic`` is not one of the supported operators.
    """
    fn = resolve_filter_logic(logic)
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns.")
    mask = fn(df[feature].to_numpy(), threshold)
    return df[mask].reset_index(drop=True)


def filter_dt(
    dt: xr.DataTree,
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> xr.DataTree:
    """Filter a DataTree, masking gates where ``feature [logic] threshold`` is False.

    Gates that do **not** satisfy the condition are set to NaN across all
    data variables in each sweep.  Gates that *do* satisfy the condition
    keep their original values unchanged — including legitimate zero values.

    .. note::
        This operates on the multidimensional DataTree structure via
        ``xr.Dataset.where()``.  For tabular (row-level) filtering use
        :func:`filter_df` instead, which drops non-matching rows entirely.

    Parameters
    ----------
    dt : xr.DataTree
        Input DataTree with ``sweep_N`` groups.
    feature : str
        Variable name to use as the filter criterion (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    xr.DataTree
        New DataTree where non-matching gates have NaN for all variables.
        Matching gates are left entirely unchanged (zeros remain zeros).

    Raises
    ------
    ValueError
        If ``logic`` is not a supported operator.
    """
    fn = resolve_filter_logic(logic)

    sweep_names = list_sweep_names(dt)
    dict_ds = {}

    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        if feature in ds:
            keep_mask = fn(ds[feature], threshold)
            # Mask only variables that share the mask's dimensions.
            # Variables on other dims (e.g. gate-edge arrays x_edges/y_edges
            # on azimuth_edge/range_edge) are static geometry — masking them
            # is meaningless and broadcasting the mask onto them explodes
            # memory (dims are disjoint).
            mask_dims = set(keep_mask.dims)
            ds = ds.assign({
                name: var.where(keep_mask)
                for name, var in ds.data_vars.items()
                if mask_dims & set(var.dims)
            })
        dict_ds[sweep_name] = ds

    return xr.DataTree.from_dict(dict_ds)


# ============================================================
# --- Profiling Utilities ---
# ============================================================

def _vprint(msg: str, verbose: bool = False) -> None:
    """Print a timestamped progress message to stdout when verbose is True."""
    if verbose:
        ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}]  {msg}", flush=True)


class StageTimer:
    """Accumulates per-stage timing records for pipeline profiling.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer.time_stage("my_stage", volume="vol_001", sweep=2):
    ...     do_work()
    >>> timer.print_summary()
    >>> plot_profiling_dashboard(timer)
    """

    def __init__(self):
        self.records: list[dict] = []

    @_contextlib.contextmanager
    def time_stage(self, stage: str, volume: str | None = None, sweep: int | None = None):
        """Context manager — records elapsed wall-clock time for *stage*."""
        t0 = _time.perf_counter()
        try:
            yield
        finally:
            self.records.append({
                "volume": volume,
                "sweep": sweep,
                "stage": stage,
                "t_start": t0,
                "duration": _time.perf_counter() - t0,
            })

    def record(self, stage: str, duration: float, volume: str | None = None, sweep: int | None = None, t_start: float | None = None):
        """Manually append a pre-measured timing entry."""
        self.records.append({"volume": volume, "sweep": sweep, "stage": stage, "t_start": t_start, "duration": duration})

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a DataFrame with columns [volume, sweep, stage, duration]."""
        if not self.records:
            return pd.DataFrame(columns=["volume", "sweep", "stage", "duration"])
        return pd.DataFrame(self.records)

    def summary(self) -> pd.DataFrame:
        """Aggregate by stage — returns (sum, mean, min, max, count) sorted by total time."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("stage")["duration"]
            .agg(["sum", "mean", "min", "max", "count"])
            .sort_values("sum", ascending=False)
        )

    def print_summary(self):
        """Print a formatted profiling table to stdout."""
        summary = self.summary()
        if summary.empty:
            print("[RadDB profiling] No timing data recorded.")
            return
        total = summary["sum"].sum()
        print("\n" + "=" * 68)
        print("  PIPELINE PROFILING SUMMARY")
        print("=" * 68)
        print(f"  {'Stage':<34} {'Total':>7}  {'Mean':>6}  {'N':>5}  {'%':>5}")
        print("-" * 68)
        for stage, row in summary.iterrows():
            pct = 100.0 * row["sum"] / total if total > 0 else 0.0
            print(
                f"  {stage:<34} {row['sum']:>6.2f}s  {row['mean']:>5.2f}s"
                f"  {int(row['count']):>5}  {pct:>4.1f}%"
            )
        print("-" * 68)
        print(f"  {'TOTAL':<34} {total:>6.2f}s")
        print("=" * 68)
