"""
raddb/helper.py
---------------
Shared utilities and configuration for RadDB.
"""
#from __future__ import annotations
import re
import time as _time
import contextlib as _contextlib
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
#from donfig import Config
#from raddb.configs import read_configs

# --- Config Initialization ---
'''def _read_donfig():
    try:
        return {k: v for k, v in read_configs().items() if v is not None}
    except Exception:
        return {}

def init_config():
    defaults = {"base_dir": None}
    defaults.update(_read_donfig())
    return Config("radar", defaults=[defaults], paths=[])

config = init_config()'''

# --- DataTree Helpers ---
def list_sweep_names(dt: xr.DataTree) -> list[str]:
    """Return sorted sweep group names from a DataTree."""
    pat = re.compile(r"^sweep_\d+$")
    return sorted(s.lstrip("/") for s in dt.groups if pat.match(s.lstrip("/")))

def nan_field_like(ds: xr.Dataset, reference_var: str = "DBZH") -> xr.DataArray:
    """Return a DataArray of NaNs with the same shape/dims as reference_var."""
    return xr.full_like(ds[reference_var], fill_value=np.nan, dtype=float)

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


def read_parquet_dataset(
    base_path: str | Path,
    pattern: str = "**/*POLAR.parquet",
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

# --- File Discovery ---
def _find_polar_files_in_range(
    radar_path: Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> list[Path]:
    """Return POLAR parquet files within the given time range, sorted by timestamp."""
    polar_files = sorted(radar_path.rglob("*_POLAR.parquet"))
    if not polar_files:
        return []

    start_dt = ensure_utc(start_time) if start_time else None
    end_dt = ensure_utc(end_time) if end_time else None

    valid = []
    for f in polar_files:
        # Filename: {radar}_{YYYYMMDD}_{HHMMSS}_POLAR.parquet
        stem = f.stem.replace("_POLAR", "")
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
                "duration": _time.perf_counter() - t0,
            })

    def record(self, stage: str, duration: float, volume: str | None = None, sweep: int | None = None):
        """Manually append a pre-measured timing entry."""
        self.records.append({"volume": volume, "sweep": sweep, "stage": stage, "duration": duration})

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


# --- Profiling Plots ---

def plot_stage_totals(
    timer: "StageTimer",
    title: str = "Pipeline — Total Time per Stage",
    save_path: str | None = None,
):
    """Horizontal bar chart: total wall-clock time per pipeline stage."""
    import matplotlib.pyplot as plt
    import numpy as np

    summary = timer.summary()
    if summary.empty:
        print("[profiling] No timing data — nothing to plot.")
        return None

    stages = summary.index.tolist()[::-1]
    totals = summary["sum"].values[::-1]
    counts = summary["count"].values[::-1].astype(int)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(stages)))

    fig, ax = plt.subplots(figsize=(11, max(4, len(stages) * 0.55 + 1.5)))
    bars = ax.barh(stages, totals, color=colors, edgecolor="white", height=0.65)
    x_max = float(max(totals)) if len(totals) > 0 else 1.0
    for bar, val, cnt in zip(bars, totals, counts):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s  (n={cnt})",
            va="center", ha="left", fontsize=8.5,
        )
    ax.set_xlim(0, x_max * 1.30)
    ax.set_xlabel("Total Time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_volume_timing(
    timer: "StageTimer",
    title: str = "Processing Time per Volume",
    save_path: str | None = None,
):
    """Stacked bar chart: per-volume time breakdown by pipeline stage."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = timer.to_dataframe()
    vol_df = df[df["volume"].notna()]
    if vol_df.empty:
        print("[profiling] No per-volume data available.")
        return None

    stage_order = (
        vol_df.groupby("stage")["duration"].sum().sort_values(ascending=False).index.tolist()
    )
    pivot = (
        vol_df.groupby(["volume", "stage"])["duration"]
        .sum().unstack(fill_value=0.0)
        .reindex(columns=stage_order, fill_value=0.0)
    )

    tab_colors = plt.cm.tab20.colors
    stage_colors = {s: tab_colors[i % len(tab_colors)] for i, s in enumerate(stage_order)}
    n_vols = len(pivot)

    fig, ax = plt.subplots(figsize=(max(9, n_vols * 0.65 + 2), 6))
    bottoms = np.zeros(n_vols)
    for stage in stage_order:
        ax.bar(
            range(n_vols), pivot[stage].values, bottom=bottoms,
            label=stage, color=stage_colors[stage], edgecolor="white", linewidth=0.4,
        )
        bottoms += pivot[stage].values

    ax.set_xticks(range(n_vols))
    ax.set_xticklabels([str(v)[-12:] for v in pivot.index], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Stage", title_fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_sweep_timing(
    timer: "StageTimer",
    title: str = "Processing Time per Sweep",
    save_path: str | None = None,
):
    """Box-plot: distribution of total sweep processing time across volumes."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = timer.to_dataframe()
    sweep_df = df[df["sweep"].notna() & (df["stage"] == "load_metranet_sweep")]
    if sweep_df.empty:
        print("[profiling] No 'load_metranet_sweep' per-sweep records found.")
        return None

    sweeps = sorted(sweep_df["sweep"].unique())
    data = [sweep_df[sweep_df["sweep"] == s]["duration"].values for s in sweeps]
    labels = [f"Sweep {int(s)}" for s in sweeps]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sweeps)))

    fig, ax = plt.subplots(figsize=(max(8, len(sweeps) * 0.75 + 2), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_profiling_dashboard(
    timer: "StageTimer",
    title_prefix: str = "",
    save_path: str | None = None,
):
    """3x2 profiling dashboard: stage totals, distribution pie, per-volume bars,
    stage breakdown per volume, and mean time per sweep line chart."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    df = timer.to_dataframe()
    if df.empty:
        print("[profiling] No timing data — nothing to plot.")
        return None

    summary = timer.summary()
    total_time = summary["sum"].sum() if not summary.empty else 0.0

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── Panel 1 (top-left): Stage totals horizontal bar ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    stages = summary.index.tolist()[::-1]
    totals = summary["sum"].values[::-1]
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(stages)))
    bars = ax1.barh(stages, totals, color=bar_colors, edgecolor="white", height=0.65)
    x_max = float(max(totals)) if len(totals) > 0 else 1.0
    for bar, val in zip(bars, totals):
        ax1.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s", va="center", ha="left", fontsize=7.5,
        )
    ax1.set_xlim(0, x_max * 1.30)
    ax1.set_xlabel("Total Time (s)", fontsize=9)
    ax1.set_title("Total Time per Stage", fontsize=11, pad=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=8)

    # ── Panel 2 (top-right): Pie chart time distribution ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pie_colors = plt.cm.tab20.colors[: len(stages)]
    wedges, _, autotexts = ax2.pie(
        totals[::-1], labels=None, autopct="%1.1f%%", startangle=90,
        colors=pie_colors, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax2.legend(
        wedges, stages[::-1],
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5,
        title="Stage", title_fontsize=8,
    )
    ax2.set_title("Time Distribution", fontsize=11, pad=8)

    # ── Panel 3 (middle-left): Per-volume total time bar ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vol_df = df[df["volume"].notna()]
    if not vol_df.empty:
        vol_totals = vol_df.groupby("volume")["duration"].sum()
        x3 = np.arange(len(vol_totals))
        ax3.bar(x3, vol_totals.values, color="steelblue", edgecolor="white", linewidth=0.4)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(
            [str(v)[-12:] for v in vol_totals.index], rotation=45, ha="right", fontsize=7,
        )
        ax3.axhline(
            vol_totals.mean(), color="crimson", linestyle="--", lw=1.5, alpha=0.8,
            label=f"mean = {vol_totals.mean():.1f}s",
        )
        ax3.set_ylabel("Time (s)", fontsize=9)
        ax3.set_title("Total Time per Volume", fontsize=11, pad=8)
        ax3.legend(fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.tick_params(labelsize=7.5)
    else:
        ax3.text(0.5, 0.5, "No per-volume data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Total Time per Volume", fontsize=11, pad=8)

    # ── Panel 4 (middle-right): Stage breakdown per volume (stacked) ─────
    ax4 = fig.add_subplot(gs[1, 1])
    if not vol_df.empty:
        stage_order = (
            vol_df.groupby("stage")["duration"].sum().sort_values(ascending=False).index.tolist()
        )
        pivot = (
            vol_df.groupby(["volume", "stage"])["duration"]
            .sum().unstack(fill_value=0.0)
            .reindex(columns=stage_order, fill_value=0.0)
        )
        tab_colors = plt.cm.tab20.colors
        bottoms4 = np.zeros(len(pivot))
        for i, stage in enumerate(stage_order):
            ax4.bar(
                range(len(pivot)), pivot[stage].values, bottom=bottoms4,
                label=stage, color=tab_colors[i % len(tab_colors)], edgecolor="white", linewidth=0.3,
            )
            bottoms4 += pivot[stage].values
        ax4.set_xticks(range(len(pivot)))
        ax4.set_xticklabels(
            [str(v)[-12:] for v in pivot.index], rotation=45, ha="right", fontsize=7,
        )
        ax4.set_ylabel("Time (s)", fontsize=9)
        ax4.set_title("Stage Breakdown per Volume", fontsize=11, pad=8)
        ax4.legend(fontsize=6.5, loc="upper right", title="Stage", title_fontsize=7)
        ax4.spines[["top", "right"]].set_visible(False)
    else:
        ax4.text(0.5, 0.5, "No per-volume data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Stage Breakdown per Volume", fontsize=11, pad=8)

    # ── Panel 5 (bottom, full width): Mean time per sweep per top stage ──
    ax5 = fig.add_subplot(gs[2, :])
    sweep_df = df[df["sweep"].notna()]
    if not sweep_df.empty:
        top_stages = (
            sweep_df.groupby("stage")["duration"].sum()
            .sort_values(ascending=False).head(6).index.tolist()
        )
        line_colors = plt.cm.tab10.colors
        for idx, stage in enumerate(top_stages):
            by_sweep = sweep_df[sweep_df["stage"] == stage].groupby("sweep")["duration"].mean()
            ax5.plot(
                by_sweep.index, by_sweep.values,
                "o-", label=stage, color=line_colors[idx % 10], markersize=5, linewidth=1.5,
            )
        ax5.set_xlabel("Sweep Number", fontsize=9)
        ax5.set_ylabel("Mean Time (s)", fontsize=9)
        ax5.set_title("Mean Processing Time per Sweep — Top Stages", fontsize=11, pad=8)
        ax5.legend(fontsize=8, loc="upper right", title="Stage", title_fontsize=8)
        ax5.grid(alpha=0.3)
        ax5.spines[["top", "right"]].set_visible(False)
    else:
        ax5.text(0.5, 0.5, "No per-sweep data", ha="center", va="center", transform=ax5.transAxes)
        ax5.set_title("Mean Processing Time per Sweep", fontsize=11, pad=8)

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Pipeline Profiling Dashboard  (total: {total_time:.1f}s)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
