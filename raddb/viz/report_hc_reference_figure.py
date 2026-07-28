"""
Generate the hydrometeor-classification reference figure for the report.

The figure shows a single PPI sweep with:
1. MeteoSwiss operational hydrometeor classification (HC_MCH)
2. PyART-based hydrometeor classification (HC_PYART)
3. Gate-level agreement between the two stored labels

The agreement panel is intended as an illustrative reminder that the two
classification products are reference labels, not absolute ground truth.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/raddb-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/raddb-cache")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import raddb


BASE_PATH = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
RADAR_NAME = "L"
PANEL_TIMESTEP = "2024-07-15 23:05:07"
PANEL_SWEEP = 4

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "hc_reference_ppi_comparison.png"

XY_LIMITS = (-150, 150)
TITLE_FONTSIZE = 13
SUBPLOT_TITLE_FONTSIZE = 11

MATCH_CMAP = ListedColormap(["#d73027", "#1a9850"])
MATCH_NORM = BoundaryNorm([-0.5, 0.5, 1.5], MATCH_CMAP.N)


def _load_archived_datatree():
    db = raddb.RadDB(base_path=BASE_PATH)
    return db.load_datatree(
        radar=RADAR_NAME,
        start_time=PANEL_TIMESTEP,
        end_time=PANEL_TIMESTEP,
    )


def _add_hc_match(dt):
    sweep_names = sorted(
        [group.lstrip("/") for group in dt.groups if group.lstrip("/").startswith("sweep_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    dict_ds = {}
    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        if "HC_MCH" in ds.variables and "HC_PYART" in ds.variables:
            hc_mch = ds["HC_MCH"].values.astype(float)
            hc_pyart = ds["HC_PYART"].values.astype(float)
            valid = np.isfinite(hc_mch) & np.isfinite(hc_pyart)
            match = np.where(valid, (hc_mch == hc_pyart).astype(float), np.nan)
            ds = ds.assign({"hc_match": (ds["HC_MCH"].dims, match)})
        dict_ds[sweep_name] = ds

    return xr.DataTree.from_dict(dict_ds)


def _match_stats(dt_match) -> tuple[int, int, int, float, float]:
    ds = dt_match[f"sweep_{PANEL_SWEEP}"].to_dataset()
    match = ds["hc_match"].values.astype(float)
    valid = np.isfinite(match)
    total = int(valid.sum())
    same = int(np.nansum(match == 1))
    different = int(np.nansum(match == 0))
    same_pct = 100 * same / total if total else float("nan")
    different_pct = 100 * different / total if total else float("nan")
    return total, same, different, same_pct, different_pct


def _plot_hc_panel(dt, dt_match, output_path: Path) -> Path:
    _, _, _, match_pct, mismatch_pct = _match_stats(dt_match)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    panels = [
        (dt, "HC_MCH", "HC MCH", {}),
        (dt, "HC_PYART", "HC PyART", {}),
        (
            dt_match,
            "hc_match",
            "HC MCH == HC PyART",
            dict(cmap=MATCH_CMAP, norm=MATCH_NORM, add_colorbar=False),
        ),
    ]

    for idx, (ax, (source, variable, title, plot_kwargs)) in enumerate(zip(axes, panels)):
        p = raddb.plot_ppi(
            source,
            sweep=PANEL_SWEEP,
            variable=variable,
            ax=ax,
            coords="cartesian",
            **plot_kwargs,
        )
        ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlim(XY_LIMITS)
        ax.set_ylim(XY_LIMITS)

        if idx != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        if variable == "hc_match":
            cbar = plt.colorbar(p, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(["Mismatch", "Match"])
            legend_handles = [
                Patch(facecolor=MATCH_CMAP(1), edgecolor="none", label=f"Match: {match_pct:.1f}%"),
                Patch(facecolor=MATCH_CMAP(0), edgecolor="none", label=f"Mismatch: {mismatch_pct:.1f}%"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower right",
                frameon=True,
                facecolor="white",
                edgecolor="0.75",
                framealpha=0.88,
                fontsize=7.5,
                borderpad=0.3,
                labelspacing=0.25,
                handlelength=1.4,
                handletextpad=0.5,
            )

    axes[2].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())

    fig.suptitle(
        f"Radar {RADAR_NAME} | {PANEL_TIMESTEP} | sweep {PANEL_SWEEP}",
        fontsize=TITLE_FONTSIZE,
        y=0.84,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _print_match_summary(dt_match) -> None:
    total, same, different, same_pct, different_pct = _match_stats(dt_match)
    print(f"Valid gates: {total:,}")
    print(f"Same label: {same:,} ({same_pct:.1f}%)")
    print(f"Different label: {different:,} ({different_pct:.1f}%)")


def main() -> int:
    dt = _load_archived_datatree()
    dt_match = _add_hc_match(dt)
    _print_match_summary(dt_match)
    output_path = _plot_hc_panel(dt, dt_match, OUTPUT_PATH)
    print(f"Figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
