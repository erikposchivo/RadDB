"""
Generate RadDB report figures.

This script is a report-focused version of the multi-feature panel section in
``examples/basic_usage.py``. It generates:

1. A 2x3 PPI panel for DBZH, ZDR, KDP, RHOHV, TEMP, HC_PYART.
2. A 2x2 RHI panel for DBZH, ZDR, KDP, RHOHV.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/raddb-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/raddb-cache")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import raddb

try:
    import pyart  # noqa: F401  # Registers Py-ART colormaps when available.
except Exception:
    pass

try:
    import cmweather  # noqa: F401  # Registers weather-radar colormaps.
except Exception:
    pass


BASE_PATH = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
RADAR_NAME = "L"
PANEL_TIMESTEP = "2024-07-15 23:05:07"
PANEL_SWEEP = 4
PANEL_AZIMUTH = 315

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PPI_FEATURES = ["DBZH", "ZDR", "KDP", "RHOHV", "TEMP", "HC_PYART"]
RHI_FEATURES = ["DBZH", "ZDR", "KDP", "RHOHV"]

COMMON_XY_LIMITS = (-150, 150)
TITLE_FONTSIZE = 13
SUBPLOT_TITLE_FONTSIZE = 11
TEMP_TICKS = [-30, -20, -10, 0, 10, 20, 30]


def _weather_cmap(name: str, fallback: str):
    if name in plt.colormaps():
        return name
    return fallback


def _first_available_cmap(*names: str) -> str:
    for name in names:
        if name in plt.colormaps():
            return name
    return names[-1]


FEATURE_KWARGS = {
    "DBZH": dict(
        cmap=_first_available_cmap("HomeyerRainbow", "turbo"),
        vmin=0,
        vmax=60,
        subtitle=r"$Z_H$",
    ),
    "ZDR": dict(
        cmap="viridis",
        vmin=-2,
        vmax=7,
        subtitle=r"$Z_{DR}$",
    ),
    "KDP": dict(
        cmap="twilight",
        vmin=-2,
        vmax=5,
        subtitle=r"$K_{dp}$",
    ),
    "RHOHV": dict(
        cmap="cividis",
        vmin=0.5,
        vmax=1.0,
        subtitle=r"$\rho_{hv}$",
    ),
    "TEMP": dict(
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
        vmin=None,
        vmax=None,
        subtitle="Temperature",
    ),
    "HC_PYART": dict(
        subtitle="HC PyART",
    ),
}

LOCAL_KEYS = {"subtitle"}


def _plot_kwargs(feature: str) -> dict:
    return {
        key: value
        for key, value in FEATURE_KWARGS[feature].items()
        if key not in LOCAL_KEYS
    }


def _set_report_colorbar_ticks(feature: str, mappable) -> None:
    if feature != "TEMP":
        return
    cbar = getattr(mappable, "colorbar", None)
    if cbar is not None:
        cbar.set_ticks(TEMP_TICKS)


def create_ppi_panel(dt, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for idx, (ax, feature) in enumerate(zip(axes.ravel(), PPI_FEATURES)):
        p = raddb.plot_ppi(
            dt,
            sweep=PANEL_SWEEP,
            variable=feature,
            ax=ax,
            coords="cartesian",
            **_plot_kwargs(feature),
        )
        _set_report_colorbar_ticks(feature, p)
        ax.set_title(FEATURE_KWARGS[feature]["subtitle"], fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlim(COMMON_XY_LIMITS)
        ax.set_ylim(COMMON_XY_LIMITS)

        row, col = divmod(idx, 3)
        if row == 0:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    fig.suptitle(
        f"Radar {RADAR_NAME} | {PANEL_TIMESTEP} | sweep {PANEL_SWEEP}",
        fontsize=TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_rhi_panel(dt, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for idx, (ax, feature) in enumerate(zip(axes.ravel(), RHI_FEATURES)):
        p = raddb.plot_rhi(
            dt,
            azimuth=PANEL_AZIMUTH,
            variable=feature,
            radar=RADAR_NAME,
            max_range_km=110,
            max_height_km=11,
            ax=ax,
            **_plot_kwargs(feature),
        )
        _set_report_colorbar_ticks(feature, p)
        ax.set_title(FEATURE_KWARGS[feature]["subtitle"], fontsize=SUBPLOT_TITLE_FONTSIZE)

        row, col = divmod(idx, 2)
        if row == 0:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Height [km]")
    for ax in axes[-1, :]:
        ax.set_xlabel("Range [km]")

    fig.suptitle(
        f"Radar {RADAR_NAME} | {PANEL_TIMESTEP} | azimuth {PANEL_AZIMUTH}$^\\circ$",
        fontsize=TITLE_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_archived_datatree():
    db = raddb.RadDB(base_path=BASE_PATH)
    return db.load_datatree(
        radar=RADAR_NAME,
        start_time=PANEL_TIMESTEP,
        end_time=PANEL_TIMESTEP,
    )


def load_panel_datatree():
    return _load_archived_datatree()


def main() -> int:
    dt = load_panel_datatree()

    ppi_path = create_ppi_panel(dt, OUTPUT_DIR / "raddb_ppi_panel.png")
    rhi_path = create_rhi_panel(dt, OUTPUT_DIR / "raddb_rhi_panel.png")
    print(f"PPI figure: {ppi_path}")
    print(f"RHI figure: {rhi_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
