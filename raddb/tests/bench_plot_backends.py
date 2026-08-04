"""
raddb/tests/bench_plot_backends.py
----------------------------------
Measured comparison of the ways a RadDB frame can be turned into a PPI.

Not a test — ``pytest`` does not collect it (the filename is not ``test_*``).
Run it directly against a real archive::

    python -m raddb.tests.bench_plot_backends --archive /path/to/archive --radar L

Five paths are compared:

1. ``polygons``  — polars -> numpy -> ``PolyCollection`` (what the four plots use)
2. ``geopandas`` — ``to_geopandas()`` -> ``GeoDataFrame.plot``
3. ``lonboard``  — ``to_geoarrow()`` -> deck.gl widget (interactive, not matplotlib)

For each: geometric deviation from the exact ``h_plane`` corners, wall time,
peak RSS, and the size of the saved PNG and PDF.  Every path is run on a full
sweep **and** on a small crop — the crop is where ``datatree`` falls apart,
because it reindexes onto the complete azimuth x range grid regardless of how
few gates survive.
"""
from __future__ import annotations

import argparse
import gc
import threading
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely

import raddb
from raddb.main import RadDB
from raddb.lut import gate_corner_table

VARIABLE = "DBZH"
SWEEP = 1


# --------------------------------------------------------------------------- util

def _rss_mb() -> float:
    """Current resident set size in MB."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1e6


class _RssSampler:
    """Per-call peak RSS.

    ``ru_maxrss`` is a monotonic high-water mark for the whole process, so it
    reports 0 for every backend that runs after a heavier one.  Sampling VmRSS in
    a side thread gives each backend its own peak, independent of run order.
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.base = _rss_mb()
        self.peak = self.base

        def poll():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, _rss_mb())

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_mb())
        return False

    @property
    def delta(self) -> float:
        return self.peak - self.base


def _sizes(fig, tmp: Path, tag: str) -> tuple[float, float]:
    """Saved PNG and PDF size in kB."""
    png, pdf = tmp / f"{tag}.png", tmp / f"{tag}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png.stat().st_size / 1e3, pdf.stat().st_size / 1e3


def _reference_corners(radar: str, base: Path, gate_ids) -> np.ndarray:
    """Exact h_plane corners for the given gates, as an (n, 4, 2) array."""
    tbl = gate_corner_table(radar, base, kind="h_plane")
    aligned = pl.DataFrame({"gate_id": np.asarray(gate_ids, dtype=np.int64)}).join(
        tbl, on="gate_id", how="left", maintain_order="left"
    )
    return np.stack([
        np.stack([aligned[f"x_{k}"].to_numpy(), aligned[f"y_{k}"].to_numpy()], axis=1)
        for k in range(1, 5)
    ], axis=1).astype(np.float64)


# ----------------------------------------------------------------------- backends

def bench_polygons(rdf, radar, base, tmp, tag):
    t0 = time.perf_counter()
    p = rdf.plot_ppi(sweep=SWEEP, variable=VARIABLE, coords="xy")
    t_draw = time.perf_counter() - t0
    fig = p.figure
    drawn = np.array([path.vertices[:4] for path in p.get_paths()])
    png, pdf = _sizes(fig, tmp, f"{tag}_polygons")
    plt.close(fig)
    return dict(n=len(drawn), t=t_draw, png=png, pdf=pdf, dev=0.0, drawn=drawn)


def bench_geopandas(rdf, radar, base, tmp, tag):
    t0 = time.perf_counter()
    gdf = rdf.to_geopandas()
    gdf = gdf[gdf[VARIABLE].notna()]
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(column=VARIABLE, ax=ax, markersize=1)
    t_draw = time.perf_counter() - t0
    png, pdf = _sizes(fig, tmp, f"{tag}_geopandas")
    plt.close(fig)
    return dict(n=len(gdf), t=t_draw, png=png, pdf=pdf, dev=float("nan"))


def bench_lonboard(rdf, radar, base, tmp, tag):
    import lonboard
    t0 = time.perf_counter()
    table = rdf.to_geoarrow(geometry="polygon")
    layer = lonboard.PolygonLayer(table=table)
    m = lonboard.Map(layers=[layer])
    t_draw = time.perf_counter() - t0
    html = tmp / f"{tag}_lonboard.html"
    try:
        m.to_html(str(html))
        size = html.stat().st_size / 1e3
    except Exception as exc:                                   # noqa: BLE001
        print(f"      (lonboard to_html failed: {exc})")
        size = float("nan")
    return dict(n=len(table), t=t_draw, png=size, pdf=float("nan"), dev=0.0)


BACKENDS = [
    ("polygons  (PolyCollection)", bench_polygons),
    ("geopandas  (GeoDataFrame)", bench_geopandas),
    ("lonboard   (deck.gl)", bench_lonboard),
]


def run(rdf, radar, base, tmp, tag):
    print(f"\n{'=' * 92}\n{tag}: {len(rdf):,} gates\n{'=' * 92}")
    print(f"{'backend':30s} {'drawn':>10s} {'time [s]':>9s} {'peakRSS':>9s} "
          f"{'PNG [kB]':>10s} {'PDF [kB]':>10s}")
    print("-" * 92)
    results = {}
    for name, fn in BACKENDS:
        gc.collect()
        try:
            with warnings.catch_warnings(), _RssSampler() as rss:
                warnings.simplefilter("ignore")
                r = fn(rdf, radar, base, tmp, tag)
        except Exception as exc:                               # noqa: BLE001
            print(f"{name:30s} {'FAILED':>10s}  {type(exc).__name__}: {exc}")
            plt.close("all")
            continue
        r["rss"] = rss.delta
        results[name] = r
        pdf = f"{r['pdf']:10.0f}" if np.isfinite(r["pdf"]) else f"{'-':>10s}"
        print(f"{name:30s} {r['n']:10,d} {r['t']:9.2f} {r['rss']:8.0f}M "
              f"{r['png']:10.0f} {pdf}")
    return results


def check_precision(rdf, radar, base):
    """How far each path's geometry sits from the exact frustum corners."""
    print(f"\n{'=' * 92}\ngeometric precision vs the exact h_plane corners\n{'=' * 92}")

    p = rdf.plot_ppi(sweep=SWEEP, variable=VARIABLE, coords="xy")
    drawn = np.array([path.vertices[:4] for path in p.get_paths()])
    ids = (
        rdf.data.select(["gate_id", VARIABLE])
        .join(gate_corner_table(radar, base, "h_plane", sweep=SWEEP).select("gate_id"),
              on="gate_id", how="semi")
        .filter(pl.col(VARIABLE).is_not_nan())["gate_id"].to_numpy()
    )
    plt.close("all")
    ref = _reference_corners(radar, base, ids)
    dev = np.abs(drawn - ref).max() if len(drawn) == len(ref) else float("nan")
    print(f"  polygons   : {dev:.3e} m   (exact — these ARE the stored corners)")

    # The centroid mesh matplotlib would infer if it had no corner nodes.
    lut = raddb.load_radar_lut(radar, base).filter(pl.col("sweep") == SWEEP)
    n_az = lut["azimuth"].n_unique()
    n_rng = lut["range"].n_unique()
    cx = lut.sort(["azimuth", "range"])["x"].to_numpy().reshape(n_az, n_rng)
    cy = lut.sort(["azimuth", "range"])["y"].to_numpy().reshape(n_az, n_rng)
    mid_x = 0.25 * (cx[:-1, :-1] + cx[:-1, 1:] + cx[1:, :-1] + cx[1:, 1:])
    mid_y = 0.25 * (cy[:-1, :-1] + cy[:-1, 1:] + cy[1:, :-1] + cy[1:, 1:])
    from raddb.lut import load_plane_nodes, _node_grids
    nodes = load_plane_nodes(radar, base, "h_plane", sweep=SWEEP)
    g = _node_grids(nodes, ["x", "y"])[SWEEP]
    off = np.hypot(mid_x - g["x"][1:-1, 1:-1], mid_y - g["y"][1:-1, 1:-1])
    print(f"  centroid   : {off.mean():.1f} m mean, {off.max():.1f} m max "
          f"(what shading='auto' invents when no corner nodes exist)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--radar", default="L")
    ap.add_argument("--crop-km", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="where to write the figures")
    args = ap.parse_args()

    base = Path(args.archive)
    tmp = Path(args.out) if args.out else Path("bench_out")
    tmp.mkdir(parents=True, exist_ok=True)

    db = RadDB(archive_dir=str(base), crs=2056)
    info = db.get_radar_info(args.radar)

    # Every backend must draw the same thing, so restrict to one sweep up front:
    # plot_ppi selects the sweep itself, but geopandas/lonboard would otherwise
    # render the whole volume and the timings would not be comparable.
    rdf = db.open(radars=args.radar).sel(sweep=SWEEP)

    check_precision(rdf, args.radar, base)
    run(rdf, args.radar, base, tmp, f"full sweep {SWEEP}")

    from raddb.aoi import _reproject_to_aoi
    site = _reproject_to_aoi(shapely.Point(info["longitude"], info["latitude"]), 4326, 2056)
    crop = rdf.crop_around_point((site.x, site.y), distance=args.crop_km * 1000)
    run(crop, args.radar, base, tmp, f"sweep {SWEEP} cropped to {args.crop_km:g} km")


if __name__ == "__main__":
    main()
