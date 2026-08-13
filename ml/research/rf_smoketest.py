"""End-to-end smoke test for the Phase 4 RF inference path.

Loads the model, builds a 60×60 grid over Dallas, runs a single prediction,
prints the startup log, latency, prediction range, and spot-checks at three
grid cells (dense north-Dallas, mid-metro, empty southeast quadrant).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import BBOX  # noqa: E402
from ml.predictor import load_model, predict_grid  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("smoketest")

HISTORY_CSV = ROOT / "ml" / "data" / "history.csv"
GRID_RES = 60


def main() -> None:
    log.info("Bootstrapping RF model")
    load_model()

    lat_grid = np.linspace(BBOX["south"], BBOX["north"], GRID_RES)
    lon_grid = np.linspace(BBOX["west"], BBOX["east"], GRID_RES)
    lons_2d, lats_2d = np.meshgrid(lon_grid, lat_grid)

    # Use the most recent history.csv row's met values so the prediction is
    # comparable to a real sensor reading.
    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    latest_ts = df["timestamp"].max()
    latest = df[df["timestamp"] == latest_ts]
    humidity = float(latest["humidity"].mean())
    wind_speed = float(latest["wind_speed"].iloc[0])
    wind_deg = float(latest["wind_deg"].iloc[0])

    timestamp = pd.Timestamp(datetime.now(timezone.utc))
    log.info(
        f"Inputs: timestamp={timestamp.isoformat()}, humidity={humidity:.1f}, "
        f"wind_speed={wind_speed:.2f}, wind_deg={wind_deg:.0f}"
    )

    grid = predict_grid(
        lats_2d=lats_2d,
        lons_2d=lons_2d,
        humidity=humidity,
        wind_speed=wind_speed,
        wind_deg=wind_deg,
        timestamp=timestamp,
    )

    log.info(
        f"Grid prediction stats: shape={grid.shape}, "
        f"min={grid.min():.2f}, max={grid.max():.2f}, "
        f"mean={grid.mean():.2f}, median={np.median(grid):.2f} µg/m³"
    )

    # Spot-check three cells. Indexing: [row, col] = [lat_idx, lon_idx].
    # row 0 = south, last row = north; col 0 = west, last col = east.
    n = GRID_RES
    cells = [
        ("dense north-Dallas",        n - 1,        n // 4),
        ("mid-metro",                 n // 2,       n // 2),
        ("empty southeast quadrant",  n // 4,       n - 1),
    ]

    print()
    print("Spot checks:")
    print(f"  {'where':<28} {'lat':>9} {'lon':>10} {'pred':>8}  {'nearest sensor reading':<35}")
    sensor_locs = (
        df.groupby("sensor_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )
    for label, ri, ci in cells:
        lat = float(lats_2d[ri, ci])
        lon = float(lons_2d[ri, ci])
        pred = float(grid[ri, ci])
        # Nearest training sensor + that sensor's most recent recorded pm25
        d2 = (sensor_locs["lat"] - lat) ** 2 + ((sensor_locs["lon"] - lon) * 0.84) ** 2
        nearest_sid = int(sensor_locs.loc[d2.idxmin(), "sensor_id"])
        nearest_dist_km = float(np.sqrt(d2.min())) * 111.0
        sensor_recent = (
            df[df["sensor_id"] == nearest_sid]
            .sort_values("timestamp")
            .iloc[-1]
        )
        comp = (
            f"sensor {nearest_sid} ({nearest_dist_km:.1f} km away): "
            f"pm25={float(sensor_recent['pm25']):.2f} @ "
            f"{pd.Timestamp(sensor_recent['timestamp']).strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"  {label:<28} {lat:>9.4f} {lon:>10.4f} {pred:>8.2f}  {comp}")
    print()


if __name__ == "__main__":
    main()
