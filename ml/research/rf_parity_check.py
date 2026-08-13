"""Phase 4 train/inference parity check.

Picks a sensor from ml/data/history.csv, builds features for that lat/lon at
that exact training-row timestamp using the live inference pipeline, and
diffs the result against the row history.csv recorded.

Structural features (lat, lon, dist_to_highway_m, all timestamp-derived
columns) should match within floating-point rounding. Meteorological
features (humidity, wind_speed, wind_deg) at training time came from
PurpleAir per-sensor + Meteostat archive; the inference pipeline uses
metro-mean PurpleAir + live OpenWeatherMap. We expect those to differ —
the script reports the magnitude rather than failing on it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.predictor import build_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("parity")

HISTORY_CSV = ROOT / "ml" / "data" / "history.csv"

STRUCTURAL = [
    "lat", "lon", "dist_to_highway_m",
    "local_hour_of_day", "day_of_week", "is_weekend",
    "is_am_rush", "is_pm_rush", "traffic_index",
]
METEO = ["humidity", "wind_speed", "wind_deg"]


def main() -> None:
    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    if df.empty:
        log.error("history.csv is empty — cannot run parity check")
        sys.exit(1)

    # Pick a recent row deterministically: the most recent timestamp for the
    # most-data sensor. Stable across runs, and a sensor with lots of rows is
    # representative.
    counts = df.groupby("sensor_id").size().sort_values(ascending=False)
    sensor_id = int(counts.index[0])
    sensor_rows = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
    train_row = sensor_rows.iloc[-1]
    timestamp = pd.Timestamp(train_row["timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    log.info(f"Parity-check sensor: {sensor_id}")
    log.info(f"Training row count for sensor: {len(sensor_rows):,}")
    log.info(f"Picked timestamp: {timestamp.isoformat()}")
    log.info(f"  lat={train_row['lat']:.6f} lon={train_row['lon']:.6f}")

    feats = build_features(
        lats=np.array([float(train_row["lat"])]),
        lons=np.array([float(train_row["lon"])]),
        humidity=float(train_row["humidity"]),
        wind_speed=float(train_row["wind_speed"]),
        wind_deg=float(train_row["wind_deg"]),
        timestamp=timestamp,
    ).iloc[0]

    print()
    print("=" * 78)
    print(f"{'feature':<22} {'training (history.csv)':>25} {'inference':>15} {'Δ':>10}")
    print("=" * 78)

    failures: list[str] = []
    for col in STRUCTURAL:
        t = float(train_row[col])
        i = float(feats[col])
        delta = i - t
        ok = abs(delta) <= 1e-6
        flag = "OK" if ok else "FAIL"
        print(f"  {col:<20} {t:>25.6f} {i:>15.6f} {delta:>10.6f}  [{flag}]")
        if not ok:
            failures.append(col)

    print()
    print("Meteorological columns (live source diverges from training source — "
          "expected; reporting magnitude):")
    for col in METEO:
        t = float(train_row[col])
        i = float(feats[col])
        delta = i - t
        print(f"  {col:<20} {t:>25.6f} {i:>15.6f} {delta:>10.6f}")
    print("=" * 78)

    if failures:
        log.error(f"Parity check FAILED for: {failures}")
        if "dist_to_highway_m" in failures:
            log.error("  → highway cache mismatch (data/.cache/dfw_highways.pkl). "
                      "Inference and training are reading different OSM snapshots.")
        if any(f in failures for f in [
            "local_hour_of_day", "day_of_week", "is_weekend",
            "is_am_rush", "is_pm_rush", "traffic_index",
        ]):
            log.error("  → timestamp-derived feature drift (likely timezone). "
                      "Training uses America/Chicago — verify ml/predictor.py "
                      "tz_convert matches.")
        sys.exit(1)

    log.info("Parity check PASSED for all structural features")


if __name__ == "__main__":
    main()
