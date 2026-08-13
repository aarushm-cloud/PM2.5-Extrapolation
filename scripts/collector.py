"""
scripts/collector.py — Headless live snapshot collector.

Runs independently of the Streamlit app. Fetches all live data sources on a
configurable interval and appends snapshots to data/dashboard_snapshots.csv.
This is NOT the RF training-data pipeline; that is handled by
ml/training/collect_training_data.py from PurpleAir's historical API.

Usage:
    python scripts/collector.py               # every 30 minutes
    python scripts/collector.py --interval 15 # every 15 minutes
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from the project root regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.ingestion.purpleair import fetch_sensors
from data.ingestion.openaq import fetch_openaq
from data.ingestion.traffic import fetch_traffic
from data.ingestion.weather import fetch_wind
from data.ingestion.history import save_snapshot, get_history_stats
from engine.features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("collector")


def _fetch_sensors() -> pd.DataFrame:
    """Try PurpleAir then OpenAQ; combine whatever succeeds."""
    frames = []

    try:
        pa_kept, _pa_dropped = fetch_sensors()
        if not pa_kept.empty:
            frames.append(pa_kept)
            log.info("PurpleAir: %d sensors", len(pa_kept))
    except Exception as e:
        log.warning("PurpleAir fetch failed — skipping. (%s)", e)

    try:
        oaq = fetch_openaq()
        if not oaq.empty:
            frames.append(oaq)
            log.info("OpenAQ: %d sensors", len(oaq))
    except Exception as e:
        log.warning("OpenAQ fetch failed — skipping. (%s)", e)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _fetch_wind() -> dict:
    try:
        wind = fetch_wind()
        log.info("Wind: %.1f m/s @ %.0f°", wind.get("wind_speed", 0), wind.get("wind_deg", 0))
        return wind
    except Exception as e:
        log.warning("Wind fetch failed — using calm conditions. (%s)", e)
        return {"wind_speed": 0.0, "wind_deg": 0.0}


def _fetch_traffic() -> pd.DataFrame:
    try:
        traffic = fetch_traffic()
        log.info("Traffic: %d points", len(traffic))
        return traffic
    except Exception as e:
        log.warning("Traffic fetch failed — skipping congestion adjustment. (%s)", e)
        return pd.DataFrame()


def run_cycle() -> int:
    """
    Execute one collection cycle.
    Returns the number of sensor records saved, or 0 on failure.
    """
    sensor_df = _fetch_sensors()
    if sensor_df.empty:
        log.warning("All sensor sources failed — skipping this cycle.")
        return 0

    wind       = _fetch_wind()
    traffic_df = _fetch_traffic()

    df = build_features(sensor_df, traffic_df, wind)

    now = datetime.now(timezone.utc)
    save_snapshot(df, traffic_df, wind, timestamp=now)

    stats = get_history_stats()
    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"saved {len(df)} sensors — "
        f"{stats['total_records']:,} total records in history"
    )

    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="DFW air quality headless collector")
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        metavar="MINUTES",
        help="Collection interval in minutes (default: 30)",
    )
    args = parser.parse_args()
    interval_seconds = args.interval * 60

    print(f"Collector started — running every {args.interval} minute(s). Ctrl+C to stop.")

    total_saved = 0
    try:
        while True:
            try:
                total_saved += run_cycle()
            except Exception as e:
                log.error("Unexpected error in collection cycle: %s", e)

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print(f"\nCollector stopped. {total_saved:,} sensor records saved this session.")
        sys.exit(0)


if __name__ == "__main__":
    main()
