# data/ingestion/history.py — Accumulate live dashboard snapshots
#
# Each call to save_snapshot() appends one row per sensor to
# data/dashboard_snapshots.csv. These are live-pipeline snapshots taken while
# the Streamlit app or scripts/collector.py is running.
#
# RF training data is NOT collected here. The canonical training set is
# ml/data/history.csv, built by ml/training/collect_training_data.py from
# PurpleAir's historical API. Live snapshots and the training set live in
# separate files on purpose, so the training script can overwrite history.csv
# without corrupting the dashboard's accumulated state.
#
# Schema policy: the row schema is defined in exactly one place,
# `_build_snapshot_record`. The `COLUMNS` constant is *derived* from a
# placeholder call to that function so it cannot drift. `save_snapshot` then
# asserts that the in-memory DataFrame's columns match `COLUMNS`, so any
# accidental schema change is loud, not silent.

import os
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from filelock import FileLock

logger = logging.getLogger(__name__)

# CSV lives at the data/ directory root (one level up from this file).
# Training data (history.csv) is owned by ml/training/collect_training_data.py;
# this writer owns dashboard_snapshots.csv so the two never collide.
HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard_snapshots.csv")

# Time zone used for `local_hour_of_day` and `day_of_week`. Matches the
# convention in ml/training/collect_training_data.py:add_traffic_features
# (`df["timestamp"].dt.tz_convert("America/Chicago")`), so live snapshots
# and the historical training set share the same hour/day values for a
# given wall-clock instant in Dallas.
DALLAS_TZ = ZoneInfo("America/Chicago")


def _build_snapshot_record(
    timestamp: datetime,
    sensor_row: pd.Series,
    wind_speed: float,
    wind_deg: float,
) -> dict:
    """
    Build one snapshot record. Single source of truth for the row schema.

    Keep the key order stable — `COLUMNS` is derived from this function via
    `_empty_record_template()` and pandas preserves dict insertion order
    when constructing a DataFrame from a list of dicts.

    `local_hour_of_day` and `day_of_week` are computed in Dallas local time
    (America/Chicago), matching the training pipeline. Storing UTC values
    under a name that promises local time would silently mis-align live
    snapshots against the training set.
    """
    local_ts = timestamp.astimezone(DALLAS_TZ)
    return {
        "timestamp":          timestamp.isoformat(),
        "sensor_id":          sensor_row["sensor_id"],
        "lat":                sensor_row["lat"],
        "lon":                sensor_row["lon"],
        # pm25 is the EPA-corrected (PurpleAir) or reference-grade (OpenAQ)
        # reading; build_features() does not modify it further.
        "pm25":               sensor_row["pm25"],
        "pm25_raw":           sensor_row.get("pm25_raw", float("nan")),
        "epa_corrected":      sensor_row.get("epa_corrected", 0),
        "source":             sensor_row.get("source", "unknown"),
        "wind_speed":         wind_speed,
        "wind_deg":           wind_deg,
        # Feature columns computed by build_features() — read directly from row.
        "nearest_congestion": sensor_row.get("nearest_congestion", float("nan")),
        "distance_to_road_m": sensor_row.get("distance_to_road_m", float("nan")),
        "traffic_factor":     sensor_row.get("traffic_factor", float("nan")),
        "wind_term":          sensor_row.get("wind_term", float("nan")),
        "direction_factor":   sensor_row.get("direction_factor", float("nan")),
        "dispersal":          sensor_row.get("dispersal", float("nan")),
        "local_hour_of_day":  local_ts.hour,
        "day_of_week":        local_ts.weekday(),
    }


def _empty_record_template() -> dict:
    """Build a placeholder record so the column schema can be introspected."""
    placeholder_sensor = pd.Series({
        "sensor_id": 0, "lat": 0.0, "lon": 0.0, "pm25": 0.0,
    })
    return _build_snapshot_record(
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        sensor_row=placeholder_sensor,
        wind_speed=0.0,
        wind_deg=float("nan"),
    )


# Single derived constant — adding a key to `_build_snapshot_record` updates
# this automatically. Renaming a key here without touching the record builder
# triggers the assertion in `save_snapshot` so the drift is surfaced loudly.
COLUMNS = list(_empty_record_template().keys())


def save_snapshot(
    sensor_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    wind: dict,
    timestamp: datetime | None = None,
) -> None:
    """
    Append one snapshot record per sensor to data/dashboard_snapshots.csv.

    sensor_df is the output of build_features(): pm25 is the EPA-corrected
    (PurpleAir) or reference-grade (OpenAQ) reading, pm25_raw holds the
    uncorrected PurpleAir value (NaN for OpenAQ), and the feature columns
    (traffic_factor, wind_term, direction_factor, etc.) are already computed
    and stored as separate columns. We read them directly from the row rather
    than recomputing them here.

    Args:
        sensor_df:  DataFrame from build_features() with pm25 (corrected) and feature columns.
        traffic_df: DataFrame with [lat, lon, congestion] (kept for signature compat).
        wind:       Dict with wind_speed and wind_deg.
        timestamp:  Snapshot time (must be tz-aware). Defaults to datetime.now(timezone.utc).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Defensive: callers must supply a tz-aware timestamp. The downstream
    # astimezone(DALLAS_TZ) would silently treat a naive datetime as local
    # time on the host machine, producing wrong `local_hour_of_day` values.
    if timestamp.tzinfo is None:
        raise ValueError("save_snapshot timestamp must be timezone-aware")

    wind_speed = float(wind.get("wind_speed") or 0.0)
    # Store NaN when wind_deg is missing rather than coercing to 0.0 (due North),
    # which would be a misleading value in the snapshot dataset.
    raw_deg  = wind.get("wind_deg")
    wind_deg = float(raw_deg) if raw_deg is not None else float("nan")

    records = [
        _build_snapshot_record(timestamp, row, wind_speed, wind_deg)
        for _, row in sensor_df.iterrows()
    ]

    if not records:
        # No sensors to write. Don't even open the file; a stray header-only
        # write would just produce a one-line file with no rows.
        return

    new_rows = pd.DataFrame(records)

    # Sanity: catch silent schema drift between _build_snapshot_record and
    # the COLUMNS constant. If this fires, the snapshot CSV header would
    # otherwise no longer match what readers (load_history) expect.
    assert list(new_rows.columns) == COLUMNS, (
        f"Snapshot record schema drifted from COLUMNS. "
        f"Got {list(new_rows.columns)}, expected {COLUMNS}."
    )

    # Hold an exclusive lock on a sidecar `.lock` file for the duration of
    # the write. Cross-platform via the `filelock` package (POSIX flock on
    # Linux/macOS, LockFileEx on Windows). Two concurrent Streamlit sessions
    # or a Streamlit + scripts/collector.py both calling save_snapshot()
    # will serialize through this lock instead of interleaving writes.
    #
    # The `file_exists` check lives inside the lock: without it, two callers
    # racing on a fresh file could each see no header and both write one,
    # producing a CSV with a duplicate header row.
    lock = FileLock(HISTORY_PATH + ".lock")
    with lock:
        file_exists = os.path.isfile(HISTORY_PATH)
        with open(HISTORY_PATH, "a", newline="") as f:
            new_rows.to_csv(f, index=False, header=not file_exists)


def load_history() -> pd.DataFrame:
    """
    Read data/dashboard_snapshots.csv and return it as a DataFrame.
    Returns an empty DataFrame with the correct columns if the file doesn't exist.
    """
    if not os.path.isfile(HISTORY_PATH):
        return pd.DataFrame(columns=COLUMNS)

    # on_bad_lines='skip' prevents a crash if the file has rows from an older
    # schema version with a different column count (e.g. after adding a new column).
    df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"], on_bad_lines="skip")
    return df


def get_history_stats() -> dict:
    """
    Return a summary of the collected dashboard snapshots.

    Returns:
        total_records:  total row count
        unique_sensors: number of distinct sensor IDs seen
        date_range:     (earliest, latest) timestamp as ISO strings, or (None, None)
        hours_covered:  number of distinct hours in the dataset
    """
    df = load_history()

    if df.empty:
        return {
            "total_records":  0,
            "unique_sensors": 0,
            "date_range":     (None, None),
            "hours_covered":  0,
        }

    return {
        "total_records":  len(df),
        "unique_sensors": df["sensor_id"].nunique(),
        "date_range":     (
            str(df["timestamp"].min()),
            str(df["timestamp"].max()),
        ),
        "hours_covered":  df["timestamp"].dt.floor("h").nunique(),
    }
