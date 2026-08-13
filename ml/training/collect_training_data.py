"""
DEPRECATED: the RF model was evaluated and shelved.
The dashboard runs on IDW + adjust_grid; this module is dormant.
See ml/docs/rf_model_result.md for the full negative-result writeup.

collect_training_data.py

Builds ml/data/history.csv — the training dataset for the Random Forest
PM2.5 interpolation model for the DFW Air Quality Dashboard.

Data sources (all free, all auditable):
    1. PurpleAir API      — hourly PM2.5 per sensor (with A/B channel validation)
    2. Meteostat          — hourly wind speed + direction at DFW airport
                            (Meteostat wraps NOAA's Integrated Surface Database,
                             the same archive NWS uses for official observations)
    3. EPA correction     — standard PurpleAir-to-reference-grade formula applied
                            per EPA's publicly documented methodology
    4. Temporal features  — traffic proxies derived from timestamps
    5. Spatial features   — dist_to_highway_m, distance from each sensor to the
                            nearest major DFW highway (OpenStreetMap via OSMnx)

Quality controls:
    - PurpleAir A/B channel disagreement flagging. Rows are dropped when
      |A - B| / mean(A, B) exceeds AB_DISAGREEMENT_THRESHOLD (a fraction in
      [0, 1], e.g. 0.50 = 50% relative difference).
    - EPA PM2.5 humidity correction applied to all readings
    - Per-sensor checkpointing so the script can resume after a crash
    - Full audit log written to ml/data/collection_log.txt
    - Data quality report written to ml/data/quality_report.json

Usage:
    python collect_training_data.py                # full 6-month collection
    python collect_training_data.py --days 30      # shorter date range
    python collect_training_data.py --resume       # resume from checkpoints

Required .env variables:
    PURPLEAIR_API_KEY   — your existing PurpleAir read key

Dependencies not already in requirements.txt:
    meteostat  (pip install meteostat)
    pyarrow    (pip install pyarrow) — for parquet checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Script lives in ml/training/ but imports project-root modules. Put the project
# root on sys.path so `from config import BBOX` works regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import BBOX, PURPLEAIR_BASE_URL, DFW_AIRPORT_LAT_LON  # noqa: E402

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — all magic numbers live here so auditors can review them
# ─────────────────────────────────────────────────────────────────────────────

PURPLEAIR_API_KEY: Optional[str] = os.getenv("PURPLEAIR_API_KEY")

# DFW International Airport coordinates (for Meteostat nearest-station lookup)
DFW_AIRPORT_LAT, DFW_AIRPORT_LON = DFW_AIRPORT_LAT_LON

# Data quality thresholds
AB_DISAGREEMENT_THRESHOLD  = 0.50    # drop rows where channels A and B differ >50%
SENSOR_AB_FAILURE_RATE_MAX = 0.50    # drop whole sensor if >50% of its rows fail A/B
PM25_MAX_VALID             = 500.0   # PM2.5 above this is almost always sensor error
PM25_MIN_VALID             = 0.0     # negative readings are sensor error

# Diagnostic thresholds — only used to fire warnings, never to drop data
UNCORRECTED_WARN_THRESHOLD_PCT = 3.0  # warn if >3% of rows lack humidity for EPA correction
WIND_FALLBACK_WARN_THRESHOLD_PCT = 3.0  # warn if >3% of rows fell back to climate-normal wind

# Network behavior
REQUEST_TIMEOUT_SEC = 30
REQUEST_PAUSE_SEC   = 0.5
MAX_RETRIES         = 3
RETRY_BACKOFF_SEC   = 2.0

# File paths
DATA_DIR          = Path("ml/data")
OUTPUT_CSV        = DATA_DIR / "history.csv"
CHECKPOINT_DIR    = DATA_DIR / ".checkpoints"
LOG_FILE          = DATA_DIR / "collection_log.txt"
QUALITY_REPORT    = DATA_DIR / "quality_report.json"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — written to both console and file for audit trails
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure logging to write to both stdout and a persistent log file."""
    DATA_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("dfw_collector")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


log = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY REPORT — running tally, saved to JSON at end
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Tracks data quality metrics so the city can audit the dataset."""
    collection_started: str = ""
    collection_finished: str = ""
    date_range_start: str = ""
    date_range_end: str = ""

    sensors_discovered: int = 0
    sensors_with_data: int = 0
    sensors_dropped_no_data: int = 0
    sensors_dropped_ab_failure: int = 0
    sensors_dropped_ab_failure_ids: list = field(default_factory=list)
    # Per-sensor breakdown for any sensor with >30% row-level A/B failure
    # rate, whether or not it survived the sensor-level cutoff. Lets us
    # decide later whether to tighten SENSOR_AB_FAILURE_RATE_MAX.
    ab_failure_borderline: list = field(default_factory=list)

    raw_purpleair_rows: int = 0
    rows_dropped_ab_nan: int = 0
    rows_dropped_bad_sensors: int = 0
    rows_dropped_ab_threshold: int = 0
    # DEPRECATED: sum of the three rows_dropped_ab_* fields above. Kept for
    # backward compat with consumers reading older quality_report.json files.
    rows_dropped_ab_disagreement: int = 0
    rows_dropped_out_of_range: int = 0
    final_row_count: int = 0

    epa_correction_applied: bool = False
    rows_uncorrected_humidity_missing: int = 0
    wind_data_source: str = ""
    wind_hours_available: int = 0
    wind_hours_gap_filled: int = 0
    wind_hours_climate_fallback: int = 0

    # Static spatial feature range (for end-of-run summary)
    dist_to_highway_min_m: float = 0.0
    dist_to_highway_max_m: float = 0.0

    warnings: list = field(default_factory=list)

    def save(self) -> None:
        """Write report as JSON for environmental department review."""
        QUALITY_REPORT.write_text(json.dumps(asdict(self), indent=2))
        log.info(f"Quality report saved to {QUALITY_REPORT}")


report = QualityReport()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS — retry with exponential backoff, rate limit handling
# ─────────────────────────────────────────────────────────────────────────────

def http_get_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> Optional[requests.Response]:
    """GET with exponential backoff on 5xx and 429 responses. Returns None if all retries fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SEC)

            if resp.status_code == 429:
                wait = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
                log.warning(f"Rate limited. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}")
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                wait = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
                log.warning(f"Server error {resp.status_code}. Retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue

            return resp

        except (requests.Timeout, requests.ConnectionError) as e:
            wait = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
            log.warning(f"Network error: {e}. Retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)

    log.error(f"All {MAX_RETRIES} retries failed for {url}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DISCOVER DFW PURPLEAIR SENSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_dfw_sensors() -> list[dict]:
    """Fetch all PurpleAir sensors within the DFW bounding box."""
    log.info("Step 1: Discovering DFW PurpleAir sensors")

    resp = http_get_with_retry(
        f"{PURPLEAIR_BASE_URL}/sensors",
        params={
            "fields": "sensor_index,name,latitude,longitude,last_seen",
            "location_type": 0,   # outdoor only — match data/ingestion/purpleair.py
            "nwlng": BBOX["west"],
            "nwlat": BBOX["north"],
            "selng": BBOX["east"],
            "selat": BBOX["south"],
        },
        headers={"X-API-Key": PURPLEAIR_API_KEY},
    )

    if resp is None or not resp.ok:
        raise RuntimeError("Failed to fetch PurpleAir sensor list.")

    data = resp.json()
    fields = data["fields"]
    rows = data["data"]

    col = {name: fields.index(name) for name in fields}

    sensors = [
        {
            "sensor_index": row[col["sensor_index"]],
            "name": row[col["name"]],
            "latitude": row[col["latitude"]],
            "longitude": row[col["longitude"]],
        }
        for row in rows
        if row[col["latitude"]] is not None and row[col["longitude"]] is not None
    ]

    report.sensors_discovered = len(sensors)
    log.info(f"  Discovered {len(sensors)} sensors in DFW bounding box")
    return sensors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PULL PURPLEAIR HISTORY (WITH CHECKPOINTING)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sensor_history(
    sensor_index: int,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Pull hourly PM2.5 for one sensor, including A and B channels (for later
    cross-validation) and humidity (for EPA correction). Chunked into 2-week
    windows — PurpleAir's API limits each request to 14 days.
    """
    url = f"{PURPLEAIR_BASE_URL}/sensors/{sensor_index}/history"
    headers = {"X-API-Key": PURPLEAIR_API_KEY}

    rows_collected: list[dict] = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=14), end_dt)

        resp = http_get_with_retry(
            url,
            params={
                "start_timestamp": int(chunk_start.timestamp()),
                "end_timestamp": int(chunk_end.timestamp()),
                "average": 60,
                "fields": "pm2.5_cf_1_a,pm2.5_cf_1_b,humidity",  # CF=1 required by Barkjohn 2021 EPA formula
            },
            headers=headers,
        )

        if resp is None:
            log.warning(f"  Skipping chunk {chunk_start.date()} for sensor {sensor_index}")
            chunk_start = chunk_end
            continue

        if resp.status_code == 404:
            chunk_start = chunk_end
            continue

        if not resp.ok:
            log.warning(f"  Sensor {sensor_index} chunk {chunk_start.date()}: HTTP {resp.status_code}")
            chunk_start = chunk_end
            continue

        data = resp.json()
        fields = data.get("fields", [])
        data_rows = data.get("data", [])

        if not data_rows or "time_stamp" not in fields:
            chunk_start = chunk_end
            continue

        col = {name: fields.index(name) for name in fields}
        for row in data_rows:
            rows_collected.append({
                "timestamp": pd.to_datetime(row[col["time_stamp"]], unit="s", utc=True),
                "sensor_index": sensor_index,
                "pm25_a": row[col["pm2.5_cf_1_a"]] if "pm2.5_cf_1_a" in col else None,
                "pm25_b": row[col["pm2.5_cf_1_b"]] if "pm2.5_cf_1_b" in col else None,
                "humidity": row[col["humidity"]] if "humidity" in col else None,
            })

        chunk_start = chunk_end
        time.sleep(REQUEST_PAUSE_SEC)

    return pd.DataFrame(rows_collected) if rows_collected else pd.DataFrame()


def collect_all_purpleair(
    sensors: list[dict],
    start_dt: datetime,
    end_dt: datetime,
    resume: bool,
) -> pd.DataFrame:
    """
    Pull history for every sensor with per-sensor checkpointing.
    If --resume is passed, skip sensors whose checkpoint already exists.

    Also computes the static distance-to-highway feature once per sensor and
    attaches it as a column (constant per sensor, since it only depends on
    location).
    """
    log.info(f"Step 2: Collecting PurpleAir history ({start_dt.date()} → {end_dt.date()})")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Imported lazily so the script can still run if OSMnx/geopy are missing
    # at startup — the failure surfaces here, scoped to this feature.
    from data.spatial.spatial_features import compute_distance_to_highway

    log.info(f"Step 1.5: Computing highway distances for {len(sensors)} sensors")
    distances: list[float] = []
    for sensor in sensors:
        d = compute_distance_to_highway(sensor["latitude"], sensor["longitude"])
        sensor["dist_to_highway_m"] = d
        distances.append(d)
    if distances:
        log.info(f"  Distance to nearest highway — "
                 f"min: {min(distances):.0f} m, "
                 f"median: {sorted(distances)[len(distances)//2]:.0f} m, "
                 f"max: {max(distances):.0f} m")
        report.dist_to_highway_min_m = float(min(distances))
        report.dist_to_highway_max_m = float(max(distances))

    dfs = []
    for i, sensor in enumerate(sensors, 1):
        sid = sensor["sensor_index"]
        checkpoint = CHECKPOINT_DIR / f"sensor_{sid}.parquet"

        if resume and checkpoint.exists():
            df = pd.read_parquet(checkpoint)
            # Re-attach in case the checkpoint pre-dates this column
            df["dist_to_highway_m"] = sensor["dist_to_highway_m"]
            log.info(f"  [{i}/{len(sensors)}] Sensor {sid}: loaded {len(df)} rows from checkpoint")
            dfs.append(df)
            continue

        log.info(f"  [{i}/{len(sensors)}] Sensor {sid} ({sensor['name']})")
        df = fetch_sensor_history(sid, start_dt, end_dt)

        if df.empty:
            report.sensors_dropped_no_data += 1
            log.info(f"    No data returned")
            continue

        df["latitude"] = sensor["latitude"]
        df["longitude"] = sensor["longitude"]
        df["dist_to_highway_m"] = sensor["dist_to_highway_m"]

        # Save checkpoint so we can resume after interruption
        df.to_parquet(checkpoint, index=False)
        dfs.append(df)
        log.info(f"    Collected {len(df)} rows (checkpoint saved)")

    if not dfs:
        raise RuntimeError("No PurpleAir data collected. Check API key and sensor list.")

    combined = pd.concat(dfs, ignore_index=True)
    report.sensors_with_data = len(dfs)
    report.raw_purpleair_rows = len(combined)
    log.info(f"  Collected {len(combined):,} raw rows across {len(dfs)} sensors")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DATA QUALITY CONTROLS ON PURPLEAIR DATA
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_LEVEL_FILTER_MIN_ROWS = 24  # one full day — below this, only row-level applies
AB_BORDERLINE_LOG_THRESHOLD  = 0.30  # log per-sensor breakdown for any sensor above this rate


def validate_ab_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    PurpleAir sensors contain two laser counters (channels A and B). Per
    EPA guidance, if the channels disagree significantly, the reading is
    not trustworthy — usually indicates a failing counter, bug ingress, or
    local contamination.

    Two-stage filter:
      1. Sensor-level: drop entire sensors where >SENSOR_AB_FAILURE_RATE_MAX
         of rows fail A/B at the threshold. Only applied to sensors with at
         least SENSOR_LEVEL_FILTER_MIN_ROWS observations — recently-online
         sensors with very small samples shouldn't get killed by 2-of-3
         flukes. These are otherwise almost always broken hardware (one
         dead laser).
      2. Row-level: among surviving sensors, drop rows where
         |A - B| / mean(A, B) > AB_DISAGREEMENT_THRESHOLD.

    The surviving reading is the average of the two channels.

    Drop counts are tracked in three separate fields on QualityReport:
      - rows_dropped_ab_nan         (missing A or B channel reading)
      - rows_dropped_bad_sensors    (entire sensor failed sensor-level filter)
      - rows_dropped_ab_threshold   (row-level threshold failure)
    """
    initial = len(df)
    valid = df.dropna(subset=["pm25_a", "pm25_b"]).copy()
    rows_dropped_nan = initial - len(valid)
    report.rows_dropped_ab_nan = rows_dropped_nan

    avg = (valid["pm25_a"] + valid["pm25_b"]) / 2
    diff = (valid["pm25_a"] - valid["pm25_b"]).abs()
    # Guard against divide-by-zero: rows with avg=0 are treated as in-agreement
    disagreement = (diff / avg.replace(0, pd.NA)).fillna(0.0)
    valid["_ab_fail"] = disagreement > AB_DISAGREEMENT_THRESHOLD

    # Sensor-level filter only applies to sensors with enough rows to make
    # the failure-rate statistic meaningful.
    sensor_row_counts = valid.groupby("sensor_index").size()
    eligible_sensors = sensor_row_counts[
        sensor_row_counts >= SENSOR_LEVEL_FILTER_MIN_ROWS
    ].index
    eligible_subset = valid[valid["sensor_index"].isin(eligible_sensors)]
    per_sensor_failure_rate = eligible_subset.groupby("sensor_index")["_ab_fail"].mean()
    bad_sensors = per_sensor_failure_rate[
        per_sensor_failure_rate > SENSOR_AB_FAILURE_RATE_MAX
    ].index.tolist()

    log.info(f"  Sensor-level filter applied to {len(eligible_sensors)} of "
             f"{sensor_row_counts.size} sensors (others below "
             f"{SENSOR_LEVEL_FILTER_MIN_ROWS}-row minimum)")
    if bad_sensors:
        log.info(f"  Dropping {len(bad_sensors)} sensors with >"
                 f"{SENSOR_AB_FAILURE_RATE_MAX*100:.0f}% A/B failure rate: "
                 f"{bad_sensors}")
    report.sensors_dropped_ab_failure = len(bad_sensors)
    report.sensors_dropped_ab_failure_ids = [int(s) for s in bad_sensors]

    # Per-sensor borderline diagnostic: any sensor with >30% row-level fail
    # rate, regardless of whether it crossed the sensor-level cutoff. Counts
    # use the row-level fail check so they reflect the same denominator as
    # the failure-rate calculation.
    bad_sensor_set = set(bad_sensors)
    rows_per_sensor = valid.groupby("sensor_index").size()
    fails_per_sensor = valid.groupby("sensor_index")["_ab_fail"].sum()
    rate_per_sensor = (fails_per_sensor / rows_per_sensor).fillna(0.0)
    borderline = (
        rate_per_sensor[rate_per_sensor > AB_BORDERLINE_LOG_THRESHOLD]
        .sort_values(ascending=False)
    )
    if len(borderline):
        log.info(f"  Borderline A/B failure rates "
                 f"(>{AB_BORDERLINE_LOG_THRESHOLD*100:.0f}%, sorted desc):")
        for sid, rate in borderline.items():
            failed = int(fails_per_sensor.loc[sid])
            total = int(rows_per_sensor.loc[sid])
            outcome = "DROPPED" if sid in bad_sensor_set else "SURVIVED"
            log.info(f"    sensor {sid}: {rate*100:5.1f}% "
                     f"({failed} of {total} rows)  [{outcome}]")
            report.ab_failure_borderline.append({
                "sensor_id": int(sid),
                "failure_rate": round(float(rate), 4),
                "rows_failed": failed,
                "rows_total": total,
                "outcome": outcome.lower(),
            })

    bad_sensor_mask = valid["sensor_index"].isin(bad_sensors)
    rows_dropped_bad_sensors = int(bad_sensor_mask.sum())
    report.rows_dropped_bad_sensors = rows_dropped_bad_sensors

    surviving = valid[~bad_sensor_mask]
    kept = surviving[~surviving["_ab_fail"]].copy()
    rows_dropped_threshold = len(surviving) - len(kept)
    report.rows_dropped_ab_threshold = rows_dropped_threshold

    kept["pm25_raw"] = (kept["pm25_a"] + kept["pm25_b"]) / 2

    total_dropped = initial - len(kept)
    report.rows_dropped_ab_disagreement = total_dropped  # deprecated sum
    pct = total_dropped / max(initial, 1) * 100
    log.info(f"  A/B validation: dropped {total_dropped:,} rows ({pct:.1f}%) "
             f"[{rows_dropped_nan:,} NaN + {rows_dropped_bad_sensors:,} bad-sensor "
             f"+ {rows_dropped_threshold:,} row-level >"
             f"{AB_DISAGREEMENT_THRESHOLD*100:.0f}%]")
    return kept.drop(columns=["pm25_a", "pm25_b", "_ab_fail"])


def apply_epa_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the EPA's PM2.5 correction formula for PurpleAir sensors and emit
    the training-pipeline-specific progress logs and report counters.

    The correction math itself lives in `data/corrections.py` so the live
    pipeline (`data/ingestion/purpleair.py`) and this training pipeline are
    guaranteed to apply byte-identical formulas. Reporting is layered on
    top here because the live pipeline does not have a `report` object.

    Rows with missing humidity fall back to the raw reading and are flagged
    (epa_corrected = 0) so downstream analysis can down-weight them.
    """
    log.info("  Applying EPA PM2.5 correction formula")

    from data.corrections import apply_epa_correction as _apply
    corrected = _apply(df)

    has_rh = corrected["epa_corrected"] == 1
    n_corrected = int(has_rh.sum())
    n_uncorrected = int((~has_rh).sum())
    n_total = len(corrected)

    report.epa_correction_applied = True
    report.rows_uncorrected_humidity_missing = n_uncorrected
    pct_corrected = n_corrected / max(n_total, 1) * 100
    pct_missing = n_uncorrected / max(n_total, 1) * 100
    log.info(f"    EPA correction: {n_corrected:,} of {n_total:,} rows corrected "
             f"({pct_corrected:.1f}%)")
    if pct_missing > UNCORRECTED_WARN_THRESHOLD_PCT:
        log.warning(f"    {n_uncorrected:,} rows ({pct_missing:.1f}%) lacked humidity "
                    f"and kept raw value — consider dropping these before training "
                    f"(they mix two distributions under one pm25 column).")
    return corrected


def filter_range(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where PM2.5 is outside physically plausible bounds."""
    initial = len(df)
    mask = (df["pm25"] >= PM25_MIN_VALID) & (df["pm25"] < PM25_MAX_VALID)
    filtered = df[mask].copy()
    dropped = initial - len(filtered)
    report.rows_dropped_out_of_range = dropped
    log.info(f"  Range filter ({PM25_MIN_VALID}–{PM25_MAX_VALID} µg/m³): "
             f"dropped {dropped:,} rows")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: PULL WIND DATA FROM METEOSTAT (NOAA ISD ARCHIVE)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wind_data(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch hourly wind speed and direction at DFW airport from Meteostat.

    Meteostat wraps NOAA's Integrated Surface Database (ISD), which is the
    same archive the National Weather Service uses for official hourly
    observations. It's free and requires no API key when used via the
    Meteostat Python library.
    """
    log.info("Step 4: Fetching wind data from Meteostat (NOAA ISD archive)")

    try:
        # Imported here so script doesn't fail at startup if not installed
        from meteostat import Point, Hourly
    except ImportError:
        msg = "meteostat not installed. Install with: pip install meteostat"
        log.warning(f"  {msg} — falling back to DFW climate averages")
        report.warnings.append(msg)
        return pd.DataFrame()

    try:
        dfw = Point(DFW_AIRPORT_LAT, DFW_AIRPORT_LON, alt=185)
        data = Hourly(dfw, start_dt.replace(tzinfo=None), end_dt.replace(tzinfo=None))
        wx = data.fetch()

        if wx.empty:
            log.warning("  Meteostat returned no data for date range")
            return pd.DataFrame()

        wx = wx.reset_index().rename(columns={"time": "timestamp"})
        wx["timestamp"] = pd.to_datetime(wx["timestamp"]).dt.tz_localize("UTC")
        # Meteostat returns wind speed in km/h — convert to m/s to match project convention
        wx["wind_speed_ms"] = wx["wspd"] / 3.6
        wx["wind_dir_deg"] = wx["wdir"]

        result = wx[["timestamp", "wind_speed_ms", "wind_dir_deg"]].dropna(
            subset=["wind_speed_ms"]
        )
        report.wind_data_source = "Meteostat (NOAA ISD)"
        report.wind_hours_available = len(result)
        log.info(f"  Retrieved {len(result):,} hourly wind observations")
        return result

    except Exception as e:
        log.error(f"  Meteostat fetch failed: {e}")
        report.warnings.append(f"Wind data fetch failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAFFIC PROXY FEATURES FROM TIMESTAMPS
# ─────────────────────────────────────────────────────────────────────────────

def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate temporal features that correlate with traffic density.
    TomTom historical data requires a paid enterprise license, so we use a
    time-based proxy. TxDOT publishes DFW congestion data showing highly
    consistent daily and weekly patterns that these features capture well.

    Vectorized (no .apply) for speed on multi-million-row datasets.
    """
    log.info("Step 5: Engineering temporal traffic features")

    # Rush hour and weekday/weekend patterns are inherently *local* concepts —
    # 8 AM Central is rush hour in Dallas regardless of what UTC says. The
    # column is named local_hour_of_day so this isn't ambiguous downstream.
    assert df["timestamp"].dt.tz is not None, "timestamp must be tz-aware"
    local = df["timestamp"].dt.tz_convert("America/Chicago")
    hour = local.dt.hour
    dow = local.dt.dayofweek

    df = df.copy()
    df["local_hour_of_day"] = hour
    df["day_of_week"] = dow
    df["is_weekend"] = (dow >= 5).astype(int)

    weekday = dow < 5
    df["is_am_rush"] = (weekday & (hour >= 7) & (hour <= 9)).astype(int)
    df["is_pm_rush"] = (weekday & (hour >= 16) & (hour <= 19)).astype(int)

    # Vectorized traffic index: defaults overridden in order of specificity
    traffic = pd.Series(0.4, index=df.index)
    traffic[df["is_weekend"] == 1] = 0.3
    traffic[(hour >= 10) & (hour <= 15) & weekday] = 0.5
    traffic[((hour >= 20) | (hour <= 5))] = 0.1
    traffic[df["is_am_rush"] == 1] = 1.0
    traffic[df["is_pm_rush"] == 1] = 1.0
    df["traffic_index"] = traffic

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: MERGE AND WRITE FINAL CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_final_dataset(pa_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
    """Join cleaned PurpleAir data with wind data and write history.csv."""
    log.info("Step 6: Merging datasets and writing final CSV")

    pa_df = pa_df.copy()
    assert pa_df["timestamp"].dt.tz is not None, "timestamp must be tz-aware"
    pa_df["timestamp"] = pa_df["timestamp"].dt.floor("h")

    if not wind_df.empty:
        merged = pa_df.merge(wind_df, on="timestamp", how="left")
        missing_wind = int(merged["wind_speed_ms"].isna().sum())
        report.wind_hours_gap_filled = missing_wind

        # Gap-fill with forward then backward fill, capped at 2 hours either
        # direction. Wind at a single metro is temporally autocorrelated for a
        # couple hours; beyond that, the next hour's true wind could be very
        # different and we'd rather hit the climate fallback than fabricate.
        merged["wind_speed_ms"] = merged["wind_speed_ms"].ffill(limit=2).bfill(limit=2)
        merged["wind_dir_deg"] = merged["wind_dir_deg"].ffill(limit=2).bfill(limit=2)

        # Last-resort fallback: DFW climate normals
        climate_mask = merged["wind_speed_ms"].isna()
        n_fallback = int(climate_mask.sum())
        report.wind_hours_climate_fallback = n_fallback
        if climate_mask.any():
            merged["wind_speed_ms"] = merged["wind_speed_ms"].fillna(4.5)
            merged["wind_dir_deg"] = merged["wind_dir_deg"].fillna(180.0)
    else:
        log.warning("  No wind data — using DFW climate normals for all rows")
        merged = pa_df.copy()
        merged["wind_speed_ms"] = 4.5
        merged["wind_dir_deg"] = 180.0
        n_fallback = len(merged)
        report.wind_hours_climate_fallback = n_fallback

    pct_fallback = n_fallback / max(len(merged), 1) * 100
    log.info(f"  Wind data: {n_fallback:,} hours used climate fallback "
             f"({pct_fallback:.1f}%)")
    if pct_fallback > WIND_FALLBACK_WARN_THRESHOLD_PCT:
        log.warning(f"  {n_fallback:,} rows ({pct_fallback:.1f}%) have synthetic wind "
                    f"(4.5 m/s, 180°). Model will learn a constant-wind artifact at "
                    f"those timestamps.")

    # Align column names to the live data/ingestion/history.py schema (the long-term
    # standard for training rows). Also tag every row with source="purpleair"
    # so concat with any future OpenAQ training set stays auditable.
    merged = merged.rename(columns={
        "sensor_index":  "sensor_id",
        "latitude":      "lat",
        "longitude":     "lon",
        "wind_speed_ms": "wind_speed",
        "wind_dir_deg":  "wind_deg",
    })
    merged["source"] = "purpleair"

    merged = merged.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    final_columns = [
        "timestamp", "sensor_id", "lat", "lon",
        "dist_to_highway_m", # static spatial feature (geodesic m to nearest highway)
        "pm25",              # EPA-corrected, model target
        "pm25_raw",          # uncorrected reading (kept for audit trail)
        "epa_corrected",     # 1 if EPA correction applied, 0 otherwise
        "source",
        "humidity",
        "wind_speed", "wind_deg",
        "local_hour_of_day", "day_of_week", "is_weekend",
        "is_am_rush", "is_pm_rush", "traffic_index",
    ]
    final_columns = [c for c in final_columns if c in merged.columns]
    result = merged[final_columns]

    DATA_DIR.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    report.final_row_count = len(result)

    log.info(f"  Wrote {len(result):,} rows to {OUTPUT_CSV}")
    log.info(f"    Date range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    log.info(f"    Unique sensors: {result['sensor_id'].nunique()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build history.csv for the DFW Air Quality Dashboard RF model."
    )
    parser.add_argument("--days", type=int, default=180,
                        help="How many days back to collect (default: 180)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from per-sensor checkpoints if they exist")
    parser.add_argument("--force", action="store_true",
                        help="Override the disabled guard. This script hits the "
                             "PurpleAir /history endpoint, the single biggest "
                             "consumer of API points — only pass this if you "
                             "intend to spend points rebuilding the training set.")
    return parser.parse_args()


def print_run_summary(final: pd.DataFrame, days: int) -> None:
    """Glanceable end-of-run digest. Surfaces the things to eyeball before
    using the dataset for training; no raw data, just counts and flags."""
    n_rows = len(final)
    n_sensors = final["sensor_id"].nunique()

    pct_uncorrected = (
        report.rows_uncorrected_humidity_missing / max(n_rows, 1) * 100
    )
    pct_fallback = report.wind_hours_climate_fallback / max(n_rows, 1) * 100
    pct_out_of_range = (
        report.rows_dropped_out_of_range / max(report.raw_purpleair_rows, 1) * 100
    )

    flag_uncorrected = " ⚠ consider dropping" if pct_uncorrected > UNCORRECTED_WARN_THRESHOLD_PCT else ""
    flag_fallback = " ⚠ synthetic wind" if pct_fallback > WIND_FALLBACK_WARN_THRESHOLD_PCT else ""

    survivors = [
        str(row["sensor_id"]) for row in report.ab_failure_borderline
        if row["outcome"] == "survived"
    ]
    survivors_str = ", ".join(survivors) if survivors else "(none)"

    log.info("=" * 70)
    log.info("Run summary — review before using this dataset for training")
    log.info("=" * 70)
    log.info(f"  Final dataset: {n_rows:,} rows, {n_sensors} sensors, {days} days")
    log.info("")
    log.info("  Sensor-level filter:")
    log.info(f"    Sensors discovered:    {report.sensors_discovered}")
    log.info(f"    Sensors with data:     {report.sensors_with_data}")
    log.info(f"    Sensors dropped (A/B): {report.sensors_dropped_ab_failure}")
    log.info(f"    Borderline survivors (>{AB_BORDERLINE_LOG_THRESHOLD*100:.0f}% A/B fail): {survivors_str}")
    log.info("")
    log.info("  Data quality flags:")
    log.info(f"    EPA-uncorrected rows:  {report.rows_uncorrected_humidity_missing:>6,} ({pct_uncorrected:.1f}%){flag_uncorrected}")
    log.info(f"    Wind climate fallback: {report.wind_hours_climate_fallback:>6,} ({pct_fallback:.1f}%){flag_fallback}")
    log.info(f"    Out-of-range PM2.5:    {report.rows_dropped_out_of_range:>6,} ({pct_out_of_range:.1f}%)")
    log.info("")
    log.info(f"  Wind data: {report.wind_data_source or 'none'}, "
             f"{report.wind_hours_available:,} hours available")
    log.info(f"  Highway feature: dist_to_highway_m, range "
             f"{report.dist_to_highway_min_m:.0f}m – {report.dist_to_highway_max_m:.0f}m")
    log.info("=" * 70)


def main() -> None:
    args = parse_args()

    # Disabled by default to protect the PurpleAir API key. The /history endpoint
    # this script calls is the biggest point drain, and the RF model is not wired up,
    # so an accidental run just burns points for nothing. Pass --force to run.
    if not args.force:
        log.error(
            "collect_training_data.py is disabled. It hits PurpleAir /history "
            "(the biggest API-point drain) and the RF model is not active. "
            "Re-run with --force only if you actually intend to rebuild the "
            "training set and spend points."
        )
        sys.exit(1)

    if not PURPLEAIR_API_KEY:
        log.error("PURPLEAIR_API_KEY not set in .env — cannot continue")
        sys.exit(1)

    end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=args.days)

    report.collection_started = datetime.now(timezone.utc).isoformat()
    report.date_range_start = start_dt.isoformat()
    report.date_range_end = end_dt.isoformat()

    log.info("=" * 70)
    log.info("DFW Air Quality Dashboard — Training Data Collection")
    log.info(f"Date range: {start_dt.date()} → {end_dt.date()} ({args.days} days)")
    log.info(f"Resume mode: {args.resume}")
    log.info("=" * 70)

    try:
        sensors = get_dfw_sensors()
        raw_pa = collect_all_purpleair(sensors, start_dt, end_dt, args.resume)

        # Quality pipeline: A/B validation → EPA correction → range filter
        clean_pa = validate_ab_channels(raw_pa)
        clean_pa = apply_epa_correction(clean_pa)
        clean_pa = filter_range(clean_pa)

        wind = fetch_wind_data(start_dt, end_dt)
        clean_pa = add_traffic_features(clean_pa)
        final = build_final_dataset(clean_pa, wind)

        report.collection_finished = datetime.now(timezone.utc).isoformat()
        report.save()

        log.info("=" * 70)
        log.info("Collection complete.")
        log.info(f"  history.csv:        {OUTPUT_CSV} ({len(final):,} rows)")
        log.info(f"  Quality report:     {QUALITY_REPORT}")
        log.info(f"  Audit log:          {LOG_FILE}")
        log.info("=" * 70)

        print_run_summary(final, args.days)

    except Exception as e:
        log.exception(f"Collection failed: {e}")
        report.warnings.append(f"Run failed: {e}")
        report.save()
        sys.exit(1)


if __name__ == "__main__":
    main()
