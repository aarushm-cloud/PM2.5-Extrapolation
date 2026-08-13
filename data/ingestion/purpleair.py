# data/ingestion/purpleair.py — PurpleAir sensor data ingestion
#
# Every PurpleAir reading returned by this module has been EPA-corrected at
# the source: pm25 is the corrected value, pm25_raw is the original laser-
# counter reading, and epa_corrected flags which rows had humidity available.
# Downstream code should treat pm25 as already corrected and NOT apply the
# formula again.
#
# The Barkjohn 2021 correction formula itself lives in data/corrections.py and
# is shared with the training pipeline (ml/training/collect_training_data.py)
# so both pipelines apply byte-identical math.

import logging
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from config import BBOX, PURPLEAIR_BASE_URL
from data.corrections import apply_epa_correction

load_dotenv()

logger = logging.getLogger(__name__)

# pm25_raw values above this are physically implausible for the DFW metro.
# EPA "hazardous" starts at 250.4 µg/m³; the 2018 Camp Fire peaked near 300
# in the Bay Area; Dallas has never seen anything close. Two PurpleAir
# sensors observed at ~5000 raw on 2026-05-12 — classic A/B-channel
# saturation, hardware fault. Filter on the raw value because the EPA
# correction can pull a 5000 down to ~2600, but the upstream fault signal
# is what matters.
PM25_RAW_SATURATION_THRESHOLD = 400.0


def get_api_key() -> str:
    """Load PurpleAir API key from .env."""
    key = os.getenv("PURPLEAIR_API_KEY")
    if not key or key == "your_key_here":
        raise ValueError("PURPLEAIR_API_KEY is not set in your .env file.")
    return key


# apply_epa_correction lives in data/corrections.py and is shared with the
# training pipeline. The earlier re-export from this module via __all__ was
# never picked up by any caller (verified by grep), so it's been dropped to
# keep the public surface honest.
__all__ = ["get_api_key", "fetch_sensors", "classify_pm25"]


def _quarantine_saturated(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (kept, dropped) based on PM25_RAW_SATURATION_THRESHOLD.

    pandas comparison treats NaN > x as False, so rows with NaN pm25_raw
    are always kept. This is defensive: this module's output should never
    have NaN pm25_raw (apply_epa_correction copies pm25 into pm25_raw at
    the start), but the invariant matters if the helper is ever reused.

    The dropped frame carries an extra `filter_reason` column for audit
    and future extensibility (currently only "saturated_raw").
    """
    mask = df["pm25_raw"] > PM25_RAW_SATURATION_THRESHOLD
    dropped = df[mask].assign(filter_reason="saturated_raw").copy()
    kept = df[~mask].copy()
    if not dropped.empty:
        offenders = dropped[["sensor_id", "name", "pm25_raw"]].to_dict("records")
        logger.warning(
            "Quarantined %d PurpleAir sensor(s) with saturated pm25_raw (> %.0f µg/m³): %s",
            len(dropped), PM25_RAW_SATURATION_THRESHOLD, offenders,
        )
    return kept, dropped


def fetch_sensors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all outdoor PurpleAir sensors within the Dallas bounding box.

    Returns a (kept, dropped) tuple of DataFrames:
      kept    — sensors that passed the saturation filter. Columns:
                sensor_id, name, lat, lon, pm25, pm25_raw, epa_corrected,
                humidity, source.
      dropped — sensors quarantined because pm25_raw exceeded
                PM25_RAW_SATURATION_THRESHOLD. Same columns as `kept`
                plus a `filter_reason` column (currently always
                "saturated_raw").

    pm25 is the EPA-corrected 10-minute average PM2.5 (µg/m³).
    pm25_raw is the original uncorrected reading.
    epa_corrected is 1 if the formula was applied, 0 if humidity was missing.
    Only outdoor sensors (location_type == 0) are included.
    """
    api_key = get_api_key()

    # Fields we want back from the API. humidity is required for the EPA
    # correction formula applied below.
    fields = [
        "sensor_index",
        "name",
        "latitude",
        "longitude",
        "pm2.5_cf_1",   # CF=1 channel, instantaneous — required input for Barkjohn 2021 EPA correction (10-min avg only exists on ATM channel)
        "humidity",
        "location_type",
    ]

    params = {
        "fields":     ",".join(fields),
        "location_type": 0,          # 0 = outdoor only
        "nwlng":      BBOX["west"],
        "nwlat":      BBOX["north"],
        "selng":      BBOX["east"],
        "selat":      BBOX["south"],
    }

    headers = {"X-API-Key": api_key}

    response = requests.get(
        f"{PURPLEAIR_BASE_URL}/sensors",
        params=params,
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()

    # The API returns {"fields": [...], "data": [[...], ...]}
    field_names = data.get("fields", [])
    rows = data.get("data", [])

    empty_cols = ["sensor_id", "name", "lat", "lon", "pm25", "pm25_raw", "epa_corrected", "humidity", "source"]
    if not rows:
        empty_kept    = pd.DataFrame(columns=empty_cols)
        empty_dropped = pd.DataFrame(columns=empty_cols + ["filter_reason"])
        return empty_kept, empty_dropped

    df = pd.DataFrame(rows, columns=field_names)

    # Rename to friendlier column names
    df = df.rename(columns={
        "sensor_index":     "sensor_id",
        "latitude":         "lat",
        "longitude":        "lon",
        "pm2.5_cf_1":       "pm25",
    })

    # Drop rows where PM2.5 reading is missing or negative.
    # Zero is a valid reading (very clean air). PurpleAir returns null for offline sensors,
    # not zero — so null is what we drop here. Negative values indicate sensor malfunction.
    df = df.dropna(subset=["pm25"])
    df = df[df["pm25"] >= 0]

    # Apply the EPA correction AFTER NaN/negative filtering so the formula only
    # sees real readings. pm25 becomes the corrected value; pm25_raw preserves
    # the original for audit.
    df = apply_epa_correction(df)

    # Keep only the columns we need and tag the source. humidity is preserved
    # as a model input for RF inference.
    keep = ["sensor_id", "name", "lat", "lon", "pm25", "pm25_raw", "epa_corrected"]
    if "humidity" in df.columns:
        keep.append("humidity")
    df = df[keep].copy()
    if "humidity" not in df.columns:
        df["humidity"] = float("nan")
    df["source"] = "purpleair"

    # NEW: quarantine after all column work, so kept and dropped have
    # the same shape (plus filter_reason on dropped).
    kept, dropped = _quarantine_saturated(df)
    return kept, dropped


def classify_pm25(pm25: float) -> str:
    """
    Classify a PM2.5 value (µg/m³) into an AQI category string.
    Mirrors the thresholds in config.py.
    """
    if pm25 <= 12.0:
        return "good"
    elif pm25 <= 35.4:
        return "moderate"
    elif pm25 <= 55.4:
        return "sensitive"
    elif pm25 <= 150.4:
        return "unhealthy"
    elif pm25 <= 250.4:
        return "very_unhealthy"
    else:
        return "hazardous"
