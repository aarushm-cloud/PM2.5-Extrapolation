# DEPRECATED: the RF model was evaluated and shelved.
# The dashboard runs on IDW + adjust_grid; this module is dormant.
# See ml/docs/rf_model_result.md for the full negative-result writeup.
#
# ml/predictor.py — Random Forest inference for grid cells.
#
# Loads models/rf.pkl once per process and predicts PM2.5 at every cell
# of the dashboard grid. Replaces the IDW + post-IDW adjustment path because
# the RF was trained with lat/lon plus the same temporal traffic proxies and
# spatial highway-distance feature, so it carries the spatial structure IDW
# used to provide.
#
# Feature order MUST match models/rf_metadata.json:feature_names
# exactly. Mismatches cause silent prediction garbage, so this module asserts
# parity at startup against a freshly-built dummy feature row.

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Project root is on sys.path via ROOT.parent in the training pipeline; here
# we add it directly because predictor.py may be imported as `ml.predictor`
# from a script whose CWD is the repo root.
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from config import DFW_AIRPORT_LAT_LON  # noqa: E402

MODEL_PATH = ROOT / "ml" / "models" / "rf.pkl"
METADATA_PATH = ROOT / "ml" / "models" / "rf_metadata.json"

PM25_MIN_PLAUSIBLE = 0.0
PM25_MAX_PLAUSIBLE = 500.0

# DFW airport — used as the dummy point for the schema parity check.
DFW_AIRPORT_LAT, DFW_AIRPORT_LON = DFW_AIRPORT_LAT_LON

log = logging.getLogger(__name__)

_MODEL: Optional[object] = None
_METADATA: Optional[dict] = None


def load_model() -> tuple[object, dict]:
    """Load the RF model and metadata once per process. ~200 MB, a few-second
    load is acceptable at startup and amortised across every request."""
    global _MODEL, _METADATA
    if _MODEL is not None and _METADATA is not None:
        return _MODEL, _METADATA

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RF model not found at {MODEL_PATH}. "
            f"Run ml/research/train_rf.py to build it."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"RF metadata not found at {METADATA_PATH}."
        )

    t0 = time.time()
    _MODEL = joblib.load(MODEL_PATH)
    with METADATA_PATH.open() as f:
        _METADATA = json.load(f)
    elapsed = time.time() - t0

    log.info(
        f"RF model loaded from {MODEL_PATH.name} in {elapsed:.2f}s "
        f"({_METADATA['training_row_count']:,} rows, "
        f"{_METADATA['training_sensor_count']} sensors, "
        f"{_METADATA['training_days_span']} days)"
    )
    _assert_schema_parity()
    return _MODEL, _METADATA


def _add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of ml/training/collect_training_data.py:add_traffic_features. Same rules,
    same column names, same dtypes — any drift here silently breaks inference."""
    # America/Chicago — matches training. Timezone is the most likely silent
    # train/inference drift; do not change without updating training too.
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

    traffic = pd.Series(0.4, index=df.index)
    traffic[df["is_weekend"] == 1] = 0.3
    traffic[(hour >= 10) & (hour <= 15) & weekday] = 0.5
    traffic[((hour >= 20) | (hour <= 5))] = 0.1
    traffic[df["is_am_rush"] == 1] = 1.0
    traffic[df["is_pm_rush"] == 1] = 1.0
    df["traffic_index"] = traffic
    return df


def build_features(
    lats: np.ndarray,
    lons: np.ndarray,
    humidity: float,
    wind_speed: float,
    wind_deg: float,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Build a feature DataFrame in the exact column order the model expects.

    `timestamp` must be a tz-aware pandas Timestamp (UTC). `lats`/`lons` are
    1-D arrays of grid-cell coordinates. `humidity`, `wind_speed`, `wind_deg`
    are scalars broadcast across all cells (matching the live data sources
    the dashboard already uses — one OWM call per metro for wind, mean
    PurpleAir humidity for the metro)."""
    from data.spatial.spatial_features import compute_distance_to_highway

    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be tz-aware (UTC)")
    ts_utc = timestamp.tz_convert("UTC") if str(timestamp.tz) != "UTC" else timestamp

    dist_to_highway_m = np.array(
        [compute_distance_to_highway(float(la), float(lo)) for la, lo in zip(lats, lons)],
        dtype=np.float64,
    )

    df = pd.DataFrame({
        "lat": lats.astype(np.float64),
        "lon": lons.astype(np.float64),
        "dist_to_highway_m": dist_to_highway_m,
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "wind_deg": float(wind_deg),
        "timestamp": pd.Series([ts_utc] * len(lats)),
    })
    df = _add_traffic_features(df)
    return df


def _assert_schema_parity() -> None:
    """Fail loudly if the live feature pipeline disagrees with the trained
    model's expected feature names or order. Correctness blocker, not a
    warning — a silent mismatch means every prediction is garbage."""
    assert _METADATA is not None
    expected = list(_METADATA["feature_names"])

    dummy = build_features(
        lats=np.array([DFW_AIRPORT_LAT]),
        lons=np.array([DFW_AIRPORT_LON]),
        humidity=50.0,
        wind_speed=4.5,
        wind_deg=180.0,
        timestamp=pd.Timestamp.now(tz="UTC"),
    )
    actual = [c for c in dummy.columns if c != "timestamp"]
    if actual != expected:
        raise RuntimeError(
            "FEATURE SCHEMA MISMATCH between live inference and trained model.\n"
            f"  Expected (rf_metadata.json): {expected}\n"
            f"  Got (ml/predictor.build_features): {actual}\n"
            "  Train/inference drift detected — fix ml/predictor.py before "
            "deploying."
        )
    log.info("  Feature schema parity check: PASS")


def predict_grid(
    lats_2d: np.ndarray,
    lons_2d: np.ndarray,
    humidity: float,
    wind_speed: float,
    wind_deg: float,
    timestamp: pd.Timestamp,
) -> np.ndarray:
    """Run the RF on every cell of a (res, res) lat/lon grid. Returns a
    (res, res) array of PM2.5 µg/m³ predictions, clipped to [0, 500]."""
    model, meta = load_model()
    expected = list(meta["feature_names"])

    flat_lats = lats_2d.ravel()
    flat_lons = lons_2d.ravel()

    feats = build_features(
        flat_lats, flat_lons, humidity, wind_speed, wind_deg, timestamp,
    )
    X = feats[expected].to_numpy()

    t0 = time.time()
    pred = model.predict(X)
    elapsed_ms = (time.time() - t0) * 1000.0
    log.info(f"  RF predicted {len(pred):,} grid cells in {elapsed_ms:.0f} ms")

    out_of_range = (pred < PM25_MIN_PLAUSIBLE) | (pred > PM25_MAX_PLAUSIBLE)
    n_oor = int(out_of_range.sum())
    if n_oor:
        log.warning(
            f"  RF produced {n_oor} prediction(s) outside "
            f"[{PM25_MIN_PLAUSIBLE}, {PM25_MAX_PLAUSIBLE}] µg/m³ — clipping"
        )
    pred = np.clip(pred, PM25_MIN_PLAUSIBLE, PM25_MAX_PLAUSIBLE)
    return pred.reshape(lats_2d.shape)
