import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from config import BBOX
from data.ingestion.openaq import fetch_openaq
from data.ingestion.purpleair import fetch_sensors
from data.ingestion.traffic import fetch_traffic
from data.ingestion.weather import fetch_wind
from engine.features import build_features
from engine.interpolation import adjust_grid, run_idw
from engine.snapshot import PipelineSnapshot

from api.schemas.responses import BBox, GridResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache window for the assembled pipeline snapshot.
#
# Widened to a 1800s TTL to bound TomTom usage: at
# SAMPLE_GRID=5 (25 TomTom calls per refresh) and a 30-min TTL, daily
# usage caps at 48 × 25 = 1,200 calls — half of TomTom's 2,500/day shared
# free-tier limit, regardless of frontend traffic.
#
# app.py's Streamlit cache stays at 300s on purpose: that file is locked
# from modification (PROJECT_STATE.md), and Streamlit sessions are light
# and transient, so cache parity isn't worth the policy violation. Worst-
# case drift between the AERIA UI and the legacy Streamlit dashboard is
# ~25 minutes, which is fine since upstream sources refresh every
# 10–30 minutes anyway.
_TTL_SECONDS = 1800


_cache: dict = {"ts": 0.0, "value": None}


def _run_full_pipeline() -> PipelineSnapshot:
    """Run the full ingest → IDW → adjust pipeline. Mirrors app.py."""
    purpleair_kept, _purpleair_dropped = fetch_sensors()
    openaq_df = fetch_openaq()
    sensor_df = pd.concat([purpleair_kept, openaq_df], ignore_index=True)

    if sensor_df.empty:
        raise RuntimeError("No sensor data available for the Dallas bounding box.")

    try:
        wind = fetch_wind()
    except Exception as e:
        logger.warning("Wind fetch failed, defaulting to calm: %s", e)
        wind = {"wind_speed": 0.0, "wind_deg": 0.0}

    try:
        traffic_df = fetch_traffic()
    except Exception as e:
        logger.warning("Traffic fetch failed, skipping congestion adjustment: %s", e)
        traffic_df = pd.DataFrame()

    feat_df = build_features(
        sensor_df,
        traffic_df if traffic_df is not None else pd.DataFrame(),
        wind,
    )

    lats_2d, lons_2d, idw_estimate, idw_hw_dist, confidence = run_idw(feat_df)
    grid = adjust_grid(
        idw_estimate,
        lats_2d,
        lons_2d,
        traffic_df if traffic_df is not None else pd.DataFrame(),
        wind,
        idw_hw_dist=idw_hw_dist,
    )

    return PipelineSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sensor_df=feat_df,
        lats_2d=lats_2d,
        lons_2d=lons_2d,
        grid=grid,
        confidence=confidence,
        wind_speed=float(wind.get("wind_speed") or 0.0),
        wind_deg=float(wind.get("wind_deg") or 0.0),
    )


def get_cached_snapshot() -> PipelineSnapshot:
    now = time.time()
    if _cache["value"] is not None and now - _cache["ts"] < _TTL_SECONDS:
        return _cache["value"]
    snap = _run_full_pipeline()
    _cache["ts"] = now
    _cache["value"] = snap
    return snap


def refresh_snapshot() -> PipelineSnapshot:
    """Force-refresh the cached snapshot. Called by the scheduler
    on a fixed interval to keep the cache hot regardless of user
    traffic. Bypasses the TTL check that get_cached_snapshot uses."""
    snap = _run_full_pipeline()
    _cache["ts"] = time.time()
    _cache["value"] = snap
    return snap


@router.get("/grid", response_model=GridResponse, tags=["grid"])
def get_grid() -> GridResponse:
    """Full IDW + traffic/wind-adjusted PM2.5 grid over the Dallas bounding box.

    Slow on first call (~5-15s end-to-end including all upstream fetches);
    cached for 30 minutes after that.
    """
    try:
        snap = get_cached_snapshot()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"Configuration error: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pipeline failure: {e}")

    # lats_2d and lons_2d come from np.meshgrid, so each row of lats_2d is
    # constant lat, and each column of lons_2d is constant lon. Collapse to
    # 1D arrays to keep the JSON payload small.
    lats_1d = snap.lats_2d[:, 0].tolist()
    lons_1d = snap.lons_2d[0, :].tolist()

    return GridResponse(
        timestamp=snap.timestamp,
        resolution=snap.grid.shape[0],
        bbox=BBox(**BBOX),
        lats=lats_1d,
        lons=lons_1d,
        pm25=snap.grid.tolist(),
        confidence=snap.confidence.tolist(),
        wind_speed=snap.wind_speed,
        wind_deg=snap.wind_deg,
        sensor_count=len(snap.sensor_df),
        avg_pm25=float(snap.sensor_df["pm25"].mean()),
    )
