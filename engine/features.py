# engine/features.py — Feature engineering: compute per-sensor traffic and wind columns
#
# TODO (RF inference): the Random Forest also trains on
# `dist_to_highway_m` from `data/spatial/spatial_features.compute_distance_to_highway`.
# Grid cells passed into the model at inference time will need the same
# lookup applied per cell. Add it here (or wherever the live feature pipeline
# feeds the model) before turning on RF inference. Static feature, cached on
# disk — same call as in `ml/training/collect_training_data.py:collect_all_purpleair`.
#
# build_features() no longer modifies pm25. PurpleAir sensors already measure the
# real-world effects of nearby traffic and wind; applying another adjustment on top
# of the raw reading would double-count those effects.
#
# Instead, this function computes traffic and wind values PER SENSOR and stores them
# as separate columns. These columns are used for:
#   1. Live dashboard snapshots — stored in data/dashboard_snapshots.csv via
#      data/ingestion/history.py:save_snapshot. (NOT the RF training set; that
#      is built separately by ml/training/collect_training_data.py from historical
#      PurpleAir data.)
#   2. The same adjustment logic is applied POST-IDW to grid cells in interpolation.py,
#      where IDW alone has no knowledge of roads or wind.
#
# The returned DataFrame keeps pm25 and pm25_raw untouched (pm25 is already
# EPA-corrected at the source for PurpleAir rows; pm25_raw is the uncorrected
# reading for PurpleAir and NaN for OpenAQ) and adds these feature columns:
#   traffic_factor      — exponential congestion factor (0–1) for the nearest road
#   wind_term           — signed wind adjustment (µg/m³) that would be subtracted
#   nearest_congestion  — raw congestion score of the nearest traffic sample point
#   distance_to_road_m  — metres to that traffic point
#   direction_factor    — signed wind direction alignment (-1 transport … +1 dispersal)
#   dispersal           — wind speed dispersal strength (0–1)

import numpy as np
import pandas as pd

from engine.adjustments import (
    WIND_WEIGHT,
    WIND_SPEED_CAP,
    traffic_factor,
    nearest_traffic_point,
    traffic_decay_multiplier,
    wind_dispersal_factor,
    wind_direction_factor,
)


def build_features(
    sensor_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    wind: dict,
) -> pd.DataFrame:
    """
    Compute traffic and wind feature columns for each sensor.
    Does NOT modify pm25 — the EPA-corrected (PurpleAir) or reference-grade
    (OpenAQ) reading is preserved as-is.

    Args:
        sensor_df:  DataFrame with [sensor_id, name, lat, lon, pm25, source]
                    and optionally [pm25_raw, epa_corrected] for PurpleAir rows.
        traffic_df: DataFrame with [lat, lon, congestion] from fetch_traffic()
        wind:       Dict with wind_speed and wind_deg from fetch_wind()

    Returns:
        Copy of sensor_df with pm25 / pm25_raw / epa_corrected untouched plus
        new feature columns: traffic_factor, wind_term, nearest_congestion,
        distance_to_road_m, direction_factor, dispersal.
    """
    df = sensor_df.copy()

    # pm25_raw comes in from data/ingestion/purpleair.py (uncorrected PurpleAir reading).
    # OpenAQ rows have no such column; after concat they carry NaN, which is
    # the right signal — OpenAQ is reference-grade and has no separate "raw"
    # reading to preserve. Do NOT overwrite the column here; that would destroy
    # the audit trail for PurpleAir rows.
    if "pm25_raw" not in df.columns:
        df["pm25_raw"] = float("nan")

    wind_speed = wind.get("wind_speed") or 0.0
    wind_deg   = wind.get("wind_deg")   # may be None if OWM didn't return it

    disp = wind_dispersal_factor(wind_speed)

    no_traffic = traffic_df is None or traffic_df.empty

    # Accumulators for each feature column
    traffic_factors    = []
    wind_terms         = []
    congestions        = []
    distances_m        = []
    direction_factors  = []
    dispersals         = []

    for _, row in df.iterrows():
        # --- Traffic term ---
        if no_traffic:
            nearest      = None
            distance_deg = None
            congestion   = 0.0
        else:
            nearest, distance_deg = nearest_traffic_point(row["lat"], row["lon"], traffic_df)
            congestion = float(nearest["congestion"])

        decay = 0.0 if nearest is None else traffic_decay_multiplier(distance_deg)
        tf    = traffic_factor(congestion) * decay  # 0–1 factor (scaled by TRAFFIC_WEIGHT elsewhere)

        # --- Wind term ---
        if wind_speed == 0.0:
            # No wind — direction is meaningless; zero out everything.
            dir_factor = 0.0
        elif wind_deg is None:
            # Wind speed exists but direction unknown — assume pure dispersal.
            dir_factor = 1.0
        elif nearest is None:
            # No traffic data — assume pure dispersal.
            dir_factor = 1.0
        else:
            dir_factor = wind_direction_factor(row["lat"], row["lon"], nearest, wind_deg)

        # wind_term is what WOULD be subtracted from pm25 if this were a grid cell.
        # Stored for training data; not applied to the sensor reading.
        wt = dir_factor * disp * WIND_WEIGHT

        # Record distance in metres for dashboard_snapshots.csv
        dist_m = float(distance_deg * 111_000) if distance_deg is not None else float("nan")

        traffic_factors.append(tf)
        wind_terms.append(wt)
        congestions.append(congestion)
        distances_m.append(dist_m)
        direction_factors.append(dir_factor)
        dispersals.append(disp)

    df["traffic_factor"]     = traffic_factors
    df["wind_term"]          = wind_terms
    df["nearest_congestion"] = congestions
    df["distance_to_road_m"] = distances_m
    df["direction_factor"]   = direction_factors
    df["dispersal"]          = dispersals

    return df
