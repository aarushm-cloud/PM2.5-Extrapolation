"""RF evaluation — IDW backbone + RF residual correction (NOT shipped).

This script was used to evaluate whether a Random Forest residual model
could improve on raw IDW for the dashboard. The conclusion was no:

    LOGO CV (19 sensors held out one at a time, pooled across folds):
      Raw IDW RMSE:            2.48 µg/m³  ← lowest
      IDW + adjust_grid RMSE:  4.57 µg/m³  (training-time proxy of the
                                            production heuristic — see
                                            ml/docs/rf_model_result.md for the
                                            caveat; this number is NOT a
                                            fair indictment of the
                                            production adjust_grid)
      IDW + RF residual RMSE:  2.91 µg/m³

The RF was trained on (lat, lon, dist_to_highway_m, humidity, wind_speed,
wind_deg, local_hour_of_day, day_of_week, is_weekend, is_am_rush,
is_pm_rush, traffic_index) to predict pm25 - idw_loo_estimate. Residuals
were well-behaved (mean -0.32, std 2.46, skew -0.08) — the RF simply has
no signal to extract from the available features beyond what IDW already
captures.

The production dashboard uses raw IDW + heuristic adjust_grid from
engine/interpolation.py; no model file is loaded at runtime. This script
is preserved so the comparison is reproducible. See ml/docs/rf_model_result.md
for the full write-up and forward-looking notes on what would need to
change before another ML attempt is worth trying.

Re-running this script: it never saves the model unless RF beats both
baselines (a guard that gated the original failed run). To re-evaluate
after data/feature changes, just `python ml/research/train_residual_rf.py`.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import skew as scipy_skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

from config import IDW_POWER, IDW_SEARCH_RADIUS_DEG, LON_CORRECTION, TRAFFIC_WEIGHT
from engine.adjustments import traffic_factor_vec

HISTORY_CSV = ROOT / "ml" / "data" / "history.csv"
MODELS_DIR = ROOT / "ml" / "models"
OUTPUT_DIR = ROOT / "ml" / "analysis" / "output"

FEATURES = [
    "lat",
    "lon",
    "dist_to_highway_m",
    "humidity",
    "wind_speed",
    "wind_deg",
    "local_hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_am_rush",
    "is_pm_rush",
    "traffic_index",
]
TARGET_RESIDUAL = "residual"
TARGET_PM25 = "pm25"
GROUP_COL = "sensor_id"

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}

LOO_IDW_BUDGET_S = 60.0


def compute_loo_idw_per_timestamp(df: pd.DataFrame) -> np.ndarray:
    """For each row, return the IDW estimate at that sensor's location using
    only the OTHER sensors reporting at the same timestamp.

    Vectorised per timestamp: one N×N distance matrix per timestamp, no
    per-row Python loop. Returns NaN for timestamps with fewer than 2
    sensors (no LOO possible) — caller drops those rows.
    """
    estimates = np.full(len(df), np.nan, dtype=np.float64)
    lats_all = df["lat"].to_numpy()
    lons_all = df["lon"].to_numpy()
    pm25_all = df["pm25"].to_numpy()

    for _ts, idx in df.groupby("timestamp", sort=False).indices.items():
        idx = np.asarray(idx)
        n = len(idx)
        if n < 2:
            continue

        lats = lats_all[idx]
        lons = lons_all[idx]
        pm25 = pm25_all[idx]

        dlat = lats[:, None] - lats[None, :]
        dlon = (lons[:, None] - lons[None, :]) * LON_CORRECTION
        D = np.sqrt(dlat ** 2 + dlon ** 2)
        np.fill_diagonal(D, np.inf)  # exclude self → weight 0

        in_radius = D <= IDW_SEARCH_RADIUS_DEG
        weights = np.where(in_radius, 1.0 / (D ** IDW_POWER), 0.0)

        weight_total = weights.sum(axis=1)
        weighted_sum = weights @ pm25

        # Fall back to the timestamp's mean for rows with no in-radius peers.
        ts_mean = pm25.mean()
        has_neighbours = weight_total > 0
        est = np.where(
            has_neighbours,
            weighted_sum / np.where(has_neighbours, weight_total, 1.0),
            ts_mean,
        )
        estimates[idx] = est

    return estimates


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HISTORY_CSV.relative_to(ROOT)}...")
    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    n_rows_raw = len(df)
    n_sensors = df[GROUP_COL].nunique()
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    n_days = int((date_max - date_min).total_seconds() // 86400)
    print(f"  rows={n_rows_raw:,}  sensors={n_sensors}  days={n_days}")

    print()
    print("Computing leave-one-out IDW per timestamp (vectorised)...")
    t0 = time.time()
    df["idw_loo"] = compute_loo_idw_per_timestamp(df)
    elapsed = time.time() - t0
    print(f"  LOO IDW phase: {elapsed:.2f}s")
    if elapsed > LOO_IDW_BUDGET_S:
        raise RuntimeError(
            f"LOO IDW took {elapsed:.1f}s (>{LOO_IDW_BUDGET_S}s budget). "
            "Likely an accidental per-row Python loop — investigate."
        )

    n_dropped = int(df["idw_loo"].isna().sum())
    if n_dropped:
        print(
            f"  Dropping {n_dropped:,} rows with no LOO IDW "
            "(single-sensor timestamps)."
        )
        df = df.dropna(subset=["idw_loo"]).reset_index(drop=True)

    df["residual"] = df[TARGET_PM25] - df["idw_loo"]

    res = df["residual"].to_numpy()
    res_mean = float(res.mean())
    res_std = float(res.std(ddof=1))
    res_skew = float(scipy_skew(res))
    print()
    print("Residual distribution (pm25 - idw_loo):")
    print(f"  mean: {res_mean:+.3f} µg/m³")
    print(f"  std:  {res_std:.3f} µg/m³")
    print(f"  skew: {res_skew:+.3f}")
    if abs(res_mean) > 0.5 or abs(res_skew) > 1.0:
        print()
        print("  ** FLAG **")
        if abs(res_mean) > 0.5:
            print("  |mean| > 0.5 µg/m³ — IDW may be systematically biased; "
                  "RF is at risk of learning a constant offset.")
        if abs(res_skew) > 1.0:
            print("  |skew| > 1 — residuals are not roughly symmetric.")

    X = df[FEATURES].to_numpy()
    y_pm25 = df[TARGET_PM25].to_numpy()
    y_residual = df[TARGET_RESIDUAL].to_numpy()
    idw_loo = df["idw_loo"].to_numpy()
    traffic_index = df["traffic_index"].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    # Training-time proxy for IDW + adjust_grid. Production adjust_grid uses
    # K-nearest TomTom road points + distance decay + wind direction; none of
    # that exists per-row in history.csv, so we use the temporal traffic_index
    # proxy. Both heuristic and RF see the same proxy column → fair comparison.
    heuristic_adj = traffic_factor_vec(traffic_index) * TRAFFIC_WEIGHT

    print()
    print(f"Leave-one-sensor-out CV ({n_sensors} folds):")
    logo = LeaveOneGroupOut()
    raw_se, heur_se, rf_se = [], [], []

    for train_idx, test_idx in logo.split(X, y_residual, groups):
        held_out = int(groups[test_idx[0]])

        model = RandomForestRegressor(**HYPERPARAMS)
        model.fit(X[train_idx], y_residual[train_idx])
        rf_pred_residual = model.predict(X[test_idx])

        truth = y_pm25[test_idx]
        pred_raw = np.clip(idw_loo[test_idx], 0.0, None)
        pred_heur = np.clip(idw_loo[test_idx] + heuristic_adj[test_idx], 0.0, None)
        pred_rf = np.clip(idw_loo[test_idx] + rf_pred_residual, 0.0, None)

        raw_se.append((truth - pred_raw) ** 2)
        heur_se.append((truth - pred_heur) ** 2)
        rf_se.append((truth - pred_rf) ** 2)

        rmse_raw = float(np.sqrt(np.mean((truth - pred_raw) ** 2)))
        rmse_heur = float(np.sqrt(np.mean((truth - pred_heur) ** 2)))
        rmse_rf = float(np.sqrt(np.mean((truth - pred_rf) ** 2)))
        print(
            f"  sensor {held_out:>7}  raw={rmse_raw:5.2f}  "
            f"heur={rmse_heur:5.2f}  rf={rmse_rf:5.2f}"
        )

    rmse_raw = float(np.sqrt(np.mean(np.concatenate(raw_se))))
    rmse_heur = float(np.sqrt(np.mean(np.concatenate(heur_se))))
    rmse_rf = float(np.sqrt(np.mean(np.concatenate(rf_se))))

    line = "=" * 70
    print()
    print(line)
    print("Three-way RMSE (LOGO CV, pooled across folds):")
    print(line)
    print(f"  Raw IDW RMSE:            {rmse_raw:.2f} µg/m³")
    print(f"  IDW + adjust_grid RMSE:  {rmse_heur:.2f} µg/m³")
    print(f"  IDW + RF residual RMSE:  {rmse_rf:.2f} µg/m³")
    print()

    rf_is_best = rmse_rf < rmse_raw and rmse_rf < rmse_heur
    if not rf_is_best:
        print("** FLAG **")
        print("  IDW + RF residual is NOT the lowest of the three.")
        print("  Stopping before saving the model. Review CV results before")
        print("  proceeding to inference integration.")
        return

    print("Final RF fit on all data...")
    final_model = RandomForestRegressor(**HYPERPARAMS)
    final_model.fit(X, y_residual)

    importances = pd.Series(
        final_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print()
    print("Feature importance:")
    for name, val in importances.items():
        print(f"  {name:<22} {val:.3f}")

    lat_lon_imp = float(importances.get("lat", 0.0) + importances.get("lon", 0.0))
    if lat_lon_imp > 0.40:
        print()
        print("** FLAG **")
        print(f"  lat+lon combined importance = {lat_lon_imp:.2f} (>0.40).")
        print("  Surface for review — model may be leaning on absolute "
              "location rather than learning location-agnostic structure.")

    joblib.dump(final_model, MODELS_DIR / "rf_phase4_residual.pkl")
    metadata = {
        "feature_names": FEATURES,
        "target_name": TARGET_RESIDUAL,
        "training_row_count": int(len(df)),
        "training_sensor_count": int(df[GROUP_COL].nunique()),
        "training_date_min": date_min.isoformat(),
        "training_date_max": date_max.isoformat(),
        "training_days_span": n_days,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": HYPERPARAMS,
        "cv_rmse_raw_idw": rmse_raw,
        "cv_rmse_idw_plus_adjust_grid": rmse_heur,
        "cv_rmse_idw_plus_rf_residual": rmse_rf,
        "residual_mean": res_mean,
        "residual_std": res_std,
        "residual_skew": res_skew,
        "feature_importance": importances.to_dict(),
    }
    with open(MODELS_DIR / "rf_phase4_residual_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"  Saved {MODELS_DIR.relative_to(ROOT)}/rf_phase4_residual.pkl")
    print(f"  Saved {MODELS_DIR.relative_to(ROOT)}/rf_phase4_residual_metadata.json")


if __name__ == "__main__":
    main()
