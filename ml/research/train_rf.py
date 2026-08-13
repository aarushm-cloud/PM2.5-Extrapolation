"""Phase 4 — Random Forest PM2.5 spatial interpolator.

Trains on ml/data/history.csv. Evaluates with leave-one-sensor-out CV, then
fits the production model on all sensors. See CLAUDE.md for context.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent.parent
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
TARGET = "pm25"
GROUP_COL = "sensor_id"

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}

WATCH_SENSOR_LOWROWS = 305450
WATCH_SENSOR_ISOLATED = 241905


def cosine_corrected_km(lat1, lon1, lat2, lon2):
    """Approx great-circle distance in km, with cos(lat) longitude correction."""
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111.0
    dx = (lon2 - lon1) * 111.0 * np.cos(lat_mid)
    return float(np.hypot(dx, dy))


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    n_rows = len(df)
    n_sensors = df[GROUP_COL].nunique()
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    n_days = int((date_max - date_min).total_seconds() // 86400)

    sensor_locs = (
        df.groupby(GROUP_COL)[["lat", "lon"]].first().reset_index()
    )
    sensor_row_counts = df.groupby(GROUP_COL).size().to_dict()

    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    logo = LeaveOneGroupOut()
    fold_records = []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = int(groups[test_idx[0]])
        held_lat, held_lon = sensor_locs.loc[
            sensor_locs[GROUP_COL] == held_out, ["lat", "lon"]
        ].iloc[0]

        train_sensors = sensor_locs[sensor_locs[GROUP_COL] != held_out]
        centroid_lat = train_sensors["lat"].mean()
        centroid_lon = train_sensors["lon"].mean()
        dist_km = cosine_corrected_km(
            held_lat, held_lon, centroid_lat, centroid_lon
        )

        model = RandomForestRegressor(**HYPERPARAMS)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
        mae = float(mean_absolute_error(y[test_idx], pred))

        fold_records.append(
            {
                "sensor_id": held_out,
                "rmse": rmse,
                "mae": mae,
                "lat": float(held_lat),
                "lon": float(held_lon),
                "row_count": int(sensor_row_counts[held_out]),
                "dist_from_train_centroid_km": dist_km,
            }
        )
        print(
            f"  fold sensor={held_out:>7}  rows={sensor_row_counts[held_out]:>5}  "
            f"dist={dist_km:5.1f}km  RMSE={rmse:5.2f}  MAE={mae:5.2f}"
        )

    cv_df = pd.DataFrame(fold_records).sort_values("rmse").reset_index(drop=True)
    cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        cv_df["dist_from_train_centroid_km"],
        cv_df["rmse"],
        s=60,
        c="tab:blue",
        edgecolor="black",
        zorder=3,
    )
    for _, row in cv_df.iterrows():
        ax.annotate(
            str(int(row["sensor_id"])),
            (row["dist_from_train_centroid_km"], row["rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Held-out sensor distance from training centroid (km)")
    ax.set_ylabel("Fold RMSE (µg/m³)")
    ax.set_title("Phase 4 RF — CV error vs. spatial isolation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_error_vs_distance.png", dpi=120)
    plt.close(fig)

    final_model = RandomForestRegressor(**HYPERPARAMS)
    final_model.fit(X, y)

    joblib.dump(final_model, MODELS_DIR / "rf_phase4.pkl")

    metadata = {
        "feature_names": FEATURES,
        "target_name": TARGET,
        "training_row_count": int(n_rows),
        "training_sensor_count": int(n_sensors),
        "training_date_min": date_min.isoformat(),
        "training_date_max": date_max.isoformat(),
        "training_days_span": n_days,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": HYPERPARAMS,
    }
    with open(MODELS_DIR / "rf_phase4_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    rmses = cv_df["rmse"].to_numpy()
    maes = cv_df["mae"].to_numpy()
    median_rmse = float(np.median(rmses))
    max_rmse = float(rmses.max())
    min_rmse = float(rmses.min())
    mean_rmse = float(rmses.mean())
    mean_mae = float(maes.mean())
    worst_row = cv_df.iloc[-1]
    best_row = cv_df.iloc[0]

    if median_rmse < 5 and max_rmse < 10:
        verdict = "Acceptable"
    elif median_rmse > 8 or max_rmse > 15:
        verdict = "Poor"
    else:
        verdict = "Marginal"

    importances = pd.Series(
        final_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    top5 = importances.head(5)

    cv_df_by_rmse = cv_df.sort_values("rmse").reset_index(drop=True)
    worst3 = cv_df_by_rmse.tail(3).iloc[::-1]
    best3 = cv_df_by_rmse.head(3)

    watch_low = cv_df_by_rmse[
        cv_df_by_rmse["sensor_id"] == WATCH_SENSOR_LOWROWS
    ]
    watch_iso = cv_df_by_rmse[
        cv_df_by_rmse["sensor_id"] == WATCH_SENSOR_ISOLATED
    ]

    notes = []
    if not watch_low.empty:
        rank = int(
            cv_df_by_rmse.index[
                cv_df_by_rmse["sensor_id"] == WATCH_SENSOR_LOWROWS
            ][0]
        ) + 1
        rank_from_worst = len(cv_df_by_rmse) - rank + 1
        watch_rmse = float(watch_low.iloc[0]["rmse"])
        next_worst = cv_df_by_rmse[
            cv_df_by_rmse["sensor_id"] != WATCH_SENSOR_LOWROWS
        ]["rmse"].max()
        if watch_rmse > 2 * next_worst:
            verdict_305450 = "unexpectedly bad — sparse-coverage decision worth revisiting"
        elif rank_from_worst <= 3:
            verdict_305450 = "matches expectation (low row count, weaker fold)"
        else:
            verdict_305450 = "unexpectedly good given the row count"
    else:
        rank_from_worst = -1
        watch_rmse = float("nan")
        verdict_305450 = "n/a"

    if not watch_iso.empty:
        iso_rmse = float(watch_iso.iloc[0]["rmse"])
        iso_dist = float(watch_iso.iloc[0]["dist_from_train_centroid_km"])
        notes.append(
            f"  Sensor {WATCH_SENSOR_ISOLATED} (isolated, {iso_dist:.1f} km from centroid): "
            f"RMSE={iso_rmse:.2f} — spatial isolation is expected; not a model failure."
        )

    lat_lon_imp = float(importances.get("lat", 0.0) + importances.get("lon", 0.0))
    if lat_lon_imp > 0.50:
        notes.append(
            f"  lat+lon importance = {lat_lon_imp:.2f} (>50%): model is leaning IDW-like, "
            "less use of meteorological/temporal features."
        )
    hwy_imp = float(importances.get("dist_to_highway_m", 0.0))
    if hwy_imp < 0.02:
        notes.append(
            f"  dist_to_highway_m importance = {hwy_imp:.3f} (<2%): didn't add training signal, "
            "but still useful at inference for grid cells far from sensors."
        )

    line = "=" * 70
    print()
    print(line)
    print("Phase 4 Random Forest — Training Complete")
    print(line)
    print(
        f"  Training set:    {n_rows:,} rows, {n_sensors} sensors, {n_days} days"
    )
    print(f"  Feature count:   {len(FEATURES)}")
    hp = HYPERPARAMS
    print(
        f"  Hyperparameters: n_estimators={hp['n_estimators']}, "
        f"max_depth={hp['max_depth']}, "
    )
    print(
        f"                   min_samples_leaf={hp['min_samples_leaf']}, "
        f"random_state={hp['random_state']}"
    )
    print()
    print("Cross-validation (leave-one-sensor-out):")
    print(f"  Folds:           {len(cv_df)}")
    print(f"  Mean RMSE:       {mean_rmse:.2f} µg/m³")
    print(f"  Median RMSE:     {median_rmse:.2f} µg/m³")
    print(
        f"  Max RMSE:        {max_rmse:.2f} µg/m³  (sensor {int(worst_row['sensor_id'])})"
    )
    print(
        f"  Min RMSE:        {min_rmse:.2f} µg/m³  (sensor {int(best_row['sensor_id'])})"
    )
    print(f"  Mean MAE:        {mean_mae:.2f} µg/m³")
    print()
    print("Worst 3 folds:")
    for _, r in worst3.iterrows():
        print(
            f"  sensor {int(r['sensor_id'])}: RMSE={r['rmse']:.2f} "
            f"(rows={int(r['row_count'])}, "
            f"dist_from_centroid={r['dist_from_train_centroid_km']:.1f} km)"
        )
    print()
    print("Best 3 folds:")
    for _, r in best3.iterrows():
        print(
            f"  sensor {int(r['sensor_id'])}: RMSE={r['rmse']:.2f} "
            f"(rows={int(r['row_count'])}, "
            f"dist_from_centroid={r['dist_from_train_centroid_km']:.1f} km)"
        )
    print()
    print(f"Sensor {WATCH_SENSOR_LOWROWS} specifically (the low-row-count survivor):")
    if not watch_low.empty:
        print(f"  RMSE: {watch_rmse:.2f}")
        print(f"  Rank: {rank_from_worst}{ordinal_suffix(rank_from_worst)} worst out of {len(cv_df)} folds")
        print(f"  Verdict: {verdict_305450}")
    else:
        print("  (sensor not present in CV)")
    print()
    print("Feature importance (top 5):")
    for name, val in top5.items():
        print(f"  {name}: {val:.3f}")
    if notes:
        print()
        print("Watchpoints:")
        for n in notes:
            print(n)
    print()
    print(f"Spatial generalization verdict:")
    print(f"  {verdict}")
    print()
    print(line)


def ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


if __name__ == "__main__":
    main()
