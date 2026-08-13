# Random Forest Model — Result

**Status:** The RF model does not ship. The dashboard stays on raw IDW + the
heuristic `adjust_grid` from [engine/interpolation.py](../../engine/interpolation.py).
This document records what was tried, why it didn't work, and what would have
to change before another ML attempt is worth running.

---

## What was attempted

Two RF architectures were considered.

### Attempt 1 — RF replacing IDW (initial wiring, reverted)

A `RandomForestRegressor` was trained on
`(lat, lon, dist_to_highway_m, humidity, wind_speed, wind_deg, local_hour_of_day, day_of_week, is_weekend, is_am_rush, is_pm_rush, traffic_index) → pm25`
using [ml/data/history.csv](../data/history.csv) (~68k rows, 19 sensors, 179 days), and
wired into the dashboard as a full replacement for IDW.

Result on the live dashboard: every grid cell predicted ~7 µg/m³ regardless of
nearby live readings, and the surface looked near-uniform across the metro. A
sensor reading 44 µg/m³ would sit next to a grid cell predicting 7.

Diagnosis: the RF was given only static spatial features (only 19 unique
`lat`/`lon` values in training, `dist_to_highway_m` collinear with sensor
location) plus metro-wide scalars at inference. It never sees live sensor
readings — at training or at inference. It is therefore a *conditional
historical-mean estimator*, not an interpolator. With PM2.5 right-skewed
toward low values (mean 7.31, median 5.84, p90 14.34), the model regresses
every cell toward the population mean.

LOGO CV at training time looked superficially fine (mean RMSE 3.38 µg/m³),
but only because every held-out sensor follows the same low-mean distribution
as the training population — "predict the mean" scores well there too.

This wiring was reverted in commit `2e2d82d`.

### Attempt 2 — RF as residual correction on top of IDW

The follow-up idea: keep IDW as the spatial anchor (so live sensor readings
drive predictions), and train the RF to predict the *residual*
`pm25 - idw_loo_estimate` from the same feature set. Final prediction at
inference would be `idw_estimate + rf_residual`. Code:
[ml/research/train_residual_rf.py](../research/train_residual_rf.py).

The script computes leave-one-out IDW per timestamp (vectorised, ~0.2 s
across 4,320 timestamps), then runs leave-one-sensor-out CV comparing three
pipelines.

Residual distribution was clean — mean **-0.319 µg/m³**, std **2.458 µg/m³**,
skew **-0.083**. IDW is not biased and residuals are symmetric.

---

## Three-way LOGO CV comparison (pooled across folds)

| Pipeline                        | RMSE (µg/m³) |
|---------------------------------|--------------|
| **Raw IDW alone**               | **2.48**     |
| IDW + adjust_grid (proxy)       | 4.57         |
| IDW + RF residual               | 2.91         |

**Raw IDW is the best of the three.** The RF residual adds ~0.43 µg/m³ of
error on average vs raw IDW.

### Caveat on the heuristic comparison

The 4.57 figure for `adjust_grid` is **not a fair indictment of the production
heuristic** and should not be cited as such. The training-time proxy strips
away the spatial part of `adjust_grid` (no live TomTom road data exists per
row in `data/history.csv` — the free TomTom tier is real-time only). The
proxy reduces to "during rush hour, push every sensor up by
`traffic_factor(traffic_index) * 8 µg/m³` regardless of location," which is
bound to hurt sensors not near a road.

The production heuristic in [engine/interpolation.py:adjust_grid](../../engine/interpolation.py)
uses K-nearest TomTom road points + per-cell distance decay + per-cell wind
direction relative to those roads. It almost certainly performs much better
than 4.57 in practice — we just can't measure that without paid TomTom
Traffic Stats history. Keeping the heuristic in the live pipeline is
defensible on those grounds.

### Why the RF didn't help

Residuals after IDW are mostly unpredictable noise *given the available
features*. With this feature set:
- `lat`, `lon` carry only 19 unique values — no per-cell signal beyond what
  IDW already encodes via sensor proximity.
- `humidity`, `wind_speed`, `wind_deg` are metro-wide scalars at inference,
  so they can't differentiate cells.
- `dist_to_highway_m` is per-sensor at training time (one value per sensor),
  collinear with lat/lon.
- `traffic_index` and the rush-hour flags are temporal, identical for every
  cell at a given timestamp.

The RF tries to fit anyway, picks up training-set quirks that don't transfer
to held-out sensors, and adds noise on average.

---

## Conclusion

**With the current feature set and 19 sensors, IDW alone outperforms ML
approaches.** The RF effort is closed for now. No model is loaded at runtime;
[ml/research/train_residual_rf.py](../research/train_residual_rf.py)
is preserved in the repo so the comparison is reproducible.

---

## What would need to change for a future ML attempt

Forward-looking only. None of these are next.

- **Per-cell wind from a regional weather model.** HRRR (NOAA's High-Resolution
  Rapid Refresh) provides ~3 km wind grids hourly over CONUS. Replacing the
  metro-mean wind scalar with a per-cell interpolated wind vector would give
  the RF actual spatial signal in the wind features. Wind history is required
  to retrain, so [ml/training/collect_training_data.py](../training/collect_training_data.py)
  would need an HRRR backfill.
- **Land-use features.** OpenStreetMap (via Overpass) road density,
  industrial-land-use proximity, building footprint density. These vary
  per cell and might help the RF learn local pollution patterns IDW can't
  pick up from sparse sensor coverage. [data/spatial/spatial_features.py](../../data/spatial/spatial_features.py)
  is the right place to add them.
- **More sensors.** 19 PurpleAir sensors over the DFW metro is sparse. With
  100+ sensors, the residuals after LOO IDW would have more spatial
  structure for the RF to learn.
- **OpenAQ as reference-grade anchors.** Reference-grade monitors are
  already pulled live in [data/ingestion/openaq.py](../../data/ingestion/openaq.py), but the historical
  pipeline only uses PurpleAir. Adding OpenAQ history to
  [ml/training/collect_training_data.py](../training/collect_training_data.py) would give the
  training set a calibration backbone independent of PurpleAir's correction
  formula.

---

## Audit-trail infrastructure stays

Independent of the RF outcome, the training data pipeline is kept:

- [ml/data/history.csv](../data/history.csv) — 6-month PurpleAir history with
  EPA correction, NOAA wind, spatial features.
- [ml/training/collect_training_data.py](../training/collect_training_data.py) — canonical
  one-shot training-data builder.
- [data/spatial/spatial_features.py](../../data/spatial/spatial_features.py) — distance-to-highway
  and other static spatial features per coordinate. Used by the training
  pipeline.

These remain useful as audit trail and as the substrate for any future ML
attempt that addresses the limitations above.
