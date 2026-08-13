# DFW Air Quality Dashboard — Algorithm Report

**Author:** Aarush Madhireddy
**Date:** May 12, 2026
**Project:** DFW Real-Time Air Quality Dashboard

---

## Table of Contents

1. [Congestion Scoring (Traffic Data)](#1-congestion-scoring-traffic-data)
2. [PM2.5 Data Cleaning & Classification](#2-pm25-data-cleaning--classification)
3. [Geographic Distance Correction](#3-geographic-distance-correction-cosine-correction)
4. [Inverse Distance Weighting (IDW) Interpolation](#4-inverse-distance-weighting-idw-interpolation)
5. [Traffic Exponential Weighting](#5-traffic-exponential-weighting)
6. [Traffic Distance Decay](#6-traffic-distance-decay)
7. [K-Nearest Traffic Blending](#7-k-nearest-traffic-blending)
8. [Wind Dispersal Factor](#8-wind-dispersal-factor)
9. [Wind Direction Factor](#9-wind-direction-factor)
10. [Weighted Circular Mean Bearing](#10-weighted-circular-mean-bearing)
11. [Final Grid Adjustment Equation](#11-final-grid-adjustment-equation)
12. [Gaussian Smoothing & Color Mapping](#12-gaussian-smoothing--color-mapping)
13. [Popup Grid Subsampling](#13-popup-grid-subsampling)
14. [Parameter Summary Table](#parameter-summary-table)

---

## 1. Congestion Scoring (Traffic Data)

**File:** `data/ingestion/traffic.py`, lines 26–34

**Formula:**

```
congestion = clamp(1 - (current_speed / free_flow_speed), 0, 1)
```

### For the Environmental Scientist

TomTom provides two speeds for each road segment: what cars are doing *right now* and what they'd do on a clear road. If cars are crawling at 15 mph on a road that normally flows at 60 mph, that's 75% congestion — and congested, slow-moving traffic means vehicles are idling longer and emitting more particulate matter per mile of road. A score of 0 means free-flowing traffic (minimal extra emissions), and 1.0 means a complete standstill (maximum emissions). This is sampled at 25 points (5x5 grid) across Dallas every 30 minutes.

### For the Programmer

Simple ratio inversion: `1 - (current / freeflow)`, clipped to `[0, 1]` via `np.clip`. The `free_flow_speed <= 0` guard returns 0.0 to handle edge cases where TomTom returns no road data for a point. The 5x5 sampling grid (`SAMPLE_GRID = 5`) produces 25 API calls per refresh; at the 30-min API cache TTL, max possible is 48 × 25 = 1,200 calls/day — half of TomTom's 2,500/day free tier.

---

## 2. PM2.5 Data Cleaning & Classification

**Files:** `data/ingestion/purpleair.py` — `fetch_sensors()` (cleaning + EPA correction) and `classify_pm25()` (classification)

**Cleaning rules:** Drop NaN, drop negative values, keep zero, then apply the EPA correction.

**EPA correction formula (PurpleAir only):**

```
PM2.5_corrected = 0.52 * PM2.5_raw - 0.085 * RH + 5.71
```

where `RH` is relative humidity in percent.

**Classification (EPA breakpoints):**

| Range (ug/m3) | Category |
|---|---|
| 0 - 12.0 | Good |
| 12.1 - 35.4 | Moderate |
| 35.5 - 55.4 | Sensitive Groups |
| 55.5 - 150.4 | Unhealthy |
| 150.5 - 250.4 | Very Unhealthy |
| 250.5+ | Hazardous |

### For the Environmental Scientist

**Dropping bad readings.** PurpleAir sensors go offline sometimes and return null — those aren't real readings, so we discard them. Negative PM2.5 values indicate a sensor malfunction (the laser particle counter can produce negative artifacts when humidity confuses it), so those are discarded too. However, a zero is kept because it genuinely means very clean air — PurpleAir returns null (not zero) for offline sensors, so zero is a real measurement.

**Why PurpleAir readings are corrected.** PurpleAir uses a low-cost laser particle counter, which systematically *overestimates* PM2.5 — especially at higher humidity. Water droplets suspended in humid air scatter the laser light and get counted as particles, inflating the reading. To make PurpleAir readings comparable to federal reference-grade monitors (the "real" PM2.5 that EPA standards and health guidance are built around), the EPA published a regression formula derived from years of co-location studies where PurpleAir sensors were placed next to regulatory monitors across the United States:

> `PM2.5_corrected = 0.52 * PM2.5_raw - 0.085 * RH + 5.71`

This formula is documented in the EPA's AirNow Fire and Smoke Map technical documentation and is the standard correction used in U.S. regulatory and public health contexts. Applying it at the data source means every downstream algorithm in this dashboard (IDW, traffic/wind grid adjustments, EPA AQI classification, color mapping) sees a PM2.5 value that genuinely matches what a reference-grade monitor would have read at the same location.

**What happens if humidity is missing.** If a given sensor didn't return humidity for a reading, the raw PurpleAir value is kept and the row is flagged with `epa_corrected = 0` so the audit trail is clear. The original uncorrected reading is always preserved in a `pm25_raw` column.

**OpenAQ readings are NOT corrected.** OpenAQ data comes from federal reference-grade monitors that are already calibrated — applying a PurpleAir correction to them would corrupt the data. A `source` column (`purpleair` vs. `openaq`) keeps the distinction auditable after the two datasets are concatenated.

**Classification.** The classification follows the EPA's standard PM2.5 breakpoints, the same thresholds used on AirNow.gov.

### For the Programmer

**Cleaning pipeline** (all in `data/ingestion/purpleair.py`):

1. `dropna(subset=["pm25"])` removes null rows.
2. Boolean filter `df["pm25"] >= 0` removes negatives.
3. `apply_epa_correction(df)` is called on the surviving rows *before* the DataFrame leaves the module.

**EPA correction implementation.** `apply_epa_correction()` copies `pm25` into `pm25_raw`, then for rows where `humidity` is not null computes `0.52 * pm25_raw - 0.085 * humidity + 5.71` and writes it back to the `pm25` column. Corrected values are clipped to `>= 0` to handle the small negatives the formula can produce at very low concentrations. Rows without humidity keep their raw value and are flagged with `epa_corrected = 0`. The PurpleAir API request in `fetch_sensors()` includes `humidity` in its `fields` list so the data is available to the correction step.

**OpenAQ handling.** `data/ingestion/openaq.py` returns data tagged with `source = "openaq"` and is never passed through `apply_epa_correction()`. For schema consistency with PurpleAir — so `pd.concat` in `app.py` produces a uniform frame — OpenAQ rows explicitly carry `pm25_raw = NaN` and `epa_corrected = 0`. The `source` column makes it possible to separate the two populations again for audit. NaN in `pm25_raw` is the canonical signal for "no laser-counter raw exists".

**Classification.** The `classify_pm25()` function is a simple if/elif cascade against EPA breakpoint constants. The 10-minute average field (`pm2.5_10minute`) is used rather than real-time to reduce noise.

**Calibration note.** `TRAFFIC_WEIGHT` (8.0 µg/m³), `WIND_WEIGHT` (10.0 µg/m³), and the classification breakpoints were all chosen against *reference-grade* PM2.5 levels reported in the literature, not PurpleAir-raw readings. Applying the EPA correction at the source aligns the live pipeline with those parameters without any recalibration.

**Training-pipeline parity.** The training-data builder (`ml/training/collect_training_data.py`) applies the same EPA PM2.5 correction formula

> `PM2.5_corrected = 0.52 * PM2.5_raw - 0.085 * RH + 5.71`

with the same `pm25` / `pm25_raw` / `epa_corrected` column convention used live, plus an additional A/B-channel disagreement filter (per EPA guidance: drop any historical hour where PurpleAir's two laser counters differ by more than 30%). The A/B filter is only available in the training pipeline because the historical API exposes both channels; the live `pm2.5_10minute` field is already an internal A/B average. **The EPA correction is implemented in two files** — `data/ingestion/purpleair.py` for the live pipeline and `ml/training/collect_training_data.py` for the training pipeline — and the two implementations must be edited in lockstep. A follow-up refactor will extract the formula into a shared `data/corrections.py` module.

---

## 3. Geographic Distance Correction (Cosine Correction)

**File:** `config.py`, lines 7–12 (`LON_CORRECTION` at line 12)

**Formula:**

```
LON_CORRECTION = cos(32.78 degrees) = 0.840

corrected_distance = sqrt(delta_lat^2 + (delta_lon * 0.840)^2)
```

### For the Environmental Scientist

The Earth is a sphere, but we're treating it as a flat grid for speed. The problem: at Dallas's latitude (32.78 degrees N), one degree of longitude is physically shorter than one degree of latitude — by about 16%. If we ignored this, the map would think east-west distances are larger than they really are, which would make the model overestimate how far east/west sensors are from a point, weakening their influence unfairly compared to north/south sensors. Multiplying every east-west distance by cos(32.78 degrees) = 0.84 corrects this distortion. It's not as precise as full spherical math (Haversine formula), but it's accurate enough for a metro-scale area and much faster.

### For the Programmer

A planar approximation with latitude-dependent longitude scaling. At latitude phi, 1 degree longitude = cos(phi) * 1 degree latitude in true distance. Rather than computing Haversine for 40,000+ grid cells x N sensors, we precompute `LON_CORRECTION = cos(radians(32.78))` once and multiply all delta-lon by it before squaring. This gives <1% error across the ~60 km Dallas bounding box — perfectly acceptable for IDW weighting. The constant is used in every distance calculation throughout the codebase: IDW, nearest-traffic-point lookup, wind direction bearings, etc.

---

## 4. Inverse Distance Weighting (IDW) Interpolation

**File:** `engine/interpolation.py`, `run_idw()` starting at line 43

**Formula:**

```
PM2.5(x) = SUM(w_i * PM2.5_i) / SUM(w_i)

where w_i = 1 / distance_i^3    (if distance_i <= 0.15 degrees)
      w_i = 0                    (if distance_i >  0.15 degrees)
```

**Parameters:**

- Power = 3 (steeper than the common default of 2)
- Search radius = 0.15 degrees (~15–17 km)
- Grid resolution = 60x60 = 3,600 cells (in production)
- Fallback for cells with no sensors in radius: global mean of all sensors

### For the Environmental Scientist

You have ~50–80 PurpleAir sensors scattered across Dallas, but you want PM2.5 estimates everywhere — including between sensors. IDW is the standard geostatistical workhorse for this: at any point on the map, it takes a weighted average of surrounding sensor readings, where closer sensors count much more than distant ones.

The "power = 3" controls how quickly a sensor's influence drops off. With power = 2 (the default in most GIS software), a sensor 10 km away still has noticeable pull. With power = 3, that same sensor's weight drops by an extra factor of 10 — so the estimate at any point is dominated by the nearest 2–3 sensors. This was chosen because PM2.5 can vary substantially over short distances (a sensor near a highway reads very differently from one in a park 2 km away), so we want the model to respect local conditions rather than averaging them away.

The 15–17 km search radius means sensors more than ~15 km away are completely ignored for a given point. Without this cutoff, a sensor in south Dallas would still have a tiny influence on north Dallas estimates, which makes no physical sense — PM2.5 plumes don't extend that far in a consistent way.

For areas at the edges of the bounding box where no sensor is within range, the model falls back to the average of all sensors rather than leaving a blank spot.

### For the Programmer

Fully vectorized NumPy implementation. The core is a 3D broadcast: grid points as `(res, res, 1)` vs sensors as `(1, 1, N)`, producing a `(res, res, N)` distance tensor. Weights are `1/distance^3` with an `np.where` mask zeroing out sensors beyond `IDW_SEARCH_RADIUS_DEG = 0.15`. Division uses a guarded `np.where(has_neighbours, weight_total, 1.0)` denominator to avoid NaN, with sparse cells falling back to `np.mean(sensor_pm25)`. Zero-distance guard sets `distances == 0` to `1e-10`. Note: `app.py` line 128 passes `grid_resolution=60` (not the config default of 200), producing a 60x60 grid of 3,600 cells for performance on each refresh.

---

## 5. Traffic Exponential Weighting

**File:** `engine/adjustments.py`, lines 29–39

**Formula:**

```
if congestion < 0.3:
    factor = 0

else:
    scaled = (congestion - 0.3) / (1.0 - 0.3)
    factor = (e^(3 * scaled) - 1) / (e^3 - 1)
```

**Parameters:**

- Threshold = 0.3 (30% congestion)
- k = 3.0 (exponential steepness)
- Output range: 0 to 1

### For the Environmental Scientist

Not all congestion is equal from an air quality perspective. Light traffic (below 30% congestion) has negligible impact on local PM2.5 — cars are moving, engines are efficient, and exhaust disperses. But as congestion climbs past that threshold, the effect on PM2.5 isn't linear — it's exponential. Stop-and-go traffic produces disproportionately more particulate matter because engines idle, accelerate, brake, repeat. A road at 90% congestion is far worse per vehicle than one at 50%.

The exponential curve captures this: below 30% congestion, the factor is zero (no adjustment). Above 30%, the factor grows slowly at first, then accelerates sharply as congestion approaches 100%. The specific curve `(e^(3s) - 1)/(e^3 - 1)` was chosen because it starts near zero, stays modest through moderate congestion (~0.15 at 50% congestion), but ramps to 1.0 at full gridlock. The constant k=3 controls this "elbow" — higher k would make the curve even more aggressive at the top end.

### For the Programmer

A normalized exponential: rescale congestion from `[0.3, 1.0]` to `[0, 1]` via `(c - 0.3) / 0.7`, then apply `(exp(k*s) - 1) / (exp(k) - 1)`. The denominator `exp(k) - 1` ensures the output is exactly 1.0 when `scaled = 1.0`. The threshold check short-circuits to 0.0 for light traffic. A vectorized version (`traffic_factor_vec`) uses `np.clip` and `np.where` for the full grid. The result is multiplied by `TRAFFIC_WEIGHT = 8.0` ug/m3 downstream, so the actual PM2.5 addition ranges from 0 to 8 ug/m3.

---

## 6. Traffic Distance Decay

**File:** `engine/adjustments.py`, lines 54–61

**Formula:**

```
distance_m = distance_degrees * 111,000
decay = max(0, 1 - distance_m / 500)
```

**Parameters:**

- Decay radius = 500 meters
- Degree-to-meter conversion: distance_m = distance_deg * 111,000

### For the Environmental Scientist

Traffic pollution doesn't just sit on the road — it drifts. But it also doesn't drift forever. Field studies (especially the EPA's near-road monitoring research) consistently show that traffic-related PM2.5 enhancement drops to background levels within about 300–500 meters of a major road. The dashboard uses 500 meters as the outer boundary.

The decay is linear: right next to the road, you get the full traffic effect. At 250 meters away, you get half. At 500+ meters, zero. This is a simplification — real near-road pollutant gradients are more exponential — but linear decay is a reasonable first approximation and avoids adding another tuning parameter.

### For the Programmer

Converts cosine-corrected degree distance to meters via `* 111_000` (approximate meters per degree latitude), then computes `max(0, 1 - dist_m / 500)`. This multiplier is applied element-wise to the traffic factor so that `traffic_adj = factor * decay * TRAFFIC_WEIGHT`. The 111,000 conversion is a simplification (actual value varies with latitude), but at 32.78 degrees N the error is <0.5%.

---

## 7. K-Nearest Traffic Blending

**File:** `engine/interpolation.py`, lines 221–239

**Formula:**

```
For each grid cell, find K=5 nearest traffic sample points, then:

weight_k = 1 / distance_k^2
blended_congestion = SUM(weight_k * congestion_k) / SUM(weight_k)
```

### For the Environmental Scientist

The dashboard samples traffic at 25 points across Dallas (a 5x5 grid). But the heatmap has 3,600 grid cells. For each heatmap cell, we need a congestion estimate, and using just the single nearest traffic point would create blocky artifacts — you'd see sharp jumps at the boundaries between traffic sample zones.

Instead, each cell looks at the 5 closest traffic points and takes a distance-weighted average of their congestion scores. Closer traffic points count more (weight = 1/distance squared). This produces a smooth congestion surface. The distance-decay multiplier (algorithm #6) is then applied based on the distance to the *nearest* of those 5 points — because what matters for air quality is how close you actually are to a road, not an average distance.

### For the Programmer

`np.argpartition(dists_deg, K-1, axis=1)[:, :K]` efficiently extracts the K=5 nearest indices per row without a full sort (O(N) vs O(N log N)). IDW weights `1/(dist^2 + 1e-10)` are normalized per row to sum to 1, then the weighted average congestion is `(k_w_norm * k_cong).sum(axis=1)`. The decay multiplier uses the minimum distance among K neighbors: `k_dists.argmin(axis=1)` converted to meters, then `np.clip(1 - dist_m/500, 0, 1)`.

---

## 8. Wind Dispersal Factor

**File:** `engine/adjustments.py`, lines 64–74

**Formula:**

```
dispersal = min(1.0, (wind_speed / 15.0) ^ 0.5)
```

**Parameters:**

- Wind speed cap = 15.0 m/s (~34 mph)
- Curve exponent = 0.5 (square root)
- Output range: 0 to 1

### For the Environmental Scientist

Wind is a powerful dispersal mechanism for PM2.5. Calm air lets pollution accumulate; strong wind blows it away (or brings it from somewhere else — that's handled by the direction factor). This formula converts wind speed into a 0-to-1 "dispersal strength" score.

The relationship uses a square-root curve rather than a linear one because atmospheric dispersion research shows that PM2.5 concentration drops sharply in the first few m/s of wind — light-to-moderate wind does most of the dispersal work, while additional wind speed has diminishing returns. Physically, the initial transition from calm to light wind breaks up stagnant air and begins turbulent mixing, which is the dominant dispersal mechanism; once that mixing is established, doubling the wind speed doesn't double the dispersal.

At 0 m/s (dead calm), dispersal is zero — no wind effect at all. At 3.75 m/s (~8 mph, a gentle breeze), dispersal is already 0.5 — half of maximum effect from just a quarter of maximum wind speed. At 7.5 m/s (~17 mph, a moderate breeze), dispersal is 0.71. At 15 m/s (~34 mph, a strong wind), dispersal maxes out at 1.0. Winds above 15 m/s don't increase the effect further because dispersal saturates once the air is already well-mixed.

The 15 m/s cap is conservative. It corresponds to roughly a sustained 30+ mph wind, which is relatively rare in Dallas outside of storm events.

### For the Programmer

`np.clip((wind_speed / WIND_SPEED_CAP) ** 0.5, 0.0, 1.0)`. Square-root scaling capped at 1.0. The `** 0.5` exponent produces a concave curve that front-loads dispersal at lower wind speeds compared to the previous linear version. This scalar is computed once per refresh cycle (wind is fetched as a single metro-wide reading from OWM) and broadcast across all grid cells. It modulates the maximum possible wind adjustment: `wind_adj = direction_factor * dispersal * WIND_WEIGHT`, where `WIND_WEIGHT = 3.0` ug/m3.

---

## 9. Wind Direction Factor

**File:** `engine/adjustments.py`, lines 77–119

**Formula:**

```
bearing       = atan2(delta_lon_corrected, delta_lat)    [traffic -> point]
wind_toward   = (wind_deg + 180) mod 360                 [convert FROM -> TOWARD]
alignment     = cos(bearing - wind_toward)
direction_factor = -alignment
```

**Output range:** -1.0 (transport: wind carries pollution toward you) to +1.0 (dispersal: wind carries it away)

### For the Environmental Scientist

This is the most nuanced algorithm in the system. It answers the question: *given where the nearest traffic source is and which way the wind is blowing, is the wind carrying that traffic's pollution toward you or away from you?*

Here's how it works step by step:

1. **Bearing:** Compute the compass direction from the nearest traffic point to the location you're evaluating. For example, if the traffic is due west of you, the bearing is 90 degrees (east).

2. **Wind direction conversion:** Weather data reports where wind comes *from* (a "west wind" comes from the west). We add 180 degrees to get the direction it's blowing *toward* (a west wind blows toward the east).

3. **Alignment:** Use the cosine of the angle between the bearing and the wind-toward direction. If they align perfectly (wind blows from traffic straight to you), cosine = +1. If they're opposite (wind blows pollution away from you), cosine = -1. Perpendicular = 0.

4. **Sign flip:** We negate the result so that +1 means dispersal (good — wind carries pollution away) and -1 means transport (bad — wind carries pollution toward you). This makes the downstream math cleaner.

**Real-world example:** Sensor at (32.80, -96.80). Traffic at (32.80, -96.81) — due west. Wind from west (270 degrees) blowing east. The wind is blowing from the traffic *toward* the sensor, so direction_factor = -1, and PM2.5 increases. This matches physical reality.

### For the Programmer

`atan2(delta_lon * LON_CORRECTION, delta_lat)` gives the bearing from traffic to point in radians, measured clockwise from north. OWM's `wind_deg` is the "from" direction, so `+180 mod 360` converts to "toward". `cos(bearing - wind_toward)` gives the alignment: +1 when co-directional, -1 when opposite. The negation ensures the sign convention `+1 = dispersal, -1 = transport` so that the final equation `pm25 -= dir_factor * disp * WIND_WEIGHT` works correctly (subtracting a negative adds PM2.5 for transport). Co-located points (distance < 1e-6) return 0.0 to avoid undefined `atan2(0,0)`.

---

## 10. Weighted Circular Mean Bearing

**File:** `engine/interpolation.py`, lines 278–283

**Formula:**

```
mean_sin        = SUM(weight_k * sin(bearing_k))
mean_cos        = SUM(weight_k * cos(bearing_k))
blended_bearing = atan2(mean_sin, mean_cos)
```

### For the Environmental Scientist

When computing the wind direction factor for each grid cell, we don't just use the single nearest traffic point — we blend the bearings from the 5 closest traffic points. But you can't simply average angles the way you average numbers, because angles wrap around (359 degrees and 1 degree are almost the same direction, but their arithmetic mean is 180 degrees — completely wrong).

The circular mean solves this by decomposing each bearing into its sine and cosine components, averaging those separately, and then recombining with atan2. This is the same technique used in meteorology to average wind directions from multiple weather stations. The weights are the same IDW weights (1/distance squared) from the traffic blending, so closer traffic points dominate the bearing estimate.

### For the Programmer

Standard circular/angular mean via Cartesian decomposition. Each bearing theta_k is decomposed to `(sin(theta_k), cos(theta_k))`, IDW-weighted, summed, then recombined with `np.arctan2(mean_sin, mean_cos)`. This avoids the 360/0 degree wrap-around artifact that breaks naive arithmetic mean. Uses the same `k_w_norm` weights as the congestion blending. The result feeds into `np.cos(blended_bearing - wind_toward_rad)` for the direction factor.

---

## 11. Final Grid Adjustment Equation

**File:** `engine/interpolation.py`, lines 295–299

**Formula:**

```
adjusted_PM2.5 = IDW_estimate + traffic_adj - wind_adj
adjusted_PM2.5 = max(0, adjusted_PM2.5)
```

Where:

- `traffic_adj = traffic_factor * decay * 8.0 ug/m3` (always >= 0)
- `wind_adj = direction_factor * dispersal * 3.0 ug/m3` (can be negative)

### For the Environmental Scientist

This is where everything comes together. Each cell on the heatmap starts with its IDW-interpolated PM2.5 (based purely on nearby sensors), then gets two corrections:

1. **Traffic addition** (0 to +8 ug/m3): If the cell is near a congested road, PM2.5 goes up. The maximum addition is 8 ug/m3, which matches field studies showing typical near-road PM2.5 enhancement of 5–10 ug/m3 even on the busiest highways. This is conservative — some studies report up to 15 ug/m3 near truck-heavy corridors.

2. **Wind adjustment** (-3 to +3 ug/m3): This can go either way. If wind is blowing pollution toward the cell from a traffic source, PM2.5 goes up (by up to +3). If wind is blowing pollution away, PM2.5 goes down (by up to -3). The sign logic works because `wind_adj` is negative when wind transports pollution toward you, so subtracting it *adds* to PM2.5. (The cap was lowered from an earlier 10 ug/m3 — wind adjustment is intentionally a smaller correction than the traffic term.)

The final clamp to >= 0 prevents the model from producing physically impossible negative PM2.5 values, which could happen if wind dispersal subtracts more than the IDW estimate.

**Why adjust the grid and not the sensors?** The sensors already measure the real world — they're already feeling the effects of traffic and wind at their physical locations. Adjusting sensor readings would double-count those effects. The adjustments are only needed for the interpolated points *between* sensors, where IDW has no knowledge of roads or wind.

### For the Programmer

Element-wise addition and subtraction on flattened `(N,)` arrays, then `np.clip(adjusted, 0.0, None)` and reshape back to `(res, res)`. The key design decision is that adjustments are post-IDW only — `build_features()` computes the same traffic/wind columns per sensor but stores them as metadata (for ML training data in `history.csv`) without modifying `pm25`. This avoids the double-counting problem documented in the `engine/features.py` header comments.

---

## 12. Gaussian Smoothing & Color Mapping

**File:** `viz/heatmap.py`, lines 128–180 (`_add_idw_overlay`)

**Smoothing:**

```
smoothed = gaussian_filter(values, sigma=1.5)
```

**Color mapping:** Linear interpolation in RGB space between 6 EPA breakpoint colors, normalized to [0, 250.4] ug/m3

**Color scale:**

| PM2.5 (ug/m3) | Hex Color | Category |
|---|---|---|
| 0 | #00e400 (green) | Good |
| 12 | #ffff00 (yellow) | Moderate |
| 35.4 | #ff7e00 (orange) | Sensitive |
| 55.4 | #ff0000 (red) | Unhealthy |
| 150.4 | #8f3f97 (purple) | Very Unhealthy |
| 250.4 | #7e0023 (dark red) | Hazardous |

### For the Environmental Scientist

Before coloring the heatmap, the raw PM2.5 grid gets a gentle Gaussian blur (sigma = 1.5 cells). This is done on purpose: if you color first and then blend, you get muddy brown artifacts where green meets yellow or orange meets red — because mixing two meaningful colors in RGB space produces a meaningless intermediate. By smoothing the *numerical values* first and then coloring, the transitions are clean and the color at every point accurately represents its (smoothed) PM2.5 level.

The color scale follows EPA conventions: green (good) through yellow (moderate) through orange (sensitive groups) through red (unhealthy) through purple (very unhealthy) to dark red (hazardous). Between breakpoints, colors are linearly interpolated channel by channel. The heatmap is rendered at 35% opacity so the street map shows through underneath.

### For the Programmer

`scipy.ndimage.gaussian_filter` with `sigma=1.5` in grid-cell units. A matplotlib `LinearSegmentedColormap` is built from 6 `(value/250.4, hex_color)` stops and a `Normalize(vmin=0, vmax=250.4)` scaler. The pipeline: smooth, colormap, set `rgba[:,:,3] = 0.35`, `np.flipud` (PNG row 0 = north), `plt.imsave` to BytesIO, base64 encode, Folium `ImageOverlay` with `opacity=1.0` (alpha already baked in). The `_pm25_to_hex()` function provides a scalar version for individual lookups but isn't used in the grid path.

---

## 13. Popup Grid Subsampling

**File:** `viz/heatmap.py`, lines 188–225

**Formula:**

```
step = max(1, grid_size // 30)    along each axis
Result: ~900 popup rectangles instead of 3,600
```

### For the Environmental Scientist

When you click on the heatmap, a popup shows the zip code, PM2.5 value, and AQI category for that location. But creating a clickable popup for every single grid cell (3,600 cells) would make the map sluggish. Instead, the map creates ~900 invisible rectangles evenly spaced across the grid. Each rectangle covers a small neighborhood and uses the *unsmoothed* PM2.5 value (for accuracy — the smoothed values are only for visual rendering). Zip codes are reverse-geocoded using a local database and cached for performance.

### For the Programmer

`POPUP_GRID_SIZE = 30` produces steps of `max(1, shape // 30)` along each axis. Each subsampled cell becomes an invisible `folium.Rectangle` (`fill_opacity=0.0`) carrying a `Popup` and `Tooltip`. `values[i, j]` uses the pre-adjustment unsmoothed grid, not the Gaussian-filtered one. `_coords_to_zip()` uses `uszipcode.SearchEngine` with an `@lru_cache(maxsize=2048)` on coordinates rounded to 2 decimal places (~1.1 km precision) to maximize cache hits.

---

## Parameter Summary Table

| Parameter | Value | File | Rationale |
|---|---|---|---|
| `LON_CORRECTION` | cos(32.78) = 0.840 | config.py:12 | Corrects ~16% east-west overstatement at Dallas latitude |
| `GRID_RESOLUTION` (default) | 200 | config.py:63 | Config default; overridden at runtime |
| `GRID_RESOLUTION` (runtime) | 60 | app.py:128 | 3,600 cells; balances detail vs. refresh speed |
| `IDW_POWER` | 3 | config.py:68 | Steeper than default 2; nearby sensors dominate local estimates |
| `IDW_SEARCH_RADIUS_DEG` | 0.15 deg (~17 km) | config.py:72 | Prevents distant sensors from smearing local variation |
| `TRAFFIC_WEIGHT` | 8.0 ug/m3 | config.py:77 | Matches EPA near-road studies (5-10 ug/m3 typical) |
| `TRAFFIC_DECAY_RADIUS_M` | 500 m | config.py:89 | Based on near-road gradient studies |
| `TRAFFIC_THRESHOLD` | 0.3 | adjustments.py:21 | Below 30% congestion, traffic effect is negligible |
| `TRAFFIC_CURVE_K` | 3.0 | adjustments.py:22 | Controls exponential steepness above threshold |
| `WIND_WEIGHT` | 3.0 ug/m3 | adjustments.py:17 | Max dispersal/transport effect of strong aligned wind (intentionally small relative to TRAFFIC_WEIGHT) |
| `WIND_SPEED_CAP` | 15.0 m/s | adjustments.py:18 | Beyond this, dispersal saturates |
| `Gaussian sigma` | 1.5 cells | heatmap.py:153 | Gentle smoothing to avoid color banding artifacts |
| `Heatmap opacity` | 0.35 (35%) | heatmap.py:158 | Basemap shows through |
| `K (traffic neighbors)` | 5 | interpolation.py:221 | Smooth congestion blending without artifacts |

---

## Data Flow Summary

`fetch_sensors()` returns `[sensor_id, name, lat, lon, pm25 (EPA-corrected), pm25_raw, epa_corrected, source]`. `fetch_openaq()` returns the same columns with `pm25_raw = NaN` and `epa_corrected = 0` so `pd.concat` produces a uniform frame.

```
PurpleAir API ──> fetch_sensors() ──┐
        (raw pm25 → apply_epa_correction → pm25, pm25_raw, epa_corrected)
                                    ├──> pd.concat ──> build_features() ──> run_idw()
OpenAQ API ─────> fetch_openaq() ──┘                        |                  |
        (reference-grade; pm25_raw=NaN, epa_corrected=0)    |                  |
                                                            |                  v
TomTom API ─────> fetch_traffic() ─────────────────────────>|           IDW grid (60x60)
                                                            |                  |
OWM API ────────> fetch_wind() ────────────────────────────>|                  v
                                                                       adjust_grid()
                                                                            |
                                                                            v
                                                                  Adjusted PM2.5 grid
                                                                            |
                                                                            v
                                                              gaussian_filter(sigma=1.5)
                                                                            |
                                                                            v
                                                              Colormap + ImageOverlay
                                                                            |
                                                                            v
                                                                    Folium Map + Popups
```
