# DFW Air Quality Dashboard — Algorithms Reference

**Author:** Aarush Madhireddy
**Date:** April 30, 2026
**Scope:** A focused walkthrough of the algorithms that turn raw sensor + traffic + wind data into the live PM2.5 heatmap. The IDW interpolation and the post-IDW traffic and wind adjustments are covered in depth; the rest of the pipeline is summarised.

A longer companion report exists at [ml/docs/DFW_Algorithm_Report.md](ml/docs/DFW_Algorithm_Report.md). Where the two disagree on a numeric value (`WIND_WEIGHT` in particular), this document reflects the current code.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Foundations (brief)](#2-foundations-brief)
3. [IDW interpolation (deep)](#3-idw-interpolation-deep)
4. [Traffic adjustments (deep)](#4-traffic-adjustments-deep)
5. [Wind adjustments (deep)](#5-wind-adjustments-deep)
6. [Final grid adjustment equation (deep)](#6-final-grid-adjustment-equation-deep)
7. [Why post-IDW only? (deep)](#7-why-post-idw-only-deep)
8. [Rendering pipeline (brief)](#8-rendering-pipeline-brief)
9. [Spatial feature (brief)](#9-spatial-feature-brief)
10. [Parameter summary table](#10-parameter-summary-table)

---

## 1. Pipeline overview

Every refresh pulls from four sources, fuses them into a single PM2.5 surface, and renders it as a translucent raster overlay with click popups.

```
PurpleAir API ──> fetch_sensors() ──┐
        (raw pm25 → apply_epa_correction → pm25, pm25_raw, epa_corrected)
                                    ├──> pd.concat ──> build_features() ──> run_idw()
OpenAQ API ─────> fetch_openaq() ──┘     (per-sensor metadata,    │             │
        (reference-grade; pm25_raw=NaN,   pm25 untouched)         │             │
         epa_corrected=0)                                          │             ▼
                                                                   │       IDW grid
TomTom API ────> fetch_traffic() ─────────────────────────────────>│      (60 × 60)
                                                                   │             │
OWM API ──────> fetch_wind() ────────────────────────────────────>│             ▼
                                                                          adjust_grid()
                                                                                │
                                                                                ▼
                                                                     Adjusted PM2.5 grid
                                                                                │
                                                                                ▼
                                                                  gaussian_filter(σ = 1.5)
                                                                                │
                                                                                ▼
                                                                  Colormap + ImageOverlay
                                                                                │
                                                                                ▼
                                                                       Folium map + popups
```

**Key design principle.** Sensors are never adjusted. They already measure the real-world effects of traffic and wind at their physical locations, so any adjustment on top would double-count. The traffic and wind corrections are applied *only* to interpolated grid cells, where IDW has no road or wind context. Section 7 explains this in more detail.

---

## 2. Foundations (brief)

### Cosine-corrected planar distance

[config.py:7-12](config.py#L7-L12)

```
LON_CORRECTION = cos(32.78°) ≈ 0.840
corrected_distance = sqrt(Δlat² + (Δlon × LON_CORRECTION)²)
```

The Earth is curved, but for a metro-scale area we treat it as flat. At Dallas's latitude one degree of longitude covers only ~84% the true distance of one degree of latitude — a ~16% east-west distortion. Multiplying every longitude delta by `LON_CORRECTION` before squaring keeps the metric approximately isotropic without paying for full Haversine math. The same constant is reused in IDW ([engine/interpolation.py:64](engine/interpolation.py#L64)), the traffic blending ([engine/interpolation.py:144](engine/interpolation.py#L144)), the wind direction factor ([engine/adjustments.py:99](engine/adjustments.py#L99)), and the per-sensor feature builder.

### PM2.5 cleaning + EPA correction (PurpleAir only)

[data/ingestion/purpleair.py:26-66](data/ingestion/purpleair.py#L26-L66)

```
PM2.5_corrected = 0.52 * PM2.5_raw - 0.085 * RH + 5.71
```

**Input: `pm2.5_cf_1` channel (not ATM).** The Barkjohn 2021 formula was derived from CF=1 co-location data. Using ATM as input overcorrects at moderate concentrations and diverges significantly at PM2.5 > 50 µg/m³.

PurpleAir uses a low-cost laser particle counter that systematically overestimates PM2.5, especially when humidity is high (water droplets scatter the laser and get counted as particles). The EPA's regression formula, derived from years of co-location with reference-grade monitors, is the standard correction in U.S. regulatory and public-health contexts.

The cleaning pipeline:

1. `dropna(subset=["pm25"])` removes offline-sensor rows (PurpleAir returns null, not zero, when a sensor is down).
2. `df[df["pm25"] >= 0]` drops negative readings (laser malfunction artefacts).
3. `apply_epa_correction()` writes the corrected value back into `pm25`, preserves the original in `pm25_raw`, and sets `epa_corrected = 1` when humidity was available (0 otherwise). Corrected values are clipped to ≥ 0 because the formula can produce small negatives at very low concentrations.

OpenAQ data is *not* corrected — it comes from federal reference-grade monitors that are already calibrated. OpenAQ rows carry `pm25_raw = NaN` and `epa_corrected = 0` for schema consistency after `pd.concat`, and a `source` column (`purpleair` / `openaq`) keeps the two populations auditable.

The same EPA formula is implemented a second time in [ml/training/collect_training_data.py](ml/training/collect_training_data.py) (the training script can't import from the live module without booting the live API). Both files must be edited in lockstep until the planned `data/corrections.py` extraction lands.

---

## 3. IDW interpolation (deep)

**File:** [engine/interpolation.py:29-89](engine/interpolation.py#L29-L89)
**Constants:** [config.py:60-67](config.py#L60-L67)

### Formula

```
PM2.5(grid_cell) = Σ(w_i × PM2.5_i) / Σ(w_i)

where:
  w_i = 1 / distance_i^IDW_POWER     if distance_i ≤ IDW_SEARCH_RADIUS_DEG
  w_i = 0                            otherwise

Fallback (no sensors in radius):
  PM2.5(grid_cell) = mean(all sensor PM2.5)
```

Distances are cosine-corrected planar Euclidean (Section 2). When a grid point happens to sit exactly on a sensor, `distances == 0` is replaced with `1e-10` to avoid division by zero ([engine/interpolation.py:68](engine/interpolation.py#L68)).

### Parameters

| Constant | Value | Source |
|---|---|---|
| `IDW_POWER` | 3 | [config.py:63](config.py#L63) |
| `IDW_SEARCH_RADIUS_DEG` | 0.15° (~15–17 km at Dallas latitude) | [config.py:67](config.py#L67) |
| `GRID_RESOLUTION` | 200 (config default), 60 (passed by `app.py` at runtime) | [config.py:58](config.py#L58) |

### For the Environmental Scientist

You have ~50–80 PurpleAir sensors plus a handful of OpenAQ reference monitors scattered across Dallas, but you want PM2.5 estimates *everywhere* — including between sensors. IDW is the standard geostatistical workhorse for that: at any point on the map, take a weighted average of nearby sensor readings, where closer sensors count much more than distant ones.

**Why power = 3 instead of the usual 2.** The exponent controls how quickly a sensor's influence drops off. With power = 2 (the default in most GIS software), a sensor 10 km away still has a noticeable pull. With power = 3, that same sensor's weight drops by another factor of 10, so the estimate at any point is dominated by its nearest 2–3 sensors. PM2.5 can vary a lot over short distances (a sensor near a highway reads very differently from one in a park 2 km away), so we want the model to respect local conditions rather than smoothing them away.

**Why a 15 km cutoff.** Sensors more than ~15 km away are completely ignored for a given point. Without this hard cutoff, a sensor in south Dallas would still have a tiny influence on north Dallas estimates — which makes no physical sense, since PM2.5 plumes don't extend that far in any consistent way. The cutoff also keeps the math honest: distant sensors contribute nothing, not "almost nothing".

**Edge fallback.** At the edges of the bounding box, some grid cells have no sensors within 15 km. Rather than render a NaN hole, the model falls back to the unweighted mean of all sensors. The cell will be coloured, but it's a weak estimate — fine for the heatmap, not load-bearing.

### For the Programmer

Fully vectorised NumPy implementation built around a 3D broadcast.

```
grid points  → shape (res, res, 1)
sensors      → shape (1, 1, N)
distances    → shape (res, res, N)
```

Weights are `1 / distance**IDW_POWER`, then `np.where(in_radius, weights, 0.0)` masks out-of-radius sensors. The denominator uses a guarded `np.where(has_neighbours, weight_total, 1.0)` to avoid NaN, with sparse cells filled in via `np.mean(sensor_pm25)` (the global mean fallback).

```python
in_radius      = distances <= IDW_SEARCH_RADIUS_DEG
weights        = 1.0 / (distances ** IDW_POWER)
weights        = np.where(in_radius, weights, 0.0)
weight_total   = weights.sum(axis=2)
has_neighbours = weight_total > 0
weighted_sum   = (weights * sensor_pm25[None, None, :]).sum(axis=2)
idw_estimate   = np.where(has_neighbours,
                          weighted_sum / np.where(has_neighbours, weight_total, 1.0),
                          global_mean)
```

`app.py` passes `grid_resolution=60` rather than the `GRID_RESOLUTION=200` config default — 3,600 cells is plenty of detail for the rendered overlay and keeps each refresh fast.

---

## 4. Traffic adjustments (deep)

The traffic pipeline has four stages: score raw congestion → blend a smooth surface from sparse sample points → convert congestion into a PM2.5 multiplier → fade with distance from the road.

### 4a. Congestion scoring

[data/ingestion/traffic.py:26-34](data/ingestion/traffic.py#L26-L34)

```
if free_flow_speed ≤ 0:
    congestion = 0
else:
    congestion = clamp(1 - current_speed / free_flow_speed, 0, 1)
```

TomTom returns two speeds for each segment: what cars are doing right now and what they'd do on a clear road. Crawling at 15 mph on a road that normally flows at 60 mph is 75% congestion. A score of 0 = free flow, 1 = full standstill. Sampling happens at 25 points (5 × 5 grid) across the bbox each refresh; at the 30-min API cache TTL that caps daily usage at 1,200 calls — half of TomTom's 2,500/day free tier.

### 4b. Exponential weighting

**File:** [engine/adjustments.py:29-39](engine/adjustments.py#L29-L39) (scalar), [engine/adjustments.py:122-135](engine/adjustments.py#L122-L135) (vectorised)
**Constants:** [engine/adjustments.py:21-22](engine/adjustments.py#L21-L22)

```
if congestion < TRAFFIC_THRESHOLD:    # 0.3
    factor = 0
else:
    s = (congestion - 0.3) / 0.7
    factor = (e^(3·s) - 1) / (e^3 - 1)
```

Output range: `[0, 1]`. Multiplied by `TRAFFIC_WEIGHT = 8.0 µg/m³` downstream.

#### For the Environmental Scientist

Not all congestion is equal from an air-quality perspective. Below ~30% congestion, traffic has negligible impact on local PM2.5: cars are moving, engines are running efficiently, exhaust disperses. Above 30%, the effect grows non-linearly — stop-and-go traffic produces disproportionately more particulate matter because engines idle, accelerate, brake, repeat. A road at 90% congestion is far worse per vehicle than one at 50%.

The exponential curve captures this. Below threshold: zero. Above threshold: starts modest (`≈ 0.15` at 50% congestion), then ramps sharply to `1.0` at full gridlock. The constant `k = 3` controls the elbow — higher would make the top-end ramp even more aggressive.

#### For the Programmer

Two-step normalisation:

1. Rescale `congestion` from `[0.3, 1.0]` to `[0, 1]` via `(c − 0.3) / 0.7`.
2. Apply the canonical normalised exponential `(exp(k·s) − 1) / (exp(k) − 1)`. The denominator `exp(k) − 1 ≈ 19.09` ensures the output is exactly `1.0` when `s = 1.0`.

`traffic_factor_vec` does the same thing across an entire NumPy array using `np.clip` and `np.where(below, 0.0, factor)` to handle the threshold branch in vectorised form. The grid-side caller is in [engine/interpolation.py:170](engine/interpolation.py#L170).

### 4c. Distance decay

[engine/adjustments.py:54-61](engine/adjustments.py#L54-L61), grid-side at [engine/interpolation.py:171-172](engine/interpolation.py#L171-L172)

```
distance_m = distance_deg × 111,000
decay      = max(0, 1 − distance_m / TRAFFIC_DECAY_RADIUS_M)   # TRAFFIC_DECAY_RADIUS_M = 500 m
```

#### For the Environmental Scientist

Traffic pollution drifts, but it doesn't drift forever. EPA near-road monitoring research consistently shows that traffic-related PM2.5 enhancement falls to background levels within about 300–500 m of a major road. We use 500 m as the outer boundary: full effect at 0 m, half at 250 m, zero past 500 m.

The decay is linear. Real-world near-road gradients are closer to exponential, but linear is a reasonable first approximation and avoids introducing another tuning parameter. The simplification matters less than the threshold + exponential weighting upstream.

#### For the Programmer

Degree-to-metres conversion uses `distance_m = distance_deg × 111_000` (the average length of one degree of latitude on Earth). At Dallas's latitude this approximation is accurate to under 0.5%. The grid-side version uses `np.clip(1 - dist_m / 500, 0, 1)` element-wise across the flattened cell array.

### 4d. K-nearest traffic blending

[engine/interpolation.py:147-167](engine/interpolation.py#L147-L167)

```
For each grid cell:
  1. Find K = 5 nearest traffic sample points via np.argpartition.
  2. IDW weights:           w_k = 1 / (distance_k² + ε)
  3. Normalise:             w_k_norm = w_k / Σ w_k
  4. Blended congestion:    Σ(w_k_norm × congestion_k)
  5. Decay distance:        nearest of the 5 neighbours (not the average)
```

#### For the Environmental Scientist

The traffic data is sparse — only 25 sample points across the whole metro — but the heatmap has 3,600 cells. If each cell snapped to its single nearest traffic point, you'd see blocky Voronoi-cell artefacts: sharp jumps in congestion at the boundaries between sample zones.

Blending the 5 closest traffic points by inverse-square distance smooths that surface. For the *decay* multiplier, we still use the distance to the very nearest neighbour rather than an average — what matters for air quality is how close you actually are to a road, not some smoothed-out distance.

#### For the Programmer

`np.argpartition(dists_deg, K-1, axis=1)[:, :K]` extracts the K smallest distances per row in O(N) instead of O(N log N) — meaningful at 3,600 × 64. Note this is a different IDW exponent (2) than the sensor IDW (3): the goal here is *smoothing* a sparse grid, not punishing distance, so a gentler curve preserves more local structure. Weights are normalised per row to sum to 1.

```python
K          = min(5, T)
k_part     = np.argpartition(dists_deg, K - 1, axis=1)[:, :K]   # (N, K) indices
k_dists    = dists_deg[np.arange(N)[:, None], k_part]            # (N, K)
k_cong     = t_cong[k_part]                                      # (N, K)
k_w        = 1.0 / (k_dists ** 2 + 1e-10)
k_w_norm   = k_w / k_w.sum(axis=1, keepdims=True)
blended    = (k_w_norm * k_cong).sum(axis=1)                     # (N,)
nearest_in_k     = k_dists.argmin(axis=1)
nearest_dist_deg = k_dists[np.arange(N), nearest_in_k]
```

The same `k_w_norm` weights are reused for the wind direction blending in Section 5c — once they're computed, both the congestion average and the bearing average can ride them.

---

## 5. Wind adjustments (deep)

Wind has two effects in the model: a **speed-driven dispersal magnitude** and a **direction-driven sign**. Strong wind disperses pollution if it's blowing the right way, transports it if it's blowing the wrong way. The two terms multiply.

### 5a. Wind dispersal factor

[engine/adjustments.py:64-74](engine/adjustments.py#L64-L74)

```
dispersal = min(1, sqrt(wind_speed / WIND_SPEED_CAP))    # WIND_SPEED_CAP = 15 m/s
```

#### For the Environmental Scientist

Wind disperses PM2.5 — calm air lets pollution accumulate, strong wind blows it away (or brings it from somewhere else; that's handled by the direction factor in 5b). This formula converts wind speed into a 0-to-1 "dispersal strength" multiplier.

The square-root curve front-loads dispersal at low wind speeds. Atmospheric dispersion research shows PM2.5 concentration drops sharply in the first few m/s of wind: light-to-moderate wind does most of the work because that initial transition from calm to breeze breaks up stagnant air and starts turbulent mixing. Once mixing is established, doubling the wind speed doesn't double the dispersal.

| wind speed | dispersal | rough description |
|---|---|---|
| 0 m/s | 0.00 | dead calm |
| 3.75 m/s (~8 mph) | 0.50 | gentle breeze, half effect |
| 7.5 m/s (~17 mph) | 0.71 | moderate breeze |
| 15 m/s (~34 mph) | 1.00 | strong wind, max effect |
| > 15 m/s | 1.00 | saturates |

The 15 m/s cap is conservative — sustained 30+ mph winds are uncommon in Dallas outside storm events.

#### For the Programmer

`np.clip((wind_speed / WIND_SPEED_CAP) ** 0.5, 0.0, 1.0)`. Computed once per refresh from the metro-wide OpenWeatherMap reading and broadcast across all 3,600 cells. Modulates the maximum possible wind adjustment: `wind_adj = direction_factor × dispersal × WIND_WEIGHT` with `WIND_WEIGHT = 3.0 µg/m³` ([engine/adjustments.py:17](engine/adjustments.py#L17)).

### 5b. Wind direction factor

[engine/adjustments.py:77-115](engine/adjustments.py#L77-L115) (scalar), [engine/adjustments.py:138-175](engine/adjustments.py#L138-L175) (vectorised)

```
Δlat            = point_lat - traffic_lat
Δlon            = (point_lon - traffic_lon) × LON_CORRECTION
bearing_rad     = atan2(Δlon, Δlat)                # traffic → point, clockwise from north
wind_toward_rad = deg2rad((wind_deg + 180) mod 360)
alignment       = cos(bearing_rad - wind_toward_rad)
direction_factor = -alignment                       # +1 dispersal, -1 transport
```

#### For the Environmental Scientist

This is the most nuanced piece of math in the system. It answers: *given where the nearest traffic source is and which way the wind is blowing, is the wind carrying that traffic's pollution toward you, or away from you?*

Step by step:

1. **Bearing from traffic to point.** Compute the compass direction from the nearest traffic source to the cell. If the traffic is due west of you, the bearing is 90° (east).

2. **Convert "wind from" to "wind toward".** Weather services (OWM included) report where wind comes *from* — a "west wind" comes from the west. Adding 180° (with wraparound) converts that to where the wind is *blowing toward*.

3. **Cosine alignment.** If the bearing and the wind-toward direction line up, `cos(0) = +1` — the wind is blowing pollution from the traffic source straight toward you. If they're opposite, `cos(180°) = -1` — the wind is carrying it away. Perpendicular = 0.

4. **Sign flip.** We negate the result so `+1` means dispersal (good) and `-1` means transport (bad). This makes the downstream equation read more naturally (Section 6).

**Worked example.** Point at (32.80, −96.80). Nearest traffic at (32.80, −96.81), due west of the point. Wind reported as 270° — coming *from* the west, blowing east. Adding 180° gives a wind-toward direction of 90° (east). The bearing from traffic to point is also 90° (east). Cosine of the difference is 1. After the sign flip, `direction_factor = -1` — wind is transporting pollution toward the point, so PM2.5 should go up. The `__main__` verification block in [engine/features.py:157-173](engine/features.py#L157-L173) actually runs this exact case as a sanity check.

#### For the Programmer

Note the argument order in `np.arctan2(delta_lon, delta_lat)` — this gives the bearing measured clockwise from north (compass convention), not the standard mathematical angle measured counter-clockwise from east. Co-located cells (`distance < 1e-6` deg) return `0.0` to avoid `atan2(0, 0)`.

The vectorised version (`wind_direction_factor_vec`) takes flattened cell arrays and a single `nearest_idx` array. The grid-side `adjust_grid` doesn't actually call this function — it inlines the same math but uses the K-blended bearing from Section 5c instead of a single nearest-point bearing.

### 5c. Weighted circular mean bearing

[engine/interpolation.py:193-198](engine/interpolation.py#L193-L198)

```
mean_sin        = Σ(w_k_norm × sin(bearing_k))
mean_cos        = Σ(w_k_norm × cos(bearing_k))
blended_bearing = atan2(mean_sin, mean_cos)
```

#### For the Environmental Scientist

When the grid uses K = 5 nearest traffic points instead of just one, we have to blend their bearings — but you can't simply average angles. 359° and 1° point in nearly the same direction, but their arithmetic mean is 180° (the exact opposite), which is wrong.

The circular mean fixes this by decomposing each bearing into its sine and cosine components, averaging those separately (with the same IDW weights from Section 4d), and recombining via `atan2`. This is the standard meteorological technique for averaging wind directions across multiple stations.

#### For the Programmer

Standard angular mean via Cartesian decomposition. The `k_w_norm` weights are the same `(N, K)` matrix from the congestion blending — closer traffic points dominate both the congestion *and* the bearing estimate. The result feeds into `np.cos(blended_bearing - wind_toward_rad)` for the per-cell direction factor.

---

## 6. Final grid adjustment equation (deep)

**File:** [engine/interpolation.py:169-217](engine/interpolation.py#L169-L217)

```
traffic_adj = traffic_factor × decay × TRAFFIC_WEIGHT     # 0 .. +8 µg/m³
wind_adj    = direction_factor × dispersal × WIND_WEIGHT  # -3 .. +3 µg/m³

adjusted    = idw_estimate + traffic_adj − wind_adj
adjusted    = max(0, adjusted)
```

`TRAFFIC_WEIGHT = 8.0 µg/m³` ([config.py:72](config.py#L72)) and `WIND_WEIGHT = 3.0 µg/m³` ([engine/adjustments.py:17](engine/adjustments.py#L17)).

> The companion report at [ml/docs/DFW_Algorithm_Report.md](ml/docs/DFW_Algorithm_Report.md) cites `WIND_WEIGHT = 10.0` — that is stale. The recent "wind weighting adjustment" commit dropped the constant to `3.0` after the old value was producing too-aggressive dispersal swings. Treat `3.0` as authoritative.

### For the Environmental Scientist

This is where everything composes. Each cell starts with its IDW PM2.5 (purely a function of nearby sensors), then gets two corrections:

- **Traffic addition (0 to +8 µg/m³).** If the cell sits near a congested road, PM2.5 goes up. The 8 µg/m³ ceiling lines up with field studies showing 5–10 µg/m³ near-road enhancement on busy corridors.

- **Wind adjustment (−3 to +3 µg/m³).** Signed. If the wind is carrying pollution from a traffic source toward this cell, PM2.5 goes up. If it's carrying it away, PM2.5 goes down. Maximum swing is ±3 µg/m³.

The final clamp to ≥ 0 prevents physically impossible negative PM2.5 if dispersal subtracts more than the IDW estimate.

### For the Programmer — sign convention walk-through

The equation is `adjusted = idw + traffic_adj − wind_adj`. The minus sign in front of `wind_adj` plus the negation inside `direction_factor` are what make the downstream math read cleanly:

| `direction_factor` | meaning | `wind_adj` sign | `idw − wind_adj` |
|---|---|---|---|
| +1 | dispersal | positive | decreases (subtract a positive) |
| −1 | transport | negative | increases (subtract a negative) |
|  0 | perpendicular | zero | unchanged |

Element-wise on `(N,)` flattened arrays, then `np.clip(adjusted, 0.0, None)` and `reshape((res, res))`.

---

## 7. Why post-IDW only? (deep)

[engine/features.py:20-41](engine/features.py#L20-L41), [engine/interpolation.py:9-12](engine/interpolation.py#L9-L12)

The single most important design decision in the pipeline: **traffic and wind adjustments are applied to interpolated grid cells, never to sensor readings.**

A sensor sitting next to the freeway already feels the freeway. Its raw PM2.5 reading reflects the local traffic. Likewise, a sensor sitting in the path of a steady west wind already reads whatever PM2.5 the wind brings or carries away. Adjusting that reading further would double-count the same physical effect.

`engine/features.py:build_features()` does still compute the same traffic and wind columns *per sensor* — `traffic_factor`, `wind_term`, `nearest_congestion`, `distance_to_road_m`, `direction_factor`, `dispersal` — but only as metadata. They're written into [data/dashboard_snapshots.csv](data/dashboard_snapshots.csv) for the live snapshot history and as candidate features for downstream ML, but `pm25` itself is left untouched. The header comment block in [engine/features.py:20-41](engine/features.py#L20-L41) spells this out explicitly.

Grid cells are different. IDW gives every interpolated point a weighted average of nearby sensor readings, but it has no idea whether that point sits on a freeway, a backyard, or a stretch of empty parkland. `adjust_grid()` injects the missing road and wind context. That's why the same math lives in two places — once as scalar helpers in `engine/adjustments.py` for the per-sensor metadata and once as fully vectorised counterparts for the post-IDW grid pass.

### Highway-proximity taper

There is one edge case where the post-IDW traffic bump still risks double-counting: when a sensor itself sits on or next to a highway. Sensors within `SENSOR_HW_PROXIMITY_M` of a highway already read elevated PM2.5 from that road. Their IDW-propagated readings carry the highway signal into surrounding cells. To prevent a second traffic bump on those same cells, the traffic adjustment is scaled by `clip(idw_hw_dist / 300, 0, 1)`, where `idw_hw_dist` is the IDW-weighted distance-to-nearest-highway across each cell's contributing sensors. The result tapers linearly to zero for cells whose dominant sensors are essentially on the highway, leaves the adjustment unchanged once those sensors are 300+ m off the road, and falls in between for partial proximity. This affects ~18–20 grid cells in the current sensor network — sensor 305450 sits 7 m from a highway and sensor 283882 sits 133 m from a highway, and IDW carries their readings into roughly 9–10 surrounding cells each.

---

## 8. Rendering pipeline (brief)

### Gaussian smoothing

[viz/heatmap.py:139](viz/heatmap.py#L139)

```python
smoothed = gaussian_filter(values, sigma=1.5)
```

Smooth in **PM2.5 space**, then colour. If you colour first and blend, you get muddy intermediate colours where adjacent EPA categories meet (yellow ↔ orange, etc.) because RGB midpoints between two meaningful colours are usually meaningless. Smoothing the *numerical values* first and then colouring keeps every pixel's colour an honest representation of its (smoothed) PM2.5 level. Sigma = 1.5 cells gives enough blur to hide cell boundaries without losing detail.

### Color mapping

[viz/heatmap.py:46-110](viz/heatmap.py#L46-L110)

| PM2.5 (µg/m³) | Hex | Category |
|---|---|---|
| 0 | `#00e400` | Good |
| 12 | `#ffff00` | Moderate |
| 35.4 | `#ff7e00` | Sensitive |
| 55.4 | `#ff0000` | Unhealthy |
| 150.4 | `#8f3f97` | Very Unhealthy |
| 250.4 | `#7e0023` | Hazardous |

Linearly interpolated channel-by-channel in RGB space via a matplotlib `LinearSegmentedColormap`. The whole grid is mapped in one vectorised call. Alpha is set to a uniform 0.35 ([viz/heatmap.py:145](viz/heatmap.py#L145)) so the basemap shows through, the array is `np.flipud`ped (PNG row 0 must be the northernmost latitude), encoded to PNG in memory, base64-embedded, and added as a Folium `ImageOverlay`. This replaces what would otherwise be 40,000 DOM rectangles.

### Popup subsampling

[viz/heatmap.py:174-208](viz/heatmap.py#L174-L208)

```
POPUP_GRID_SIZE = 30
row_step        = max(1, lats.shape[0] // 30)
col_step        = max(1, lats.shape[1] // 30)
```

Each subsampled cell becomes an invisible `folium.Rectangle` (`fill_opacity = 0.0`) carrying a popup with zip code, PM2.5 value, and AQI category. Popups use the **unsmoothed** values for accuracy — smoothing is for visual rendering only. Zip code lookups go through `_coords_to_zip()`, which uses `uszipcode.SearchEngine` with an `@lru_cache(maxsize=2048)` keyed on coordinates rounded to 2 decimal places (~1.1 km precision) to maximise cache hits across the grid.

---

## 9. Spatial feature (brief)

[data/spatial/spatial_features.py:104-116](data/spatial/spatial_features.py#L104-L116)

```python
@lru_cache(maxsize=4096)
def compute_distance_to_highway(lat, lon) -> float:
    point        = Point(lon, lat)
    nearest_line = min(highways, key=point.distance)            # planar ranking
    on_line, _   = nearest_points(nearest_line, point)
    return geodesic((lat, lon), (on_line.y, on_line.x)).meters  # geodesic refinement
```

`dist_to_highway_m` is the spatial feature that bridges the live and historical pipelines. The Random Forest needs features that can be computed identically at training time (per sensor, from history) and at inference time (per grid cell, live). Because TomTom's free tier has no historical traffic, we can't reconstruct historical congestion — but distance to the nearest interstate or US highway is a property of *location alone*, no time component, so the same function works for both.

Highways are pulled from OpenStreetMap via OSMnx with the filter `["highway"~"motorway|motorway_link|trunk|trunk_link"]`, cached on disk for 30 days, and held in a module-level `_HIGHWAYS` list to load once per process. The lookup picks the nearest LineString by cheap planar distance (good enough for *ranking*) and then computes geodesic distance to the actual nearest point on that line for the precise value.

This feature is computed by the training pipeline today but is not yet wired into a live model — the TODO in [engine/features.py:3-8](engine/features.py#L3-L8) tracks the inference-time integration that has to happen before RF inference is turned on.

---

## 10. Parameter summary table

| Parameter | Value | File:Line | Rationale |
|---|---|---|---|
| `LON_CORRECTION` | cos(32.78°) ≈ 0.840 | [config.py:12](config.py#L12) | Corrects ~16% east-west distortion at Dallas latitude |
| `IDW_POWER` | 3 | [config.py:63](config.py#L63) | Steeper than the default 2 — nearby sensors dominate |
| `IDW_SEARCH_RADIUS_DEG` | 0.15° (~17 km) | [config.py:67](config.py#L67) | Beyond this, sensor influence is unphysical |
| `GRID_RESOLUTION` | 200 (config) / 60 (runtime) | [config.py:58](config.py#L58) | Detail vs. refresh speed |
| `TRAFFIC_WEIGHT` | 8.0 µg/m³ | [config.py:72](config.py#L72) | Matches near-road EPA studies (5–10 µg/m³) |
| `TRAFFIC_DECAY_RADIUS_M` | 500 m | [config.py:76](config.py#L76) | Near-road gradient research |
| `TRAFFIC_THRESHOLD` | 0.3 | [engine/adjustments.py:21](engine/adjustments.py#L21) | Below 30% congestion: negligible PM2.5 effect |
| `TRAFFIC_CURVE_K` | 3.0 | [engine/adjustments.py:22](engine/adjustments.py#L22) | Exponential steepness above threshold |
| `WIND_WEIGHT` | **3.0 µg/m³** | [engine/adjustments.py:17](engine/adjustments.py#L17) | Updated from the old 10.0 calibration |
| `WIND_SPEED_CAP` | 15.0 m/s | [engine/adjustments.py:18](engine/adjustments.py#L18) | Dispersal saturates above this |
| K (traffic neighbours) | 5 | [engine/interpolation.py:149](engine/interpolation.py#L149) | Smooth blending without Voronoi artefacts |
| Gaussian sigma | 1.5 cells | [viz/heatmap.py:139](viz/heatmap.py#L139) | Avoid colour banding |
| Heatmap opacity | 0.35 | [viz/heatmap.py:145](viz/heatmap.py#L145) | Basemap visibility |
| `POPUP_GRID_SIZE` | 30 | [viz/heatmap.py:174](viz/heatmap.py#L174) | ~900 popups vs. 3,600 cells |
| `SAMPLE_GRID` (traffic) | 5 (5×5 = 25 points) | [data/ingestion/traffic.py:23](data/ingestion/traffic.py#L23) | At 30-min API TTL, 48 × 25 = 1,200/day — half of TomTom's 2,500/day free tier |
| Cosine guard ε | 1e-6 deg | [engine/adjustments.py:102](engine/adjustments.py#L102) | Co-location guard for `atan2(0,0)` |
| Distance ε | 1e-10 | [engine/interpolation.py:68](engine/interpolation.py#L68) | Divide-by-zero guard for IDW |
