# config.py — Project-wide constants
import os
from math import cos, radians
from dotenv import load_dotenv
load_dotenv()

# --- Geospatial correction ---
# Dallas sits at ~32.78° N. Raw degree differences overstate east-west distances
# by ~1/cos(lat). Multiply all longitude deltas by LON_CORRECTION before squaring
# to get an approximately isotropic distance metric without full Haversine.
LAT_CENTER     = 32.78
LON_CORRECTION = cos(radians(LAT_CENTER))  # ≈ 0.840

# Dallas metro bounding box
BBOX = {
    "north": 33.08,
    "south": 32.55,
    "east": -96.46,
    "west": -97.05,
}

# Map center (approximate center of Dallas)
MAP_CENTER = [32.815, -96.755]
MAP_ZOOM = 10

# DFW International Airport — used by the training pipeline as the
# Meteostat lookup point and by ml/predictor.py as the schema-parity
# probe coordinate. Single source so both stay in lockstep.
DFW_AIRPORT_LAT_LON = (32.8998, -97.0403)

# Data refresh interval in seconds (used by APScheduler)
REFRESH_INTERVAL_SECONDS = 300  # 5 minutes

# AQI / PM2.5 thresholds (µg/m³) — EPA breakpoints
AQI_THRESHOLDS = {
    "good":        (0, 12),
    "moderate":    (12.1, 35.4),
    "sensitive":   (35.5, 55.4),
    "unhealthy":   (55.5, 150.4),
    "very_unhealthy": (150.5, 250.4),
    "hazardous":   (250.5, 9999),
}

# Color mapping for AQI categories (used on the map markers)
AQI_COLORS = {
    "good":           "green",
    "moderate":       "yellow",
    "sensitive":      "orange",
    "unhealthy":      "red",
    "very_unhealthy": "purple",
    "hazardous":      "darkred",
}

# PurpleAir API base URL
PURPLEAIR_BASE_URL = "https://api.purpleair.com/v1"

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

# --- Grid resolution ---
# Number of lat/lon points along each axis of the interpolation grid.
# 200 → 200×200 = 40,000 cells, fine enough that individual cell edges
# are invisible even when zoomed in over a neighbourhood.
GRID_RESOLUTION = 200

# --- IDW interpolation ---
# Power controls how steeply sensor influence drops with distance.
# 3 (vs the old 2) gives a steeper falloff so nearby sensors dominate more.
IDW_POWER = 3

# Maximum radius (degrees) within which a sensor influences a grid cell.
# ~0.15° ≈ 15–17 km at Dallas latitude. Sensors beyond this get zero weight.
IDW_SEARCH_RADIUS_DEG = 0.15

# --- Traffic feature ---
# Real-world near-road PM2.5 enhancement is typically 5–10 µg/m³ even at
# the busiest highways. 20 was too aggressive and created artificial hotspots.
TRAFFIC_WEIGHT = 8.0

# --- route comparator ---
# Per-edge cleanest-route weight is `length × (pm_midpoint + ROUTE_PM_ALPHA)`.
# Small α (≈0.1) → PM dominates → cleanest aggressively detours.
# Large α (≈10) → length dominates → cleanest collapses to shortest.
# 1.0 keeps PM in the lead at typical Dallas levels (PM₂.₅ ≈ 8 µg/m³) while
# still letting a short detour win when PM is otherwise close.
ROUTE_PM_ALPHA = 1.0

# Distance (meters) at which the traffic adjustment fades to zero.
# At 0 m: full effect. At 250 m: half effect. At 500 m+: zero effect.
TRAFFIC_DECAY_RADIUS_M = 500

# Below this IDW-weighted distance-to-highway, the post-IDW traffic adjustment
# is scaled down: sensors near a highway already read its elevated PM2.5, and
# IDW propagates that signal to surrounding cells. Without this taper, those
# cells would also receive the post-IDW traffic bump and double-count the road.
SENSOR_HW_PROXIMITY_M = 300