# app.py — Streamlit entry point

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from data.ingestion.purpleair import fetch_sensors
from data.ingestion.openaq import fetch_openaq
from data.ingestion.weather import fetch_wind
from data.ingestion.traffic import fetch_traffic
from data.ingestion.history import save_snapshot, get_history_stats
from engine.features import build_features
from engine.interpolation import run_idw, adjust_grid
from viz.heatmap import build_sensor_map

# --- Page config ---
st.set_page_config(
    page_title="DFW Air Quality",
    page_icon="💨",
    layout="wide",
)

st.title("💨 DFW Real-Time Air Quality")
st.caption("PM2.5 adjusted for live traffic congestion and wind conditions.")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    st.markdown("---")
    st.markdown(
        "**Sources:** PurpleAir · OpenAQ · TomTom · OpenWeatherMap  \n"
        "PM2.5 is adjusted for nearby traffic and wind dispersal.  \n"
        "Refreshes every 5 minutes."
    )

    st.markdown("---")
    st.markdown("**ML Training Data**")
    stats = get_history_stats()
    if stats["total_records"] == 0:
        st.caption("No snapshots collected yet.")
    else:
        st.caption(f"Records: {stats['total_records']:,}")
        st.caption(f"Sensors: {stats['unique_sensors']}")
        st.caption(f"Hours: {stats['hours_covered']}")
        earliest, latest = stats["date_range"]
        if earliest:
            st.caption(f"From: {earliest[:16]}")
            st.caption(f"To:   {latest[:16]}")

# --- Cached data fetches (all TTL 5 minutes) ---
#
# The FastAPI backend (api/routes/grid.py) caches the assembled pipeline
# snapshot with the same 300s TTL. The two caches are intentionally
# independent — under simultaneous use, worst-case drift between the
# Streamlit view and the React/AERIA view is ~5 minutes, which is fine
# because upstream sources (PurpleAir/OpenAQ/TomTom/OpenWeatherMap) only
# refresh every 10–30 minutes anyway. See api/routes/grid.py for the
# matching note.
@st.cache_data(ttl=300, show_spinner="Fetching sensor data...")
def load_sensors():
    return fetch_sensors()

@st.cache_data(ttl=300, show_spinner="Fetching OpenAQ data...")
def load_openaq():
    return fetch_openaq()

@st.cache_data(ttl=300, show_spinner="Fetching wind data...")
def load_wind():
    return fetch_wind()

@st.cache_data(ttl=300, show_spinner="Fetching traffic data...")
def load_traffic():
    return fetch_traffic()

# --- Fetch all data sources ---
try:
    purpleair_df = load_sensors()
except ValueError as e:
    st.error(str(e))
    st.info("Add your PurpleAir API key to the `.env` file and restart the app.")
    st.stop()
except Exception as e:
    st.error(f"Failed to fetch PurpleAir data: {e}")
    st.stop()

openaq_df = load_openaq()  # returns empty DataFrame on failure — non-fatal

sensor_df = pd.concat([purpleair_df, openaq_df], ignore_index=True)

try:
    wind = load_wind()
except Exception as e:
    st.warning(f"Wind data unavailable — defaulting to calm conditions. ({e})")
    wind = {"wind_speed": 0.0, "wind_deg": 0.0}

try:
    traffic_df = load_traffic()
except Exception as e:
    st.warning(f"Traffic data unavailable — skipping congestion adjustment. ({e})")
    traffic_df = None

# --- Fuse data sources ---
if sensor_df.empty:
    st.warning("No sensor data found for the Dallas bounding box. Check your API key and try again.")
    st.stop()

df = build_features(
    sensor_df,
    traffic_df if traffic_df is not None else pd.DataFrame(),
    wind,
)

# --- Persist snapshot for ML training ---
try:
    save_snapshot(df, traffic_df if traffic_df is not None else pd.DataFrame(), wind)
except Exception as e:
    st.warning(f"Could not save training snapshot: {e}")

# --- Stats row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Sensors",       len(df))
col2.metric("Avg PM2.5 (EPA-corrected)", f"{df['pm25'].mean():.1f} µg/m³")
col3.metric("Wind Speed",           f"{wind['wind_speed']:.1f} m/s")
col4.metric("Wind Direction",       f"{wind['wind_deg']:.0f}°")

# --- Build map (IDW + post-IDW traffic/wind adjustments) ---
lats_2d, lons_2d, idw_estimate, idw_hw_dist, confidence_grid = run_idw(df, grid_resolution=60)
grid = adjust_grid(
    idw_estimate,
    lats_2d,
    lons_2d,
    traffic_df if traffic_df is not None else pd.DataFrame(),
    wind,
    idw_hw_dist=idw_hw_dist,
)
folium_map = build_sensor_map(df, lats_2d, lons_2d, grid, confidence_grid)
st_folium(folium_map, width="100%", height=600, returned_objects=[])
