"""
sensor_coverage_check.py

One-off spatial-coverage analysis of the surviving sensors in
ml/data/history.csv. Reports grid-cell uniformity, nearest-neighbor
distances, and shows where the A/B-rejected sensors sat geographically
so we can judge whether the holes left behind are dense or sparse.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from geopy.distance import geodesic
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BBOX, PURPLEAIR_BASE_URL  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

HISTORY_CSV     = PROJECT_ROOT / "ml" / "data" / "history.csv"
QUALITY_REPORT  = PROJECT_ROOT / "ml" / "data" / "quality_report.json"
OUTPUT_DIR      = PROJECT_ROOT / "ml" / "analysis" / "output"
PLOT_PATH       = OUTPUT_DIR / "sensor_coverage.png"

LANDMARKS = {
    "DFW Airport":       (32.8998, -97.0403),
    "Downtown Dallas":   (32.7767, -96.7970),
    "Downtown Ft Worth": (32.7555, -97.3308),
    "Plano":             (33.0198, -96.6989),
    "Arlington":         (32.7357, -97.1081),
}

GRID_N           = 4
CV_CLUSTER_BAR   = 0.7
NN_OUTLIER_KM    = 8.0


def load_surviving_sensors() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(f"{HISTORY_CSV} not found — run collect_training_data.py first")
    df = pd.read_csv(HISTORY_CSV)
    sensors = (
        df[["sensor_id", "lat", "lon"]]
        .drop_duplicates(subset=["sensor_id"])
        .sort_values("sensor_id")
        .reset_index(drop=True)
    )
    return sensors


def load_dropped_ids() -> list[int]:
    if not QUALITY_REPORT.exists():
        return []
    data = json.loads(QUALITY_REPORT.read_text())
    return [int(s) for s in data.get("sensors_dropped_ab_failure_ids", [])]


def fetch_dropped_sensor_locations(dropped_ids: list[int]) -> pd.DataFrame:
    """One PurpleAir call to recover lat/lon for the IDs we cut at A/B."""
    if not dropped_ids:
        return pd.DataFrame(columns=["sensor_id", "lat", "lon"])

    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        print("  (PURPLEAIR_API_KEY not set — skipping dropped-sensor lookup)")
        return pd.DataFrame(columns=["sensor_id", "lat", "lon"])

    resp = requests.get(
        f"{PURPLEAIR_BASE_URL}/sensors",
        params={
            "fields": "sensor_index,latitude,longitude",
            "show_only": ",".join(str(s) for s in dropped_ids),
        },
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    if not resp.ok:
        print(f"  (PurpleAir lookup for dropped sensors failed: HTTP {resp.status_code})")
        return pd.DataFrame(columns=["sensor_id", "lat", "lon"])

    data = resp.json()
    fields = data["fields"]
    col = {n: fields.index(n) for n in fields}
    rows = [
        {
            "sensor_id": r[col["sensor_index"]],
            "lat":       r[col["latitude"]],
            "lon":       r[col["longitude"]],
        }
        for r in data["data"]
        if r[col["latitude"]] is not None and r[col["longitude"]] is not None
    ]
    return pd.DataFrame(rows)


def grid_cell_counts(sensors: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Count sensors in a GRID_N x GRID_N tiling of the bounding box."""
    lat_edges = pd.cut(
        sensors["lat"],
        bins=[BBOX["south"] + i * (BBOX["north"] - BBOX["south"]) / GRID_N for i in range(GRID_N + 1)],
        include_lowest=True, labels=False,
    )
    lon_edges = pd.cut(
        sensors["lon"],
        bins=[BBOX["west"] + i * (BBOX["east"] - BBOX["west"]) / GRID_N for i in range(GRID_N + 1)],
        include_lowest=True, labels=False,
    )

    counts = pd.DataFrame(0, index=range(GRID_N), columns=range(GRID_N))
    for lat_idx, lon_idx in zip(lat_edges, lon_edges):
        if pd.notna(lat_idx) and pd.notna(lon_idx):
            counts.iloc[int(lat_idx), int(lon_idx)] += 1

    # Flip so row 0 = north (matches map orientation when printed)
    counts = counts.iloc[::-1].reset_index(drop=True)
    empty = [(int(r), int(c)) for r in range(GRID_N) for c in range(GRID_N) if counts.iloc[r, c] == 0]
    return counts, empty


def describe_empty_cells(empty_cells: list[tuple[int, int]]) -> str:
    """Human-readable description of which corners/regions have no sensors."""
    if not empty_cells:
        return "no empty cells"

    # Row 0 = north, Row 3 = south. Col 0 = west, Col 3 = east.
    region_for = lambda r, c: (
        f"{'far ' if r in (0, 3) else ''}"
        f"{'north' if r < 2 else 'south'}"
        f"{'west' if c < 2 else 'east'}"
        + (f" ({'NW' if (r,c)==(0,0) else 'NE' if (r,c)==(0,3) else 'SW' if (r,c)==(3,0) else 'SE' if (r,c)==(3,3) else 'inner'} corner)"
           if (r in (0,3) and c in (0,3)) else "")
    )
    descriptions = sorted({region_for(r, c) for r, c in empty_cells})
    return ", ".join(descriptions)


def nearest_neighbor_distances(sensors: pd.DataFrame) -> pd.DataFrame:
    coords = list(zip(sensors["lat"], sensors["lon"]))
    ids = sensors["sensor_id"].tolist()
    rows = []
    for i, (lat_i, lon_i) in enumerate(coords):
        best_d = float("inf")
        best_j = None
        for j, (lat_j, lon_j) in enumerate(coords):
            if i == j:
                continue
            d = geodesic((lat_i, lon_i), (lat_j, lon_j)).km
            if d < best_d:
                best_d = d
                best_j = j
        rows.append({
            "sensor_id":     ids[i],
            "nearest_id":    ids[best_j] if best_j is not None else None,
            "distance_km":   round(best_d, 2),
        })
    return pd.DataFrame(rows).sort_values("distance_km", ascending=False).reset_index(drop=True)


def plot_coverage(sensors: pd.DataFrame, dropped: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    bbox_rect = Rectangle(
        (BBOX["west"], BBOX["south"]),
        BBOX["east"] - BBOX["west"],
        BBOX["north"] - BBOX["south"],
        linewidth=1.5, edgecolor="#444", facecolor="#f7f7f7", zorder=1,
    )
    ax.add_patch(bbox_rect)

    # 4x4 grid lines
    for i in range(1, GRID_N):
        lat = BBOX["south"] + i * (BBOX["north"] - BBOX["south"]) / GRID_N
        lon = BBOX["west"]  + i * (BBOX["east"]  - BBOX["west"])  / GRID_N
        ax.axhline(lat, color="#bbb", linewidth=0.6, zorder=2)
        ax.axvline(lon, color="#bbb", linewidth=0.6, zorder=2)

    ax.scatter(
        sensors["lon"], sensors["lat"],
        s=70, color="#1f77b4", edgecolor="white", linewidth=1,
        zorder=5, label=f"Surviving sensors (n={len(sensors)})",
    )
    for _, r in sensors.iterrows():
        ax.annotate(
            str(int(r["sensor_id"])),
            (r["lon"], r["lat"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#1f3a5f", zorder=6,
        )

    if not dropped.empty:
        ax.scatter(
            dropped["lon"], dropped["lat"],
            s=70, marker="x", color="#d62728", linewidth=2,
            zorder=4, label=f"Dropped at A/B (n={len(dropped)})",
        )
        for _, r in dropped.iterrows():
            ax.annotate(
                str(int(r["sensor_id"])),
                (r["lon"], r["lat"]),
                xytext=(4, -10), textcoords="offset points",
                fontsize=7, color="#7a1818", zorder=6,
            )

    for name, (lat, lon) in LANDMARKS.items():
        ax.scatter(lon, lat, s=110, marker="*", color="#ffae00",
                   edgecolor="#5a3e00", linewidth=0.8, zorder=7)
        ax.annotate(
            name, (lon, lat),
            xytext=(7, 7), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#5a3e00", zorder=8,
        )

    ax.set_xlim(BBOX["west"]  - 0.04, BBOX["east"]  + 0.04)
    ax.set_ylim(BBOX["south"] - 0.04, BBOX["north"] + 0.04)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "DFW Air Quality Dashboard — Sensor Spatial Coverage\n"
        f"Surviving training sensors with {GRID_N}x{GRID_N} coverage grid",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print(" DFW SENSOR COVERAGE CHECK")
    print("=" * 72)

    sensors = load_surviving_sensors()
    print(f"\nLoaded {len(sensors)} unique surviving sensors from {HISTORY_CSV.name}")
    print(f"Bounding box: lat [{BBOX['south']}, {BBOX['north']}], "
          f"lon [{BBOX['west']}, {BBOX['east']}]")

    bbox_height_km = geodesic(
        (BBOX["south"], BBOX["west"]), (BBOX["north"], BBOX["west"])
    ).km
    bbox_width_km = geodesic(
        (BBOX["south"], BBOX["west"]), (BBOX["south"], BBOX["east"])
    ).km
    bbox_area_km2 = bbox_height_km * bbox_width_km
    print(f"Approx. coverage area: {bbox_width_km:.1f} km E-W x "
          f"{bbox_height_km:.1f} km N-S = {bbox_area_km2:.0f} km²")

    # -------- Grid coverage --------
    print("\n" + "-" * 72)
    print(f" {GRID_N}x{GRID_N} GRID COVERAGE (rows: north→south, cols: west→east)")
    print("-" * 72)
    counts, empty = grid_cell_counts(sensors)
    print(counts.to_string(index=False, header=False))

    counts_flat = counts.values.flatten()
    mean = counts_flat.mean()
    std  = counts_flat.std()
    cv   = std / mean if mean > 0 else float("inf")

    print(f"\nMean sensors per cell: {mean:.2f}")
    print(f"Std  sensors per cell: {std:.2f}")
    print(f"Coefficient of variation (std/mean): {cv:.2f}")
    if empty:
        print(f"Empty cells: {len(empty)} of {GRID_N*GRID_N}  → {empty}")
    else:
        print("Empty cells: none")

    if cv > CV_CLUSTER_BAR:
        clustering_verdict = "coverage is clustered"
    else:
        clustering_verdict = "coverage is reasonably uniform"
    print(f"Verdict: {clustering_verdict} (threshold CV>{CV_CLUSTER_BAR})")

    # -------- Nearest-neighbor distances --------
    print("\n" + "-" * 72)
    print(" NEAREST-NEIGHBOR DISTANCES (km, geodesic)")
    print("-" * 72)
    nn = nearest_neighbor_distances(sensors)
    print(nn.to_string(index=False))

    print(f"\n  Min:    {nn['distance_km'].min():.2f} km")
    print(f"  Median: {nn['distance_km'].median():.2f} km")
    print(f"  Max:    {nn['distance_km'].max():.2f} km")

    outliers = nn[nn["distance_km"] > NN_OUTLIER_KM]
    if not outliers.empty:
        print(f"\nSpatial outliers (>{NN_OUTLIER_KM} km from nearest neighbor):")
        print(outliers.to_string(index=False))
    else:
        print(f"\nNo sensors are isolated by more than {NN_OUTLIER_KM} km.")

    # -------- A/B-rejected sensors --------
    print("\n" + "-" * 72)
    print(" SENSORS DROPPED AT A/B-FAILURE STAGE")
    print("-" * 72)
    dropped_ids = load_dropped_ids()
    print(f"IDs from quality_report.json: {dropped_ids}")
    dropped_df = fetch_dropped_sensor_locations(dropped_ids)
    if dropped_df.empty:
        print("(no locations recovered)")
    else:
        print("\nLocations of dropped sensors:")
        print(dropped_df.to_string(index=False))

        # Did dropping them open new holes?
        merged = pd.concat([
            sensors[["sensor_id", "lat", "lon"]],
            dropped_df[["sensor_id", "lat", "lon"]],
        ], ignore_index=True)
        counts_with, empty_with = grid_cell_counts(merged)
        print(f"\nIf dropped sensors were healthy, empty cells would be: "
              f"{len(empty_with)} (vs {len(empty)} now)")
        gap_filling = len(empty) - len(empty_with)
        if gap_filling > 0:
            print(f"  → Dropped sensors were filling {gap_filling} cell(s) "
                  f"that are now empty. Holes are real losses, not redundancy.")
        else:
            print("  → Dropped sensors were in already-covered cells. "
                  "Their loss is redundancy, not new gaps.")

    # -------- Plot --------
    print("\n" + "-" * 72)
    print(" PLOT")
    print("-" * 72)
    plot_coverage(sensors, dropped_df)
    print(f"Saved to {PLOT_PATH.relative_to(PROJECT_ROOT)}")

    # -------- Summary --------
    print("\n" + "=" * 72)
    print(" COVERAGE ASSESSMENT")
    print("=" * 72)

    region = describe_empty_cells(empty)

    if cv > CV_CLUSTER_BAR or len(empty) >= 4 or not outliers.empty:
        recommendation = (
            "Reconsider before training. Coverage gaps will cause the IDW "
            "(and the RF model) to extrapolate heavily in empty regions. "
            "Either accept that the model is unreliable in those areas, "
            "supplement with OpenAQ reference monitors there, or weight "
            "training rows by spatial density."
        )
    elif len(empty) >= 1:
        recommendation = (
            "Proceed with caution. Coverage is mostly even but a few cells "
            "are empty — flag those regions as low-confidence at inference time."
        )
    else:
        recommendation = "Proceed. Coverage is uniform enough for training."

    print(
        f"\n{len(sensors)} sensors over ~{bbox_area_km2:.0f} km² of DFW. "
        f"{len(empty)} of {GRID_N*GRID_N} grid cells have zero sensors "
        f"(largest gap: {region}). "
        f"Median nearest-neighbor distance is {nn['distance_km'].median():.1f} km, "
        f"max is {nn['distance_km'].max():.1f} km. "
        f"Spatial layout is {clustering_verdict} (CV={cv:.2f}).\n"
    )
    print(f"Recommendation: {recommendation}\n")


if __name__ == "__main__":
    main()
