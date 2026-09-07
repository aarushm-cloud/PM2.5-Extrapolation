# Highway-distance source: a committed geometry bundle, with OSMnx (live
# OSM pull, cached on disk) as the refresh path. To widen the highway set,
# change HIGHWAY_FILTER below and rerun scripts/build_highway_bundle.py.
"""
Static spatial features derived from OpenStreetMap geometry.

The RF training pipeline needs spatial signals that can be computed
identically at inference time for IDW grid cells. Distance-to-nearest-highway
fits: it's a property of location alone (no time component, no API), so the
same lookup works for historical sensor rows AND live grid cells.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from geopy.distance import geodesic
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# DFW bounding box (matches config.BBOX). Hardcoded here so this module
# stays usable without booting the project's import graph.
DFW_BBOX = {
    "north": 33.08,
    "south": 32.55,
    "east":  -96.46,
    "west":  -97.05,
}

CACHE_DIR  = Path("data/.cache")
CACHE_FILE = CACHE_DIR / "dfw_highways.pkl"

# Committed highway geometry. `data/.cache/` is gitignored, so a fresh
# deploy has no disk cache and would otherwise hit Overpass live — which
# refuses connections from datacenter IPs, taking the grid pipeline down.
# Regenerate with scripts/build_highway_bundle.py.
BUNDLE_FILE = Path(__file__).resolve().parent / "dfw_highways.json.gz"
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

# OSM tag filter — interstates and US highways in the DFW area
HIGHWAY_FILTER = '["highway"~"motorway|motorway_link|trunk|trunk_link"]'

log = logging.getLogger("dfw_collector")


def _fetch_and_cache_highways() -> list[LineString]:
    """Pull the major-highway network from OSM, save edge geometries to disk."""
    import osmnx as ox  # imported lazily so module import doesn't require it

    # Keep OSMnx's per-request HTTP cache under data/.cache/ instead of
    # littering the repo root with a top-level cache/ directory.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(CACHE_DIR / "osmnx_http")

    log.info("  Fetching DFW highway network from OpenStreetMap (one-time, ~30s)")
    bbox = (DFW_BBOX["west"], DFW_BBOX["south"], DFW_BBOX["east"], DFW_BBOX["north"])
    G = ox.graph_from_bbox(
        bbox,
        network_type="drive",
        custom_filter=HIGHWAY_FILTER,
        simplify=True,
    )

    # Convert each edge into a LineString in (lon, lat) order. Edges with a
    # 'geometry' attribute carry the full polyline; the rest are straight
    # lines between their two endpoint nodes.
    geoms: list[LineString] = []
    for u, v, _key, data in G.edges(keys=True, data=True):
        if "geometry" in data:
            geoms.append(data["geometry"])
        else:
            u_node, v_node = G.nodes[u], G.nodes[v]
            geoms.append(LineString([(u_node["x"], u_node["y"]),
                                     (v_node["x"], v_node["y"])]))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as f:
        pickle.dump(geoms, f)
    log.info(f"  Cached {len(geoms)} highway edges to {CACHE_FILE}")
    return geoms


def _load_bundled_highways() -> list[LineString]:
    """Load the committed highway bundle. No network, no OSMnx import."""
    with gzip.open(BUNDLE_FILE, "rt", encoding="utf-8") as f:
        return [LineString(coords) for coords in json.load(f)]


def _load_highways() -> list[LineString]:
    """Return highway LineStrings, preferring the freshest source available:
    disk cache, then a refetch, then the committed bundle. Overpass is never
    allowed to be the difference between a working and a broken pipeline."""
    if CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            with CACHE_FILE.open("rb") as f:
                return pickle.load(f)
        try:
            return _fetch_and_cache_highways()
        except Exception as e:
            log.warning("Highway refetch failed (%s) — using bundled geometry", e)
            return _load_bundled_highways()

    if BUNDLE_FILE.exists():
        return _load_bundled_highways()
    return _fetch_and_cache_highways()


# Module-level cache: load the network once per process, not once per call.
# `_HIGHWAYS_MTIME` records the disk-cache file's mtime at load time so
# `_highways()` can cheaply detect a refreshed disk cache and self-refresh.
_HIGHWAYS: Optional[list[LineString]] = None
_HIGHWAYS_MTIME: Optional[float] = None


def refresh_highways() -> None:
    """
    Drop the in-process highway cache so the next `_highways()` call reloads
    from disk (or refetches if the disk cache is stale).

    Also clears the `compute_distance_to_highway` lru_cache, which would
    otherwise keep returning stale per-coordinate distances even after the
    underlying geometry was reloaded.

    Called automatically from `_highways()` when the on-disk cache file's
    mtime advances past the in-process snapshot's mtime — see #18 in the
    audit. Also exposed publicly so a caller can force a reload (e.g. after
    explicitly invalidating the disk cache).
    """
    global _HIGHWAYS, _HIGHWAYS_MTIME
    _HIGHWAYS = None
    _HIGHWAYS_MTIME = None
    compute_distance_to_highway.cache_clear()


def _maybe_refresh_on_mtime_change() -> None:
    """
    Cheap freshness probe — stat the disk cache file and compare its mtime
    to the snapshot we took at load time. If the disk file is newer, drop
    both the in-process highway list and the lru_cache so the next lookup
    pulls fresh geometry.

    Called from `compute_distance_to_highway` *before* the lru_cache layer
    so it actually runs on cache hits — embedding it inside the cached body
    would never fire for a coordinate already in the cache, which is
    exactly the long-running-process staleness window #18 cares about.

    A single `os.stat` on a small file is on the order of µs on macOS/Linux
    (well under the cost of a returned cached lookup), so this is safe to
    run on every public call.
    """
    if _HIGHWAYS is None or _HIGHWAYS_MTIME is None or not CACHE_FILE.exists():
        return
    disk_mtime = CACHE_FILE.stat().st_mtime
    if disk_mtime > _HIGHWAYS_MTIME:
        log.info("Highway disk cache refreshed since load — reloading")
        refresh_highways()


def _highways() -> list[LineString]:
    """Return the cached list of highway LineStrings, lazy-loading from
    disk on first access. Mtime-driven invalidation is handled separately
    by `_maybe_refresh_on_mtime_change` so it can run before the
    `compute_distance_to_highway` lru_cache."""
    global _HIGHWAYS, _HIGHWAYS_MTIME
    if _HIGHWAYS is None:
        _HIGHWAYS = _load_highways()
        _HIGHWAYS_MTIME = (
            CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else None
        )
    return _HIGHWAYS


def _distance_to_highway_uncached(lat: float, lon: float) -> float:
    """Pure geometry — distance from (lat, lon) to the nearest major DFW
    highway segment, in meters. Wrapped by an lru_cache below."""
    point = Point(lon, lat)
    highways = _highways()

    # Pick the nearest LineString by planar distance (cheap, accurate enough
    # at this scale for *ranking*), then compute geodesic distance to the
    # actual nearest point on that LineString.
    nearest_line = min(highways, key=point.distance)
    on_line, _ = nearest_points(nearest_line, point)
    return geodesic((lat, lon), (on_line.y, on_line.x)).meters


_distance_cached = lru_cache(maxsize=4096)(_distance_to_highway_uncached)


def compute_distance_to_highway(lat: float, lon: float) -> float:
    """Returns geodesic distance in meters from (lat, lon) to the
    nearest major DFW highway segment.

    Two-layer cache:
      1. Per-coordinate `lru_cache` on the inner geometry function.
      2. The `_HIGHWAYS` snapshot, refreshed automatically when the disk
         cache file's mtime advances past our recorded snapshot.

    The mtime probe runs *before* the lru_cache lookup so a refresh
    invalidates per-coordinate results too — otherwise long-running
    processes would keep serving stale distances forever.
    """
    _maybe_refresh_on_mtime_change()
    return _distance_cached(lat, lon)


# Forward the lru_cache control attributes to the public name so callers
# (refresh_highways, tests) can use compute_distance_to_highway.cache_clear()
# / .cache_info() the same way they would on a directly-decorated function.
compute_distance_to_highway.cache_clear = _distance_cached.cache_clear
compute_distance_to_highway.cache_info = _distance_cached.cache_info
