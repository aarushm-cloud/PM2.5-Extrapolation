# engine/router.py — walking-route comparator.
#
# Two-route comparator over an OSMnx walking graph:
#   shortest:  Dijkstra on edge length only.
#   cleanest:  Dijkstra on length × (pm_midpoint + ROUTE_PM_ALPHA), with
#              pm_midpoint sampled from the post-IDW grid at each edge's
#              midpoint by nearest neighbor (mirroring api/routes/cells.py).
#
# Public surface:
#   - find_routes(start, end, grid=None, alpha=None) -> RouteComparison
#   - preload_graph()  — for the FastAPI lifespan hook.
#   - GeocodeFailure / OutOfDFWError / DisconnectedRouteError — narrow
#     exception types so the api layer can map each to a distinct status.
#
# Walking graph and geocode results are cached at module level. The graph is
# also persisted to data/.cache/walking_graph.graphml with a 30-day TTL,
# mirroring the pattern in data/spatial/spatial_features.py.
#
# Smoke test (requires LOCATIONIQ_API_KEY in .env):
#   python -m engine.router --start "Mockingbird Station Dallas" \
#       --end "Klyde Warren Park Dallas" --mock

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from cachetools import LRUCache
from dotenv import load_dotenv

from config import BBOX, LON_CORRECTION, ROUTE_PM_ALPHA
from engine.snapshot import PipelineSnapshot

load_dotenv()

logger = logging.getLogger(__name__)
# Named logger for cold-boot / preload / perf telemetry — same convention as
# api/main.py's "aeria.cors". Render free-tier 512 MB RAM means graph load
# time and edge counts will matter to track.
perf_logger = logging.getLogger("aeria.router")


# --- Constants ---

# LocationIQ forward geocoding. The us1 endpoint has lower latency from US
# datacenters; eu1 is the European mirror if we ever need it.
#
# We hit this with `format=json` and parse the flat array response — NOT
# `format=geojson`, which the free tier rejects with HTTP 400 (paid plans
# unlock it). Upgrading to a paid plan would let us drop the float() casts
# below since geojson returns coordinates as proper floats inside a
# FeatureCollection. For now the trade-off favors zero cost over parser
# elegance.
LOCATIONIQ_GEOCODE_URL = "https://us1.locationiq.com/v1/search"

# 1.4 m/s ≈ 5.0 km/h — standard average pedestrian speed used by transit
# agencies. Hardcoded; not a tuning knob.
WALK_SPEED_MS = 1.4

CACHE_DIR = Path("data/.cache")
GRAPH_CACHE_FILE = CACHE_DIR / "walking_graph.graphml"
GRAPH_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days


# --- Errors ---

class GeocodeFailure(Exception):
    """LocationIQ failed or returned no match for the address."""


class OutOfDFWError(Exception):
    """Geocoded address falls outside config.BBOX."""


class DisconnectedRouteError(Exception):
    """No walking-network path connects start and end."""


# --- Result types ---

@dataclass
class Route:
    geometry: dict          # GeoJSON LineString {"type": "LineString", "coordinates": [[lon, lat], ...]}
    distance_m: float
    mean_pm25: float        # exposure-weighted mean = total_exposure / distance_m
    walk_seconds: float     # distance_m / WALK_SPEED_MS
    total_exposure: float   # Σ pm_midpoint × edge_length — line integral of PM along the path


@dataclass
class RouteComparison:
    cleanest: Route
    shortest: Route


# ---------------------------------------------------------------------------
# Geocoding (LocationIQ /v1/search, requests + cachetools.LRUCache)
# ---------------------------------------------------------------------------
# 10k entries is generous — DFW has on the order of 50k unique street
# addresses worth caching, and entries are tiny tuples.
_geocode_cache: LRUCache = LRUCache(maxsize=10_000)


def _locationiq_key() -> str:
    # LocationIQ free tier: 5,000 requests/day, ~2 req/s.
    # Daily quota resets at 00:00 UTC, NOT local midnight — if the cap looks
    # like it "reset at the wrong time" while debugging, that's why.
    key = os.getenv("LOCATIONIQ_API_KEY")
    if not key or key == "your_key_here":
        raise GeocodeFailure(
            "LOCATIONIQ_API_KEY is not set in your .env file. "
            "Sign up free at https://locationiq.com/register."
        )
    return key


def geocode(address: str) -> tuple[float, float]:
    """Forward-geocode an address via LocationIQ /v1/search, biased to DFW.

    Returns (lat, lon). Raises GeocodeFailure on LocationIQ error or no match.
    Identical inputs hit the process-local LRU cache.
    """
    cached = _geocode_cache.get(address)
    if cached is not None:
        return cached

    params = {
        "key": _locationiq_key(),
        "q": address,
        # format=json returns a flat array with lat/lon as STRINGS. We don't
        # use format=geojson because the LocationIQ free tier rejects it
        # with HTTP 400 (paid plans unlock geojson and would let us drop
        # the float() casts below). See the URL constant for context.
        "format": "json",
        "limit": 1,
        # Bias only — `bounded=1` would HARD filter to the bbox and
        # collapse the OutOfDFWError vs GeocodeFailure distinction we use
        # downstream. Leave biasing soft so out-of-DFW results still come
        # back and `_geocode_in_dfw()` can raise the more specific error.
        "viewbox": f"{BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}",
    }
    try:
        resp = requests.get(LOCATIONIQ_GEOCODE_URL, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise GeocodeFailure(f"LocationIQ request failed for {address!r}: {e}") from e

    # LocationIQ returns either a list of results or, for a no-match query
    # on some endpoints, an HTTP 404 (already turned into GeocodeFailure
    # above by raise_for_status) or an empty list.
    if not isinstance(payload, list) or not payload:
        raise GeocodeFailure(f"No geocoding match for {address!r}.")

    top = payload[0]
    try:
        result = (float(top["lat"]), float(top["lon"]))
    except (KeyError, TypeError, ValueError) as e:
        raise GeocodeFailure(f"Malformed coordinates for {address!r}: {e}") from e

    _geocode_cache[address] = result
    return result


def _geocode_in_dfw(address: str) -> tuple[float, float]:
    """geocode + bbox enforcement. LocationIQ's viewbox is a soft bias
    (we deliberately do not pass `bounded=1`), so out-of-DFW results can
    still come back — this raises OutOfDFWError so the API layer can
    distinguish "address not found" from "address found but not in DFW"."""
    lat, lon = geocode(address)
    if not (BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]):
        raise OutOfDFWError(
            f"{address!r} resolved to ({lat:.4f}, {lon:.4f}), outside the DFW bbox."
        )
    return lat, lon


# ---------------------------------------------------------------------------
# Walking graph — module singleton + on-disk graphml cache.
# Mirrors the pattern in data/spatial/spatial_features.py:36-90:
#   * 30-day TTL on the disk file (mtime-based).
#   * Module-level singleton holding the loaded graph + its mtime snapshot.
#   * mtime probe before each access so an out-of-band disk refresh
#     auto-invalidates the in-process snapshot.
# ---------------------------------------------------------------------------
_GRAPH = None
_GRAPH_MTIME: Optional[float] = None


def _fetch_and_cache_graph():
    """One-time OSMnx fetch over the full DFW bbox; persisted to graphml."""
    import osmnx as ox  # imported lazily so module import doesn't require it

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(CACHE_DIR / "osmnx_http")
    perf_logger.info("Fetching DFW walking graph from OSM (one-time, ~60–180s)...")
    t0 = time.time()
    bbox = (BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])
    G = ox.graph_from_bbox(bbox, network_type="walk")
    elapsed = time.time() - t0
    perf_logger.info(
        "Walking graph fetched in %.1fs (%d nodes, %d edges).",
        elapsed, G.number_of_nodes(), G.number_of_edges(),
    )
    ox.save_graphml(G, GRAPH_CACHE_FILE)
    return G


def _load_graph():
    """Load from on-disk graphml, or refetch if missing or stale."""
    import osmnx as ox

    if GRAPH_CACHE_FILE.exists():
        age = time.time() - GRAPH_CACHE_FILE.stat().st_mtime
        if age < GRAPH_CACHE_TTL_SECONDS:
            t0 = time.time()
            G = ox.load_graphml(GRAPH_CACHE_FILE)
            elapsed = time.time() - t0
            perf_logger.info(
                "Walking graph loaded from disk in %.1fs (%d nodes, %d edges).",
                elapsed, G.number_of_nodes(), G.number_of_edges(),
            )
            return G
    return _fetch_and_cache_graph()


def _maybe_refresh_on_mtime_change() -> None:
    """If the disk cache's mtime advanced past our recorded snapshot, drop
    the in-process graph + indexes so the next access reloads. Cheap stat
    (well under µs) so safe to call on every public entry point."""
    global _GRAPH, _GRAPH_MTIME
    if _GRAPH is None or _GRAPH_MTIME is None or not GRAPH_CACHE_FILE.exists():
        return
    disk_mtime = GRAPH_CACHE_FILE.stat().st_mtime
    if disk_mtime > _GRAPH_MTIME:
        perf_logger.info("Walking graph disk cache refreshed since load — reloading.")
        _GRAPH = None
        _GRAPH_MTIME = None
        _reset_node_index()
        _reset_annotation_state()


def get_walking_graph():
    """Module-level singleton accessor. Lazy-loads from disk on first
    access; mtime-checks against the disk cache on every call so an
    out-of-band refresh is picked up without restart."""
    global _GRAPH, _GRAPH_MTIME
    _maybe_refresh_on_mtime_change()
    if _GRAPH is None:
        _GRAPH = _load_graph()
        _GRAPH_MTIME = (
            GRAPH_CACHE_FILE.stat().st_mtime if GRAPH_CACHE_FILE.exists() else None
        )
    return _GRAPH


def preload_graph() -> None:
    """Force the walking graph to load. Exposed for the FastAPI lifespan
    hook — calling this at startup means the first
    user request doesn't pay the cold-load cost.
    """
    get_walking_graph()


def refresh_walking_graph() -> None:
    """Drop in-process graph, node index, and annotation state. Test/forced-
    reload entry point — does NOT delete the on-disk cache."""
    global _GRAPH, _GRAPH_MTIME
    _GRAPH = None
    _GRAPH_MTIME = None
    _reset_node_index()
    _reset_annotation_state()


# ---------------------------------------------------------------------------
# Node index — vectorised nearest-node lookup. Memoised by graph identity
# so a fresh graph (eg in tests) rebuilds without leaking stale arrays.
# ---------------------------------------------------------------------------
_NODE_INDEX: Optional[tuple[list, np.ndarray, np.ndarray]] = None
_NODE_INDEX_GRAPH_ID: Optional[int] = None


def _reset_node_index() -> None:
    global _NODE_INDEX, _NODE_INDEX_GRAPH_ID
    _NODE_INDEX = None
    _NODE_INDEX_GRAPH_ID = None


def _get_node_index(G):
    global _NODE_INDEX, _NODE_INDEX_GRAPH_ID
    gid = id(G)
    if _NODE_INDEX is not None and _NODE_INDEX_GRAPH_ID == gid:
        return _NODE_INDEX
    nodes = list(G.nodes(data=True))
    node_ids = [n for n, _ in nodes]
    lats = np.fromiter((float(d["y"]) for _, d in nodes), dtype=np.float64, count=len(nodes))
    lons = np.fromiter((float(d["x"]) for _, d in nodes), dtype=np.float64, count=len(nodes))
    _NODE_INDEX = (node_ids, lats, lons)
    _NODE_INDEX_GRAPH_ID = gid
    return _NODE_INDEX


def _nearest_node(G, lat: float, lon: float):
    """Find the graph node closest to (lat, lon) under the cosine-corrected
    planar metric used everywhere else in the engine."""
    node_ids, lats, lons = _get_node_index(G)
    dlat = lat - lats
    dlon = (lon - lons) * LON_CORRECTION
    sq = dlat * dlat + dlon * dlon
    return node_ids[int(np.argmin(sq))]


# ---------------------------------------------------------------------------
# PM annotation — nearest-neighbor lookup at edge midpoints, mirrored from
# api/routes/cells.py:120-126 so route stats agree with what the cell card
# would show for the same point.
# ---------------------------------------------------------------------------
_ANNOTATED_TIMESTAMP: Optional[str] = None
_ANNOTATED_GRAPH_ID: Optional[int] = None


def _reset_annotation_state() -> None:
    global _ANNOTATED_TIMESTAMP, _ANNOTATED_GRAPH_ID
    _ANNOTATED_TIMESTAMP = None
    _ANNOTATED_GRAPH_ID = None


def _annotate_edges(G, snap: PipelineSnapshot) -> None:
    """Stamp each edge with `pm_midpoint`, the snapshot's PM2.5 at the edge
    midpoint by nearest neighbor. Memoised by (graph identity, snapshot
    timestamp) — second call with the same pair is a no-op."""
    global _ANNOTATED_TIMESTAMP, _ANNOTATED_GRAPH_ID

    gid = id(G)
    if _ANNOTATED_GRAPH_ID == gid and _ANNOTATED_TIMESTAMP == snap.timestamp:
        return

    lat_arr = snap.lats_2d[:, 0]
    lon_arr = snap.lons_2d[0, :]

    for u, v, _key, data in G.edges(keys=True, data=True):
        u_node = G.nodes[u]
        v_node = G.nodes[v]
        mid_lat = (float(u_node["y"]) + float(v_node["y"])) / 2.0
        mid_lon = (float(u_node["x"]) + float(v_node["x"])) / 2.0
        i = int(np.argmin(np.abs(lat_arr - mid_lat)))
        j = int(np.argmin(np.abs(lon_arr - mid_lon)))
        data["pm_midpoint"] = float(snap.grid[i, j])

    _ANNOTATED_TIMESTAMP = snap.timestamp
    _ANNOTATED_GRAPH_ID = gid


# ---------------------------------------------------------------------------
# Route construction
# ---------------------------------------------------------------------------

def _build_route(G, nodes: list, edge_weight_fn: Callable[[dict], float]) -> Route:
    """Walk a node sequence, picking the parallel edge with min weight at
    each step (matches the choice nx.shortest_path made under the hood),
    and accumulate distance + exposure."""
    if not nodes:
        raise DisconnectedRouteError("Empty node sequence.")

    coords = [[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in nodes]
    distance_m = 0.0
    total_exposure = 0.0

    is_multi = G.is_multigraph()
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        edge_dict = G.get_edge_data(u, v)
        if not edge_dict:
            raise DisconnectedRouteError(
                f"Path traversal hit missing edge between {u} and {v}."
            )
        # MultiDiGraph: edge_dict is {key: attrs}. Plain DiGraph: edge_dict is attrs.
        if is_multi:
            chosen = min(edge_dict.values(), key=edge_weight_fn)
        else:
            chosen = edge_dict
        length = float(chosen.get("length", 0.0))
        pm = float(chosen.get("pm_midpoint", 0.0))
        distance_m += length
        total_exposure += pm * length

    mean_pm25 = total_exposure / distance_m if distance_m > 0 else 0.0
    walk_seconds = distance_m / WALK_SPEED_MS if distance_m > 0 else 0.0

    return Route(
        geometry={"type": "LineString", "coordinates": coords},
        distance_m=distance_m,
        mean_pm25=mean_pm25,
        walk_seconds=walk_seconds,
        total_exposure=total_exposure,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def find_routes(
    start: str,
    end: str,
    grid: Optional[PipelineSnapshot] = None,
    alpha: Optional[float] = None,
) -> RouteComparison:
    """Compare a length-only shortest path against a length × (PM + α)
    cleanest path between two geocoded addresses.

    grid=None opts into mock-PM mode (clean north / polluted south gradient
    over the BBOX) so the CLI has a deterministic offline smoke path.

    Raises GeocodeFailure / OutOfDFWError / DisconnectedRouteError. The api
    layer (item 2) translates each into an HTTP status — this layer never
    imports HTTPException.
    """
    import networkx as nx

    if alpha is None:
        alpha = ROUTE_PM_ALPHA

    s_lat, s_lon = _geocode_in_dfw(start)
    e_lat, e_lon = _geocode_in_dfw(end)

    G = get_walking_graph()
    if grid is None:
        grid = _mock_snapshot()
    _annotate_edges(G, grid)

    s_node = _nearest_node(G, s_lat, s_lon)
    e_node = _nearest_node(G, e_lat, e_lon)

    def _edge_cleanest(d: dict) -> float:
        return float(d.get("length", 0.0)) * (float(d.get("pm_midpoint", 0.0)) + alpha)

    def _edge_shortest(d: dict) -> float:
        return float(d.get("length", 0.0))

    def _path_cleanest(u, v, multi_d: dict) -> float:
        # MultiDiGraph: networkx hands us the {key: attrs} dict; pick min
        # over parallel edges. For a plain Graph, multi_d is just attrs —
        # wrap it into a 1-element dict so the same code path works.
        attrs = multi_d.values() if any(isinstance(x, dict) for x in multi_d.values()) else [multi_d]
        return min(_edge_cleanest(a) for a in attrs)

    try:
        shortest_nodes = nx.shortest_path(G, s_node, e_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        raise DisconnectedRouteError(
            f"No walking path between {start!r} and {end!r}: {e}"
        ) from e

    try:
        cleanest_nodes = nx.shortest_path(G, s_node, e_node, weight=_path_cleanest)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        raise DisconnectedRouteError(
            f"No walking path between {start!r} and {end!r}: {e}"
        ) from e

    return RouteComparison(
        cleanest=_build_route(G, cleanest_nodes, _edge_cleanest),
        shortest=_build_route(G, shortest_nodes, _edge_shortest),
    )


def _mock_snapshot() -> PipelineSnapshot:
    """5×5 synthetic snapshot with a clean-north / polluted-south PM
    gradient. Used by the CLI for offline smoke and as the find_routes
    fallback when grid=None."""
    import pandas as pd

    res = 5
    lat_grid = np.linspace(BBOX["south"], BBOX["north"], res)
    lon_grid = np.linspace(BBOX["west"], BBOX["east"], res)
    lons_2d, lats_2d = np.meshgrid(lon_grid, lat_grid)
    grid = np.empty((res, res))
    for i in range(res):
        # Row 0 = south (~30 µg/m³), row res-1 = north (~5 µg/m³).
        grid[i, :] = 30.0 - (i / (res - 1)) * 25.0

    return PipelineSnapshot(
        timestamp="mock-snapshot",
        sensor_df=pd.DataFrame(),
        lats_2d=lats_2d,
        lons_2d=lons_2d,
        grid=grid,
        confidence=np.ones((res, res)),
        wind_speed=0.0,
        wind_deg=0.0,
    )


# ---------------------------------------------------------------------------
# CLI — `python -m engine.router --start ... --end ... --mock`
# ---------------------------------------------------------------------------

def _print_route(label: str, route: Route) -> None:
    minutes = route.walk_seconds / 60.0
    print(f"{label}:")
    print(f"  distance       {route.distance_m:8.0f} m")
    print(f"  walk time      {minutes:8.1f} min")
    print(f"  mean PM2.5     {route.mean_pm25:8.2f} µg/m³")
    print(f"  total exposure {route.total_exposure:8.0f} µg/m³·m")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFW air quality route comparator")
    parser.add_argument(
        "--start", required=True,
        help="Start address — must geocode inside the DFW bbox",
    )
    parser.add_argument(
        "--end", required=True,
        help="End address — must geocode inside the DFW bbox",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a synthetic clean-north/polluted-south PM gradient instead of a live grid",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        metavar="A",
        help=f"Override ROUTE_PM_ALPHA (config default: {ROUTE_PM_ALPHA})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        metavar="PATH",
        help="Write the cleanest+shortest GeoJSON FeatureCollection to this path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    grid = _mock_snapshot() if args.mock else None
    try:
        result = find_routes(args.start, args.end, grid=grid, alpha=args.alpha)
    except GeocodeFailure as e:
        print(f"Geocoding failed: {e}", file=sys.stderr)
        sys.exit(1)
    except OutOfDFWError as e:
        print(f"Out of DFW: {e}", file=sys.stderr)
        sys.exit(1)
    except DisconnectedRouteError as e:
        print(f"No walking path: {e}", file=sys.stderr)
        sys.exit(1)

    _print_route("Shortest", result.shortest)
    print()
    _print_route("Cleanest", result.cleanest)

    if args.out:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"label": "shortest"}, "geometry": result.shortest.geometry},
                {"type": "Feature", "properties": {"label": "cleanest"}, "geometry": result.cleanest.geometry},
            ],
        }
        Path(args.out).write_text(json.dumps(fc, indent=2))
        print(f"\nGeoJSON written to {args.out}")


if __name__ == "__main__":
    main()
