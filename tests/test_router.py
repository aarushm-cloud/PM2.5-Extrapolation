"""Tests for engine.router — walking-route comparator.

All network calls (LocationIQ, OSMnx) are mocked. Routing tests use a
tiny hand-built MultiDiGraph and a 5×5 PipelineSnapshot built inline so
the expected behavior is hand-checkable from the constants below.

The diamond fixture sets up two competing detours between the same
start/end pair, with the south detour shorter (so length-only Dijkstra
prefers it) and the north detour cleaner (so PM-weighted Dijkstra
prefers it under a clean-north/polluted-south gradient).
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import requests

from config import BBOX
from engine import router
from engine.snapshot import PipelineSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_snapshot(grid_values: np.ndarray | None = None,
                        timestamp: str | None = None) -> PipelineSnapshot:
    """5×5 PipelineSnapshot over BBOX. Default PM is a linear gradient from
    30 µg/m³ at the southernmost row to 5 µg/m³ at the northernmost row."""
    res = 5
    lat_grid = np.linspace(BBOX["south"], BBOX["north"], res)
    lon_grid = np.linspace(BBOX["west"], BBOX["east"], res)
    lons_2d, lats_2d = np.meshgrid(lon_grid, lat_grid)
    if grid_values is None:
        grid_values = np.empty((res, res))
        for i in range(res):
            grid_values[i, :] = 30.0 - (i / (res - 1)) * 25.0
    return PipelineSnapshot(
        # id() of the array is unique per call so the annotation memoisation
        # in router._annotate_edges doesn't mistake two distinct test grids
        # for the same one.
        timestamp=timestamp or f"test-{id(grid_values)}",
        sensor_df=pd.DataFrame(),
        lats_2d=lats_2d,
        lons_2d=lons_2d,
        grid=grid_values,
        confidence=np.ones((res, res)),
        wind_speed=0.0,
        wind_deg=0.0,
    )


def _diamond_graph() -> nx.MultiDiGraph:
    """Two-detour test graph between start (32.78, -96.80) and end
    (32.78, -96.70):

        start ──N1──> end       north detour (32.95, -96.75) — clean
        start ──S1──> end       south detour (32.62, -96.75) — polluted

    Edge lengths: south detour 1000+1000=2000 m, north detour 1100+1100=2200 m.
    Length-only Dijkstra prefers south. PM-weighted Dijkstra under a
    clean-north gradient prefers north.
    """
    G = nx.MultiDiGraph()
    G.add_node("start", y=32.78, x=-96.80)
    G.add_node("end",   y=32.78, x=-96.70)
    G.add_node("N1",    y=32.95, x=-96.75)  # well north — sees clean PM
    G.add_node("S1",    y=32.62, x=-96.75)  # well south — sees polluted PM
    # Both directions so MultiDiGraph isn't asymmetric for these tests.
    G.add_edge("start", "N1", length=1100.0)
    G.add_edge("N1", "end",   length=1100.0)
    G.add_edge("N1", "start", length=1100.0)
    G.add_edge("end", "N1",   length=1100.0)
    G.add_edge("start", "S1", length=1000.0)
    G.add_edge("S1", "end",   length=1000.0)
    G.add_edge("S1", "start", length=1000.0)
    G.add_edge("end", "S1",   length=1000.0)
    return G


@pytest.fixture(autouse=True)
def _reset_router_state():
    """Reset every module-level singleton in engine.router so tests don't
    leak graph / annotation / cache state into each other."""
    router.refresh_walking_graph()
    router._geocode_cache.clear()
    yield
    router.refresh_walking_graph()
    router._geocode_cache.clear()


def _stub_geocode(monkeypatch, table: dict[str, tuple[float, float]]) -> None:
    """Replace router.geocode with a dict lookup. Honours the LRU cache so
    repeated lookups still observe cache-hit semantics, just without
    actually hitting the network."""
    def fake(address: str) -> tuple[float, float]:
        cached = router._geocode_cache.get(address)
        if cached is not None:
            return cached
        if address not in table:
            raise router.GeocodeFailure(f"no stub for {address!r}")
        coords = table[address]
        router._geocode_cache[address] = coords
        return coords
    monkeypatch.setattr(router, "geocode", fake)


# ---------------------------------------------------------------------------
# Routing behavior
# ---------------------------------------------------------------------------

def test_cleanest_deviates_from_shortest_under_pm_gradient(monkeypatch):
    """Under a clean-north / polluted-south gradient with the south
    detour 200 m shorter than the north detour, shortest must take the
    south path and cleanest must take the north."""
    G = _diamond_graph()
    monkeypatch.setattr(router, "get_walking_graph", lambda: G)
    _stub_geocode(monkeypatch, {
        "start_addr": (32.78, -96.80),
        "end_addr":   (32.78, -96.70),
    })

    snap = _synthetic_snapshot()
    result = router.find_routes("start_addr", "end_addr", grid=snap)

    cleanest_lats = [lat for _, lat in result.cleanest.geometry["coordinates"]]
    shortest_lats = [lat for _, lat in result.shortest.geometry["coordinates"]]

    assert max(cleanest_lats) > 32.85, (
        f"cleanest must detour through N1 (lat 32.95); got max lat {max(cleanest_lats):.3f}"
    )
    assert min(shortest_lats) < 32.70, (
        f"shortest must detour through S1 (lat 32.62); got min lat {min(shortest_lats):.3f}"
    )
    assert result.cleanest.mean_pm25 < result.shortest.mean_pm25, (
        f"cleanest mean PM {result.cleanest.mean_pm25:.2f} should beat "
        f"shortest mean PM {result.shortest.mean_pm25:.2f}"
    )


def test_cleanest_collapses_to_shortest_under_uniform_pm(monkeypatch):
    """With uniform PM across the grid, cleanest's per-edge weight is just
    a constant (PM + α) scaling of length, so cleanest must produce the
    same path as shortest."""
    G = _diamond_graph()
    monkeypatch.setattr(router, "get_walking_graph", lambda: G)
    _stub_geocode(monkeypatch, {
        "start_addr": (32.78, -96.80),
        "end_addr":   (32.78, -96.70),
    })

    snap = _synthetic_snapshot(grid_values=np.full((5, 5), 12.0))
    result = router.find_routes("start_addr", "end_addr", grid=snap)

    assert result.cleanest.geometry["coordinates"] == result.shortest.geometry["coordinates"]
    assert result.cleanest.distance_m == pytest.approx(result.shortest.distance_m)
    assert result.cleanest.mean_pm25 == pytest.approx(result.shortest.mean_pm25)


def test_alpha_scaling_changes_route_choice(monkeypatch):
    """Small α → PM dominates the per-edge cost → cleanest detours north.
    Large α → length dominates → cleanest collapses onto shortest (south)."""
    G = _diamond_graph()
    monkeypatch.setattr(router, "get_walking_graph", lambda: G)
    _stub_geocode(monkeypatch, {
        "start_addr": (32.78, -96.80),
        "end_addr":   (32.78, -96.70),
    })
    snap = _synthetic_snapshot()

    small = router.find_routes("start_addr", "end_addr", grid=snap, alpha=0.01)
    large = router.find_routes("start_addr", "end_addr", grid=snap, alpha=10_000.0)

    small_max_lat = max(lat for _, lat in small.cleanest.geometry["coordinates"])
    assert small_max_lat > 32.85, (
        f"small α should detour north; got max lat {small_max_lat:.3f}"
    )
    assert large.cleanest.geometry["coordinates"] == large.shortest.geometry["coordinates"], (
        "large α should collapse cleanest onto shortest"
    )


# ---------------------------------------------------------------------------
# Geocoding cache
# ---------------------------------------------------------------------------

def test_geocode_cache_hits_on_repeated_input(monkeypatch):
    """Two geocode calls for the same address must hit LocationIQ once
    and serve from the cache the second time.

    cachetools.LRUCache doesn't expose a `cache_info()` API like
    functools.lru_cache, so we observe cache behavior the same way
    test_spatial_cache pins it: explicit before/after counters plus a
    direct currsize read on the cache. Same intent, different surface.
    """
    call_count = {"n": 0}

    class FakeResp:
        status_code = 200
        def raise_for_status(self): return None
        # LocationIQ free-tier shape: flat list, lat/lon as strings.
        def json(self): return [
            {"lat": "32.78", "lon": "-96.80", "display_name": "downtown"}
        ]

    def fake_get(url, params=None, timeout=None):
        call_count["n"] += 1
        return FakeResp()

    monkeypatch.setenv("LOCATIONIQ_API_KEY", "test-key")
    monkeypatch.setattr(router.requests, "get", fake_get)

    coords1 = router.geocode("downtown")
    assert call_count["n"] == 1
    assert router._geocode_cache.currsize == 1

    coords2 = router.geocode("downtown")
    assert coords1 == coords2 == (32.78, -96.80)
    assert call_count["n"] == 1, "second call must hit the cache, not LocationIQ"
    assert router._geocode_cache.currsize == 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_out_of_dfw_error_for_address_outside_bbox(monkeypatch):
    """LocationIQ's viewbox is only a soft bias, not a hard filter — the
    router enforces the DFW bbox itself and raises OutOfDFWError when a
    result falls outside."""
    G = _diamond_graph()
    monkeypatch.setattr(router, "get_walking_graph", lambda: G)
    _stub_geocode(monkeypatch, {
        "ny_address": (40.71, -74.00),  # Manhattan
    })
    snap = _synthetic_snapshot()

    with pytest.raises(router.OutOfDFWError):
        router.find_routes("ny_address", "ny_address", grid=snap)


def test_geocode_failure_surfaces_locationiq_4xx(monkeypatch):
    """A LocationIQ HTTP error (e.g. 401 bad key, 429 rate limit) must
    surface as GeocodeFailure rather than bubbling up as a raw
    requests.HTTPError."""
    class FakeBadResp:
        status_code = 401
        def raise_for_status(self):
            raise requests.HTTPError("401 Unauthorized")
        def json(self):
            return {}

    monkeypatch.setenv("LOCATIONIQ_API_KEY", "test-key")
    monkeypatch.setattr(
        router.requests, "get",
        lambda url, params=None, timeout=None: FakeBadResp(),
    )

    with pytest.raises(router.GeocodeFailure):
        router.geocode("anywhere")


def test_missing_api_key_raises_geocode_failure(monkeypatch):
    """A missing or placeholder LOCATIONIQ_API_KEY must raise the same
    GeocodeFailure type as any other geocoding error, so the API layer
    has one exception class to translate. Both the unset and the
    `your_key_here` placeholder cases should fail the same way."""
    monkeypatch.delenv("LOCATIONIQ_API_KEY", raising=False)
    with pytest.raises(router.GeocodeFailure):
        router.geocode("anywhere")

    monkeypatch.setenv("LOCATIONIQ_API_KEY", "your_key_here")
    with pytest.raises(router.GeocodeFailure):
        router.geocode("anywhere")


def test_disconnected_route_error_on_two_component_graph(monkeypatch):
    """Start and end in disjoint components must surface as
    DisconnectedRouteError, not nx.NetworkXNoPath."""
    G = nx.MultiDiGraph()
    G.add_node("A", y=32.78, x=-96.80)
    G.add_node("C", y=32.79, x=-96.80)
    G.add_node("B", y=32.78, x=-96.70)
    G.add_node("D", y=32.79, x=-96.70)
    # Two islands: {A,C} and {B,D}, no edge between them.
    G.add_edge("A", "C", length=10.0)
    G.add_edge("C", "A", length=10.0)
    G.add_edge("B", "D", length=10.0)
    G.add_edge("D", "B", length=10.0)

    monkeypatch.setattr(router, "get_walking_graph", lambda: G)
    _stub_geocode(monkeypatch, {
        "a_addr": (32.78, -96.80),
        "b_addr": (32.78, -96.70),
    })
    snap = _synthetic_snapshot()

    with pytest.raises(router.DisconnectedRouteError):
        router.find_routes("a_addr", "b_addr", grid=snap)
