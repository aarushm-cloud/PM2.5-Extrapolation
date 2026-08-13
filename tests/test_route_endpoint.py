"""Tests for POST /api/route.

Both engine.router.find_routes and api.routes.grid.get_cached_snapshot
are mocked at api.routes.route's import sites so these tests don't need
a real grid pipeline, OSMnx graph, or LocationIQ key. The router's own
tests cover the geocoding + Dijkstra logic; this file pins the HTTP
contract: status codes, response shape, error mapping, CORS preflight.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from engine.router import (
    DisconnectedRouteError,
    GeocodeFailure,
    OutOfDFWError,
    Route,
    RouteComparison,
)
from engine.snapshot import PipelineSnapshot

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stub_snapshot() -> PipelineSnapshot:
    """Minimal PipelineSnapshot — the route module only reads `.timestamp`
    once we've mocked find_routes, so the rest can be cheap defaults."""
    return PipelineSnapshot(
        timestamp="2026-05-08T17:23:11+00:00",
        sensor_df=pd.DataFrame(),
        lats_2d=np.zeros((2, 2)),
        lons_2d=np.zeros((2, 2)),
        grid=np.zeros((2, 2)),
        confidence=np.ones((2, 2)),
        wind_speed=0.0,
        wind_deg=0.0,
    )


def _stub_route(distance_m: float, mean_pm25: float) -> Route:
    return Route(
        geometry={"type": "LineString", "coordinates": [[-96.78, 32.84], [-96.80, 32.79]]},
        distance_m=distance_m,
        mean_pm25=mean_pm25,
        walk_seconds=distance_m / 1.4,
        total_exposure=mean_pm25 * distance_m,
    )


def _stub_comparison() -> RouteComparison:
    return RouteComparison(
        cleanest=_stub_route(distance_m=6342.0, mean_pm25=7.8),
        shortest=_stub_route(distance_m=5418.0, mean_pm25=9.6),
    )


@pytest.fixture
def patched_snapshot():
    """Mock the shared grid snapshot so we don't fetch real PurpleAir/OpenAQ
    data. Note: api.routes.route imports get_cached_snapshot directly, so
    patch it where it's bound, not at the api.routes.grid origin."""
    with patch("api.routes.route.get_cached_snapshot", return_value=_stub_snapshot()) as m:
        yield m


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_post_route_happy_path_returns_route_response(patched_snapshot):
    with patch("api.routes.route.find_routes", return_value=_stub_comparison()):
        resp = client.post(
            "/api/route",
            json={"start": "Mockingbird Station Dallas", "end": "Klyde Warren Park Dallas"},
        )
    assert resp.status_code == 200
    body = resp.json()
    # RouteResponse shape
    assert set(body.keys()) == {"cleanest", "shortest", "timestamp"}
    assert body["timestamp"] == "2026-05-08T17:23:11+00:00"
    # Both routes populated with full RouteStats shape
    for label in ("cleanest", "shortest"):
        stats = body[label]
        assert set(stats.keys()) == {
            "geometry", "distance_m", "mean_pm25", "walk_seconds", "total_exposure",
        }
        assert stats["geometry"]["type"] == "LineString"
        assert isinstance(stats["geometry"]["coordinates"], list)
        assert stats["geometry"]["coordinates"][0] == [-96.78, 32.84]
    assert body["cleanest"]["distance_m"] == pytest.approx(6342.0)
    assert body["shortest"]["distance_m"] == pytest.approx(5418.0)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def test_post_route_geocode_failure_maps_to_400(patched_snapshot):
    with patch(
        "api.routes.route.find_routes",
        side_effect=GeocodeFailure("No geocoding match for 'asdfqwer'."),
    ):
        resp = client.post(
            "/api/route",
            json={"start": "asdfqwer", "end": "Klyde Warren Park Dallas"},
        )
    assert resp.status_code == 400
    assert "Could not geocode address" in resp.json()["detail"]
    assert "asdfqwer" in resp.json()["detail"]


def test_post_route_out_of_dfw_maps_to_404(patched_snapshot):
    with patch(
        "api.routes.route.find_routes",
        side_effect=OutOfDFWError(
            "'Times Square NY' resolved to (40.7580, -73.9855), outside the DFW bbox."
        ),
    ):
        resp = client.post(
            "/api/route",
            json={"start": "Times Square NY", "end": "Klyde Warren Park Dallas"},
        )
    assert resp.status_code == 404
    assert "outside the DFW bounding box" in resp.json()["detail"]


def test_post_route_disconnected_maps_to_404(patched_snapshot):
    with patch(
        "api.routes.route.find_routes",
        side_effect=DisconnectedRouteError("No walking path between A and B."),
    ):
        resp = client.post(
            "/api/route",
            json={"start": "A", "end": "B"},
        )
    assert resp.status_code == 404
    assert "No walking path exists" in resp.json()["detail"]


def test_post_route_generic_exception_maps_to_502(patched_snapshot):
    """Anything the router didn't model must surface as 502, not a 500
    traceback. The detail string should at least include the exception
    type so the client sees something more than a generic 'failure'."""
    with patch(
        "api.routes.route.find_routes",
        side_effect=ValueError("unexpected pipeline state"),
    ):
        resp = client.post(
            "/api/route",
            json={"start": "A", "end": "B"},
        )
    assert resp.status_code == 502
    assert "Routing pipeline failure" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

def test_post_route_missing_start_field_returns_422():
    resp = client.post("/api/route", json={"end": "Klyde Warren Park Dallas"})
    assert resp.status_code == 422


def test_post_route_wrong_field_types_return_422():
    resp = client.post("/api/route", json={"start": 12345, "end": ["not", "a", "string"]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint-layer route cache
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_route_cache():
    """Reset the module-level route cache before and after each test that
    exercises caching behaviour. Cleanup runs in both directions so a
    cached entry can never leak into the rest of the suite."""
    from api.routes.route import _route_cache
    _route_cache.clear()
    yield _route_cache
    _route_cache.clear()


def test_route_cache_hits_on_repeated_request(patched_snapshot, empty_route_cache):
    """Two POSTs with the same start/end against the same grid snapshot
    must call find_routes only once; the second response must equal the first."""
    with patch(
        "api.routes.route.find_routes",
        return_value=_stub_comparison(),
    ) as mock_find:
        body = {"start": "Mockingbird Station Dallas", "end": "Klyde Warren Park Dallas"}
        r1 = client.post("/api/route", json=body)
        r2 = client.post("/api/route", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    assert mock_find.call_count == 1


def test_route_cache_normalizes_address_casing(patched_snapshot, empty_route_cache):
    """Inputs that differ only in casing or surrounding whitespace must
    collapse to the same cache key — so a uppercase-then-lowercase pair
    only computes once."""
    with patch(
        "api.routes.route.find_routes",
        return_value=_stub_comparison(),
    ) as mock_find:
        client.post(
            "/api/route",
            json={"start": "Mockingbird Station", "end": "Klyde Warren Park"},
        )
        client.post(
            "/api/route",
            json={"start": "  mockingbird station  ", "end": "klyde warren park"},
        )
    assert mock_find.call_count == 1


def test_route_cache_distinct_keys(patched_snapshot, empty_route_cache):
    """Different address pairs must each compute independently."""
    with patch(
        "api.routes.route.find_routes",
        return_value=_stub_comparison(),
    ) as mock_find:
        client.post("/api/route", json={"start": "Pair One Start", "end": "Pair One End"})
        client.post("/api/route", json={"start": "Pair Two Start", "end": "Pair Two End"})
    assert mock_find.call_count == 2


def test_route_cache_invalidates_on_new_grid_snapshot(empty_route_cache):
    """When the underlying grid snapshot rotates between calls, the cached
    entry's timestamp no longer matches and must be discarded — find_routes
    runs again. After both calls, the cache holds exactly one entry for the
    key (the T1 response overwrote the T0 entry in place)."""
    snap_t0 = _stub_snapshot()
    snap_t1 = PipelineSnapshot(
        timestamp="2026-05-08T18:00:00+00:00",
        sensor_df=snap_t0.sensor_df,
        lats_2d=snap_t0.lats_2d,
        lons_2d=snap_t0.lons_2d,
        grid=snap_t0.grid,
        confidence=snap_t0.confidence,
        wind_speed=snap_t0.wind_speed,
        wind_deg=snap_t0.wind_deg,
    )
    with patch(
        "api.routes.route.get_cached_snapshot",
        side_effect=[snap_t0, snap_t1],
    ), patch(
        "api.routes.route.find_routes",
        return_value=_stub_comparison(),
    ) as mock_find:
        body = {"start": "Rotation Start", "end": "Rotation End"}
        r1 = client.post("/api/route", json=body)
        r2 = client.post("/api/route", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert mock_find.call_count == 2
    assert len(empty_route_cache) == 1
    assert empty_route_cache[("rotation start", "rotation end")].timestamp == snap_t1.timestamp


# ---------------------------------------------------------------------------
# CORS preflight
# ---------------------------------------------------------------------------

def test_cors_preflight_advertises_post_method():
    """A browser preflight OPTIONS request from the dev origin must come
    back advertising POST as an allowed method. If CORS is still locked
    to GET-only, the browser will silently block /api/route."""
    resp = client.options(
        "/api/route",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    # CORSMiddleware returns 200 for an authorized preflight.
    assert resp.status_code == 200
    allow_methods = resp.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods, (
        f"preflight must advertise POST; got methods={allow_methods!r}"
    )
