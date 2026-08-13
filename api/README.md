# AERIA Backend

FastAPI service that wraps the existing DFW air quality pipeline as a JSON API
for the AERIA frontend. The legacy Streamlit app at
`app.py` keeps running unchanged — this backend only re-exposes `engine/`,
`data/`, and `config.py` as HTTP endpoints.

## Run locally

From the project root with the project venv activated:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Then open:

- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json

The server reads the same `.env` file as the Streamlit app (see `CLAUDE.md`),
so `PURPLEAIR_API_KEY`, `OPENAQ_API_KEY`, `OPENWEATHERMAP_API_KEY`, and
`TOMTOM_API_KEY` must all be set.

### Grid cache warmup (automatic)

The grid pipeline is primed in a background daemon thread on every startup,
so `/api/health` reports `cache_warm=true` within ~5–15 s and the frontend's
`cache_warm` gate doesn't deadlock on cold boot. No env var required. The
warmup runs on a daemon thread, so `/api/health` and other endpoints stay
responsive while it's priming.

### Optional: pre-load the walking graph

`AERIA_PRELOAD_GRAPH=1` triggers `engine.router.preload_graph()` in a
background thread at startup so the first `/api/route` request doesn't pay
the OSM walking-graph cold-load cost (typically 60–180 s on first run, then
5–15 s reloads from the on-disk graphml cache).

```bash
AERIA_PRELOAD_GRAPH=1 uvicorn api.main:app --reload --port 8000
```

Preload failures are non-fatal — they're logged via the `aeria.router`
named logger and the first `/api/route` call retries the load
synchronously.

## Environment

| Variable | Default | Description |
|---|---|---|
| `AERIA_PRELOAD_GRAPH` | unset | If set to `1`, pre-loads the OSM walking graph (used by `/api/route`) in a background thread at startup. |
| `AERIA_CORS_ORIGINS` | unset | Comma-separated list of additional CORS origins. Localhost dev origins (`http://localhost:5173` and `http://127.0.0.1:5173`) are always included. Set in deploy environments to add the production frontend origin (e.g. `https://aeria.vercel.app`). |
| `LOCATIONIQ_API_KEY` | required for `/api/route` | LocationIQ forward-geocoding key. Free tier (5,000/day) is sufficient. |

The resolved CORS allowlist is logged once at module load as `[cors] active origins: ...` so a misconfigured deploy is immediately obvious in the logs.

## Endpoints

All endpoints live under `/api`. GET responses are cached in memory for
30 minutes (matches the pipeline's existing refresh cadence). The single
POST endpoint (`/api/route`) is uncached at this layer — it reuses the
shared grid snapshot's cache for PM data.

### `GET /api/sensors`

Live PM2.5 readings from PurpleAir + OpenAQ inside the Dallas bounding box.
PurpleAir values are EPA-corrected at ingest; OpenAQ values are reference-grade.

```json
{
  "count": 87,
  "timestamp": "2026-05-01T17:23:11+00:00",
  "sensors": [
    {
      "sensor_id": "12345",
      "name": "Downtown Dallas",
      "lat": 32.78,
      "lon": -96.80,
      "pm25": 9.4,
      "pm25_raw": 12.1,
      "epa_corrected": 1,
      "source": "purpleair"
    }
  ]
}
```

### `GET /api/grid`

Full pipeline result: ingest → IDW → traffic/wind adjustment. Returns the
`GRID_RESOLUTION` × `GRID_RESOLUTION` PM2.5 grid plus a matching confidence
grid. **First call is slow** (5–15 seconds end-to-end including TomTom polling);
cached calls are instant.

```json
{
  "timestamp": "2026-05-01T17:23:11+00:00",
  "resolution": 200,
  "bbox": {"north": 33.08, "south": 32.55, "east": -96.46, "west": -97.05},
  "lats": [32.55, ..., 33.08],
  "lons": [-97.05, ..., -96.46],
  "pm25": [[9.4, 9.5, ...], ...],
  "confidence": [[0.81, 0.79, ...], ...],
  "wind_speed": 3.2,
  "wind_deg": 180.0,
  "sensor_count": 87,
  "avg_pm25": 9.8
}
```

`lats` and `lons` are returned as 1D arrays because both axes are regular
linspaces — full 2D meshes are reconstructable on the frontend without
shipping 40k duplicated floats.

### `GET /api/cells/{zip}`

Look up a US zip code, find the closest cell on the grid, and return that
cell's PM2.5, AQI category, neighborhood, and confidence. Uses the same
cached pipeline snapshot as `/api/grid`.

```json
{
  "zip": "75201",
  "lat": 32.78,
  "lon": -96.80,
  "cell_lat": 32.781,
  "cell_lon": -96.798,
  "cell_i": 137,
  "cell_j": 95,
  "pm25": 11.2,
  "aqi_category": "good",
  "confidence": 0.92,
  "neighborhood": "Dallas",
  "timestamp": "2026-05-01T17:23:11+00:00"
}
```

Returns 404 if the zip is unknown or falls outside the Dallas bounding box.

### `GET /api/geocode/suggest`

Server-side proxy for LocationIQ autocomplete. Used by the Route Lab
typeahead so the LocationIQ key stays off the wire and identical queries
share a 10-minute TTL cache. Normalizes results to `{display_name, lat,
lon}` only — provider-specific fields (`place_id`, `class`, `boundingbox`,
etc.) are not surfaced.

Query params:

- `q` — partial address text, min 2 chars (required)
- `limit` — max suggestions, 1–10 (default 5)

```
GET /api/geocode/suggest?q=Klyde&limit=3
```

```json
[
  {
    "display_name": "Klyde Warren Park, 2012, Woodall Rodgers Freeway, Dallas, TX, 75201, USA",
    "lat": 32.7898,
    "lon": -96.8012
  },
  ...
]
```

Empty results return `200 []` (empty typeahead is a valid state, not an
error). `503` if `LOCATIONIQ_API_KEY` is unset; `502` on any LocationIQ
upstream failure (4xx, 5xx, 429, network).

This proxy is for typeahead UX only — `POST /api/route` re-geocodes the
final picked address authoritatively via `/v1/search`.

### `POST /api/route`

Compare a length-only shortest walking path against a PM-weighted cleanest
path between two DFW addresses. Both addresses are forward-geocoded
server-side via LocationIQ; PM₂.₅ is sampled from the same grid snapshot
`/api/grid` returns. First call after a cold boot pays the OSM
walking-graph load (set `AERIA_PRELOAD_GRAPH=1` to amortize at startup).

Request:

```json
{
  "start": "Mockingbird Station Dallas",
  "end": "Klyde Warren Park Dallas"
}
```

Response:

```json
{
  "cleanest": {
    "geometry": {
      "type": "LineString",
      "coordinates": [[-96.7764, 32.8377], [-96.78, 32.83], ...]
    },
    "distance_m": 6342.0,
    "mean_pm25": 7.8,
    "walk_seconds": 4530.0,
    "total_exposure": 49467.6
  },
  "shortest": {
    "geometry": { "type": "LineString", "coordinates": [...] },
    "distance_m": 5418.0,
    "mean_pm25": 9.6,
    "walk_seconds": 3870.0,
    "total_exposure": 52012.8
  },
  "timestamp": "2026-05-08T17:23:11+00:00"
}
```

Errors:

- `400` — could not geocode an address.
- `404` — address resolves outside the DFW bbox, or no walking path exists.
- `503` — walking graph not yet loaded (transient on cold boot).
- `502` — generic routing pipeline failure.
- `422` — malformed request body (FastAPI default).

### `GET /api/health`

Cheap liveness + cache-warm probe. Hit by the frontend on every page load.
Does no I/O.

```json
{ "status": "ok", "cache_warm": true, "uptime_seconds": 142 }
```

`cache_warm` flips to `true` once the grid pipeline has been run at least
once. The startup warmup primes it within ~5–15 s of boot.

## Development tools

### OpenAPI snapshot

`api/scripts/snapshot_openapi.py` writes `api/openapi.snapshot.json` —
a pretty-printed, sorted-keys dump of the live FastAPI schema. Run it
manually whenever you want a fresh contract baseline:

```bash
python api/scripts/snapshot_openapi.py
```

Diffing the snapshot across commits surfaces unintended API contract
changes. Not automated — the developer runs it on demand.

## Architecture notes

- The backend is a **thin wrapper** — it imports `engine/`, `data/`, and
  `config.py` and does not duplicate logic. Any change to the pipeline lands
  here automatically the next time the cache expires.
- Caching is a small in-memory TTL dict per route. `routes/grid.py` owns the
  shared pipeline snapshot; `routes/cells.py` reads from the same snapshot
  so the two endpoints stay coherent within a single 30-minute window.
- CORS is built from two sources at startup: the always-included localhost
  dev origins and the comma-separated `AERIA_CORS_ORIGINS` env var. See the
  Environment section above for deploy configuration.
