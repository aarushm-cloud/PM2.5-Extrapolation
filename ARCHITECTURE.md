# DFW Air Quality — Architecture

How the pieces fit together. For *what* the project does, see [README.md](README.md). For the math behind interpolation and adjustments, see [ALGORITHMS.md](ALGORITHMS.md).

---

## System topology

```
                ┌─────────────────────────────────────────────┐
                │   External APIs (free tier)                 │
                │   PurpleAir · OpenAQ · OpenWeatherMap       │
                │   TomTom Traffic · OSM/Overpass · Meteostat │
                └────────────────────┬────────────────────────┘
                                     │ HTTP (requests + requests-cache)
                                     ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  data/ingestion/                                                 │
   │    purpleair · openaq · weather · traffic · history              │
   │  data/spatial/  spatial_features (OSMnx)                         │
   │  data/corrections.py  EPA correction (shared)                    │
   └────────────────────────────────┬─────────────────────────────────┘
                                    │  pandas DataFrames
                                    ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  engine/                                                         │
   │    features      per-sensor live feature columns                 │
   │    interpolation IDW grid + post-IDW grid adjustment             │
   │    adjustments   traffic/wind math (scalar + vectorised)         │
   │    router        OSM walking-graph cleanest-path optimizer       │
   │    snapshot      single-call pipeline assembly                   │
   └──────────────────────────────────────────────────────────────────┘
                  ▲                                ▲
                  │ direct import                  │ direct import
                  │                                │
   ┌──────────────┴────────────┐      ┌────────────┴───────────────────┐
   │  app.py + viz/heatmap.py  │      │  api/  (FastAPI)               │
   │  Streamlit (local only)   │      │   routes: sensors · grid       │
   │                           │      │           cells · route        │
   │                           │      │           geocode · health     │
   │                           │      │   schemas: pydantic models     │
   │                           │      └────────────┬───────────────────┘
   │                           │                   │ HTTPS / JSON
   │                           │                   ▼
   │                           │      ┌────────────────────────────────┐
   │                           │      │  web/  (Vite + React + R3F)    │
   │                           │      │   AERIA dashboard (Vercel)     │
   │                           │      └────────────────────────────────┘
   └───────────────────────────┘
```

Two frontends share one engine. The Streamlit app stays for local cross-comparison; AERIA is the deployed product.

---

## Layer responsibilities

| Layer | Role | What it must NOT do |
|---|---|---|
| `data/ingestion/` | Talk to external APIs, return clean pandas DataFrames | Compute IDW, adjust readings, hold business logic |
| `data/corrections.py` | EPA correction formula (single source of truth) | Be re-implemented elsewhere |
| `data/spatial/` | OSM-derived static features (highway distance, etc.) | Touch live readings |
| `engine/` | All numerical pipeline: features, IDW, adjustments, routing | Make network calls, know about HTTP or React |
| `api/` | Thin JSON wrapper over `engine/` and `data/` | Duplicate engine logic — must import and reuse |
| `web/` | UI, state, 3D scene, fetch layer | Compute air-quality logic client-side |
| `ml/` | Offline training, evaluation, audit trail | Run in the live request path (RF was evaluated and shelved) |
| `viz/` | Folium rendering for the Streamlit app | Be wired into AERIA |
| `scripts/` | Headless jobs (snapshot collector) | Bypass `engine/` or duplicate ingestion |

The `api/` boundary is load-bearing: if you find yourself porting engine math into a route handler, the route handler is wrong.

---

## Request lifecycle (AERIA)

1. **Frontend mounts.** `web/src/App.tsx` boots; Zustand stores in `web/src/state/` initialise empty. The client polls `GET /api/health` and gates rendering on `cache_warm=true`.
2. **Backend warmup.** On startup, `api/main.py` spawns a daemon thread that calls `get_cached_snapshot()` — this runs the full pipeline once and stores the result. `/api/health` flips to `cache_warm=true` once the prime completes (~5–15s typical).
3. **Initial data fetch.** Frontend issues `GET /api/sensors` and `GET /api/grid` in parallel through `web/src/api/client.ts`. Both read from the cached snapshot.
4. **Render.** `components/scene/` builds the R3F city: cell grid, generated buildings, AQI-driven particles. `components/ui/` renders the top status bar and persistent left panel from the same snapshot.
5. **Interaction.** Cell hover/click is local state. ZIP search hits `/api/geocode/suggest` → updates the camera + selection. Drill-down panel pulls from `/api/cells/{id}` for per-cell breakdown (traffic adjustment, wind adjustment, attribution).
6. **Refresh.** APScheduler (`api/main.py`, single-worker assumption) re-runs the pipeline every 25 minutes and replaces the cached snapshot. Frontend re-fetches on a timer; users see the "updated N min ago" stamp slide.

---

## The pipeline (single request through `engine/`)

```
1. fetch_sensors()  (PurpleAir, EPA-corrected at the source)
2. fetch_openaq()   (reference-grade, NOT EPA-corrected)
3. pd.concat        (source column preserved: purpleair | openaq)
4. fetch_traffic()  (TomTom live congestion grid)
5. fetch_wind()     (OpenWeatherMap live wind speed + direction)
6. build_features() (per-sensor metadata: nearest road, distance, etc.
                     pm25 is NOT modified here)
7. run_idw()        (200×200 grid over Dallas bbox, k=5 nearest sensors,
                     cosine-corrected distance)
8. adjust_grid()    (traffic + wind adjustments on interpolated cells only)
9. (Streamlit)      gaussian_filter → colormap → Folium ImageOverlay
   (AERIA)          serialise to JSON → R3F renders cell grid + particles
```

Steps 1–8 are pure Python and identical for both frontends. Step 9 diverges by consumer.

---

## Load-bearing invariants

These are conventions enforced by code review, not the type system. Breaking one silently corrupts output.

- **Sensors are never adjusted.** PurpleAir/OpenAQ readings reflect real-world traffic and wind at their physical locations. Adjustments apply only to interpolated grid cells where IDW has no road or wind context. `build_features()` computes traffic/wind *metadata* per sensor; it does not modify `pm25`.
- **EPA correction is applied once, at ingestion.** PurpleAir raw → `data/corrections.py` → corrected `pm25`. The raw value is preserved as `pm25_raw` and `epa_corrected=1`. OpenAQ rows have `pm25_raw=NaN, epa_corrected=0`. Downstream code must read `pm25` and trust it; re-applying the formula double-corrects.
- **Live and training pipelines are separate files writing separate outputs.** `data/ingestion/history.py` → `data/dashboard_snapshots.csv` (live, append-only). `ml/training/collect_training_data.py` → `ml/data/history.csv` (training, overwritten per run). A training rebuild must never corrupt the dashboard's accumulated state.
- **RF feature boundary.** RF infers only from columns the training collector produces. Live-only columns from `engine/features.py` (`nearest_congestion`, `distance_to_road_m`, `traffic_factor`, `wind_term`, `direction_factor`, `dispersal`) are NOT model inputs — they can't be reconstructed historically without a paid TomTom Traffic Stats license. The two feature sets do not mix.
- **CF=1 channel for EPA correction.** Both live (`pm2.5_cf_1_10minute`) and training (`pm2.5_cf_1_a`/`_b`) pipelines feed CF=1 data into the Barkjohn 2021 formula. Substituting ATM channels yields a biased correction.
- **Cosine-corrected distance everywhere.** Every lat/lon distance calculation in the codebase multiplies longitude deltas by `cos(32.78°)` to correct ~16% east-west distortion at Dallas latitude. Adding a new distance computation without it produces silently wrong geometry.
- **Median for headline aggregate.** The top status bar's network PM₂.₅ is the median across display cells, not the mean. Sparse network + known sensor quality issues mean a single broken sensor can shift the mean materially; median absorbs it.
- **`api/` is a thin wrapper.** Route handlers compose calls into `engine/`, `data/`, and `config.py`. They serialise to Pydantic models in `api/schemas/`. They do not compute.

---

## Caching and concurrency

- **`requests-cache`** wraps HTTP calls to external APIs in `data/ingestion/*`. TTLs are per-source.
- **In-process grid cache** in `api/routes/grid.py` (`get_cached_snapshot()`) holds the most recent pipeline output. Reads are O(1); a request never triggers a synchronous pipeline run.
- **Background scheduler** (`APScheduler` in `api/main.py`) refreshes the cache every 25 minutes. `max_instances=1, coalesce=True` prevents overlap.
- **Single-worker assumption.** The scheduler runs once per process. Horizontal scaling (gunicorn `--workers N` or paid Render) requires a DB-backed jobstore or external lock — otherwise N workers run N pipelines per cycle.
- **Optional graph preload.** `AERIA_PRELOAD_GRAPH=1` loads the OSMnx walking graph in a daemon thread at startup so the first `/api/route` call doesn't pay the 60–180s cold load.

---

## Deployment topology

| Component | Host | Notes |
|---|---|---|
| `web/` (AERIA frontend) | Vercel | Built with Vite; static assets + edge serving |
| `api/` (FastAPI backend) | Render free tier | `render.yaml` + `runtime.txt` pin runtime; single worker |
| Streamlit app | Not deployed | Local-only, for cross-comparison |
| Scheduler | In-process inside the API | No separate worker dyno |

CORS: dev origins (`localhost:5173`) are always permitted; production origins are added via `AERIA_CORS_ORIGINS` env var. Brotli compression is added after CORS so headers are set before the body is compressed.

---

## Local development topology

```
./dev.sh --with-frontend                 # api:8000 + web:5173
./dev.sh --with-streamlit                # api:8000 + Streamlit
./dev.sh --with-streamlit --with-frontend  # all three
```

Each child process is prefixed in the multiplexed log (`[api]`, `[web]`). Ctrl+C cleans up the process group.

---

## Where things live (at a glance)

| Question | File |
|---|---|
| What's the Dallas bounding box? | [`config.py`](config.py) |
| How is IDW computed? | [`engine/interpolation.py`](engine/interpolation.py) |
| Where are traffic/wind adjustments? | [`engine/adjustments.py`](engine/adjustments.py) |
| Where is the EPA correction formula? | [`data/corrections.py`](data/corrections.py) |
| How is the grid cached? | [`api/routes/grid.py`](api/routes/grid.py) |
| What endpoints exist? | [`api/main.py`](api/main.py) + [`api/routes/`](api/routes/) |
| How does the frontend fetch data? | [`web/src/api/client.ts`](web/src/api/client.ts) |
| Where is the 3D scene defined? | [`web/src/components/scene/`](web/src/components/scene/) |
| How was the training set built? | [`ml/training/collect_training_data.py`](ml/training/collect_training_data.py) |
| Why didn't Random Forest ship? | [`ml/docs/rf_model_result.md`](ml/docs/rf_model_result.md) |
