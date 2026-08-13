"""AERIA FastAPI backend — thin JSON wrapper around the existing pipeline.

Run from the project root:
    uvicorn api.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs

The grid cache is primed in a background daemon thread on every startup so
/api/health reports cache_warm=true within ~5–15s and the frontend's gate on
cache_warm doesn't deadlock on cold boot.

Optional startup flags:
  AERIA_PRELOAD_GRAPH=1  — pre-load the OSM walking graph in a daemon
                            thread (first /api/route
                            otherwise pays the cold-load cost).
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from brotli_asgi import BrotliMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.cells import router as cells_router
from api.routes.geocode import router as geocode_router
from api.routes.grid import get_cached_snapshot, refresh_snapshot, router as grid_router
from api.routes.health import router as health_router
from api.routes.route import router as route_router
from api.routes.sensors import router as sensors_router

logging.basicConfig(level=logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
cors_logger = logging.getLogger("aeria.cors")
router_logger = logging.getLogger("aeria.router")

# Single-worker assumption: this scheduler runs once per process.
# If scaled to multiple workers (gunicorn --workers N, paid Render
# with horizontal scale), switch to a DB-backed jobstore or
# external lock to prevent N concurrent pipeline runs every cycle.
SCHEDULER_INTERVAL_MINUTES = 25

_scheduler: BackgroundScheduler | None = None


def _start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    try:
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.add_job(
            refresh_snapshot,
            trigger="interval",
            minutes=SCHEDULER_INTERVAL_MINUTES,
            max_instances=1,
            coalesce=True,
            id="grid-refresh",
        )
        _scheduler.start()
        logger.info(
            "Background scheduler started — grid refresh every %d min",
            SCHEDULER_INTERVAL_MINUTES,
        )
    except Exception as e:
        logger.warning("Scheduler failed to start, continuing without it: %s", e)
        _scheduler = None


def _stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception as e:
        logger.warning("Scheduler shutdown error: %s", e)
    _scheduler = None

# Local Vite dev server origins — always permitted so a misconfigured deploy
# env can never break local development.
DEV_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def resolve_cors_origins() -> list[str]:
    """Build the CORS allowlist from dev defaults + the AERIA_CORS_ORIGINS env var.

    AERIA_CORS_ORIGINS is a comma-separated list of additional origins (e.g. the
    production frontend at https://aeria.vercel.app). Dev defaults are always
    included; duplicates are dropped while preserving order.
    """
    raw = os.environ.get("AERIA_CORS_ORIGINS", "")
    extras = [o.strip() for o in raw.split(",") if o.strip()]
    origins = list(dict.fromkeys([*DEV_CORS_ORIGINS, *extras]))
    cors_logger.info("[cors] active origins: %s", origins)
    return origins


def _warmup_pipeline() -> None:
    try:
        logger.info("priming grid cache in background...")
        get_cached_snapshot()
        logger.info("grid cache primed.")
    except Exception as e:
        logger.warning("pipeline prime failed: %s", e)


def _preload_walking_graph() -> None:
    # Imported here, not at module load, so a missing osmnx (or a slow
    # graph load) never blocks startup of the rest of the API.
    from engine.router import preload_graph

    try:
        router_logger.info("AERIA_PRELOAD_GRAPH=1 — loading walking graph in background...")
        t0 = time.time()
        preload_graph()
        router_logger.info(
            "AERIA_PRELOAD_GRAPH — walking graph ready in %.1fs.",
            time.time() - t0,
        )
    except Exception as e:
        # First /api/route call will retry the load synchronously through
        # the same find_routes codepath, so a preload failure is not fatal.
        router_logger.warning("AERIA_PRELOAD_GRAPH — preload failed: %s", e)


def _start_warmup() -> None:
    # Run in a daemon thread so startup doesn't block the event loop.
    # Endpoints stay responsive (e.g. /api/health) while the prime runs.
    # Always-on so Render free-tier cold boots don't deadlock the frontend's
    # cache_warm gate.
    threading.Thread(target=_warmup_pipeline, name="aeria-warmup", daemon=True).start()


def _maybe_preload_graph() -> None:
    if os.getenv("AERIA_PRELOAD_GRAPH", "0") == "1":
        # Same daemon-thread pattern as _start_warmup. Walking-graph load
        # can take 60–180s on a cold cache, so doing this synchronously
        # would block uvicorn's event loop for that whole window.
        threading.Thread(
            target=_preload_walking_graph,
            name="aeria-preload-graph",
            daemon=True,
        ).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: spawn both warmups as daemon threads and yield immediately so
    # endpoints come up while the work runs in the background. The scheduler
    # is stopped explicitly on shutdown; the daemon threads die with the
    # parent process.
    _start_warmup()
    _start_scheduler()
    _maybe_preload_graph()
    try:
        yield
    finally:
        _stop_scheduler()


app = FastAPI(
    title="AERIA · DFW Air Quality API",
    description="JSON wrapper around the DFW air quality pipeline (PurpleAir + OpenAQ + IDW + traffic/wind adjustment).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Registered after CORS so it's outermost on the response path: CORS headers
# are set first, then the body is compressed with Content-Length recomputed
# on the compressed payload. Falls back to identity for clients that don't
# send Accept-Encoding: br.
app.add_middleware(
    BrotliMiddleware,
    quality=4,
    minimum_size=500,
)

app.include_router(sensors_router, prefix="/api")
app.include_router(grid_router, prefix="/api")
app.include_router(cells_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(route_router, prefix="/api")
app.include_router(geocode_router, prefix="/api")


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "AERIA · DFW Air Quality API",
        "version": "0.1.0",
        "endpoints": [
            "/api/sensors",
            "/api/grid",
            "/api/cells/{zip}",
            "/api/health",
            "/api/route",
            "/api/geocode/suggest",
        ],
        "docs": "/docs",
    }
