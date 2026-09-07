"""
Regenerate the committed highway bundle used by data/spatial/spatial_features.py.

`data/.cache/` is gitignored, so a deployed instance has no OSMnx disk cache
and would hit Overpass live on every cold start — which Render's egress IP
gets refused on. This bakes the geometry into the repo instead.

    python scripts/build_highway_bundle.py            # reuse disk cache if present
    python scripts/build_highway_bundle.py --refetch  # force a fresh OSM pull
"""

import argparse
import gzip
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.spatial.spatial_features import (  # noqa: E402
    BUNDLE_FILE,
    CACHE_FILE,
    _fetch_and_cache_highways,
)

COORD_PRECISION = 7  # ~0.01 m at this latitude


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true",
                        help="pull fresh geometry from OSM instead of the disk cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.refetch or not CACHE_FILE.exists():
        print("Fetching highway network from OpenStreetMap...")
        geoms = _fetch_and_cache_highways()
    else:
        print(f"Reading disk cache: {CACHE_FILE}")
        with CACHE_FILE.open("rb") as f:
            geoms = pickle.load(f)

    payload = [
        [[round(x, COORD_PRECISION), round(y, COORD_PRECISION)] for x, y in g.coords]
        for g in geoms
    ]

    BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(BUNDLE_FILE, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = BUNDLE_FILE.stat().st_size / 1024
    coords = sum(len(p) for p in payload)
    print(f"Wrote {len(payload):,} linestrings ({coords:,} coords) "
          f"to {BUNDLE_FILE} — {size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
