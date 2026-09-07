"""
Tests for the committed highway-geometry bundle.

`data/.cache/` is gitignored, so the deployed instance has no disk cache and
previously fell through to a live Overpass fetch — which Render's egress IP
gets refused on, taking down the whole grid pipeline. The bundle is a
committed artifact that makes the highway lookup work with no network.
"""

import pickle

import pytest
from shapely.geometry import LineString

import data.spatial.spatial_features as sf


def test_bundle_file_is_committed():
    """The bundle must exist in the repo, not under gitignored data/.cache/."""
    assert sf.BUNDLE_FILE.exists(), f"missing bundle: {sf.BUNDLE_FILE}"
    assert ".cache" not in sf.BUNDLE_FILE.parts


def test_bundle_loads_to_linestrings():
    geoms = sf._load_bundled_highways()
    assert len(geoms) > 1000
    assert all(isinstance(g, LineString) for g in geoms[:50])


def test_load_highways_works_with_no_cache_and_no_network(tmp_path, monkeypatch):
    """The production failure mode: no disk cache, Overpass unreachable.

    Must return real geometry from the bundle instead of raising.
    """
    monkeypatch.setattr(sf, "CACHE_FILE", tmp_path / "absent.pkl")

    def _boom():
        raise ConnectionError("overpass-api.de: Connection refused")

    monkeypatch.setattr(sf, "_fetch_and_cache_highways", _boom)

    geoms = sf._load_highways()
    assert len(geoms) > 1000


def test_distance_is_sane_from_bundle_only(tmp_path, monkeypatch):
    """Downtown Dallas is near a highway; the bundle must reproduce that."""
    monkeypatch.setattr(sf, "CACHE_FILE", tmp_path / "absent.pkl")
    monkeypatch.setattr(
        sf, "_fetch_and_cache_highways",
        lambda: (_ for _ in ()).throw(ConnectionError("refused")),
    )
    sf.refresh_highways()

    d = sf.compute_distance_to_highway(32.78, -96.80)
    assert 0.0 <= d < 5000.0
    sf.refresh_highways()


def test_bundle_matches_disk_cache_geometry():
    """Bundle is a faithful round-trip of the OSMnx pull, within rounding."""
    if not sf.CACHE_FILE.exists():
        pytest.skip("no local OSMnx disk cache to compare against")
    with sf.CACHE_FILE.open("rb") as f:
        cached = pickle.load(f)
    bundled = sf._load_bundled_highways()
    assert len(bundled) == len(cached)
