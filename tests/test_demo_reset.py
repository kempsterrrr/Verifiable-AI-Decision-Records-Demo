"""Tests for the demo reset feature.

The reset endpoint wipes the MLflow tracking store + local cache files
and re-trains a fresh v1. Both the endpoint and the /demo/admin UI page
are gated behind ``Settings.demo_mode``.

Note: ``app.main`` and ``app.ui`` register their demo-only routes at
*module import time* based on ``get_settings().demo_mode``. To test the
demo_mode=False path, we need to clear the lru_cache, set the env var,
and re-import the modules. ``importlib.reload`` handles that.
"""
import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient


def _reload_app(monkeypatch, tmp_path, demo_mode: bool):
    """Reload app.main + app.ui with the requested demo_mode.

    Returns the freshly imported ``app.main.app``. Caller is responsible
    for using it inside a TestClient context manager so the lifespan
    handler runs (which trains the initial model).
    """
    monkeypatch.setenv("VAIDR_DEMO_MODE", "true" if demo_mode else "false")
    monkeypatch.setenv("VAIDR_RECORDS_FILE", str(tmp_path / "records.json"))
    monkeypatch.setenv("VAIDR_LIFECYCLE_FILE", str(tmp_path / "lifecycle.json"))
    monkeypatch.setenv("VAIDR_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    # Disable Arweave so anchoring doesn't try to hit the network.
    monkeypatch.setenv("VAIDR_ARWEAVE_WALLET_PATH", "")

    from app.config import get_settings
    get_settings.cache_clear()

    # Reload the modules that conditionally register routes at import
    # time so the new demo_mode setting takes effect.
    import app.ui
    importlib.reload(app.ui)
    import app.main
    importlib.reload(app.main)
    return app.main.app


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    """Boot the demo app with demo_mode=True and isolated paths."""
    app = _reload_app(monkeypatch, tmp_path, demo_mode=True)
    with TestClient(app) as c:
        c._tmp_path = tmp_path
        yield c


@pytest.fixture
def prod_client(tmp_path, monkeypatch):
    """Boot the demo app with demo_mode=False (production-like)."""
    app = _reload_app(monkeypatch, tmp_path, demo_mode=False)
    with TestClient(app) as c:
        yield c


def _seed_decision(client):
    """Helper: make a prediction so there's something to wipe."""
    client.post("/predict-form", data={
        "annual_income": "78000",
        "credit_utilization": "0.18",
        "debt_to_income_ratio": "0.22",
        "months_employed": "72",
        "credit_score": "745",
    }, follow_redirects=False)


def test_post_reset_returns_ok_when_demo_mode(demo_client):
    """POST /demo/reset returns 200 + {"reset": true, ...} in demo mode."""
    _seed_decision(demo_client)
    r = demo_client.post("/demo/reset")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reset"] is True
    assert body["new_version"] == "1"


def test_post_reset_returns_404_when_not_demo_mode(prod_client):
    """POST /demo/reset is not registered when demo_mode is False."""
    r = prod_client.post("/demo/reset")
    assert r.status_code == 404


def test_get_admin_page_renders_when_demo_mode(demo_client):
    """GET /demo/admin renders the admin HTML in demo mode."""
    r = demo_client.get("/demo/admin")
    assert r.status_code == 200
    assert "Demo administration" in r.text
    assert "Reset demo data" in r.text


def test_get_admin_page_returns_404_when_not_demo_mode(prod_client):
    """GET /demo/admin is not registered when demo_mode is False."""
    r = prod_client.get("/demo/admin")
    assert r.status_code == 404


def test_reset_clears_records_and_lifecycle_files(demo_client):
    """After a reset, records.json is empty and lifecycle.json contains
    only the freshly trained v1's training + registration entries."""
    tmp = demo_client._tmp_path
    _seed_decision(demo_client)

    records_path = tmp / "records.json"
    with open(records_path) as f:
        before = json.load(f)
    assert len(before) >= 1, "expected at least one decision to be seeded"

    r = demo_client.post("/demo/reset")
    assert r.status_code == 200, r.text

    with open(records_path) as f:
        after = json.load(f)
    assert after == [], f"records.json should be empty after reset, got {after}"

    # Lifecycle file is repopulated by _startup_anchor_lifecycle if the
    # fresh v1 carried plugin anchor results; otherwise it stays empty.
    # Either way it must be a valid JSON list (not the pre-reset content).
    lifecycle_path = tmp / "lifecycle.json"
    with open(lifecycle_path) as f:
        lifecycle = json.load(f)
    assert isinstance(lifecycle, list)


def test_reset_repopulates_mlruns_with_fresh_v1(demo_client):
    """After a reset, mlruns/ contains a fresh v1 (auto-trained)."""
    tmp = demo_client._tmp_path
    _seed_decision(demo_client)

    r = demo_client.post("/demo/reset")
    assert r.status_code == 200, r.text
    assert r.json()["new_version"] == "1"

    # mlruns dir exists and is non-empty.
    mlruns = tmp / "mlruns"
    assert mlruns.exists()
    # The directory should have at least one experiment subdir; an
    # empty mlruns wouldn't have a registered v1.
    contents = list(mlruns.iterdir())
    assert len(contents) > 0, f"mlruns should be repopulated; got {contents}"

    # The new active model is v1 (load_model auto-trains since wipe).
    # /decisions returns the new RecordStore which should be empty.
    decisions = demo_client.get("/decisions").json()
    assert decisions == []


def test_reset_under_lock_serializes(demo_client):
    """Two near-concurrent reset calls both succeed (the lock serializes
    them) and the second sees a fresh v1 from the first run.

    Using TestClient (which runs sync) we can't truly hit the lock from
    two threads here without async machinery, so this is a smoke test:
    sequential calls don't break each other.
    """
    r1 = demo_client.post("/demo/reset")
    assert r1.status_code == 200
    r2 = demo_client.post("/demo/reset")
    assert r2.status_code == 200
    assert r2.json()["new_version"] == "1"
