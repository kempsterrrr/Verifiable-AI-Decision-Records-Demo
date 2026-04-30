"""Tamper endpoint tests.

Each tamper mutates real MLflow state and the verifier should catch it.
Reset restores the original state. Auto-revert (background timer) is
tested separately via direct call to the revert helper, not via real
sleep.
"""
import os
import json
import tempfile
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Boot the demo app with an isolated MLflow + records directory.

    Each test gets a fresh tracking store and a fresh model trained
    automatically by the lifespan handler (existing behavior).
    """
    monkeypatch.setenv("VAIDR_RECORDS_FILE", str(tmp_path / "records.json"))
    monkeypatch.setenv("VAIDR_LIFECYCLE_FILE", str(tmp_path / "lifecycle.json"))
    monkeypatch.setenv("VAIDR_MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    # Disable Arweave so anchoring doesn't try to hit the network.
    monkeypatch.setenv("VAIDR_ARWEAVE_WALLET_PATH", "")
    from app.main import app
    return TestClient(app)


def _make_decision(client):
    """Helper: make a prediction and return its decision_id."""
    client.post("/predict-form", data={
        "annual_income": "78000",
        "credit_utilization": "0.18",
        "debt_to_income_ratio": "0.22",
        "months_employed": "72",
        "credit_score": "745",
    }, follow_redirects=False)
    decisions = client.get("/decisions").json()
    assert len(decisions) >= 1, "expected at least one decision after predict"
    return decisions[0]["record"]["decision_id"]


def test_tamper_saved_record_returns_ok(client):
    """POST /tamper/saved/decision/{event_id} writes garbage to payload.json
    and returns success."""
    decision_id = _make_decision(client)
    r = client.post(f"/tamper/saved/decision/{decision_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tampered"] is True
    assert body["kind"] == "saved"


def test_tamper_live_data_returns_ok(client):
    """POST /tamper/live/decision/{event_id} mutates the trace's
    ario.payload_json tag and returns success."""
    decision_id = _make_decision(client)
    r = client.post(f"/tamper/live/decision/{decision_id}")
    assert r.status_code == 200, r.text
    assert r.json()["tampered"] is True


def test_tamper_reset_restores_state(client):
    """After tamper + reset, the verification should pass again."""
    decision_id = _make_decision(client)
    client.post(f"/tamper/saved/decision/{decision_id}")
    r = client.post(f"/tamper/reset/decision/{decision_id}")
    assert r.status_code == 200, r.text
    assert r.json()["reset"] is True


def test_tamper_unknown_event_id_returns_404(client):
    """Tampering an event that doesn't exist returns 404."""
    r = client.post("/tamper/saved/decision/no-such-id")
    assert r.status_code == 404


def test_tamper_unknown_event_type_returns_400(client):
    """Tampering an unknown event_type returns 400."""
    r = client.post("/tamper/saved/banana/some-id")
    assert r.status_code == 400
