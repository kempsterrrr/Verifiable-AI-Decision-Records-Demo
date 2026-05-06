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
    # Clear the lru_cache so get_settings() picks up the monkeypatched env.
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import app
    # Use as context manager to trigger lifespan (which trains the model,
    # sets app.state.settings, etc.).
    with TestClient(app) as c:
        yield c


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


def _wait_for_registration(app, timeout=30.0):
    """Block until the registration daemon thread has finished writing
    the canonical artifact to MLflow.

    The post-condition we actually need for the swap-artifact regression
    is that ``ario/registration_payload.json`` exists on the source run.
    Don't wait for ``arweave_tx_id`` — under unfavorable conditions
    (offline, gateway reject, bad wallet state) the Arweave upload may
    fail while the local artifact write still succeeds, and that's
    enough to exercise the verifier's source-of-truth path.
    """
    import os, time, mlflow
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = app.state.lifecycle_store.list_all()
        registration = next(
            (r for r in records if r["record"]["event_type"] == "model_registered"),
            None,
        )
        if registration:
            source_run_id = registration["record"].get("source_run_id")
            if source_run_id:
                # Check the canonical artifact actually landed on disk.
                tracking_root = app.state.settings.mlflow_tracking_uri.removeprefix("file://")
                # MLflow file backend: <root>/<exp>/<run_id>/artifacts/...
                # We don't know experiment_id from here; query MLflow.
                try:
                    mlflow.set_tracking_uri(app.state.settings.mlflow_tracking_uri)
                    mlflow.tracking.MlflowClient().download_artifacts(
                        source_run_id, "ario/registration_payload.json"
                    )
                    return registration
                except Exception:
                    pass
        time.sleep(0.1)
    raise AssertionError(
        "ario/registration_payload.json never appeared on the source run "
        "within timeout"
    )


def test_swap_artifact_tamper_breaks_registration_source_of_truth(client):
    """Regression test for the swap-deployed-model-artifact tamper.

    The ``tamper_live(event_type="registration", ...)`` call swaps the
    bytes of ``model.pkl`` in MLflow, which should make the verifier's
    source-of-truth check FAIL when it re-derives ``artifact_verified``
    against the new on-disk hash. Pre-fix bug: the tamper resolved the
    target path via ``mlflow.artifacts.download_artifacts(run_id, "model")``
    which in MLflow 3.x returns an ephemeral *temp copy* — writing there
    doesn't mutate the canonical LoggedModel store, so the verifier's
    next ``download_artifacts`` call read the still-untouched bytes and
    the tamper went undetected.
    """
    import mlflow
    from ario_mlflow.verify import verify_source_of_truth

    app = client.app
    registration = _wait_for_registration(app)
    event_id = registration["record"]["event_id"]
    rec = registration["record"]
    source_run_id = rec["source_run_id"]

    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    # The anchored canonical bytes for a registration live on the source
    # training run as ario/registration_payload.json (operator side of
    # the chain — registration happens by tagging an existing run, not
    # creating a new one). Download directly using source_run_id.
    local_path = mlflow.artifacts.download_artifacts(
        run_id=source_run_id, artifact_path="ario/registration_payload.json",
    )
    with open(local_path, "rb") as f:
        canonical_bytes = f.read()

    envelope = {
        "event_type": "model_registered",
        "subject": {
            "type": "mlflow_model_version",
            "name": rec["model_name"],
            "version": str(rec["model_version"]),
        },
    }

    # BEFORE tamper: source-of-truth must PASS — the LoggedModel store
    # still holds the artifact bytes that were anchored.
    sot_before = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert sot_before["ok"] is True, f"baseline source-of-truth should pass: {sot_before}"

    # Apply swap-artifact tamper by calling the tamper_live function
    # directly, bypassing the FastAPI route. This avoids the route's
    # auto-revert BackgroundTask, which can race with the post-tamper
    # source-of-truth check we're trying to observe. (The auto-revert
    # itself is exercised separately by reset-related tests.)
    from app.tamper import tamper_live
    snap = tamper_live(
        event_type="registration", event_id=event_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    assert snap is not None, "tamper_live returned no snapshot"
    # The snapshot's live_field_name encodes the on-disk path the
    # tamper wrote to — sanity check that file is now non-original.
    assert snap.live_field_name and snap.live_field_name.startswith("artifact_swap_path:"), snap.live_field_name
    target_path = snap.live_field_name.split(":", 1)[1]
    with open(target_path, "rb") as f:
        post_tamper = f.read()
    assert post_tamper.startswith(b"TAMPERED"), (
        f"tamper_live did not write the expected bytes to {target_path}"
    )

    # AFTER tamper: source-of-truth MUST FAIL. The artifact_checksums
    # re-derived from the canonical store must differ from what was
    # anchored, flipping artifact_verified to False, which makes the
    # rebuilt canonical bytes diverge.
    sot_after = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert sot_after["ok"] is False, (
        "source-of-truth should FAIL after model.pkl was swapped on the "
        f"canonical LoggedModel store, but got {sot_after}"
    )
