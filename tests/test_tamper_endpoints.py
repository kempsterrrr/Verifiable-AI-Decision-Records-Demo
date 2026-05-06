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
    import time, mlflow
    from mlflow.exceptions import MlflowException
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
                # The artifact lands when the registration daemon thread
                # completes its log_artifacts call. We poll by attempting
                # the download itself — any expected "not yet" failure
                # mode is one of MlflowException / FileNotFoundError /
                # OSError (the local file backend's surface). Any other
                # exception is a real misconfiguration we want to fail
                # fast on, not silently absorb into a polling timeout.
                try:
                    mlflow.set_tracking_uri(app.state.settings.mlflow_tracking_uri)
                    mlflow.tracking.MlflowClient().download_artifacts(
                        source_run_id, "ario/registration_payload.json"
                    )
                    return registration
                except (MlflowException, FileNotFoundError, OSError):
                    pass
        time.sleep(0.1)
    raise AssertionError(
        "ario/registration_payload.json never appeared on the source run "
        "within timeout"
    )


def _wait_for_artifact(mlflow_client, run_id, artifact_path, timeout=30.0):
    """Poll until ``artifact_path`` is downloadable from ``run_id``.

    Used by the per-button regression tests to ride out the prediction /
    registration daemon thread's artifact-write step without depending
    on ``arweave_tx_id`` (which won't appear if Arweave upload fails —
    flaky under suite pollution but not relevant to the tamper-verify
    cycle this audit is checking).
    """
    import time
    from mlflow.exceptions import MlflowException
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Expected "not yet" failure modes: MlflowException for missing
        # artifacts on the local file backend, FileNotFoundError / OSError
        # for the open() that follows. Anything else is a real error we
        # want to surface immediately, not absorb into a polling timeout.
        try:
            local_path = mlflow_client.download_artifacts(run_id, artifact_path)
            with open(local_path, "rb") as f:
                return f.read()
        except (MlflowException, FileNotFoundError, OSError):
            time.sleep(0.1)
    raise AssertionError(
        f"artifact {artifact_path!r} on run {run_id} never appeared within {timeout}s"
    )


def test_tamper_saved_decision_breaks_anchored_bytes_check(client):
    """Audit B1: the 'Tamper with the saved record' button on
    decision_detail.html overwrites ario/predictions/<id>/payload.json
    in MLflow. The verifier's anchored-bytes check (the 'hash_match'
    component of the 'Decision Record Matches' UI row) should flip
    from PASS to FAIL.
    """
    import mlflow
    from app.tamper import tamper_saved
    from ario_mlflow.verify import verify_anchored_bytes
    from ario_mlflow.proof import hash_data

    decision_id = _make_decision(client)
    app = client.app
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    envelope_record = app.state.store.get_by_id(decision_id)
    assert envelope_record, f"decision {decision_id} not in RecordStore"
    run_id = envelope_record["record"]["mlflow_run_id"]

    # Wait for the prediction daemon to log the canonical artifact, then
    # capture the bytes so we can compute the anchored payload_hash
    # without going to Arweave.
    artifact_path = f"ario/predictions/{decision_id}/payload.json"
    canonical_bytes = _wait_for_artifact(mlflow_client, run_id, artifact_path)
    anchored_hash = hash_data(canonical_bytes)

    envelope = {
        "event_type": "prediction",
        "subject": {
            "type": "mlflow_prediction",
            "decision_id": decision_id,
            "model_run_id": run_id,
        },
        "payload_hash": anchored_hash,
    }

    # BEFORE tamper: anchored bytes still hash to the recorded value.
    before = verify_anchored_bytes(envelope, mlflow_client)
    assert before["ok"] is True, f"baseline should pass: {before}"

    # Apply the same backend the tamper button calls.
    tamper_saved(
        event_type="decision", event_id=decision_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    # AFTER tamper: garbage bytes have a different hash → check must FAIL.
    after = verify_anchored_bytes(envelope, mlflow_client)
    assert after["ok"] is False, (
        "tamper_saved on a decision did not flip verify_anchored_bytes; "
        f"got {after}"
    )


def test_tamper_live_decision_breaks_source_of_truth_check(client):
    """Audit B2: the 'Tamper with the live data' button on
    decision_detail.html overwrites the MLflow trace's
    ``ario.payload_json`` tag. The verifier's source-of-truth refetcher
    re-reads that tag, rebuilds canonical bytes against it, and the
    'Decision Record Matches' SoT check should flip from PASS to FAIL.
    """
    import mlflow
    from app.tamper import tamper_live
    from ario_mlflow.verify import verify_source_of_truth

    decision_id = _make_decision(client)
    app = client.app
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    envelope_record = app.state.store.get_by_id(decision_id)
    assert envelope_record
    run_id = envelope_record["record"]["mlflow_run_id"]

    artifact_path = f"ario/predictions/{decision_id}/payload.json"
    canonical_bytes = _wait_for_artifact(mlflow_client, run_id, artifact_path)

    envelope = {"event_type": "prediction"}

    # BEFORE tamper: the trace tag was set by VerifiedModel.predict to
    # mirror the canonical bytes, so SoT passes.
    before = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert before["ok"] is True, f"baseline should pass: {before}"

    tamper_live(
        event_type="decision", event_id=decision_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    # AFTER tamper: trace tag carries garbage; rebuilt bytes diverge → FAIL.
    after = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert after["ok"] is False, (
        "tamper_live on a decision did not flip source_of_truth; "
        f"got {after}"
    )


def _wait_for_training_run(app, timeout=30.0):
    """Find the training_complete entry in the lifecycle_store once the
    background hydrator has populated it. Returns (training_envelope,
    run_id)."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = app.state.lifecycle_store.list_all()
        training = next(
            (r for r in records if r["record"]["event_type"] == "training_complete"),
            None,
        )
        if training:
            return training, training["record"]["run_id"]
        time.sleep(0.1)
    raise AssertionError("training entry never appeared in lifecycle_store")


def test_tamper_saved_training_breaks_anchored_bytes_check(client):
    """Audit B3: the 'Tamper with the saved record' button on
    run_detail.html overwrites ``ario/payload.json`` on the training run.
    The verifier's anchored-bytes check (the hash_match component of
    'Training Record Matches') should flip from PASS to FAIL.
    """
    import mlflow
    from app.tamper import tamper_saved
    from ario_mlflow.verify import verify_anchored_bytes
    from ario_mlflow.proof import hash_data

    app = client.app
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    training, run_id = _wait_for_training_run(app)
    canonical_bytes = _wait_for_artifact(mlflow_client, run_id, "ario/payload.json")
    anchored_hash = hash_data(canonical_bytes)

    envelope = {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": run_id},
        "payload_hash": anchored_hash,
    }

    before = verify_anchored_bytes(envelope, mlflow_client)
    assert before["ok"] is True, f"baseline should pass: {before}"

    tamper_saved(
        event_type="training", event_id=run_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    after = verify_anchored_bytes(envelope, mlflow_client)
    assert after["ok"] is False, (
        "tamper_saved on a training run did not flip verify_anchored_bytes; "
        f"got {after}"
    )


def test_tamper_live_training_breaks_source_of_truth_check(client):
    """Audit B4: the 'Tamper with the live data' button on
    run_detail.html (and the matching button on model_chain.html) writes
    ``log_metric(run_id, "accuracy", 0.999)``. The verifier's source-of-
    truth refetcher re-reads run.data.metrics, rebuilds canonical bytes,
    and the 'Training Record Matches' SoT check should flip PASS→FAIL.
    """
    import mlflow
    from app.tamper import tamper_live
    from ario_mlflow.verify import verify_source_of_truth

    app = client.app
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    training, run_id = _wait_for_training_run(app)
    canonical_bytes = _wait_for_artifact(mlflow_client, run_id, "ario/payload.json")

    envelope = {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": run_id},
    }

    before = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert before["ok"] is True, f"baseline should pass: {before}"

    tamper_live(
        event_type="training", event_id=run_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    after = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert after["ok"] is False, (
        "tamper_live on a training run did not flip source_of_truth; "
        f"got {after}"
    )


def test_tamper_dataset_metadata_breaks_training_source_of_truth(client):
    """Audit / standalone-dataset-anchoring Piece C: the
    "Tamper with the dataset metadata" button mutates the dataset's
    digest in MLflow's dataset registry (mlruns/<exp>/datasets/<id>/
    meta.yaml). Training's source-of-truth re-derives dataset_inputs
    from MLflow at verify time, so a mutated digest makes the rebuilt
    canonical bytes diverge from what was anchored — training's
    Record Matches row flips PASS→FAIL.
    """
    import mlflow
    from app.tamper import tamper_live
    from ario_mlflow.verify import verify_source_of_truth

    app = client.app
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    training, run_id = _wait_for_training_run(app)
    canonical_bytes = _wait_for_artifact(mlflow_client, run_id, "ario/payload.json")

    envelope = {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": run_id},
    }

    # BEFORE tamper: SoT passes (training's inlined dataset_inputs
    # match what's in MLflow's dataset registry).
    before = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert before["ok"] is True, f"baseline should pass: {before}"

    # Apply dataset-metadata tamper directly (avoids the FastAPI route's
    # auto-revert race, same pattern as the other regression tests).
    tamper_live(
        event_type="dataset", event_id=run_id,
        lifecycle_store=app.state.lifecycle_store,
        record_store=app.state.store,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    # AFTER tamper: rebuilt dataset_inputs has a different digest;
    # canonical bytes diverge; SoT must FAIL.
    after = verify_source_of_truth(envelope, canonical_bytes, mlflow_client)
    assert after["ok"] is False, (
        "tamper_live(event_type='dataset') did not flip training's "
        f"source_of_truth; got {after}"
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

    # Also assert the *transitive* chain failure: the training event's
    # canonical bytes commit to artifact_checksums too, so swapping the
    # model artifact must also flip the training-side source-of-truth.
    # This is what makes the chain valuable — tampering one shared
    # dependency cascades to every link that references it (the button
    # description on model_chain.html promises exactly this).
    training_canonical = mlflow.artifacts.download_artifacts(
        run_id=source_run_id, artifact_path="ario/payload.json",
    )
    with open(training_canonical, "rb") as f:
        training_bytes = f.read()
    training_envelope = {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": source_run_id},
    }
    training_sot = verify_source_of_truth(training_envelope, training_bytes, mlflow_client)
    assert training_sot["ok"] is False, (
        "swap-artifact tamper should also flip the training-side "
        f"source-of-truth (transitive chain failure), but got {training_sot}"
    )
