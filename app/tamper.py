"""Tamper state management for the demo's tamper buttons.

Each tamper mutates real MLflow state so the plugin's verifier catches
it organically. Pre-tamper snapshots live in-memory; reset writes them
back. Auto-revert is a background task that calls reset after a short
window (default 60s).

This module is demo-only — production deployments should never expose
these endpoints.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Literal, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


TAMPER_TTL_SECONDS = int(os.environ.get("VAIDR_TAMPER_TTL_SECONDS", "60"))


@dataclass
class TamperSnapshot:
    """Pre-tamper state captured so reset can restore it."""
    event_type: Literal["decision", "training", "registration"]
    event_id: str
    kind: Literal["saved", "live"]
    saved_artifact_bytes: Optional[bytes] = None
    live_field_name: Optional[str] = None
    live_field_old_value: Optional[str] = None


_snapshots: dict[tuple[str, str, str], TamperSnapshot] = {}
_lock = threading.Lock()


def _resolve_run_id(event_type, event_id, lifecycle_store, record_store):
    """Look up the MLflow run_id for a given event."""
    if event_type == "decision":
        envelope = record_store.get_by_id(event_id)
        if envelope is None:
            raise KeyError(f"decision {event_id} not found")
        return envelope["record"]["mlflow_run_id"]
    elif event_type == "training":
        envelope = lifecycle_store.get_by_event_id(event_id)
        if envelope is not None:
            return envelope["record"]["run_id"]
        # Fallback: event_id may actually be a run_id passed directly
        # (the run_detail page sends run_id as event_id).
        return event_id
    elif event_type == "registration":
        envelope = lifecycle_store.get_by_event_id(event_id)
        if envelope is None:
            raise KeyError(f"registration {event_id} not found")
        return envelope["record"]["source_run_id"]
    raise ValueError(f"unknown event_type: {event_type}")


def _payload_artifact_path(event_type, event_id):
    """The MLflow artifact path for the canonical bytes per event type."""
    if event_type == "decision":
        return f"ario/predictions/{event_id}/payload.json"
    elif event_type == "training":
        return "ario/payload.json"
    elif event_type == "registration":
        return "ario/registration_payload.json"
    raise ValueError(f"unknown event_type: {event_type}")


def tamper_saved(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Overwrite the canonical bytes artifact in MLflow with garbage.

    Snapshots the original bytes so reset can restore. Idempotent.
    """
    key = (event_type, event_id, "saved")
    with _lock:
        if key in _snapshots:
            return _snapshots[key]

        run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
        artifact_path = _payload_artifact_path(event_type, event_id)

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
                with open(local_path, "rb") as f:
                    original_bytes = f.read()
            except Exception as e:
                raise KeyError(f"could not download {artifact_path} for run {run_id}: {e}")

            artifact_dir = os.path.dirname(artifact_path)
            artifact_name = os.path.basename(artifact_path)
            tampered_local = os.path.join(tmpdir, artifact_name)
            with open(tampered_local, "wb") as f:
                f.write(b'{"tampered": true, "this is not the original payload": "garbage"}')
            client.log_artifact(run_id, tampered_local, artifact_path=artifact_dir)

        snapshot = TamperSnapshot(
            event_type=event_type, event_id=event_id, kind="saved",
            saved_artifact_bytes=original_bytes,
        )
        _snapshots[key] = snapshot
        logger.info(f"Tamper SAVED applied: {event_type}/{event_id}")
        return snapshot


def tamper_live(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Mutate a live MLflow field per event type.

    - decision: overwrite the trace's ario.payload_json tag.
    - training: overwrite logged accuracy metric to 0.999.
    - registration: overwrite the model version's source_run_id tag.
    """
    key = (event_type, event_id, "live")
    with _lock:
        if key in _snapshots:
            return _snapshots[key]

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        snapshot: TamperSnapshot

        if event_type == "decision":
            envelope = record_store.get_by_id(event_id)
            if envelope is None:
                raise KeyError(f"decision {event_id} not found")
            # The MLflow trace_id is stored in the canonical payload artifact,
            # not directly in the RecordStore (which stores OTel trace_id).
            # Read the payload artifact to find the mlflow_trace_id.
            run_id = envelope["record"]["mlflow_run_id"]
            artifact_path = _payload_artifact_path("decision", event_id)
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
                with open(local_path, "r") as f:
                    payload = json.load(f)
            trace_id = payload.get("mlflow_trace_id")
            if not trace_id:
                raise KeyError(
                    f"decision {event_id} has no MLflow trace_id available; cannot tamper live data"
                )

            # Read the current tag value for the snapshot (best-effort; fall back to "").
            old = ""
            try:
                trace = client.get_trace(trace_id)
                tags = {}
                info = getattr(trace, "info", None)
                if info is not None and getattr(info, "tags", None):
                    tags = dict(info.tags)
                elif getattr(trace, "tags", None):
                    tags = dict(trace.tags)
                old = tags.get("ario.payload_json", "")
            except Exception:
                old = ""

            # Write mutation — must not be swallowed; if it fails the tamper did not happen.
            try:
                client.set_trace_tag(trace_id, "ario.payload_json",
                                     '{"tampered": "this is no longer the canonical bytes"}')
            except Exception as e:
                raise RuntimeError(f"failed to mutate trace tag for decision {event_id}: {e}") from e

            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"trace_tag:{trace_id}:ario.payload_json",
                live_field_old_value=old,
            )

        elif event_type == "training":
            run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
            run = client.get_run(run_id)
            old = str(run.data.metrics.get("accuracy", "0.0"))
            client.log_metric(run_id, "accuracy", 0.999)
            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"run_metric:{run_id}:accuracy",
                live_field_old_value=old,
            )

        elif event_type == "registration":
            # Model-swap tamper: replace model.pkl bytes on the source run.
            # This is the visceral "model swapping" attack — someone swaps
            # the deployed model artifact while keeping the same registered
            # version. The artifact_checksums hash changes, which breaks
            # both Registration Record Matches AND Training Record Matches
            # (since both canonical bytes include artifact_checksums).
            #
            # In MLflow 3.x, ``mlflow.artifacts.download_artifacts(run_id,
            # "model")`` returns an *ephemeral temp copy* of the LoggedModel
            # store on every call — writing to that path is a no-op
            # (the temp dir gets cleaned up). We must resolve the real
            # on-disk LoggedModel artifact_location and write there for
            # the verifier's next ``download_artifacts`` to pick up the
            # mutated bytes.
            envelope = lifecycle_store.get_by_event_id(event_id)
            if envelope is None:
                raise KeyError(f"registration {event_id} not found")
            source_run_id = envelope["record"]["source_run_id"]

            run = client.get_run(source_run_id)
            try:
                logged_models = client.search_logged_models(
                    experiment_ids=[run.info.experiment_id],
                    filter_string=f"source_run_id = '{source_run_id}'",
                )
            except Exception as e:
                raise KeyError(
                    f"could not search logged models for run {source_run_id}: {e}"
                )
            if not logged_models:
                raise KeyError(
                    f"no LoggedModel found for source run {source_run_id}; "
                    "swap-artifact tamper requires a registered model"
                )
            # MLflow 3.x supports multiple models per run, but the demo logs
            # exactly one (`name="model"`). Take the first match — if a
            # future caller logs multiple, the tamper still demonstrates
            # "swap one of the deployed artifacts."
            canonical_dir = logged_models[0].artifact_location
            # artifact_location is an absolute filesystem path for file://
            # tracking stores ("/.../mlruns/<exp>/models/m-<uuid>/artifacts").
            # Strip a possible "file://" scheme defensively for portability.
            if canonical_dir.startswith("file://"):
                canonical_dir = canonical_dir[len("file://"):]

            model_pkl_path = os.path.join(canonical_dir, "model.pkl")
            if not os.path.isfile(model_pkl_path):
                raise KeyError(
                    f"model.pkl not found at canonical LoggedModel path "
                    f"{model_pkl_path} for run {source_run_id}"
                )

            with open(model_pkl_path, "rb") as f:
                original_bytes = f.read()

            # Direct filesystem overwrite of the canonical LoggedModel
            # artifact. The next ``download_artifacts`` call by the
            # verifier copies these mutated bytes into a fresh temp dir
            # and re-hashes them — the new hash diverges from the
            # anchored one, flipping ``artifact_verified`` to False and
            # breaking the source-of-truth check on both the registration
            # and any training/registration that references the artifact.
            with open(model_pkl_path, "wb") as f:
                f.write(
                    b"TAMPERED MODEL ARTIFACT - this is not the registered "
                    b"model weights. The artifact hash that was anchored "
                    b"at registration time no longer matches."
                )

            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"artifact_swap_path:{model_pkl_path}",
                saved_artifact_bytes=original_bytes,
            )
        else:
            raise ValueError(f"unknown event_type: {event_type}")

        _snapshots[key] = snapshot
        logger.info(f"Tamper LIVE applied: {event_type}/{event_id}")
        return snapshot


def reset(event_type, event_id, *, lifecycle_store, record_store, tracking_uri):
    """Restore both saved and live state for an event from snapshots."""
    reverted = 0
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    with _lock:
        for kind in ("saved", "live"):
            key = (event_type, event_id, kind)
            # Peek the snapshot — only pop after the restore actually succeeds,
            # so a transient MLflow failure leaves the snapshot in place for
            # retry rather than dropping the only path back to a clean state.
            snap = _snapshots.get(key)
            if snap is None:
                continue

            try:
                if snap.kind == "saved" and snap.saved_artifact_bytes is not None:
                    run_id = _resolve_run_id(event_type, event_id, lifecycle_store, record_store)
                    artifact_path = _payload_artifact_path(event_type, event_id)
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        local_path = os.path.join(tmpdir, os.path.basename(artifact_path))
                        with open(local_path, "wb") as f:
                            f.write(snap.saved_artifact_bytes)
                        client.log_artifact(run_id, local_path,
                                            artifact_path=os.path.dirname(artifact_path))
                elif snap.kind == "live" and snap.live_field_name:
                    parts = snap.live_field_name.split(":", 3)
                    kind_prefix = parts[0]
                    if kind_prefix == "trace_tag":
                        _, trace_id, tag = parts
                        try:
                            client.set_trace_tag(trace_id, tag, snap.live_field_old_value or "")
                        except Exception as e:
                            logger.warning(f"Could not restore trace tag {tag} on {trace_id}: {e}")
                    elif kind_prefix == "run_metric":
                        _, run_id, metric = parts
                        client.log_metric(run_id, metric, float(snap.live_field_old_value or 0))
                    elif kind_prefix == "mv_tag":
                        _, name, version, tag = parts
                        client.set_model_version_tag(name, version, tag,
                                                     snap.live_field_old_value or "")
                    elif kind_prefix == "artifact_swap_path" and snap.saved_artifact_bytes is not None:
                        # The live_field_name encodes the absolute on-disk path
                        # we wrote to. Restore by writing the snapshot bytes back.
                        _, target_path = snap.live_field_name.split(":", 1)
                        try:
                            with open(target_path, "wb") as f:
                                f.write(snap.saved_artifact_bytes)
                        except Exception as e:
                            logger.warning(f"Could not restore artifact at {target_path}: {e}")
                    elif kind_prefix == "artifact_swap" and snap.saved_artifact_bytes is not None:
                        # Legacy log_artifact-based variant (pre-fix); kept so
                        # any in-flight snapshot from before the upgrade restores.
                        _, run_id, artifact_path = parts[0], parts[1], parts[2]
                        import tempfile as _tempfile
                        with _tempfile.TemporaryDirectory() as tmpdir:
                            local_path = os.path.join(tmpdir, os.path.basename(artifact_path))
                            with open(local_path, "wb") as f:
                                f.write(snap.saved_artifact_bytes)
                            client.log_artifact(
                                run_id, local_path,
                                artifact_path=os.path.dirname(artifact_path),
                            )
                # Restore succeeded — now drop the snapshot.
                _snapshots.pop(key, None)
                reverted += 1
                logger.info(f"Tamper RESET: {event_type}/{event_id}/{kind}")
            except Exception as e:
                logger.warning(
                    f"Reset failed for {key}; keeping snapshot for retry: {e}"
                )

    return reverted
