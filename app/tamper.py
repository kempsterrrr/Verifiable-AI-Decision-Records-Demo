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
            trace_id = None
            old = ""
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
                    with open(local_path, "r") as f:
                        payload = json.load(f)
                    trace_id = payload.get("mlflow_trace_id")
            except Exception as e:
                logger.warning(f"Could not read payload artifact for decision {event_id}: {e}")

            if trace_id:
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
                try:
                    client.set_trace_tag(trace_id, "ario.payload_json",
                                         '{"tampered": "this is no longer the canonical bytes"}')
                except Exception as e:
                    logger.warning(
                        f"set_trace_tag unavailable for decision {event_id}: {e}. "
                        f"Live tamper may be a no-op on this MLflow version."
                    )
            else:
                logger.warning(
                    f"No mlflow_trace_id in payload for decision {event_id}; "
                    f"live tamper degraded (no trace tag to mutate)."
                )

            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"trace_tag:{trace_id}:ario.payload_json" if trace_id else None,
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
            envelope = lifecycle_store.get_by_event_id(event_id)
            if envelope is None:
                raise KeyError(f"registration {event_id} not found")
            model_name = envelope["record"]["model_name"]
            model_version = envelope["record"]["model_version"]
            mv = client.get_model_version(model_name, model_version)
            tags = mv.tags or {}
            old = tags.get("source_run_id", "")
            client.set_model_version_tag(model_name, model_version,
                                          "source_run_id", "tampered-fake-run-id")
            snapshot = TamperSnapshot(
                event_type=event_type, event_id=event_id, kind="live",
                live_field_name=f"mv_tag:{model_name}:{model_version}:source_run_id",
                live_field_old_value=old,
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
            snap = _snapshots.pop(key, None)
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
                reverted += 1
                logger.info(f"Tamper RESET: {event_type}/{event_id}/{kind}")
            except Exception as e:
                logger.warning(f"Reset failed for {key}: {e}")

    return reverted
