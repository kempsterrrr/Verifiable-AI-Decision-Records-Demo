"""Build proof records for ML lifecycle events (training, registration)."""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

import mlflow

from ario_mlflow.proof import canonical_json, hash_data


def _artifact_checksums(tracking_uri: str, run_id: str) -> dict[str, str]:
    """Compute SHA-256 checksums of all artifacts in an MLflow run."""
    mlflow.set_tracking_uri(os.path.abspath(tracking_uri))
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "")
    checksums = {}
    for root, _dirs, files in os.walk(local_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_path)
            with open(fpath, "rb") as f:
                checksums[rel] = hashlib.sha256(f.read()).hexdigest()
    return checksums


def build_training_record(
    tracking_uri: str,
    run_id: str,
    model_name: str,
    model_version: str,
) -> dict:
    """Build a proof record for a completed training run."""
    mlflow.set_tracking_uri(os.path.abspath(tracking_uri))
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    params = dict(run.data.params)
    metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in run.data.metrics.items()}
    artifact_checksums = _artifact_checksums(tracking_uri, run_id)

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "training_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "params": params,
        "metrics": metrics,
        "artifact_checksums": artifact_checksums,
        "artifact_hash": hash_data(canonical_json(artifact_checksums)),
        "source_name": run.data.tags.get("mlflow.source.name", ""),
        "git_commit": run.data.tags.get("mlflow.source.git.commit", ""),
    }


def build_registration_record(
    tracking_uri: str,
    model_name: str,
    model_version: str,
    training_tx: str | None = None,
) -> dict:
    """Build a proof record for a model registration event."""
    mlflow.set_tracking_uri(os.path.abspath(tracking_uri))
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    version_info = None
    for v in versions:
        if str(v.version) == str(model_version):
            version_info = v
            break

    run_id = version_info.run_id if version_info else None
    source = version_info.source if version_info else None

    # Compute artifact hash from the source run
    artifact_hash = None
    if run_id:
        checksums = _artifact_checksums(tracking_uri, run_id)
        artifact_hash = hash_data(canonical_json(checksums))

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "model_registered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "source_run_id": run_id,
        "source": source,
        "artifact_hash": artifact_hash,
        "previous_tx": training_tx,
    }
