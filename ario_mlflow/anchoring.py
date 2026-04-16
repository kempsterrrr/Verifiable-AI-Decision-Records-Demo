"""Public anchor() API and artifact checksum utilities."""

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

import mlflow

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.report import generate_verification_html

logger = logging.getLogger(__name__)


def artifact_checksums(client_or_run_id, run_id: str | None = None, artifact_path: str = "model") -> dict[str, str]:
    """Compute SHA-256 checksums of model artifacts in an MLflow run.

    Uses ``mlflow.artifacts.download_artifacts`` which works with both
    file-based and database-backed tracking stores in MLflow 3.x.

    Args:
        client_or_run_id: An MlflowClient (ignored, kept for backward compat) or a run_id string.
        run_id: The run ID. If client_or_run_id is a string, this is ignored.
        artifact_path: Artifact subdirectory to hash (default "model").
    """
    if isinstance(client_or_run_id, str):
        run_id = client_or_run_id
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        checksums = {}
        for root, _dirs, files in os.walk(local_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_path)
                with open(fpath, "rb") as f:
                    checksums[rel] = hashlib.sha256(f.read()).hexdigest()
        return checksums
    except Exception:
        return {}


def anchor(proof_engine: ProofEngine | None = None, arweave: ArweaveAnchor | None = None) -> dict:
    """Create a verifiable proof of the current training run.

    Must be called inside an active ``mlflow.start_run()`` block, after
    artifacts have been logged.  Signs a proof envelope, optionally uploads
    to Arweave, and writes rich tags + artifacts to the run.
    """
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("anchor() must be called inside an active MLflow run")

    run_id = active.info.run_id
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    if proof_engine is None:
        proof_engine = ProofEngine()
    if arweave is None:
        arweave = ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

    params = dict(run_data.data.params)
    metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in run_data.data.metrics.items()}
    checksums = artifact_checksums(run_id)
    art_hash = hash_data(canonical_json(checksums))

    record = {
        "event_id": str(uuid.uuid4()),
        "event_type": "training_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "params": params,
        "metrics": metrics,
        "artifact_checksums": checksums,
        "artifact_hash": art_hash,
        "source_name": run_data.data.tags.get("mlflow.source.name", ""),
        "git_commit": run_data.data.tags.get("mlflow.source.git.commit", ""),
    }

    proof = proof_engine.create_proof(record, "GENESIS")

    anchor_result = arweave.upload_proof(proof) if arweave.enabled else None

    tags = {
        "ario.artifact_hash": art_hash,
        "ario.public_key": proof["public_key"],
        "ario.verify_status": "anchored" if anchor_result else "signed",
    }
    if anchor_result:
        tags["ario.training_tx"] = anchor_result["tx_id"]
        tags["ario.arweave_url"] = anchor_result["url"]

    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        ario_dir = os.path.join(tmpdir, "ario")
        os.makedirs(ario_dir)

        # proof.json — only the cryptographic proof (matches what's on Arweave)
        with open(os.path.join(ario_dir, "proof.json"), "w") as f:
            json.dump(proof, f, indent=2)

        # receipt.json — Turbo upload receipt (independent timestamp witness)
        if anchor_result and anchor_result.get("receipt"):
            with open(os.path.join(ario_dir, "receipt.json"), "w") as f:
                json.dump(anchor_result["receipt"], f, indent=2)

        # verification.html — human-readable report
        html_content = generate_verification_html(proof, anchor_result, artifact_hash=art_hash)
        with open(os.path.join(ario_dir, "verification.html"), "w") as f:
            f.write(html_content)

        mlflow.log_artifacts(ario_dir, "ario")

    logger.info(f"Run {run_id} anchored: status={tags['ario.verify_status']}")

    return {"proof": proof, "anchor_result": anchor_result, "tags": tags}
