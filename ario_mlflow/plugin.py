"""MLflow RunContextProvider — auto-anchors training runs on completion."""

import hashlib
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

import mlflow
from mlflow.tracking.context.abstract_context import RunContextProvider

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.anchor import ArweaveAnchor

logger = logging.getLogger(__name__)


def _artifact_checksums(client, run_id: str) -> dict[str, str]:
    """Compute SHA-256 checksums of all artifacts in a run."""
    try:
        local_path = client.download_artifacts(run_id, "")
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


def _anchor_training_run(run_id: str, proof_engine: ProofEngine, anchor: ArweaveAnchor):
    """Background: build and anchor a training run proof record."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        params = dict(run.data.params)
        metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in run.data.metrics.items()}
        checksums = _artifact_checksums(client, run_id)

        record = {
            "event_id": str(uuid.uuid4()),
            "event_type": "training_complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
            "artifact_checksums": checksums,
            "artifact_hash": hash_data(canonical_json(checksums)),
            "source_name": run.data.tags.get("mlflow.source.name", ""),
            "git_commit": run.data.tags.get("mlflow.source.git.commit", ""),
        }

        proof = proof_engine.create_proof(record, "GENESIS")
        result = anchor.upload_proof(proof)

        if result:
            client.set_tag(run_id, "ario.training_tx", result["tx_id"])
            client.set_tag(run_id, "ario.artifact_hash", record["artifact_hash"])
            logger.info(f"Training run {run_id} anchored: tx={result['tx_id']}")
        else:
            logger.warning(f"Training run {run_id}: anchoring disabled or failed")

    except Exception as e:
        logger.error(f"Failed to anchor training run {run_id}: {e}")


class ArioContextProvider(RunContextProvider):
    """Auto-injects ar.io metadata and triggers async anchoring on run completion."""

    _proof_engine = None
    _anchor = None

    @classmethod
    def _init_components(cls):
        if cls._proof_engine is None:
            cls._proof_engine = ProofEngine()
        if cls._anchor is None:
            wallet = os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", "")
            gateway = os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com")
            cls._anchor = ArweaveAnchor(wallet, gateway)

    def in_context(self) -> bool:
        return True

    def tags(self) -> dict[str, str]:
        return {
            "ario.enabled": "true",
            "ario.version": "0.1.0",
        }


# Hook into MLflow's run completion to trigger async anchoring
_original_end_run = None


def _patched_end_run(status="FINISHED"):
    """Patched mlflow.end_run that triggers anchoring after run completion."""
    _original_end_run(status)

    if status == "FINISHED":
        run = mlflow.active_run()
        if run is None:
            # Run just ended, get the last run from the client
            client = mlflow.tracking.MlflowClient()
            # Check if already anchored
            try:
                experiment_id = mlflow.get_experiment_by_name(
                    os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default")
                )
                if experiment_id:
                    runs = client.search_runs([experiment_id.experiment_id], max_results=1)
                    if runs:
                        last_run = runs[0]
                        if "ario.training_tx" not in last_run.data.tags:
                            ArioContextProvider._init_components()
                            threading.Thread(
                                target=_anchor_training_run,
                                args=(last_run.info.run_id, ArioContextProvider._proof_engine, ArioContextProvider._anchor),
                                daemon=True,
                            ).start()
            except Exception as e:
                logger.debug(f"Post-run anchoring check failed: {e}")


def _install_hook():
    """Install the end_run hook. Called on import."""
    global _original_end_run
    if _original_end_run is None:
        _original_end_run = mlflow.end_run
        mlflow.end_run = _patched_end_run


# Auto-install on import
_install_hook()
