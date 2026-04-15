"""ArioMlflowClient — wraps MlflowClient with automatic proof anchoring."""

import logging
import os
import threading
import uuid
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.anchor import ArweaveAnchor

logger = logging.getLogger(__name__)


class ArioMlflowClient(MlflowClient):
    """MlflowClient that auto-anchors model registration and promotion events."""

    def __init__(self, *args, proof_engine: ProofEngine | None = None, anchor: ArweaveAnchor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

    def create_model_version(self, name, source, run_id=None, **kwargs):
        """Register a model version and anchor a proof record."""
        mv = super().create_model_version(name, source, run_id=run_id, **kwargs)

        # Build and anchor registration proof in background
        threading.Thread(
            target=self._anchor_registration,
            args=(name, str(mv.version), run_id, source),
            daemon=True,
        ).start()

        return mv

    def transition_model_version_stage(self, name, version, stage, **kwargs):
        """Transition a model stage and anchor a proof record."""
        # Get current stage before transition
        current = self.get_model_version(name, version)
        from_stage = current.current_stage

        result = super().transition_model_version_stage(name, version, stage, **kwargs)

        # Build and anchor promotion proof in background
        threading.Thread(
            target=self._anchor_promotion,
            args=(name, str(version), from_stage, stage),
            daemon=True,
        ).start()

        return result

    def _anchor_registration(self, model_name: str, version: str, run_id: str | None, source: str | None):
        """Background: anchor a model registration proof."""
        try:
            # Get training TX from the source run's tags
            training_tx = None
            if run_id:
                try:
                    run = self.get_run(run_id)
                    training_tx = run.data.tags.get("ario.training_tx")
                except Exception:
                    pass

            record = {
                "event_id": str(uuid.uuid4()),
                "event_type": "model_registered",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "model_version": version,
                "source_run_id": run_id,
                "source": source,
                "previous_tx": training_tx,
            }

            proof = self._proof_engine.create_proof(record, training_tx or "GENESIS")
            result = self._anchor.upload_proof(proof)

            if result:
                self.set_model_version_tag(model_name, version, "ario.registration_tx", result["tx_id"])
                logger.info(f"Registration {model_name}/v{version} anchored: tx={result['tx_id']}")

        except Exception as e:
            logger.error(f"Failed to anchor registration {model_name}/v{version}: {e}")

    def _anchor_promotion(self, model_name: str, version: str, from_stage: str, to_stage: str):
        """Background: anchor a stage transition proof."""
        try:
            # Get registration TX from model version tags
            mv = self.get_model_version(model_name, version)
            registration_tx = mv.tags.get("ario.registration_tx")

            record = {
                "event_id": str(uuid.uuid4()),
                "event_type": "stage_transition",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "model_version": version,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "previous_tx": registration_tx,
            }

            proof = self._proof_engine.create_proof(record, registration_tx or "GENESIS")
            result = self._anchor.upload_proof(proof)

            if result:
                self.set_model_version_tag(model_name, version, "ario.promotion_tx", result["tx_id"])
                logger.info(f"Promotion {model_name}/v{version} ({from_stage}→{to_stage}) anchored: tx={result['tx_id']}")

        except Exception as e:
            logger.error(f"Failed to anchor promotion {model_name}/v{version}: {e}")
