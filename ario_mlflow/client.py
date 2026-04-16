"""ArioMlflowClient — wraps MlflowClient with automatic proof anchoring."""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.anchoring import artifact_checksums
from ario_mlflow.report import generate_verification_html

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

        threading.Thread(
            target=self._anchor_registration,
            args=(name, str(mv.version), run_id, source),
            daemon=True,
        ).start()

        return mv

    def transition_model_version_stage(self, name, version, stage, **kwargs):
        """Transition a model stage and anchor a proof record."""
        current = self.get_model_version(name, version)
        from_stage = current.current_stage

        result = super().transition_model_version_stage(name, version, stage, **kwargs)

        threading.Thread(
            target=self._anchor_promotion,
            args=(name, str(version), from_stage, stage),
            daemon=True,
        ).start()

        return result

    def _anchor_registration(self, model_name: str, version: str, run_id: str | None, source: str | None):
        """Background: verify artifact integrity, anchor a registration proof."""
        try:
            training_tx = None
            expected_hash = None
            artifact_verified = None

            if run_id:
                try:
                    run = self.get_run(run_id)
                    training_tx = run.data.tags.get("ario.training_tx")
                    expected_hash = run.data.tags.get("ario.artifact_hash")
                except Exception:
                    pass

                checksums = artifact_checksums(self, run_id)
                if checksums:
                    computed_hash = hash_data(canonical_json(checksums))
                    artifact_verified = expected_hash is not None and computed_hash == expected_hash

            record = {
                "event_id": str(uuid.uuid4()),
                "event_type": "model_registered",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "model_version": version,
                "source_run_id": run_id,
                "source": source,
                "artifact_verified": artifact_verified,
                "artifact_hash": expected_hash,
                "previous_tx": training_tx,
            }

            proof = self._proof_engine.create_proof(record, training_tx or "GENESIS")
            result = self._anchor.upload_proof(proof) if self._anchor.enabled else None

            tags = {
                "ario.verify_status": "anchored" if result else "signed",
                "ario.public_key": proof["public_key"],
            }
            if artifact_verified is not None:
                tags["ario.artifact_verified"] = str(artifact_verified).lower()
            if result:
                tags["ario.registration_tx"] = result["tx_id"]
                tags["ario.arweave_url"] = result["url"]

            for key, value in tags.items():
                self.set_model_version_tag(model_name, version, key, value)

            with tempfile.TemporaryDirectory() as tmpdir:
                ario_dir = os.path.join(tmpdir, "ario")
                os.makedirs(ario_dir)

                with open(os.path.join(ario_dir, "proof.json"), "w") as f:
                    json.dump(proof, f, indent=2)

                html = generate_verification_html(
                    proof, result,
                    artifact_hash=expected_hash,
                    artifact_verified=artifact_verified,
                )
                with open(os.path.join(ario_dir, "verification.html"), "w") as f:
                    f.write(html)

                if run_id:
                    mlflow.log_artifacts(ario_dir, "ario")

            logger.info(f"Registration {model_name}/v{version} anchored: verified={artifact_verified}")

        except Exception as e:
            logger.error(f"Failed to anchor registration {model_name}/v{version}: {e}")

    def _anchor_promotion(self, model_name: str, version: str, from_stage: str, to_stage: str):
        """Background: anchor a stage transition proof."""
        try:
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
                logger.info(f"Promotion {model_name}/v{version} ({from_stage}->{to_stage}) anchored: tx={result['tx_id']}")

        except Exception as e:
            logger.error(f"Failed to anchor promotion {model_name}/v{version}: {e}")
