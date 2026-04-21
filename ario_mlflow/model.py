"""VerifiedModel — inference wrapper with integrity checking and proof anchoring."""

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from time import time

import mlflow
import numpy as np

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.anchoring import artifact_checksums

logger = logging.getLogger(__name__)


class IntegrityError(Exception):
    """Raised when model artifacts fail integrity verification."""


@dataclass
class VerifiedPrediction:
    """Result of a verified prediction."""
    prediction: any
    decision_id: str
    proof_status: str  # "anchoring" | "anchored" | "disabled"
    record: dict | None = None
    tx_id: str | None = None


class VerifiedModel:
    """Wraps an MLflow model with integrity checking and proof anchoring on predict()."""

    def __init__(
        self,
        model_uri: str,
        proof_engine: ProofEngine | None = None,
        anchor: ArweaveAnchor | None = None,
    ):
        self._model_uri = model_uri
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

        client = mlflow.tracking.MlflowClient()
        parts = model_uri.replace("models:/", "").split("/")
        self.model_name = parts[0] if parts else "unknown"
        self.model_version = parts[1] if len(parts) > 1 else "unknown"
        self.run_id = "unknown"

        # Resolve the run_id and load model via runs:/ URI for compatibility
        # with both file-based and database-backed MLflow stores
        load_uri = model_uri
        try:
            mv = client.get_model_version(self.model_name, self.model_version)
            self.run_id = mv.run_id or "unknown"
            if self.run_id != "unknown":
                load_uri = f"runs:/{self.run_id}/model"
        except Exception:
            pass

        self._model = mlflow.pyfunc.load_model(load_uri)

        self._artifact_verified = None
        if self.run_id != "unknown":
            try:
                run = client.get_run(self.run_id)
                expected_hash = run.data.tags.get("ario.artifact_hash")
                if expected_hash:
                    checksums = artifact_checksums(self.run_id)
                    computed_hash = hash_data(canonical_json(checksums))
                    if computed_hash != expected_hash:
                        raise IntegrityError(
                            f"Model artifact integrity check failed for {model_uri}. "
                            f"Expected {expected_hash}, got {computed_hash}"
                        )
                    self._artifact_verified = True
                    logger.info(f"Artifact integrity verified for {model_uri}")
            except IntegrityError:
                raise
            except Exception as e:
                logger.warning(f"Could not verify artifact integrity: {e}")

        self._last_hash = "GENESIS"
        self._lock = threading.Lock()

    @mlflow.trace(name="VerifiedModel.predict")
    def predict(self, input_data) -> VerifiedPrediction:
        """Run inference, create cryptographic proof, and log an MLflow trace."""
        decision_id = str(uuid.uuid4())
        start = time()

        if isinstance(input_data, dict):
            input_array = np.array([list(input_data.values())])
        elif isinstance(input_data, (list, tuple)):
            input_array = np.array([input_data])
        else:
            input_array = input_data

        prediction = self._model.predict(input_array)
        latency_ms = (time() - start) * 1000

        if hasattr(prediction, 'tolist'):
            pred_serializable = prediction.tolist()
        else:
            pred_serializable = prediction

        input_serializable = input_data if isinstance(input_data, dict) else {"features": list(input_data) if hasattr(input_data, '__iter__') else input_data}

        record = {
            "decision_id": decision_id,
            "event_type": "prediction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "model_uri": self._model_uri,
            "input_hash": hash_data(canonical_json(input_serializable)),
            "output_hash": hash_data(canonical_json({"prediction": pred_serializable})),
            "latency_ms": round(latency_ms, 2),
            "artifact_verified": self._artifact_verified,
        }

        with self._lock:
            proof = self._proof_engine.create_proof(record, self._last_hash)
            self._last_hash = proof["record_hash"]

        result = VerifiedPrediction(
            prediction=prediction,
            decision_id=decision_id,
            proof_status="disabled" if not self._anchor.enabled else "anchoring",
            record=record,
        )

        # Tag the MLflow trace with proof metadata — links trace to proof
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            mlflow.set_trace_tag(trace_id, "ario.decision_id", decision_id)
            mlflow.set_trace_tag(trace_id, "ario.model_name", self.model_name)
            mlflow.set_trace_tag(trace_id, "ario.model_version", self.model_version)
            mlflow.set_trace_tag(trace_id, "ario.input_hash", record["input_hash"])
            mlflow.set_trace_tag(trace_id, "ario.output_hash", record["output_hash"])
            mlflow.set_trace_tag(trace_id, "ario.record_hash", proof["record_hash"])
            mlflow.set_trace_tag(trace_id, "ario.proof_status", result.proof_status)
            if self._artifact_verified is not None:
                mlflow.set_trace_tag(trace_id, "ario.artifact_verified", str(self._artifact_verified).lower())

        if self._anchor.enabled:
            threading.Thread(
                target=self._anchor_prediction,
                args=(result, proof, trace_id),
                daemon=True,
            ).start()

        return result

    def _anchor_prediction(self, result: VerifiedPrediction, proof: dict, trace_id: str | None = None):
        """Background: upload prediction proof to Arweave, update trace."""
        try:
            anchor_result = self._anchor.upload_proof(proof)
            if anchor_result:
                result.tx_id = anchor_result["tx_id"]
                result.proof_status = "anchored"
                # Update the trace with the Arweave TX — links trace to on-chain proof
                if trace_id:
                    try:
                        mlflow.set_trace_tag(trace_id, "ario.arweave_tx", anchor_result["tx_id"])
                        mlflow.set_trace_tag(trace_id, "ario.arweave_url", anchor_result["url"])
                        mlflow.set_trace_tag(trace_id, "ario.proof_status", "anchored")
                    except Exception:
                        pass  # Trace may have been flushed already
                logger.info(f"Prediction {result.decision_id} anchored: tx={anchor_result['tx_id']}")
        except Exception as e:
            logger.error(f"Prediction anchoring failed for {result.decision_id}: {e}")
