"""VerifiedModel — inference wrapper with async proof anchoring."""

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
from ario_mlflow.anchor import ArweaveAnchor

logger = logging.getLogger(__name__)


@dataclass
class VerifiedPrediction:
    """Result of a verified prediction."""
    prediction: any
    decision_id: str
    proof_status: str  # "anchoring" | "anchored" | "disabled"
    record: dict | None = None
    tx_id: str | None = None


class VerifiedModel:
    """Wraps an MLflow model with automatic proof anchoring on predict()."""

    def __init__(
        self,
        model_uri: str,
        proof_engine: ProofEngine | None = None,
        anchor: ArweaveAnchor | None = None,
    ):
        self._model = mlflow.pyfunc.load_model(model_uri)
        self._model_uri = model_uri
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

        # Extract model info from the URI
        client = mlflow.tracking.MlflowClient()
        parts = model_uri.replace("models:/", "").split("/")
        self.model_name = parts[0] if parts else "unknown"
        self.model_version = "unknown"
        self.run_id = "unknown"

        try:
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                self.model_version = str(latest.version)
                self.run_id = latest.run_id or "unknown"
        except Exception:
            pass

        # Proof chain tracking
        self._last_hash = "GENESIS"
        self._lock = threading.Lock()

    def predict(self, input_data) -> VerifiedPrediction:
        """Run inference and anchor a decision record in the background."""
        decision_id = str(uuid.uuid4())
        start = time()

        # Run inference
        if isinstance(input_data, dict):
            input_array = np.array([list(input_data.values())])
        elif isinstance(input_data, (list, tuple)):
            input_array = np.array([input_data])
        else:
            input_array = input_data

        prediction = self._model.predict(input_array)
        latency_ms = (time() - start) * 1000

        # Normalize prediction for hashing
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
        }

        # Create proof with chain linking
        with self._lock:
            proof = self._proof_engine.create_proof(record, self._last_hash)
            self._last_hash = proof["record_hash"]

        # Anchor in background
        result = VerifiedPrediction(
            prediction=prediction,
            decision_id=decision_id,
            proof_status="disabled" if not self._anchor.enabled else "anchoring",
            record=record,
        )

        if self._anchor.enabled:
            threading.Thread(
                target=self._anchor_prediction,
                args=(result, proof),
                daemon=True,
            ).start()

        return result

    def _anchor_prediction(self, result: VerifiedPrediction, proof: dict):
        """Background: upload prediction proof to Arweave."""
        try:
            anchor_result = self._anchor.upload_proof(proof)
            if anchor_result:
                result.tx_id = anchor_result["tx_id"]
                result.proof_status = "anchored"
                logger.info(f"Prediction {result.decision_id} anchored: tx={anchor_result['tx_id']}")
        except Exception as e:
            logger.error(f"Prediction anchoring failed for {result.decision_id}: {e}")
