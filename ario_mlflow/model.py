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
from ario_mlflow.anchoring import artifact_checksums, parse_runs_uri

logger = logging.getLogger(__name__)


class IntegrityError(Exception):
    """Raised when model artifacts fail integrity verification."""


def _resolve_model_version(client, model_uri: str):
    """Resolve a ``models:/`` URI to a ``ModelVersion`` using the correct MLflow API.

    Supports numeric versions (``models:/name/1``), aliases
    (``models:/name@champion``), and legacy stage URIs
    (``models:/name/Production``). Returns the resolved ``ModelVersion`` or
    ``None`` if the URI cannot be parsed or the registry lookup fails.
    """
    if not model_uri.startswith("models:/"):
        return None
    rest = model_uri[len("models:/"):]
    if not rest:
        return None

    if "@" in rest:
        name, alias = rest.split("@", 1)
        if not name or not alias:
            return None
        try:
            return client.get_model_version_by_alias(name, alias)
        except Exception as e:
            logger.warning(f"Could not resolve alias {model_uri}: {e}")
            return None

    parts = rest.split("/", 1)
    name = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    if not name or not suffix:
        return None

    if suffix.isdigit():
        try:
            return client.get_model_version(name, suffix)
        except Exception as e:
            logger.warning(f"Could not resolve version {model_uri}: {e}")
            return None

    # Stage URI (deprecated in MLflow 2.9+ but still supported).
    try:
        results = client.search_model_versions(
            f"name='{name}' and current_stage='{suffix}'"
        )
    except Exception as e:
        logger.warning(f"Could not resolve stage {model_uri}: {e}")
        return None
    if not results:
        return None
    # MLflow returns latest-first; take the most recent version in the stage.
    return results[0]


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
        self.model_name = "unknown"
        self.model_version = "unknown"
        self.run_id = "unknown"

        # Resolve the models:/ URI via the correct MLflow registry API for each
        # supported URI form:
        #   models:/<name>/<numeric_version>  → get_model_version
        #   models:/<name>@<alias>            → get_model_version_by_alias
        #   models:/<name>/<stage>            → search_model_versions (deprecated)
        mv = _resolve_model_version(client, model_uri)
        if mv is not None:
            self.model_name = mv.name
            self.model_version = str(mv.version)
            self.run_id = mv.run_id or "unknown"

        # ModelVersion.source preserves the original artifact path from
        # registration (e.g. "sklearn-model") — we must use it rather than
        # hardcoding "/model".
        load_uri = model_uri
        artifact_path = "model"
        if mv is not None and mv.source:
            load_uri = mv.source
            _src_run_id, src_artifact_path = parse_runs_uri(mv.source)
            if src_artifact_path:
                artifact_path = src_artifact_path

        # Verify artifact integrity BEFORE loading the model. pyfunc models can
        # execute user code during load (PythonModel subclasses, custom loaders),
        # so a tampered artifact must be rejected before mlflow.pyfunc.load_model
        # is given a chance to run it.
        self._artifact_verified = None
        if self.run_id != "unknown":
            try:
                run = client.get_run(self.run_id)
                expected_hash = run.data.tags.get("ario.artifact_hash")
                if expected_hash:
                    checksums = artifact_checksums(self.run_id, artifact_path=artifact_path)
                    if not checksums:
                        logger.warning(
                            f"Could not download artifacts for integrity check of {model_uri}; "
                            f"treating status as unknown"
                        )
                    else:
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

        # Integrity has passed (or was unverifiable with a logged warning).
        # Only now load the model.
        self._model = mlflow.pyfunc.load_model(load_uri)

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

        # Tag the MLflow trace with proof metadata — links trace to proof.
        # Best-effort: tracing backends can fail (network, backend down); we
        # don't want that to surface as an inference failure.
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            try:
                mlflow.set_trace_tag(trace_id, "ario.decision_id", decision_id)
                mlflow.set_trace_tag(trace_id, "ario.model_name", self.model_name)
                mlflow.set_trace_tag(trace_id, "ario.model_version", self.model_version)
                mlflow.set_trace_tag(trace_id, "ario.input_hash", record["input_hash"])
                mlflow.set_trace_tag(trace_id, "ario.output_hash", record["output_hash"])
                mlflow.set_trace_tag(trace_id, "ario.record_hash", proof["record_hash"])
                mlflow.set_trace_tag(trace_id, "ario.proof_status", result.proof_status)
                if self._artifact_verified is not None:
                    mlflow.set_trace_tag(trace_id, "ario.artifact_verified", str(self._artifact_verified).lower())
            except Exception as e:
                logger.warning(f"Failed to tag MLflow trace {trace_id}: {e}")

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
