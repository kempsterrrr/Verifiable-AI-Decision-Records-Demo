"""VerifiedModel — inference wrapper with integrity checking and proof anchoring."""

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import time
from typing import Any

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
    """Result of a verified prediction, including background anchoring status.

    Fields:
        prediction: The model's output (whatever ``pyfunc.predict`` returned).
        decision_id: UUID4 string uniquely identifying this prediction. Mirrors
            the ``ario.decision_id`` trace tag written on the MLflow trace.
        proof_status: One of:
            - ``"disabled"`` — anchoring is off (no wallet / no Turbo client).
            - ``"anchoring"`` — background upload in progress.
            - ``"anchored"`` — uploaded successfully; ``tx_id`` is set.
            - ``"failed"`` — upload raised; ``anchor_error`` is set.
        record: The canonical decision record that was signed. ``None`` only
            in exotic failure cases.
        tx_id: Arweave transaction ID, populated after a successful anchor.
        anchor_error: Stringified exception from the background anchor when
            ``proof_status == "failed"``. ``None`` otherwise.

    Use :meth:`wait_for_anchor` to block until the background thread
    finishes. The underlying :class:`threading.Event` is hidden from
    ``repr()`` and equality so it behaves like plain data otherwise.
    """
    prediction: Any
    decision_id: str
    proof_status: str  # "anchoring" | "anchored" | "disabled" | "failed"
    record: dict | None = None
    tx_id: str | None = None
    anchor_error: str | None = None
    _anchor_done: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def wait_for_anchor(self, timeout: float | None = None) -> bool:
        """Block until the background anchor completes or the timeout expires.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits forever.

        Returns:
            ``True`` if the background anchor finished (check ``proof_status``,
            ``tx_id``, and ``anchor_error`` for outcome). ``False`` if the
            timeout expired while still ``"anchoring"``.

        When anchoring is disabled (``proof_status == "disabled"``) the event
        is already set and this returns ``True`` immediately.
        """
        return self._anchor_done.wait(timeout=timeout)


class VerifiedModel:
    """Wraps an MLflow model with integrity checking and proof anchoring on predict()."""

    def __init__(
        self,
        model_uri: str,
        proof_engine: ProofEngine | None = None,
        anchor: ArweaveAnchor | None = None,
    ):
        """Load an MLflow model and verify its artifacts against the anchored hash.

        Resolves ``model_uri`` through the MLflow registry, re-hashes the
        model artifacts, and compares the result to the ``ario.artifact_hash``
        tag from the source training run. The integrity check runs **before**
        :func:`mlflow.pyfunc.load_model`, so a tampered artifact is rejected
        before any user code (``PythonModel`` subclasses, custom loaders) can
        execute.

        Args:
            model_uri: A ``models:/`` URI in any of these forms:

                - ``models:/<name>/<version>`` — numeric version.
                - ``models:/<name>@<alias>`` — registry alias.
                - ``models:/<name>/<stage>`` — legacy stage URI (MLflow's
                  ``search_model_versions`` is used; deprecated in 2.9+).
            proof_engine: Override for the signing engine. Defaults to a
                :class:`ProofEngine` using the process-local Ed25519 key.
            anchor: Override for the Arweave anchor client. Defaults to an
                :class:`ArweaveAnchor` configured from the
                ``ARIO_MLFLOW_ARWEAVE_WALLET`` /
                ``ARIO_MLFLOW_GATEWAY_HOST`` env vars.

        Raises:
            IntegrityError: If the re-hashed artifacts do not match the
                ``ario.artifact_hash`` anchored at training time. The underlying
                pyfunc model is never loaded in this case.
        """
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
        """Run inference, sign a decision record, and anchor it asynchronously.

        Args:
            input_data: A dict of named features, a list/tuple of positional
                features, or any array-like the underlying pyfunc model
                accepts. Dicts and single-row lists are wrapped into a
                2-D array (``[[values]]``) before passing to the model.

        Returns:
            A :class:`VerifiedPrediction`. ``prediction`` is whatever the
            wrapped model returned. The Arweave upload runs in a background
            thread; callers that need the ``tx_id`` immediately should call
            :meth:`VerifiedPrediction.wait_for_anchor` before reading it.

        Side effects:
            - An ``@mlflow.trace`` span is emitted with ``ario.*`` tags
              linking the trace to the proof: ``decision_id``, ``model_name``,
              ``model_version``, ``input_hash``, ``output_hash``,
              ``record_hash``, ``proof_status``, and ``artifact_verified``
              (when known). After a successful background anchor the trace
              is updated with ``ario.arweave_tx`` / ``ario.arweave_url``.
            - If the :class:`ArweaveAnchor` is enabled, the proof is
              uploaded to Arweave in a daemon thread. Upload errors are
              captured on the returned object (``proof_status="failed"`` +
              ``anchor_error``), not raised to the caller.
        """
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
        if not self._anchor.enabled:
            # Nothing will mark this done in the background, so callers
            # waiting on wait_for_anchor() get an immediate True.
            result._anchor_done.set()

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
                    except Exception as e:
                        # Trace may have been flushed by the backend already.
                        logger.debug(f"Could not update trace {trace_id} with anchor tags: {e}")
                logger.info(f"Prediction {result.decision_id} anchored: tx={anchor_result['tx_id']}")
            else:
                result.proof_status = "failed"
                result.anchor_error = "upload returned no result"
                logger.error(
                    f"Prediction anchoring failed for {result.decision_id}: upload returned no result"
                )
        except Exception as e:
            result.proof_status = "failed"
            result.anchor_error = str(e)
            logger.error(f"Prediction anchoring failed for {result.decision_id}: {e}")
        finally:
            # Always release wait_for_anchor() callers, whatever the outcome.
            result._anchor_done.set()
