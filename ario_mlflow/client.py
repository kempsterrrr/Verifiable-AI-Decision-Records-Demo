"""ArioMlflowClient — wraps MlflowClient with automatic proof anchoring."""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone

from mlflow.tracking import MlflowClient

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.anchoring import artifact_checksums, parse_runs_uri, ArtifactAccessError
from ario_mlflow.report import generate_verification_html

logger = logging.getLogger(__name__)


class ArioMlflowClient(MlflowClient):
    """MlflowClient that auto-anchors model registration and promotion events.

    Anchoring runs in a daemon thread so the MLflow call returns immediately.
    Because the return value (an MLflow ``ModelVersion``) has no room for an
    anchor future, the client exposes status via two methods:

    - :meth:`anchor_status` — returns the latest status for a given event.
    - :meth:`wait_for_anchor` — blocks until that event finishes.

    Statuses are keyed by ``(event_type, name, version)``, where
    ``event_type`` is ``"registration"`` or ``"promotion"``.
    """

    def __init__(self, *args, proof_engine: ProofEngine | None = None, anchor: ArweaveAnchor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )
        # Per-(event_type, name, version) status and completion events, so the
        # caller can observe async anchoring outcomes without the ModelVersion
        # return value carrying a future.
        self._anchor_events: dict[tuple[str, str, str], threading.Event] = {}
        self._anchor_statuses: dict[tuple[str, str, str], dict] = {}
        self._anchor_state_lock = threading.Lock()

    def _status_key(self, event_type: str, name: str, version: str) -> tuple[str, str, str]:
        return (event_type, name, str(version))

    def _ensure_anchor_state(self) -> None:
        """Lazily initialize the status-tracking attributes.

        Subclasses that override ``__init__`` without calling super (e.g.
        test doubles) still work — the first call that needs these
        attributes creates them.
        """
        if not hasattr(self, "_anchor_state_lock"):
            self._anchor_state_lock = threading.Lock()
            self._anchor_events = {}
            self._anchor_statuses = {}

    def _register_pending(self, event_type: str, name: str, version: str) -> threading.Event:
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        event = threading.Event()
        with self._anchor_state_lock:
            self._anchor_events[key] = event
            self._anchor_statuses[key] = {
                "status": "anchoring",
                "error": None,
                "tx_id": None,
            }
        return event

    def _record_status(
        self,
        event_type: str,
        name: str,
        version: str,
        status: str,
        tx_id: str | None = None,
        error: str | None = None,
    ) -> None:
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            self._anchor_statuses[key] = {
                "status": status,
                "error": error,
                "tx_id": tx_id,
            }

    def anchor_status(self, event_type: str, name: str, version: str) -> dict:
        """Return the latest anchor status for a registration or promotion.

        Args:
            event_type: ``"registration"`` or ``"promotion"``.
            name: Registered model name.
            version: Model version (int or string).

        Returns:
            A dict with keys ``status`` (``"anchoring"`` | ``"anchored"`` |
            ``"signed"`` | ``"failed"`` | ``"unknown"``), ``tx_id``,
            ``error``, and ``done`` (bool — has the background thread
            finished).
        """
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            status = dict(self._anchor_statuses.get(key, {"status": "unknown", "error": None, "tx_id": None}))
            event = self._anchor_events.get(key)
        status["done"] = bool(event and event.is_set())
        return status

    def wait_for_anchor(
        self,
        event_type: str,
        name: str,
        version: str,
        timeout: float | None = None,
    ) -> bool:
        """Block until the background anchor for this event completes.

        Returns ``True`` if the anchor finished (check :meth:`anchor_status`
        for the outcome). ``False`` if the timeout expired, or if no anchor
        was ever queued for this key.
        """
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            event = self._anchor_events.get(key)
        if event is None:
            return False
        return event.wait(timeout=timeout)

    def create_model_version(self, name, source, run_id=None, **kwargs):
        """Register a model version and anchor a proof record."""
        mv = super().create_model_version(name, source, run_id=run_id, **kwargs)
        event = self._register_pending("registration", name, str(mv.version))

        threading.Thread(
            target=self._anchor_registration,
            args=(name, str(mv.version), run_id, source, event),
            daemon=True,
        ).start()

        return mv

    def transition_model_version_stage(self, name, version, stage, **kwargs):
        """Transition a model stage and anchor a proof record."""
        current = self.get_model_version(name, version)
        from_stage = current.current_stage

        result = super().transition_model_version_stage(name, version, stage, **kwargs)
        event = self._register_pending("promotion", name, str(version))

        threading.Thread(
            target=self._anchor_promotion,
            args=(name, str(version), from_stage, stage, event),
            daemon=True,
        ).start()

        return result

    def _anchor_registration(
        self,
        model_name: str,
        version: str,
        run_id: str | None,
        source: str | None,
        done_event: threading.Event | None = None,
    ):
        """Background: verify artifact integrity, anchor a registration proof."""
        try:
            training_tx = None
            expected_hash = None
            artifact_verified = None

            # create_model_version's run_id parameter is optional. When absent,
            # derive it from the source URI so we still link the registration
            # proof back to the training run (and its ario.training_tx) instead
            # of fabricating a fresh GENESIS chain.
            src_run_id, src_artifact_path = parse_runs_uri(source)
            source_run_id = run_id or src_run_id

            if source_run_id:
                try:
                    run = self.get_run(source_run_id)
                    training_tx = run.data.tags.get("ario.training_tx")
                    expected_hash = run.data.tags.get("ario.artifact_hash")
                except Exception as e:
                    # A transient tracking-store failure must NOT silently drop
                    # training_tx and cause us to mint a fresh GENESIS chain —
                    # that would permanently break provenance for this model
                    # version. Skip anchoring this attempt instead.
                    logger.warning(
                        f"Skipping registration anchoring for {model_name}/v{version}: "
                        f"could not load source run {source_run_id}: {e}"
                    )
                    self._record_status(
                        "registration", model_name, version,
                        status="failed",
                        error=f"Could not load source run {source_run_id}: {e}",
                    )
                    return

                # Hash the artifact path that was actually registered. MLflow's
                # source URI (runs:/<run_id>/<artifact_path>) preserves the
                # original path — it is not always "model".
                try:
                    checksums = artifact_checksums(source_run_id, artifact_path=src_artifact_path or "model")
                except ArtifactAccessError as e:
                    # We can't verify — leave artifact_verified as None (unknown)
                    # and continue anchoring the registration event itself.
                    logger.warning(
                        f"Could not re-hash artifacts for {model_name}/v{version}: {e}"
                    )
                    checksums = {}
                if checksums and expected_hash is not None:
                    computed_hash = hash_data(canonical_json(checksums))
                    artifact_verified = computed_hash == expected_hash

            record = {
                "event_id": str(uuid.uuid4()),
                "event_type": "model_registered",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "model_version": version,
                "source_run_id": source_run_id,
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
            wallet_mode = getattr(self._anchor, "wallet_mode", None)
            if wallet_mode:
                tags["ario.wallet_mode"] = wallet_mode

            for key, value in tags.items():
                self.set_model_version_tag(model_name, version, key, value)

            with tempfile.TemporaryDirectory() as tmpdir:
                ario_dir = os.path.join(tmpdir, "ario")
                os.makedirs(ario_dir)

                with open(os.path.join(ario_dir, "registration_proof.json"), "w") as f:
                    json.dump(proof, f, indent=2)

                if result and result.get("receipt"):
                    with open(os.path.join(ario_dir, "registration_receipt.json"), "w") as f:
                        json.dump(result["receipt"], f, indent=2)

                report = generate_verification_html(
                    proof, result,
                    artifact_hash=expected_hash,
                    artifact_verified=artifact_verified,
                    cli_verify_cmd=f"ario-mlflow verify model {model_name}/{version}",
                    wallet_mode=wallet_mode,
                )
                with open(os.path.join(ario_dir, "registration_verification.html"), "w") as f:
                    f.write(report)

                if source_run_id:
                    self.log_artifacts(source_run_id, ario_dir, "ario")

            status = "anchored" if result else "signed (anchoring disabled or upload failed)"
            logger.info(f"Registration {model_name}/v{version} {status}: verified={artifact_verified}")

            self._record_status(
                "registration", model_name, version,
                status="anchored" if result else "signed",
                tx_id=result["tx_id"] if result else None,
            )

        except Exception as e:
            logger.error(f"Failed to anchor registration {model_name}/v{version}: {e}")
            self._record_status(
                "registration", model_name, version,
                status="failed",
                error=str(e),
            )
        finally:
            if done_event is not None:
                done_event.set()

    def _anchor_promotion(
        self,
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
        done_event: threading.Event | None = None,
    ):
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
                self._record_status(
                    "promotion", model_name, version,
                    status="anchored", tx_id=result["tx_id"],
                )
            else:
                self._record_status(
                    "promotion", model_name, version,
                    status="failed",
                    error="upload returned no result",
                )

        except Exception as e:
            logger.error(f"Failed to anchor promotion {model_name}/v{version}: {e}")
            self._record_status(
                "promotion", model_name, version,
                status="failed",
                error=str(e),
            )
        finally:
            if done_event is not None:
                done_event.set()
