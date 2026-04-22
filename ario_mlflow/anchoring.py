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


class ArtifactAccessError(RuntimeError):
    """Raised when an MLflow run's artifacts cannot be downloaded or read for hashing.

    Callers must NOT treat this as "no artifacts" — the true state is unknown and
    they should skip writing an `ario.artifact_hash` rather than anchor a hash of
    an empty tree as if it were a real provenance record.
    """


def _logged_model_paths(run_data) -> list[str]:
    """Return the artifact paths of every model logged in this run.

    MLflow writes a ``mlflow.log-model.history`` tag whose value is a JSON list
    describing each ``mlflow.<flavor>.log_model`` call in the run. Reading this
    tag lets ``anchor()`` hash whatever the user actually logged, rather than
    silently defaulting to ``"model"`` and skipping the hash when the caller
    used a different name.
    """
    history_json = run_data.data.tags.get("mlflow.log-model.history")
    if not history_json:
        return []
    try:
        history = json.loads(history_json)
    except (ValueError, TypeError):
        return []
    paths = []
    for entry in history:
        if isinstance(entry, dict) and entry.get("artifact_path"):
            paths.append(entry["artifact_path"])
    return paths


def parse_runs_uri(source: str | None) -> tuple[str | None, str | None]:
    """Parse a ``runs:/<run_id>/<artifact_path>`` URI.

    Returns ``(run_id, artifact_path)`` where either element may be ``None`` if
    the source is missing, not a ``runs:/`` URI, or has no artifact path. This
    matters because MLflow's ``ModelVersion.source`` preserves the original
    artifact path from registration (e.g. ``sklearn-model``, ``keras-model``)
    and we must not assume it is always ``model``.
    """
    if not source or not source.startswith("runs:/"):
        return None, None
    rest = source[len("runs:/"):].lstrip("/")
    if "/" not in rest:
        return (rest or None), None
    run_id, artifact_path = rest.split("/", 1)
    return (run_id or None), (artifact_path or None)


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
    except Exception as e:
        # Callers must not silently anchor an empty tree as if it were the
        # artifact's real hash — surface the failure so they can skip.
        raise ArtifactAccessError(
            f"Could not download artifacts for run {run_id!r} at path {artifact_path!r}: {e}"
        ) from e

    checksums: dict[str, str] = {}
    for root, _dirs, files in os.walk(local_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_path)
            try:
                with open(fpath, "rb") as f:
                    checksums[rel] = hashlib.sha256(f.read()).hexdigest()
            except OSError as e:
                raise ArtifactAccessError(
                    f"Failed to read artifact file {fpath!r} for run {run_id!r}: {e}"
                ) from e
    return checksums


def anchor(
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
    artifact_path: str | None = None,
) -> dict:
    """Create a verifiable proof of the current training run.

    Must be called inside an active ``mlflow.start_run()`` block, after
    artifacts have been logged. Signs a proof envelope, optionally uploads
    to Arweave, and writes rich tags + artifacts to the run.

    Args:
        proof_engine: Optional override for the signing engine.
        arweave: Optional override for the Arweave anchor client.
        artifact_path: The MLflow artifact subdirectory that was logged
            (e.g. ``"model"``, ``"sklearn-model"``). When ``None`` (the
            default), the path is auto-resolved from MLflow's
            ``mlflow.log-model.history`` tag — which records exactly what
            was logged — so callers who used a custom path no longer need
            to re-specify it. Falls back to ``"model"`` if no model was
            logged. Pass explicitly to hash a non-model subdirectory, or
            to pick one path when multiple models were logged.

    Returns:
        A dict with keys:

        - ``proof`` — the signed proof envelope
        - ``anchor_result`` — Turbo upload result (``None`` if disabled/failed)
        - ``tags`` — the MLflow tags written on the run
        - ``artifact_path`` — the path actually used for hashing
        - ``artifact_status`` — one of ``"hashed"`` (artifacts hashed
          successfully), ``"no_artifacts"`` (no files found at the path),
          or ``"hash_failed"`` (download/read error; details in
          ``artifact_error``)
        - ``artifact_error`` — the error message when
          ``artifact_status == "hash_failed"``, otherwise ``None``
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

    # Auto-resolve the artifact path from the run's logged-model history when
    # the caller did not specify one. This replaces the old hardcoded default
    # of "model", which silently minted proofs with no artifact hash when the
    # caller logged under a different name.
    resolved_path = artifact_path
    if resolved_path is None:
        logged_paths = _logged_model_paths(run_data)
        if len(logged_paths) == 1:
            resolved_path = logged_paths[0]
        elif len(logged_paths) > 1:
            logger.warning(
                f"Run {run_id} logged {len(logged_paths)} models: {logged_paths}. "
                f"Hashing first; pass artifact_path explicitly to choose another."
            )
            resolved_path = logged_paths[0]
        else:
            resolved_path = "model"

    params = dict(run_data.data.params)
    metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in run_data.data.metrics.items()}
    artifact_status = "no_artifacts"
    artifact_error: str | None = None
    try:
        checksums = artifact_checksums(run_id, artifact_path=resolved_path)
        artifact_status = "hashed" if checksums else "no_artifacts"
    except ArtifactAccessError as e:
        # Artifact download/read failed. Anchor params/metrics as a record of
        # the run, but flag the status so callers can surface the failure.
        logger.warning(f"Skipping artifact_hash in proof for run {run_id}: {e}")
        checksums = {}
        artifact_status = "hash_failed"
        artifact_error = str(e)
    art_hash = hash_data(canonical_json(checksums)) if checksums else None

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
        "ario.public_key": proof["public_key"],
        "ario.verify_status": "anchored" if anchor_result else "signed",
    }
    if art_hash is not None:
        tags["ario.artifact_hash"] = art_hash
    if anchor_result:
        tags["ario.training_tx"] = anchor_result["tx_id"]
        tags["ario.arweave_url"] = anchor_result["url"]
    wallet_mode = getattr(arweave, "wallet_mode", None)
    if wallet_mode:
        tags["ario.wallet_mode"] = wallet_mode

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
        html_content = generate_verification_html(
            proof, anchor_result,
            artifact_hash=art_hash,
            wallet_mode=wallet_mode,
        )
        with open(os.path.join(ario_dir, "verification.html"), "w") as f:
            f.write(html_content)

        mlflow.log_artifacts(ario_dir, "ario")

    logger.info(
        f"Run {run_id} anchored: status={tags['ario.verify_status']}, "
        f"artifacts={artifact_status} (path={resolved_path!r})"
    )

    return {
        "proof": proof,
        "anchor_result": anchor_result,
        "tags": tags,
        "artifact_path": resolved_path,
        "artifact_status": artifact_status,
        "artifact_error": artifact_error,
    }
