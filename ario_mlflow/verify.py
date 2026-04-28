"""Four-check verification helpers for the pure-commitment design.

The four checks (see plan Part 3 verification flow):

1. **Cryptographic signature.** Validate Ed25519 signature on the
   envelope. Local, instant.
2. **Anchored bytes intact.** Download ``ario/payload.json`` from MLflow,
   re-hash, compare to the envelope's ``payload_hash``. Catches
   tampering with the canonical witness in MLflow's artifact store.
3. **Live MLflow matches anchored bytes.** Re-fetch the "live" fields
   (params, metrics, artifact checksums) from MLflow and rebuild the
   canonical payload, holding "snapshot" fields (caller metadata,
   trace IDs) constant from the original. Compare bytes. Catches
   tampering with MLflow's tracking-store data after anchoring.
4. **(Optional) ar.io Verify Level 3 attestation.** Independent
   third-party confirmation that the Arweave TX exists and is
   permanently stored. Implemented by ``ArioVerifyClient`` — unchanged
   from v1.

Verifiers wanting only signature + envelope-internal consistency can
call :func:`verify_signature` alone. Auditors with MLflow access run
:func:`full_verify` to get all four. The legacy three-level helpers
(``verify_record``, ``verify_arweave``, ``verify_ario``,
``full_verify`` in their v1 form) are deleted in this redesign — they
covered a different design where the proof carried the source data.
"""

import json
import logging
import os

import requests

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data, normalize_floats
from ario_mlflow.arweave import ArweaveAnchor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live-field re-derivation per event type
# ---------------------------------------------------------------------------
#
# For check 3 we need to know which fields in the canonical payload are
# "live" (re-fetchable from MLflow at verify time) vs. "snapshot" (set at
# anchor time and unchanged thereafter). The hash of the rebuilt payload
# equals the original only if every live field still matches; any
# tampered live field flips check 3 to FAIL.
#
# Snapshot fields (taken from the downloaded payload.json as-is): event_id,
# event_type, signed_at, caller metadata (otel_trace_id, service_name,
# etc.), mlflow_trace_id (per-call, can't be re-derived from the run).


def _refetch_training_live_fields(payload: dict, mlflow_client) -> dict:
    """Re-fetch the live training fields from MLflow's current state."""
    from ario_mlflow.anchoring import artifact_checksums, ArtifactAccessError, _logged_model_paths

    run_id = payload.get("run_id")
    if not run_id:
        return {}
    run = mlflow_client.get_run(run_id)

    # Resolve artifact path the same way anchor() did — from the run's
    # logged-model history. Necessary for non-default paths.
    logged_paths = list(dict.fromkeys(_logged_model_paths(run)))
    if len(logged_paths) == 1:
        artifact_path = logged_paths[0]
    elif len(logged_paths) > 1:
        # Multiple models logged: hash the artifact_checksums map by
        # walking each path. Anchor() rejects this case at write time;
        # at verify time we follow the original payload's checksum map's
        # key prefixes if present.
        artifact_path = "model"
    else:
        artifact_path = "model"

    fresh: dict = {
        "params": dict(run.data.params),
        "metrics": normalize_floats(dict(run.data.metrics), precision=6),
        "source_name": run.data.tags.get("mlflow.source.name", ""),
        "git_commit": run.data.tags.get("mlflow.source.git.commit", ""),
    }
    try:
        fresh["artifact_checksums"] = artifact_checksums(run_id, artifact_path=artifact_path)
    except ArtifactAccessError:
        # If artifacts can't be re-hashed, leave the field unset — the
        # original payload's value will be used as-is, which means this
        # check can't catch artifact tampering. Other checks still run.
        logger.warning(
            f"Could not re-hash artifacts for run {run_id} during verify; "
            f"check 3 will not catch artifact tampering for this proof."
        )
    return fresh


def _refetch_registration_live_fields(payload: dict, mlflow_client) -> dict:
    """Re-fetch the live registration fields from MLflow's current state."""
    from ario_mlflow.anchoring import artifact_checksums, ArtifactAccessError, parse_runs_uri

    source_run_id = payload.get("source_run_id")
    fresh: dict = {}
    if not source_run_id:
        return fresh

    try:
        run = mlflow_client.get_run(source_run_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not fetch source run {source_run_id} during verify: {e}")
        return fresh

    expected_hash = run.data.tags.get("ario.artifact_hash")
    fresh["artifact_hash"] = expected_hash

    src_run_id, src_artifact_path = parse_runs_uri(payload.get("source"))
    artifact_path = src_artifact_path or "model"
    try:
        checksums = artifact_checksums(source_run_id, artifact_path=artifact_path)
    except ArtifactAccessError:
        checksums = {}
    if checksums and expected_hash is not None:
        computed = hash_data(canonical_json(checksums))
        fresh["artifact_verified"] = computed == expected_hash
    return fresh


_LIVE_FIELD_REFETCHERS = {
    "training_complete": _refetch_training_live_fields,
    "model_registered": _refetch_registration_live_fields,
    # Predictions have no plugin-level live fields. Their canonical
    # payload contains hashes of input/output that can only be re-derived
    # if the caller has the raw values. Auditors verifying predictions
    # should run check 3 themselves by hashing their copy of the raw
    # input/output and comparing to the payload's input_hash / output_hash.
}


# ---------------------------------------------------------------------------
# Payload-download helpers per subject type
# ---------------------------------------------------------------------------

def _download_payload_for_envelope(envelope: dict, mlflow_client) -> bytes | None:
    """Download the ``ario/payload.json`` artifact for an envelope's subject.

    Returns the raw bytes (so the caller can hash them directly without
    risking re-canonicalization drift). ``None`` if the artifact can't be
    located — distinguishes "downloaded but mismatched" from "not found"
    in the verification result.
    """
    import mlflow

    subject = envelope.get("subject", {})
    subject_type = subject.get("type")
    event_type = envelope.get("event_type")

    # Map (subject_type, event_type) → (run_id, artifact_path).
    if subject_type == "mlflow_run":
        run_id = subject.get("run_id")
        artifact_path = "ario/payload.json"
    elif subject_type == "mlflow_model_version":
        # Registration / promotion proofs were anchored against the
        # source run; payload.json lives under that run's ario/ tree
        # with an event-specific filename.
        name = subject.get("name")
        version = subject.get("version")
        if not name or not version:
            return None
        try:
            mv = mlflow_client.get_model_version(name, str(version))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not resolve {name}/v{version} during verify: {e}")
            return None
        run_id = mv.run_id
        # Filename varies by event type — anchor() writes payload.json,
        # ArioMlflowClient writes registration_payload.json or
        # promotion_<version>_payload.json.
        if event_type == "model_registered":
            artifact_path = "ario/registration_payload.json"
        elif event_type == "stage_transition":
            artifact_path = f"ario/promotion_{version}_payload.json"
        else:
            artifact_path = "ario/payload.json"
    elif subject_type == "mlflow_prediction":
        # Predictions write canonical bytes as an artifact on the
        # model's source run at ario/predictions/<decision_id>/payload.json.
        # Trace tags exist for observability but are not the source of
        # truth — the artifact is.
        decision_id = subject.get("decision_id")
        run_id = subject.get("model_run_id")
        if not decision_id or not run_id:
            return None
        artifact_path = f"ario/predictions/{decision_id}/payload.json"
    elif subject_type in ("mlflow_trace", "mlflow_decision"):
        # Legacy prediction subject types (pre-Phase-1.14). Kept as a
        # graceful path for proofs anchored under the older subject
        # format; they don't have payload.json artifacts.
        return None
    else:
        logger.warning(f"Unknown subject type for download: {subject_type!r}")
        return None

    if not run_id:
        return None

    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not download {artifact_path} for run {run_id} during verify: {e}"
        )
        return None

    try:
        with open(local_path, "rb") as f:
            return f.read()
    except OSError as e:
        logger.warning(f"Could not read downloaded payload at {local_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# The four checks
# ---------------------------------------------------------------------------

def verify_signature(envelope: dict, proof_engine: ProofEngine) -> dict:
    """Check 1: cryptographic signature on the envelope.

    Wraps :meth:`ProofEngine.verify_commitment` (without payload bytes —
    that's check 2). Returns ``{ok: bool, ...}`` for uniform composition
    with the other checks.
    """
    result = proof_engine.verify_commitment(envelope)
    return {
        "ok": result["signature_valid"],
        "signature_valid": result["signature_valid"],
    }


def verify_anchored_bytes(envelope: dict, mlflow_client) -> dict:
    """Check 2: anchored bytes intact.

    Downloads ``ario/payload.json`` from MLflow, re-hashes the bytes,
    compares to ``envelope["payload_hash"]``. Catches tampering with the
    canonical witness in MLflow's artifact store.

    For event types that don't write payload.json (predictions,
    promotions) the result is ``{ok: None, reason: "not_applicable", ...}``
    — these proofs are verified by check 1 plus event-type-specific
    means (raw input/output hashes for predictions; the registration
    chain for promotions).
    """
    payload_bytes = _download_payload_for_envelope(envelope, mlflow_client)
    if payload_bytes is None:
        return {
            "ok": None,
            "reason": "payload_artifact_not_available",
            "computed_hash": None,
            "stored_hash": envelope.get("payload_hash"),
            "payload_bytes": None,
        }
    computed = hash_data(payload_bytes)
    stored = envelope.get("payload_hash")
    return {
        "ok": computed == stored,
        "computed_hash": computed,
        "stored_hash": stored,
        "payload_bytes": payload_bytes,
    }


def verify_source_of_truth(
    envelope: dict,
    payload_bytes: bytes,
    mlflow_client,
) -> dict:
    """Check 3: live MLflow data still matches anchored bytes.

    Parses ``payload_bytes`` (from check 2's download), re-fetches the
    "live" fields from MLflow's current state, and compares the rebuilt
    canonical bytes to the original. Catches MLflow-side tampering after
    anchoring — the central guarantee of the redesign.

    Args:
        envelope: The signed envelope (for event_type / subject).
        payload_bytes: The canonical bytes downloaded by check 2. Must
            be the exact bytes — re-canonicalizing parsed JSON could
            produce different output if the parsed dict's iteration
            order differs.
        mlflow_client: An ``MlflowClient`` for live re-fetching.
    """
    if not payload_bytes:
        return {"ok": None, "reason": "no_payload_to_compare"}

    event_type = envelope.get("event_type")
    refetcher = _LIVE_FIELD_REFETCHERS.get(event_type)
    if refetcher is None:
        return {
            "ok": None,
            "reason": "no_live_fields_for_event_type",
            "event_type": event_type,
            "note": (
                "Predictions commit to hashes of input/output. To run check 3 "
                "for a prediction, hash your copy of the raw input/output and "
                "compare to the payload's input_hash / output_hash directly."
            ),
        }

    try:
        original_payload = json.loads(payload_bytes)
    except (ValueError, TypeError) as e:
        return {"ok": False, "reason": f"payload_parse_failed: {e}"}

    try:
        fresh_fields = refetcher(original_payload, mlflow_client)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"live_refetch_failed: {e}"}

    # Build the rebuilt payload: original + live overrides. Snapshot
    # fields (anything not in fresh_fields) flow through unchanged.
    rebuilt = dict(original_payload)
    rebuilt.update(fresh_fields)
    rebuilt_bytes = canonical_json(rebuilt)

    return {
        "ok": rebuilt_bytes == payload_bytes,
        "rebuilt_bytes": rebuilt_bytes,
        "live_fields_refetched": list(fresh_fields.keys()),
    }


def verify_ario_attestation(envelope: dict, ario_client: "ArioVerifyClient | None") -> dict:
    """Check 4 (optional): ar.io Verify Level 3 attestation.

    Independent third-party confirmation that the Arweave TX exists and
    is permanently stored. Returns the attestation result (``level``,
    ``attested_by``, ``report_url``, ...) or ``None`` if the client is
    disabled / not provided.
    """
    if ario_client is None or not getattr(ario_client, "enabled", False):
        return {"ok": None, "reason": "ario_verify_not_enabled"}

    # Pull the Arweave TX ID from wherever it's stored. The envelope
    # itself doesn't carry the TX (the TX is the address ON Arweave) —
    # callers passing an envelope here are expected to also know the
    # TX, typically from MLflow tags (ario.training_tx, etc.). For now
    # we accept it via a special key the caller adds; future API may
    # split the envelope and the TX more cleanly.
    tx_id = envelope.get("_tx_id") or envelope.get("arweave_tx_id")
    if not tx_id:
        return {"ok": None, "reason": "no_tx_id_provided"}

    result = ario_client.submit_verification(tx_id)
    if not result:
        return {"ok": False, "reason": "ario_verify_returned_no_result"}
    return {
        "ok": True,
        "attestation_level": result.get("attestation_level"),
        "attested_by": result.get("attested_by"),
        "attested_at": result.get("attested_at"),
        "report_url": result.get("report_url"),
        "pdf_url": result.get("pdf_url"),
    }


def full_verify(
    envelope: dict,
    *,
    proof_engine: ProofEngine,
    mlflow_client=None,
    ario_client: "ArioVerifyClient | None" = None,
) -> dict:
    """Run all four checks and return a combined result.

    Each check is independent — failures in one don't short-circuit the
    others, so the caller sees the complete state. ``overall`` is
    ``True`` only when every applicable check passed (checks that are
    not applicable to this event type don't count against ``overall``).
    """
    sig = verify_signature(envelope, proof_engine)
    bytes_check = (
        verify_anchored_bytes(envelope, mlflow_client) if mlflow_client else
        {"ok": None, "reason": "no_mlflow_client"}
    )
    sot = (
        verify_source_of_truth(envelope, bytes_check.get("payload_bytes") or b"", mlflow_client)
        if mlflow_client and bytes_check.get("payload_bytes")
        else {"ok": None, "reason": "no_payload_to_compare"}
    )
    ario = verify_ario_attestation(envelope, ario_client) if ario_client else {"ok": None, "reason": "no_ario_client"}

    # Overall: all "True"s pass, any "False" fails. None means "not
    # applicable" and is neutral.
    statuses = [sig["ok"], bytes_check["ok"], sot["ok"], ario["ok"]]
    if any(s is False for s in statuses):
        overall = False
    elif any(s is True for s in statuses):
        overall = True
    else:
        overall = None  # nothing was checked

    return {
        "signature": sig,
        "anchored_bytes": bytes_check,
        "source_of_truth": sot,
        "ario_attestation": ario,
        "overall": overall,
    }


class ArioVerifyClient:
    """Client for AR.IO Verify REST API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.environ.get("ARIO_MLFLOW_ARIO_VERIFY_URL", "")).rstrip("/")
        self.enabled = False

        if not self.base_url:
            return

        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                self.enabled = True
                logger.info(f"ar.io Verify connected at {self.base_url}")
        except Exception as e:
            logger.warning(f"ar.io Verify unavailable: {e}")

    def submit_verification(self, tx_id: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/verify",
                json={"txId": tx_id},
                timeout=30,
            )
            resp.raise_for_status()
            return self._normalize(resp.json())
        except Exception as e:
            logger.error(f"ar.io Verify failed: {e}")
            return None

    def _normalize(self, data: dict) -> dict:
        # ar.io Verify returns explicit nulls for sub-objects when no
        # attestation is yet available (e.g. for fresh TXs not yet
        # indexed). dict.get(key, {}) returns the key's actual value
        # when present — None, not the {} default. Use ``or {}`` to
        # collapse both None and missing into {}.
        links = data.get("links") or {}
        attestation = data.get("attestation") or {}
        existence = data.get("existence") or {}

        def resolve(path):
            if not path:
                return None
            return path if path.startswith("http") else f"{self.base_url}{path}"

        return {
            "verification_id": data.get("verificationId"),
            "status": existence.get("status", "unknown"),
            "attestation_level": data.get("level"),
            "report_url": resolve(links.get("dashboard")),
            "pdf_url": resolve(links.get("pdf")),
            "attested_by": attestation.get("gateway"),
            "attested_at": attestation.get("attestedAt"),
        }


# Note on deleted v1 helpers: the legacy three-level helpers
# (verify_record / verify_arweave / verify_ario / legacy full_verify)
# lived at the bottom of this module in v1. They covered the v1
# envelope shape where source data was inside the proof itself; the
# pure-commitment design needs different checks (download payload.json,
# compare hash, re-derive from MLflow). The new helpers near the top
# of this file (verify_signature / verify_anchored_bytes /
# verify_source_of_truth / verify_ario_attestation / new full_verify)
# are the replacements. CLI consumers updated in Phase 1.9.
