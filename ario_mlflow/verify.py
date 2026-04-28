"""Three-level verification: local, Arweave, ar.io Verify."""

import logging
import os

import requests

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor

logger = logging.getLogger(__name__)


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
        links = data.get("links", {})
        attestation = data.get("attestation", {})

        def resolve(path):
            if not path:
                return None
            return path if path.startswith("http") else f"{self.base_url}{path}"

        return {
            "verification_id": data.get("verificationId"),
            "status": data.get("existence", {}).get("status", "unknown"),
            "attestation_level": data.get("level"),
            "report_url": resolve(links.get("dashboard")),
            "pdf_url": resolve(links.get("pdf")),
            "attested_by": attestation.get("gateway"),
            "attested_at": attestation.get("attestedAt"),
        }


def verify_record(envelope: dict, proof_engine: ProofEngine) -> dict:
    """Level 1: Local verification — re-hash and check signature."""
    return proof_engine.verify_local(envelope)


def verify_arweave(envelope: dict, anchor: ArweaveAnchor) -> dict:
    """Level 2: Fetch from Arweave gateway and compare hashes."""
    tx_id = envelope.get("arweave_tx_id")
    if not tx_id:
        return {"arweave_data_found": False, "reason": "no_tx_id"}

    arweave_data = anchor.fetch_proof(tx_id)
    if not arweave_data:
        return {"arweave_data_found": False, "reason": "fetch_failed"}

    arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
    return {
        "arweave_data_found": True,
        "arweave_record_hash": arweave_hash,
        "hash_match": arweave_hash == arweave_data.get("record_hash"),
    }


def verify_ario(envelope: dict, ario_client: ArioVerifyClient) -> dict | None:
    """Level 3: ar.io Verify attestation."""
    tx_id = envelope.get("arweave_tx_id")
    if not tx_id:
        return None
    return ario_client.submit_verification(tx_id)


def full_verify(envelope: dict, proof_engine: ProofEngine, anchor: ArweaveAnchor, ario_client: ArioVerifyClient) -> dict:
    """Run all three verification levels and return a combined result."""
    local = verify_record(envelope, proof_engine)
    arweave = verify_arweave(envelope, anchor)
    ario = verify_ario(envelope, ario_client)

    return {
        "local": local,
        "arweave": arweave,
        "ario": ario,
        "overall": (
            local.get("overall", False)
            and arweave.get("hash_match", False)
        ),
    }


def verify_envelope(
    envelope: dict,
    proof_engine: ProofEngine,
    anchor: ArweaveAnchor,
    ario_verify: "ArioVerifyClient | None" = None,
) -> dict:
    """Three-level verification of a proof envelope.

    Runs:
    1. Local hash + signature check via ``proof_engine.verify_local``.
    2. If the envelope carries an ``arweave_tx_id``: fetch the canonical
       proof from the gateway, recompute the record hash, and compare.
    3. If ``ario_verify`` is provided AND enabled: request a level
       attestation from ar.io Verify.

    Returns a flat dict suitable for direct UI rendering. The shape is
    intentionally stable — UI templates can read fields without nested
    lookups, and a missing capability (e.g., ``ario_verify=None``) just
    leaves the corresponding fields as ``None``.

    Args:
        envelope: A signed proof envelope (with ``record``, ``record_hash``,
            ``signature``, ``public_key``, optionally ``arweave_tx_id``).
        proof_engine: The :class:`ProofEngine` that signed (or can verify)
            the envelope.
        anchor: An :class:`ArweaveAnchor` used to fetch the canonical
            proof from the gateway when an ``arweave_tx_id`` is present.
        ario_verify: Optional :class:`ArioVerifyClient` for the level
            attestation. ``None`` (or a disabled client) skips the
            attestation step.

    Returns:
        A dict with keys ``hash_valid``, ``signature_valid``,
        ``permanent_copy_found``, ``hash_match``, ``attestation_level``,
        ``report_url``, ``pdf_url``, ``attested_by``, ``attested_at``.
        Boolean fields default to ``False``; level/url/operator fields
        default to ``None``.
    """
    from ario_mlflow.proof import canonical_json, hash_data

    local = proof_engine.verify_local(envelope)
    result = {
        "hash_valid": local["hash_valid"],
        "signature_valid": local["signature_valid"],
        "permanent_copy_found": False,
        "hash_match": False,
        "attestation_level": None,
        "report_url": None,
        "pdf_url": None,
        "attested_by": None,
        "attested_at": None,
    }

    tx_id = envelope.get("arweave_tx_id")
    if tx_id:
        arweave_data = anchor.fetch_proof(tx_id)
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            result["permanent_copy_found"] = True
            result["hash_match"] = arweave_hash == arweave_data.get("record_hash")

        if ario_verify is not None and getattr(ario_verify, "enabled", False):
            normalized = ario_verify.submit_verification(tx_id)
            if normalized:
                result["attestation_level"] = normalized.get("attestation_level")
                result["report_url"] = normalized.get("report_url")
                result["pdf_url"] = normalized.get("pdf_url")
                result["attested_by"] = normalized.get("attested_by")
                result["attested_at"] = normalized.get("attested_at")

    return result
