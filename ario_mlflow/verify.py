"""Three-level verification: local, Arweave, ar.io Verify."""

import logging
import os

import requests

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.anchor import ArweaveAnchor

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
            "level": data.get("level"),
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
