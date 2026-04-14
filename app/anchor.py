import json
import logging
import os

import requests

from app.decision_record import canonical_json

logger = logging.getLogger(__name__)


class ArweaveAnchor:
    """Upload proof payloads to Arweave via Turbo SDK."""

    def __init__(self, wallet_path: str, gateway_host: str = "arweave.net"):
        self.gateway_host = gateway_host
        self.enabled = False
        self._signer = None
        self._upload_url = None
        self._token = None

        if not os.path.exists(wallet_path):
            logger.warning(f"Arweave wallet not found at {wallet_path}. Anchoring disabled.")
            return

        try:
            from turbo_sdk import ArweaveSigner, Turbo

            with open(wallet_path) as f:
                jwk = json.load(f)
            self._signer = ArweaveSigner(jwk)
            # Get upload URL and token from a Turbo instance
            turbo = Turbo(self._signer)
            self._upload_url = turbo.upload_url
            self._token = turbo.token
            self.enabled = True
            logger.info("Arweave anchoring enabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Arweave anchor: {e}")

    def upload_proof(self, proof: dict) -> dict | None:
        """Upload proof to Arweave. Returns dict with tx_id, url, and full receipt."""
        if not self.enabled or not self._signer:
            return None

        try:
            from turbo_sdk.bundle import create_data, sign

            data_bytes = canonical_json(proof)
            decision_id = proof.get("record", {}).get("decision_id", "unknown")
            record_hash = proof.get("record_hash", "unknown")

            # Create and sign data item using the SDK
            data_item = create_data(
                bytearray(data_bytes),
                self._signer,
                [
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "App-Name", "value": "Verifiable-AI-Demo"},
                    {"name": "Record-Type", "value": "DecisionProof"},
                    {"name": "Decision-ID", "value": decision_id},
                    {"name": "Record-Hash", "value": record_hash},
                ],
            )
            sign(data_item, self._signer)

            # Upload directly via HTTP to capture the full receipt
            url = f"{self._upload_url}/tx/{self._token}"
            raw_data = data_item.get_raw()
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(raw_data)),
            }

            response = requests.post(url, data=raw_data, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")

            # Capture the full receipt (all fields, not just the SDK's subset)
            receipt = response.json()
            tx_id = receipt["id"]
            gateway_url = f"https://{self.gateway_host}/{tx_id}"

            logger.info(f"Uploaded to Arweave: tx_id={tx_id}")
            return {
                "tx_id": tx_id,
                "url": gateway_url,
                "receipt": receipt,
            }

        except Exception as e:
            logger.error(f"Arweave upload failed: {e}")
            return None

    def fetch_proof(self, tx_id: str) -> dict | None:
        """Fetch proof envelope from Arweave gateway."""
        try:
            url = f"https://{self.gateway_host}/raw/{tx_id}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch from Arweave: {e}")
            return None

    def check_status(self, tx_id: str) -> dict:
        """Check upload status via Turbo status endpoint."""
        try:
            resp = requests.get(
                f"https://turbo.ardrive.io/tx/{tx_id}/status",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": data.get("status", "UNKNOWN"),
                    "info": data.get("info"),
                    "bundle_id": data.get("bundleId"),
                }
            return {"status": "NOT_FOUND"}
        except Exception as e:
            logger.error(f"Failed to check Turbo status: {e}")
            return {"status": "UNKNOWN"}
