"""Arweave upload and retrieval via ar.io Turbo."""

import json
import logging
import os

import requests

from ario_mlflow.proof import canonical_json

logger = logging.getLogger(__name__)


class ArweaveAnchor:
    """Upload proof payloads to Arweave via Turbo SDK."""

    def __init__(self, wallet_path: str | None = None, gateway_host: str = "turbo-gateway.com"):
        self.gateway_host = gateway_host
        self.enabled = False
        self._signer = None
        self._upload_url = None
        self._token = None

        wallet_path = wallet_path or os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", "")
        if not wallet_path or not os.path.exists(wallet_path):
            logger.warning("Arweave wallet not found. Anchoring disabled.")
            return

        try:
            from turbo_sdk import ArweaveSigner, Turbo

            with open(wallet_path) as f:
                jwk = json.load(f)
            self._signer = ArweaveSigner(jwk)
            turbo = Turbo(self._signer)
            self._upload_url = turbo.upload_url
            self._token = turbo.token
            self.enabled = True
            logger.info("Arweave anchoring enabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Arweave anchor: {e}")

    def upload_proof(self, proof: dict, tags: list[dict] | None = None) -> dict | None:
        if not self.enabled or not self._signer:
            return None

        try:
            from turbo_sdk.bundle import create_data, sign

            data_bytes = canonical_json(proof)
            record = proof.get("record", {})
            event_type = record.get("event_type", record.get("decision_id", "unknown"))
            record_hash = proof.get("record_hash", "unknown")

            default_tags = [
                {"name": "Content-Type", "value": "application/json"},
                {"name": "App-Name", "value": "ario-mlflow"},
                {"name": "Record-Type", "value": event_type},
                {"name": "Record-Hash", "value": record_hash},
            ]

            data_item = create_data(bytearray(data_bytes), self._signer, tags or default_tags)
            sign(data_item, self._signer)

            url = f"{self._upload_url}/tx/{self._token}"
            raw_data = data_item.get_raw()
            response = requests.post(
                url,
                data=raw_data,
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(raw_data))},
            )

            if response.status_code != 200:
                raise Exception(f"Upload failed: {response.status_code} - {response.text}")

            receipt = response.json()
            tx_id = receipt["id"]
            logger.info(f"Uploaded to Arweave: tx_id={tx_id}")
            return {"tx_id": tx_id, "url": f"https://{self.gateway_host}/{tx_id}", "receipt": receipt}

        except Exception as e:
            logger.error(f"Arweave upload failed: {e}")
            return None

    def fetch_proof(self, tx_id: str) -> dict | None:
        try:
            resp = requests.get(f"https://{self.gateway_host}/raw/{tx_id}", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch from Arweave: {e}")
            return None

    def check_status(self, tx_id: str) -> dict:
        try:
            resp = requests.get(f"https://turbo.ardrive.io/tx/{tx_id}/status", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"status": data.get("status", "UNKNOWN"), "info": data.get("info")}
            return {"status": "NOT_FOUND"}
        except Exception as e:
            logger.error(f"Failed to check Turbo status: {e}")
            return {"status": "UNKNOWN"}
