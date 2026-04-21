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

        try:
            from turbo_sdk import ArweaveSigner, Turbo

            jwk = None
            if wallet_path and os.path.exists(wallet_path):
                try:
                    with open(wallet_path) as f:
                        jwk = json.load(f)
                    logger.info(f"Using Arweave wallet from {wallet_path}")
                except (OSError, json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"Invalid Arweave wallet at {wallet_path}: {e}; generating a fresh in-memory wallet"
                    )
            if jwk is None:
                jwk = self._generate_wallet()
                logger.info("Auto-generated in-memory Arweave wallet for anchoring")

            self._signer = ArweaveSigner(jwk)
            turbo = Turbo(self._signer)
            self._upload_url = turbo.upload_url
            self._token = turbo.token
            self.enabled = True
            logger.info(f"Arweave anchoring enabled (wallet: {self._signer.get_wallet_address()})")
        except Exception as e:
            logger.warning(f"Failed to initialize Arweave anchor: {e}")

    @staticmethod
    def _generate_wallet() -> dict:
        """Generate a fresh Arweave RSA-4096 wallet in JWK format."""
        import base64
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        pn = private_key.private_numbers()
        pub = pn.public_numbers

        def to_b64(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, "big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

        return {
            "kty": "RSA",
            "n": to_b64(pub.n),
            "e": to_b64(pub.e),
            "d": to_b64(pn.d),
            "p": to_b64(pn.p),
            "q": to_b64(pn.q),
            "dp": to_b64(pn.dmp1),
            "dq": to_b64(pn.dmq1),
            "qi": to_b64(pn.iqmp),
        }

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
