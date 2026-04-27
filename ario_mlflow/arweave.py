"""Arweave upload and retrieval via ar.io Turbo."""

import json
import logging
import os

import requests

from ario_mlflow.proof import canonical_json

logger = logging.getLogger(__name__)

# Where the plugin keeps its auto-generated wallet so the same address is
# reused across sessions. Matches the pattern used by proof.py for signing
# keys (~/.ario-mlflow/keys/).
DEFAULT_WALLET_PATH = os.path.expanduser("~/.ario-mlflow/wallet.json")

# The three wallet_mode values exposed in logs / tags / reports:
#   user-configured — loaded from a caller-supplied wallet path.
#   persistent      — auto-generated at DEFAULT_WALLET_PATH and reused across runs.
#   ephemeral       — in-memory only (filesystem not writable); rotates every restart.
WALLET_MODE_USER = "user-configured"
WALLET_MODE_PERSISTENT = "persistent"
WALLET_MODE_EPHEMERAL = "ephemeral"

_REQUIRED_JWK_FIELDS = {"kty", "n", "e", "d", "p", "q", "dp", "dq", "qi"}


class ArweaveAnchor:
    """Upload proof payloads to Arweave via Turbo SDK."""

    def __init__(self, wallet_path: str | None = None, gateway_host: str = "turbo-gateway.com"):
        self.gateway_host = gateway_host
        self.enabled = False
        self.wallet_mode: str | None = None
        self._signer = None
        self._upload_url = None
        self._token = None

        wallet_path = wallet_path or os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", "")

        try:
            from turbo_sdk import ArweaveSigner, Turbo

            jwk, mode = self._load_or_create_wallet(wallet_path)

            self._signer = ArweaveSigner(jwk)
            turbo = Turbo(self._signer)
            self._upload_url = turbo.upload_url
            self._token = turbo.token
            self.enabled = True
            self.wallet_mode = mode

            address = self._signer.get_wallet_address()
            if mode == WALLET_MODE_USER:
                logger.info(f"Arweave anchoring enabled (wallet: {address}, mode=user-configured)")
            elif mode == WALLET_MODE_PERSISTENT:
                logger.info(
                    f"Arweave anchoring enabled (wallet: {address}, mode=persistent, "
                    f"path={DEFAULT_WALLET_PATH}) — set ARIO_MLFLOW_ARWEAVE_WALLET to use your own"
                )
            else:
                logger.warning(
                    f"Arweave anchoring enabled (wallet: {address}, mode=ephemeral) — "
                    f"wallet is in-memory only and will rotate on restart. "
                    f"Persistent wallet path {DEFAULT_WALLET_PATH} was not writable."
                )
        except Exception as e:
            logger.warning(f"Failed to initialize Arweave anchor: {e}")

    @classmethod
    def _load_or_create_wallet(cls, wallet_path: str) -> tuple[dict, str]:
        """Return ``(jwk, mode)`` for the wallet to use.

        Resolution order:

        1. Caller-supplied ``wallet_path`` (or ``ARIO_MLFLOW_ARWEAVE_WALLET``)
           — validate and use if well-formed; otherwise log a warning and
           fall through to (2).
        2. ``DEFAULT_WALLET_PATH`` — if it already exists, reuse it; if it
           doesn't, generate a new wallet and persist it there.
        3. If step (2)'s filesystem write fails, fall back to a pure
           in-memory wallet (``ephemeral`` mode).
        """
        if wallet_path and os.path.exists(wallet_path):
            try:
                with open(wallet_path) as f:
                    jwk = json.load(f)
                if not isinstance(jwk, dict) or not _REQUIRED_JWK_FIELDS.issubset(jwk):
                    raise ValueError("wallet file is not a complete RSA JWK")
                return jwk, WALLET_MODE_USER
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Invalid Arweave wallet at {wallet_path}: {e}; "
                    f"falling back to auto-generated wallet"
                )

        # No user-configured wallet (or it was invalid). Try to reuse or
        # create a persistent one.
        if os.path.exists(DEFAULT_WALLET_PATH):
            try:
                with open(DEFAULT_WALLET_PATH) as f:
                    jwk = json.load(f)
                if isinstance(jwk, dict) and _REQUIRED_JWK_FIELDS.issubset(jwk):
                    return jwk, WALLET_MODE_PERSISTENT
                logger.warning(
                    f"Persistent wallet at {DEFAULT_WALLET_PATH} is malformed; regenerating"
                )
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Could not read persistent wallet at {DEFAULT_WALLET_PATH}: {e}; regenerating"
                )

        jwk = cls._generate_wallet()
        try:
            os.makedirs(os.path.dirname(DEFAULT_WALLET_PATH), exist_ok=True)
            with open(DEFAULT_WALLET_PATH, "w") as f:
                json.dump(jwk, f)
            os.chmod(DEFAULT_WALLET_PATH, 0o600)
            logger.info(
                f"Auto-generated Arweave wallet at {DEFAULT_WALLET_PATH} — "
                f"back this up or set ARIO_MLFLOW_ARWEAVE_WALLET for production use"
            )
            return jwk, WALLET_MODE_PERSISTENT
        except OSError as e:
            logger.warning(
                f"Could not persist auto-generated wallet to {DEFAULT_WALLET_PATH}: {e}; "
                f"using in-memory wallet for this session only"
            )
            return jwk, WALLET_MODE_EPHEMERAL

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
                timeout=60,
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
