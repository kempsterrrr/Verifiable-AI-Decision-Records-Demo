import logging

import requests

logger = logging.getLogger(__name__)


class ArioVerifyClient:
    """Client for AR.IO Verify REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.enabled = False

        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                self.enabled = True
                logger.info(f"AR.IO Verify connected at {self.base_url}")
            else:
                logger.warning(f"AR.IO Verify health check returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"AR.IO Verify unavailable at {self.base_url}: {e}")

    def submit_verification(self, tx_id: str) -> dict | None:
        """Submit a transaction for verification. Returns verification result or None."""
        if not self.enabled:
            return None

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/verify",
                json={"txId": tx_id},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"AR.IO Verify submission failed: {e}")
            return None

    def check_verification(self, verification_id: str) -> dict | None:
        """Check status of a verification by ID."""
        if not self.enabled:
            return None

        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/verify/{verification_id}",
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"AR.IO Verify check failed: {e}")
            return None

    @staticmethod
    def _normalize_result(data: dict) -> dict:
        """Extract key fields from AR.IO Verify response."""
        links = data.get("links", {})
        return {
            "verification_id": data.get("verificationId"),
            "status": data.get("existence", {}).get("status", "unknown"),
            "level": data.get("level"),
            "attestation_url": links.get("dashboard"),
            "pdf_url": links.get("pdf"),
        }
