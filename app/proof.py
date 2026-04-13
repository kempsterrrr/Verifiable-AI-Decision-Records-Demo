import base64
import json
import os

from nacl.signing import SigningKey, VerifyKey

from app.decision_record import canonical_json, hash_data


def generate_keypair(private_path: str, public_path: str) -> tuple[SigningKey, VerifyKey]:
    """Generate Ed25519 keypair and save to JSON files."""
    os.makedirs(os.path.dirname(private_path), exist_ok=True)
    sk = SigningKey.generate()
    vk = sk.verify_key

    with open(private_path, "w") as f:
        json.dump({"seed": base64.b64encode(bytes(sk)).decode()}, f)
    with open(public_path, "w") as f:
        json.dump({"key": base64.b64encode(bytes(vk)).decode()}, f)

    return sk, vk


def load_signing_key(path: str) -> SigningKey:
    with open(path, "r") as f:
        data = json.load(f)
    seed = base64.b64decode(data["seed"])
    return SigningKey(seed)


def load_verify_key(path: str) -> VerifyKey:
    with open(path, "r") as f:
        data = json.load(f)
    key_bytes = base64.b64decode(data["key"])
    return VerifyKey(key_bytes)


class ProofEngine:
    """Creates and verifies hash-chained, Ed25519-signed proof envelopes."""

    def __init__(self, private_key_path: str, public_key_path: str):
        if not os.path.exists(private_key_path):
            self._sk, self._vk = generate_keypair(private_key_path, public_key_path)
        else:
            self._sk = load_signing_key(private_key_path)
            self._vk = load_verify_key(public_key_path)

    def create_proof(self, record: dict, previous_hash: str) -> dict:
        """Create a proof envelope for a decision record."""
        record_hash = hash_data(canonical_json(record))

        sign_payload = canonical_json({
            "record_hash": record_hash,
            "previous_hash": previous_hash,
            "timestamp": record["timestamp"],
        })
        signed = self._sk.sign(sign_payload)
        signature = signed.signature.hex()
        public_key = bytes(self._vk).hex()

        return {
            "record": record,
            "record_hash": record_hash,
            "previous_hash": previous_hash,
            "signature": signature,
            "public_key": public_key,
        }

    def verify_local(self, envelope: dict) -> dict:
        """Verify record hash and signature locally. Returns verification result."""
        record = envelope["record"]
        stored_hash = envelope["record_hash"]

        # 1. Verify record hash
        computed_hash = hash_data(canonical_json(record))
        hash_valid = computed_hash == stored_hash

        # 2. Verify signature
        sig_valid = False
        try:
            vk = VerifyKey(bytes.fromhex(envelope["public_key"]))
            sign_payload = canonical_json({
                "record_hash": stored_hash,
                "previous_hash": envelope["previous_hash"],
                "timestamp": record["timestamp"],
            })
            vk.verify(sign_payload, bytes.fromhex(envelope["signature"]))
            sig_valid = True
        except Exception:
            sig_valid = False

        return {
            "hash_valid": hash_valid,
            "signature_valid": sig_valid,
            "computed_hash": computed_hash,
            "stored_hash": stored_hash,
            "chain_previous": envelope["previous_hash"],
            "overall": hash_valid and sig_valid,
        }
