"""Ed25519 signing, canonical JSON, and SHA-256 hashing for proof records."""

import base64
import hashlib
import json
import os

from nacl.signing import SigningKey, VerifyKey


def normalize_floats(obj, precision=6):
    """Recursively round floats for deterministic hashing."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: normalize_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_floats(v, precision) for v in obj]
    return obj


def canonical_json(obj: dict) -> bytes:
    """Deterministic JSON serialization: sorted keys, compact, UTF-8."""
    normalized = normalize_floats(obj)
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def hash_data(data: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


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
    return SigningKey(base64.b64decode(data["seed"]))


def load_signing_key_from_env(env_var: str = "ARIO_MLFLOW_SIGNING_KEY") -> SigningKey | None:
    """Load signing key from base64-encoded environment variable."""
    val = os.environ.get(env_var)
    if val:
        return SigningKey(base64.b64decode(val))
    return None


def load_verify_key(path: str) -> VerifyKey:
    with open(path, "r") as f:
        data = json.load(f)
    return VerifyKey(base64.b64decode(data["key"]))


class ProofEngine:
    """Creates and verifies hash-chained, Ed25519-signed proof envelopes."""

    def __init__(self, private_key_path: str | None = None, public_key_path: str | None = None):
        # Try env var first, then key files, then auto-generate
        sk = load_signing_key_from_env()
        if sk:
            self._sk = sk
            self._vk = sk.verify_key
        elif private_key_path and os.path.exists(private_key_path):
            self._sk = load_signing_key(private_key_path)
            self._vk = load_verify_key(public_key_path)
        else:
            priv = private_key_path or os.path.expanduser("~/.ario-mlflow/keys/ed25519_private.json")
            pub = public_key_path or os.path.expanduser("~/.ario-mlflow/keys/ed25519_public.json")
            self._sk, self._vk = generate_keypair(priv, pub)

    def create_proof(self, record: dict, previous_hash: str) -> dict:
        record_hash = hash_data(canonical_json(record))
        sign_payload = canonical_json({
            "record_hash": record_hash,
            "previous_hash": previous_hash,
            "timestamp": record["timestamp"],
        })
        signed = self._sk.sign(sign_payload)
        return {
            "record": record,
            "record_hash": record_hash,
            "previous_hash": previous_hash,
            "signature": signed.signature.hex(),
            "public_key": bytes(self._vk).hex(),
        }

    def verify_local(self, envelope: dict) -> dict:
        record = envelope["record"]
        stored_hash = envelope["record_hash"]
        computed_hash = hash_data(canonical_json(record))
        hash_valid = computed_hash == stored_hash

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
            pass

        return {
            "hash_valid": hash_valid,
            "signature_valid": sig_valid,
            "computed_hash": computed_hash,
            "stored_hash": stored_hash,
            "overall": hash_valid and sig_valid,
        }
