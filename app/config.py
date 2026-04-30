import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Settings:
    app_name: str = "Verifiable-AI-Demo"
    otel_service_name: str = "verifiable-ai-demo"

    # Key paths
    ed25519_private_key_path: str = "keys/ed25519_private.json"
    ed25519_public_key_path: str = "keys/ed25519_public.json"
    arweave_wallet_path: str = "keys/arweave_wallet.json"

    # Storage
    records_file: str = "data/records.json"
    lifecycle_file: str = "data/lifecycle.json"

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_model_name: str = "credit-scorer"

    # Demo-only routes (e.g. /tamper/*) are gated behind this flag.
    # Defaults to True for the public demo on Railway. Set
    # VAIDR_DEMO_MODE=false in any production deployment to disable
    # the routes that mutate live MLflow state.
    demo_mode: bool = True

    # AR.IO
    ario_gateway_host: str = "turbo-gateway.com"
    # Public ar.io Verify instance operated by an ar.io gateway (vilenarios).
    # Reachability is tested at startup via /health; the topbar shows an offline
    # indicator when the service isn't responding. Override with
    # VAIDR_ARIO_VERIFY_URL to point at a different attestation endpoint.
    ario_verify_url: str = "https://vilenarios.com/local/verify"

    @property
    def arweave_enabled(self) -> bool:
        return os.path.exists(self.arweave_wallet_path)

    @property
    def ario_verify_enabled(self) -> bool:
        return bool(self.ario_verify_url)

    @classmethod
    def from_env(cls) -> "Settings":
        prefix = "VAIDR_"
        kwargs = {}
        for f, field_def in cls.__dataclass_fields__.items():
            env_key = prefix + f.upper()
            val = os.environ.get(env_key)
            if val is not None:
                # Coerce booleans from common string values; leave other
                # types as strings (the dataclass declares them str).
                if field_def.type is bool or field_def.type == "bool":
                    kwargs[f] = val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    kwargs[f] = val
        return cls(**kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
