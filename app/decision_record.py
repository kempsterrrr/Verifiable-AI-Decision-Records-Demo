"""Demo-specific decision record builder.

The cryptographic primitives (``canonical_json``, ``hash_data``,
``normalize_floats``) live in the ``ario_mlflow`` plugin — this module imports
them so both the demo and the plugin share a single source of truth.
"""

import uuid
from datetime import datetime, timezone

from ario_mlflow.proof import canonical_json, hash_data


def build_decision_record(
    input_data: dict,
    prediction: dict,
    model_name: str,
    model_version: str,
    mlflow_run_id: str,
    artifact_uri: str,
    trace_id: str,
    span_id: str,
    latency_ms: float,
    service_name: str = "verifiable-ai-demo",
) -> dict:
    """Build a canonical decision record dict."""
    return {
        "decision_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "span_id": span_id,
        "service_name": service_name,
        "mlflow_run_id": mlflow_run_id,
        "model_name": model_name,
        "model_version": model_version,
        "artifact_uri": artifact_uri,
        "input_hash": hash_data(canonical_json(input_data)),
        "output_hash": hash_data(canonical_json(prediction)),
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "human_override": False,
    }
