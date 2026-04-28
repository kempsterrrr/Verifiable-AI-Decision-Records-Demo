"""ar.io MLflow plugin — verifiable provenance for the ML lifecycle."""


def __getattr__(name):
    if name == "anchor":
        from ario_mlflow.anchoring import anchor
        return anchor
    if name == "anchor_dataset":
        from ario_mlflow.anchoring import anchor_dataset
        return anchor_dataset
    if name == "VerifiedModel":
        from ario_mlflow.model import VerifiedModel
        return VerifiedModel
    if name == "IntegrityError":
        from ario_mlflow.model import IntegrityError
        return IntegrityError
    if name == "ArioMlflowClient":
        from ario_mlflow.client import ArioMlflowClient
        return ArioMlflowClient
    if name == "verify_prediction":
        from ario_mlflow.decision_verify import verify_prediction
        return verify_prediction
    if name == "PredictionVerificationResult":
        from ario_mlflow.decision_verify import PredictionVerificationResult
        return PredictionVerificationResult
    if name == "verify_model_lifecycle":
        from ario_mlflow.decision_verify import verify_model_lifecycle
        return verify_model_lifecycle
    if name == "verify_envelope":
        from ario_mlflow.verify import verify_envelope
        return verify_envelope
    if name == "verify_run_artifact_integrity":
        from ario_mlflow.anchoring import verify_run_artifact_integrity
        return verify_run_artifact_integrity
    raise AttributeError(f"module 'ario_mlflow' has no attribute {name!r}")


__all__ = [
    "anchor",
    "anchor_dataset",
    "VerifiedModel",
    "IntegrityError",
    "ArioMlflowClient",
    "verify_prediction",
    "PredictionVerificationResult",
    "verify_envelope",
    "verify_run_artifact_integrity",
    "verify_model_lifecycle",
]
