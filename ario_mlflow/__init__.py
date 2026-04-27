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
    raise AttributeError(f"module 'ario_mlflow' has no attribute {name!r}")


__all__ = ["anchor", "anchor_dataset", "VerifiedModel", "IntegrityError", "ArioMlflowClient"]
