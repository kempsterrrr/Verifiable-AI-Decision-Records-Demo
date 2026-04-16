"""MLflow RunContextProvider — auto-injects ar.io metadata tags on every run."""

from mlflow.tracking.context.abstract_context import RunContextProvider


class ArioContextProvider(RunContextProvider):
    """Auto-injects ar.io tags on every MLflow run via the entry point."""

    def in_context(self) -> bool:
        return True

    def tags(self) -> dict[str, str]:
        return {
            "ario.enabled": "true",
            "ario.version": "0.1.0",
        }
