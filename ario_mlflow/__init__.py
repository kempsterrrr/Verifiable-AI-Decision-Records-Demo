"""ar.io MLflow plugin — verifiable provenance for the ML lifecycle.

Public API:

- :func:`anchor` — call inside ``mlflow.start_run()`` to anchor a
  training-complete commitment to Arweave.
- :class:`VerifiedModel` — wraps an MLflow model with load-time
  integrity check + per-prediction commitments.
- :class:`ArioMlflowClient` — drop-in ``MlflowClient`` subclass that
  auto-anchors registration and promotion events.
- :class:`IntegrityError` — raised by ``VerifiedModel`` when artifact
  integrity fails.
- :func:`verify_signature`, :func:`verify_anchored_bytes`,
  :func:`verify_source_of_truth`, :func:`verify_ario_attestation`,
  :func:`full_verify` — the four-check verification helpers. Re-exported
  here so consumers can import them without spelunking
  ``ario_mlflow.verify``.
- :class:`ArioVerifyClient` — ar.io Verify REST client.
"""


def __getattr__(name):
    if name == "anchor":
        from ario_mlflow.anchoring import anchor
        return anchor
    if name == "VerifiedModel":
        from ario_mlflow.model import VerifiedModel
        return VerifiedModel
    if name == "IntegrityError":
        from ario_mlflow.model import IntegrityError
        return IntegrityError
    if name == "ArioMlflowClient":
        from ario_mlflow.client import ArioMlflowClient
        return ArioMlflowClient
    if name in (
        "verify_signature",
        "verify_anchored_bytes",
        "verify_source_of_truth",
        "verify_ario_attestation",
        "full_verify",
        "ArioVerifyClient",
    ):
        from ario_mlflow import verify as _verify
        return getattr(_verify, name)
    raise AttributeError(f"module 'ario_mlflow' has no attribute {name!r}")


__all__ = [
    "anchor",
    "VerifiedModel",
    "IntegrityError",
    "ArioMlflowClient",
    "verify_signature",
    "verify_anchored_bytes",
    "verify_source_of_truth",
    "verify_ario_attestation",
    "full_verify",
    "ArioVerifyClient",
]
