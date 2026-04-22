"""Minimal end-to-end example: train a model, anchor a proof, verify it.

Run:
    python examples/sklearn-quickstart/train_and_anchor.py

The script trains a toy classifier, logs it to a local MLflow store at
./mlruns-quickstart, calls ``ario_mlflow.anchor()`` to create a signed proof,
and prints the CLI command you can run to verify the proof after the fact.

No Arweave wallet is required — the plugin will auto-generate one at
``~/.ario-mlflow/wallet.json`` and reuse it on subsequent runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ario_mlflow

TRACKING_URI = Path(__file__).parent / "mlruns-quickstart"


def main() -> int:
    mlflow.set_tracking_uri(f"file://{TRACKING_URI.resolve()}")
    mlflow.set_experiment("ario-mlflow-quickstart")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200).fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        mlflow.log_params({"max_iter": 200, "random_state": 42})
        mlflow.log_metric("accuracy", round(acc, 6))
        mlflow.sklearn.log_model(model, "model")

        result = ario_mlflow.anchor()

    print()
    print("Run ID:              ", run.info.run_id)
    print("Accuracy:            ", round(acc, 4))
    print("Wallet mode:         ", result["tags"].get("ario.wallet_mode", "unknown"))
    print("Verify status:       ", result["tags"]["ario.verify_status"])
    print("Artifact status:     ", result["artifact_status"])
    print("Artifact hash:       ", result["tags"].get("ario.artifact_hash", "n/a"))
    if "ario.training_tx" in result["tags"]:
        print("Arweave TX:          ", result["tags"]["ario.training_tx"])
        print("Arweave URL:         ", result["tags"]["ario.arweave_url"])

    print()
    print("Verify this proof later with:")
    print(f"  MLFLOW_TRACKING_URI={TRACKING_URI.resolve()} \\")
    print(f"  ario-mlflow verify run {run.info.run_id}")
    print()
    print("Open the MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
