"""Toy credit-decision classifier backing the demo.

The model is deliberately small and synthetic — the demo's point is the
verifiable-provenance pipeline (hash, sign, anchor, verify), not the ML.
The feature set and labels are chosen so that the numbers a visitor sees
on the prediction form read as a plausible credit-scoring scenario rather
than flower measurements.
"""

import logging
import os
import tempfile

import ario_mlflow
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature order is load-bearing: the prediction handlers build the input
# vector by iterating FEATURE_NAMES, and the HTML form inputs share these
# names. Change one, change the others.
FEATURE_NAMES = [
    "annual_income",          # USD
    "credit_utilization",     # 0.0 – 1.0
    "debt_to_income_ratio",   # 0.0 – 1.0
    "months_employed",        # integer, 0 – 240
    "credit_score",           # 300 – 850
]

CLASS_NAMES = ["deny", "approve"]


def _generate_credit_data(n_samples: int = 800, random_state: int = 42):
    """Synthetic credit-application dataset.

    The ground-truth rule is legible: higher credit_score, longer employment,
    lower debt-to-income, and lower credit_utilization all push the decision
    toward 'approve'. Noise is added so the classifier has something
    non-trivial to fit (we expect accuracy in the high 80s / low 90s).
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    income = rng.normal(65_000, 25_000, n).clip(15_000, 250_000)
    utilization = rng.beta(2, 5, n)                           # skewed toward low
    dti = rng.beta(3, 5, n) * 0.7                             # 0 – 0.7
    months = rng.integers(0, 240, n)                          # 0 – 20 years
    score = rng.normal(700, 70, n).clip(350, 830)

    z = (
        (score - 650) / 80.0
        - 2.5 * utilization
        - 2.5 * dti
        + (months / 240.0)
        + (np.log1p(income) - np.log1p(65_000)) * 0.5
    )
    z += rng.normal(0, 0.6, n)
    labels = (z > 0.2).astype(int)

    features = np.column_stack([income, utilization, dti, months, score])
    return features, labels


def train_and_register(tracking_uri: str, model_name: str) -> dict:
    """Train the credit classifier and register it with MLflow (default params)."""
    return train_and_register_with_params(tracking_uri, model_name)


def train_and_register_with_params(
    tracking_uri: str,
    model_name: str,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict:
    """Train the credit classifier with configurable params, anchor data + run, register."""
    mlflow.set_tracking_uri(tracking_uri)

    X, y = _generate_credit_data(n_samples=800, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    # StandardScaler is essential here because features span three orders of
    # magnitude (income ~10^4, utilization ~10^-1). Without scaling the
    # classifier's fit is dominated by income and essentially ignores the
    # ratio features — bad for a demo meant to illustrate sensible decisions.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    # Persist the training split to disk so we can hash it as the dataset link.
    # Using the test split would also be valid; we anchor the train split as the
    # one the model was actually fit on.
    with tempfile.TemporaryDirectory() as data_dir:
        train_path = os.path.join(data_dir, "credit_train.csv")
        header = ",".join(FEATURE_NAMES + ["label"])
        rows = np.column_stack([X_train, y_train])
        np.savetxt(train_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")

        # Link 1: dataset.
        dataset_result = ario_mlflow.anchor_dataset(
            name=f"{model_name}_train_rs{random_state}",
            path=train_path,
        )

        with mlflow.start_run() as run:
            mlflow.log_param("model_type", "LogisticRegression+StandardScaler")
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_training_samples", len(X_train))
            mlflow.log_param("feature_names", ",".join(FEATURE_NAMES))
            mlflow.log_metric("accuracy", accuracy)

            # Tag the run with dataset_tx so anchor() chains to it.
            if dataset_result["tx_id"]:
                mlflow.set_tag("ario.dataset_tx", dataset_result["tx_id"])
            mlflow.set_tag("ario.dataset_hash", dataset_result["dataset_hash"])

            # Log dataset itself as an artifact so verifiers can re-hash it.
            mlflow.log_artifact(train_path, artifact_path="dataset")

            model_info = mlflow.sklearn.log_model(
                pipeline,
                "model",
                input_example=X_train[:1],
            )

            # Link 2: training run. anchor() reads ario.dataset_tx and chains.
            ario_mlflow.anchor(artifact_path="model")

            # Register via ArioMlflowClient so registration is anchored and
            # chained to ario.training_tx automatically.
            from ario_mlflow.client import ArioMlflowClient
            ario_client = ArioMlflowClient(tracking_uri=tracking_uri)
            # Ensure the registered model exists (create_model_version requires it).
            try:
                ario_client.create_registered_model(model_name)
            except Exception:
                pass  # Already exists — that's fine.
            mv = ario_client.create_model_version(
                name=model_name,
                source=model_info.model_uri,
                run_id=run.info.run_id,
            )
            latest_version = mv.version

            logger.info(
                f"Credit model trained: accuracy={accuracy:.4f}, "
                f"run_id={run.info.run_id}, version={latest_version}, "
                f"dataset_tx={dataset_result['tx_id']}"
            )

            return {
                "run_id": run.info.run_id,
                "model_name": model_name,
                "model_version": str(latest_version),
                "artifact_uri": model_info.model_uri,
                "accuracy": accuracy,
                "dataset_tx": dataset_result["tx_id"],
            }


class _IncompatibleSchemaError(Exception):
    """Raised when the registered model expects a different feature shape."""


def _assert_credit_schema(model) -> None:
    """Fail fast if the loaded model doesn't match the 5-feature credit schema.

    Protects against a stale Iris-era model (or any other shape) being
    registered under ``model_name``: the app would otherwise boot fine
    and crash on the first prediction with a shape-mismatch.
    """
    expected = len(FEATURE_NAMES)
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and actual != expected:
        raise _IncompatibleSchemaError(
            f"registered model expects {actual} features; "
            f"credit-scorer needs {expected} ({FEATURE_NAMES})"
        )


def load_model(tracking_uri: str, model_name: str) -> dict:
    """Load the latest model from MLflow. Auto-trains if none found or schema stale."""
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/latest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        _assert_credit_schema(model)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        return {
            "model": model,
            "model_name": model_name,
            "model_version": str(latest.version),
            "run_id": latest.run_id,
            "artifact_uri": f"models:/{model_name}/{latest.version}",
        }
    except Exception as e:
        if isinstance(e, _IncompatibleSchemaError):
            logger.warning(f"Incompatible registered model for {model_name}: {e}. Re-training.")
        else:
            logger.info(f"No model found ({e}), training new model...")
        info = train_and_register(tracking_uri, model_name)
        model = mlflow.sklearn.load_model(model_uri)
        _assert_credit_schema(model)
        return {
            "model": model,
            "model_name": info["model_name"],
            "model_version": info["model_version"],
            "run_id": info["run_id"],
            "artifact_uri": info["artifact_uri"],
        }


def predict(model, features: list[float]) -> dict:
    """Run prediction and return structured result.

    ``features`` is an ordered list matching :data:`FEATURE_NAMES`.
    Expects ``model`` to be the native scikit-learn estimator loaded via
    ``mlflow.sklearn.load_model``, which exposes ``predict`` and
    ``predict_proba`` directly — no pyfunc internal digging required.
    """
    input_array = np.array([features])

    pred = model.predict(input_array)
    class_idx = int(pred[0]) if isinstance(pred[0], (int, np.integer)) else int(np.argmax(pred[0]))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_array)[0]
        probabilities = {
            CLASS_NAMES[i]: round(float(p), 6) for i, p in enumerate(probs)
        }
    else:
        probabilities = {CLASS_NAMES[class_idx]: 1.0}

    return {
        "class": CLASS_NAMES[class_idx],
        "class_index": class_idx,
        "probabilities": probabilities,
        "features_used": FEATURE_NAMES,
    }
