import os
import logging

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def train_and_register(tracking_uri: str, model_name: str) -> dict:
    """Train an iris classifier and register it with MLflow (default params)."""
    return train_and_register_with_params(tracking_uri, model_name)


def train_and_register_with_params(
    tracking_uri: str,
    model_name: str,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict:
    """Train an iris classifier with configurable params and register it with MLflow."""
    mlflow.set_tracking_uri(os.path.abspath(tracking_uri))

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state,
    )

    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name,
            input_example=X_train[:1],
        )

        # Get the actual registered version number
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(int(v.version) for v in versions)

        logger.info(f"Model trained: accuracy={accuracy:.4f}, run_id={run.info.run_id}, version={latest_version}")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "model_version": str(latest_version),
            "artifact_uri": model_info.model_uri,
            "accuracy": accuracy,
        }


def load_model(tracking_uri: str, model_name: str) -> dict:
    """Load the latest model from MLflow. Auto-trains if none found."""
    mlflow.set_tracking_uri(os.path.abspath(tracking_uri))

    model_uri = f"models:/{model_name}/latest"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        # Get model version info
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
        logger.info(f"No model found ({e}), training new model...")
        info = train_and_register(tracking_uri, model_name)
        model = mlflow.pyfunc.load_model(model_uri)
        return {
            "model": model,
            "model_name": info["model_name"],
            "model_version": info["model_version"],
            "run_id": info["run_id"],
            "artifact_uri": info["artifact_uri"],
        }


def predict(model, features: list[float]) -> dict:
    """Run prediction and return structured result."""
    input_array = np.array([features])

    # Get the underlying sklearn model for probability access
    sklearn_model = None
    try:
        sklearn_model = model._model_impl.sklearn_model
    except AttributeError:
        pass

    # Prediction via pyfunc
    pred = model.predict(input_array)
    class_idx = int(pred[0]) if isinstance(pred[0], (int, np.integer)) else int(np.argmax(pred[0]))

    # Probabilities
    probabilities = {}
    if sklearn_model and hasattr(sklearn_model, "predict_proba"):
        probs = sklearn_model.predict_proba(input_array)[0]
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
