"""Toy credit-decision classifier backing the demo.

The model is deliberately small and synthetic — the demo's point is the
verifiable-provenance pipeline (hash, sign, anchor, verify), not the ML.
The feature set and labels are chosen so that the numbers a visitor sees
on the prediction form read as a plausible credit-scoring scenario rather
than flower measurements.
"""

import logging
import os

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ario_mlflow
from ario_mlflow import ArioMlflowClient
from ario_mlflow.proof import ProofEngine
from ario_mlflow.arweave import ArweaveAnchor

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


def train_and_register(
    tracking_uri: str,
    model_name: str,
    *,
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
) -> dict:
    """Train the credit classifier and register it with MLflow (default params)."""
    return train_and_register_with_params(
        tracking_uri, model_name, proof_engine=proof_engine, arweave=arweave,
    )


def train_and_register_with_params(
    tracking_uri: str,
    model_name: str,
    *,
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict:
    """Train the credit classifier and register it via the plugin's headline API.

    Phase 2.A migration: instead of using ``mlflow.sklearn.log_model(...,
    registered_model_name=...)`` (which auto-registers but doesn't anchor)
    + manual proof creation, this function now:

    1. Logs the model under the run *without* auto-registration, so we
       can use ``ArioMlflowClient.create_model_version`` for the
       registration step.
    2. Calls ``ario_mlflow.anchor()`` inside the run — produces the
       pure-commitment training proof, writes ``ario/payload.json``,
       sets ``ario.training_tx`` / ``ario.payload_hash`` on the run.
    3. Calls ``ArioMlflowClient.create_model_version()`` after the run
       closes — auto-anchors the registration proof in a daemon thread,
       chained to the training's ``ario.training_tx``.

    Args:
        tracking_uri: MLflow tracking URI.
        model_name: Registered model name to use.
        proof_engine: Optional override. When ``None``, ``anchor()`` and
            ``ArioMlflowClient`` create their own.
        arweave: Optional override. Same semantics.
        max_iter: Logistic regression max iterations.
        random_state: Seed for data + classifier.

    Returns:
        Dict with ``run_id``, ``model_name``, ``model_version``,
        ``artifact_uri``, ``accuracy``, plus the new
        ``training_envelope``, ``training_payload_hash``,
        ``training_anchor_result`` from the plugin's anchor() so the
        caller can hydrate UI state.
    """
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

    # Wrap the training data as a labelled MLflow Dataset so the run
    # carries a structured dataset reference. The plugin's anchor() reads
    # run.inputs.dataset_inputs and folds the dataset's digest, source,
    # name, and JCS-canonicalized schema fingerprint into the signed
    # canonical payload — closing the input-side honesty gap.
    train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    train_df["approved"] = y_train
    training_dataset = mlflow.data.from_pandas(
        train_df,
        # Plain identifier (no scheme) so MLflow's source registry
        # resolves it as a LocalArtifactDatasetSource. The data is
        # synthetic and generated in-process; the source string is
        # informational, not a real fetch URI.
        source="credit_scorer_demo_synthetic.csv",
        name="credit_scorer_demo_training_data",
    )

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LogisticRegression+StandardScaler")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_training_samples", len(X_train))
        mlflow.log_param("feature_names", ",".join(FEATURE_NAMES))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_input(training_dataset, context="training")

        # Log the model WITHOUT auto-registration. We register
        # explicitly via ArioMlflowClient below so the registration
        # event gets its own anchored proof chained to the training.
        model_info = mlflow.sklearn.log_model(
            pipeline,
            "model",
            input_example=X_train[:1],
        )

        # Anchor the training run via the plugin's headline API.
        # Writes ario.training_tx, ario.payload_hash, ario.artifact_hash
        # tags + the ario/payload.json artifact + uploads the
        # pure-commitment envelope to Arweave.
        training_anchor = ario_mlflow.anchor(
            proof_engine=proof_engine,
            arweave=arweave,
            metadata={"service_name": "verifiable-ai-demo"},
        )

    # Register the model AFTER the run closes via ArioMlflowClient. This
    # spawns a daemon thread that anchors the registration proof,
    # chained back to ario.training_tx that anchor() just set on the run.
    #
    # MLflow's create_model_version requires the registered model to
    # already exist (the legacy log_model(registered_model_name=...)
    # auto-created it; we don't pass that any more). Create it
    # idempotently here for first-training cases.
    ario_client = ArioMlflowClient(
        tracking_uri,
        proof_engine=proof_engine,
        anchor=arweave,
    )
    try:
        ario_client.get_registered_model(model_name)
    except Exception:  # noqa: BLE001
        # Most common: RestException / MlflowException "not found"
        # for first training. Some other failure modes (e.g., backend
        # transiently unavailable) would be retried by create_model_version
        # itself. Suppress here; only handle the not-found case.
        try:
            ario_client.create_registered_model(model_name)
            logger.info(f"Created new registered model {model_name!r}")
        except Exception as e:  # noqa: BLE001
            # If the create races with another caller (e.g. concurrent
            # first-training requests), this can raise "already exists" —
            # also fine; let create_model_version proceed.
            logger.debug(
                f"create_registered_model({model_name!r}) raised "
                f"(probably already exists): {e}"
            )

    mv = ario_client.create_model_version(
        name=model_name,
        source=model_info.model_uri,
        run_id=run.info.run_id,
    )

    logger.info(
        f"Credit model trained: accuracy={accuracy:.4f}, "
        f"run_id={run.info.run_id}, version={mv.version}, "
        f"training_tx={training_anchor.get('anchor_result', {}).get('tx_id') if training_anchor.get('anchor_result') else None}"
    )

    return {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "model_version": str(mv.version),
        "artifact_uri": model_info.model_uri,
        "accuracy": accuracy,
        # Phase 2.A: surface the plugin's anchor() output so /api/train
        # can populate lifecycle_store entries from the new flow.
        "training_envelope": training_anchor["envelope"],
        "training_payload": training_anchor["payload"],
        "training_payload_hash": training_anchor["payload_hash"],
        "training_anchor_result": training_anchor.get("anchor_result"),
        "ario_client": ario_client,
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


def load_model(
    tracking_uri: str,
    model_name: str,
    *,
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
) -> dict:
    """Load the latest model from MLflow + wrap it in a VerifiedModel.

    Returns a dict with both the raw sklearn estimator (used for the
    demo's UI predict that needs ``predict_proba`` for class
    probabilities) AND a ``VerifiedModel`` instance (used for the
    cryptographic prediction proof + Arweave anchoring on every
    ``predict()`` call).

    Auto-trains a new model if no version is registered yet or the
    registered version's schema doesn't match the credit-scorer.
    """
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/latest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        _assert_credit_schema(model)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        version_uri = f"models:/{model_name}/{latest.version}"
        verified_model = _build_verified_model(version_uri, proof_engine, arweave)
        return {
            "model": model,
            "verified_model": verified_model,
            "model_name": model_name,
            "model_version": str(latest.version),
            "run_id": latest.run_id,
            "artifact_uri": version_uri,
        }
    except Exception as e:
        if isinstance(e, _IncompatibleSchemaError):
            logger.warning(f"Incompatible registered model for {model_name}: {e}. Re-training.")
        else:
            logger.info(f"No model found ({e}), training new model...")
        info = train_and_register(tracking_uri, model_name, proof_engine=proof_engine, arweave=arweave)
        model = mlflow.sklearn.load_model(model_uri)
        _assert_credit_schema(model)
        version_uri = f"models:/{info['model_name']}/{info['model_version']}"
        verified_model = _build_verified_model(version_uri, proof_engine, arweave)
        return {
            "model": model,
            "verified_model": verified_model,
            "model_name": info["model_name"],
            "model_version": info["model_version"],
            "run_id": info["run_id"],
            "artifact_uri": version_uri,
            # Surface the plugin's anchor results so the lifespan startup
            # flow can populate lifecycle_store with the real new-shape
            # TX without re-anchoring (which would generate legacy-shape
            # duplicates on Arweave).
            "training_anchor_result": info.get("training_anchor_result"),
            "training_envelope": info.get("training_envelope"),
            "training_payload": info.get("training_payload"),
            "ario_client": info.get("ario_client"),
        }


def _build_verified_model(
    model_uri: str,
    proof_engine: ProofEngine | None,
    arweave: ArweaveAnchor | None,
):
    """Construct a VerifiedModel wrapping the given URI, using the
    demo's app-state proof_engine + arweave so the anchored predictions
    sign with the same key + upload via the same wallet as the rest of
    the demo's plugin flow."""
    return ario_mlflow.VerifiedModel(
        model_uri,
        proof_engine=proof_engine,
        anchor=arweave,
    )


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
