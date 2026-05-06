import asyncio
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.config import get_settings
from app.storage import RecordStore
from app.lifecycle_store import LifecycleStore
from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient
from app.model import load_model, predict, train_and_register_with_params, FEATURE_NAMES
from app.ui import router as ui_router
from app import tamper as tamper_mod
from app.reset import reset_demo_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_training_cache_record(model_info: dict) -> dict:
    """Build the lifecycle_store ``record`` sub-dict for a training event.

    Phase 2.D: derived directly from the plugin's canonical payload
    (``model_info["training_payload"]``) instead of re-querying MLflow
    via the legacy ``build_training_record`` helper. The plugin's
    payload already contains run_id, params, metrics, artifact_checksums,
    source_name, git_commit. We add the demo-specific UI display fields
    (event_id, timestamp, model_name, model_version, artifact_hash) on
    top.
    """
    payload = model_info.get("training_payload") or {}
    artifact_checksums = payload.get("artifact_checksums", {}) or {}
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "training_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": model_info["run_id"],
        "model_name": model_info["model_name"],
        "model_version": model_info["model_version"],
        "params": payload.get("params", {}),
        "metrics": payload.get("metrics", {}),
        "artifact_checksums": artifact_checksums,
        "artifact_hash": hash_data(canonical_json(artifact_checksums)),
        "source_name": payload.get("source_name", ""),
        "git_commit": payload.get("git_commit", ""),
    }


def _build_registration_cache_record(
    model_name: str,
    model_version: str,
    run_id: str,
    artifact_hash: str | None,
    training_tx: str | None,
) -> dict:
    """Build the lifecycle_store ``record`` sub-dict for a registration event."""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "model_registered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "source_run_id": run_id,
        "artifact_hash": artifact_hash,
        "previous_tx": training_tx,
    }


def _startup_anchor_lifecycle(settings, model_info, lifecycle_store):
    """Populate lifecycle_store from a freshly auto-trained model's plugin
    anchor results.

    Per the redesign (2026-04-28): this helper NO LONGER uploads to
    Arweave and NO LONGER builds a legacy local proof. The plugin's
    ``anchor()`` and ``ArioMlflowClient`` already uploaded the canonical
    new-shape envelope inside ``train_and_register_with_params``. This
    helper just writes a thin display cache entry pointing at the real
    TX so the UI can render and the verify path can fetch by
    ``arweave_tx_id``.

    For pre-existing models loaded from an existing ``mlruns/`` (no
    ``training_anchor_result`` in ``model_info``), this is a no-op —
    per the redesign, pre-existing models are not retro-anchored.
    """
    training_anchor = model_info.get("training_anchor_result")
    if not training_anchor:
        logger.info(
            "Pre-existing model loaded; skipping lifecycle_store population "
            "(per redesign — only models auto-trained on this boot are tracked)."
        )
        return

    run_id = model_info["run_id"]
    model_name = model_info["model_name"]
    model_version = model_info["model_version"]

    # Training cache entry — record sub-dict derived from plugin payload,
    # arweave fields stamped from plugin's anchor result.
    if not lifecycle_store.get_by_run_id(run_id):
        record = _build_training_cache_record(model_info)
        envelope = {
            "record": record,
            "arweave_tx_id": training_anchor.get("tx_id"),
            "arweave_url": training_anchor.get("url"),
            "turbo_receipt": training_anchor.get("receipt"),
        }
        lifecycle_store.append(envelope)
        logger.info(
            f"Tracked training {run_id} in lifecycle_store: "
            f"tx={training_anchor.get('tx_id')}"
        )

    # Registration cache entry — plugin's ArioMlflowClient kicked off the
    # registration anchor in a daemon thread; arweave_tx_id starts None
    # and gets hydrated by the bridge task once the daemon settles.
    if not lifecycle_store.get_by_model_version(model_name, model_version):
        registration_record = _build_registration_cache_record(
            model_name=model_name,
            model_version=model_version,
            run_id=run_id,
            artifact_hash=hash_data(canonical_json(
                (model_info.get("training_payload") or {}).get("artifact_checksums", {}) or {}
            )),
            training_tx=training_anchor.get("tx_id"),
        )
        registration_envelope = {
            "record": registration_record,
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }
        lifecycle_store.append(registration_envelope)
        logger.info(
            f"Tracked registration {model_name}/v{model_version} in "
            f"lifecycle_store; awaiting plugin daemon for TX hydration."
        )

        ario_client = model_info.get("ario_client")
        if ario_client is not None:
            threading.Thread(
                target=_hydrate_registration_envelope_from_plugin,
                args=(lifecycle_store, ario_client, model_name, model_version,
                      registration_record["event_id"]),
                daemon=True,
            ).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # OpenTelemetry
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # Core components
    app.state.settings = settings
    app.state.store = RecordStore(settings.records_file)
    app.state.lifecycle_store = LifecycleStore(settings.lifecycle_file)
    app.state.proof_engine = ProofEngine(
        settings.ed25519_private_key_path,
        settings.ed25519_public_key_path,
    )

    # Arweave anchor — initialised BEFORE load_model so the demo's
    # signing key + wallet are threaded through to VerifiedModel and
    # any auto-train fallback path. (Phase 2.B)
    app.state.anchor = ArweaveAnchor(settings.arweave_wallet_path, settings.ario_gateway_host)

    # MLflow model — load_model now returns a VerifiedModel alongside
    # the raw sklearn estimator. The sklearn one is used for the UI's
    # probability display; the VerifiedModel handles inference-time
    # commitment anchoring on every predict() call.
    logger.info("Loading MLflow model...")
    app.state.model_info = load_model(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        proof_engine=app.state.proof_engine,
        arweave=app.state.anchor,
    )
    logger.info(f"Model loaded: {settings.mlflow_model_name}/v{app.state.model_info['model_version']}")

    # AR.IO Verify
    app.state.ario_verify = ArioVerifyClient(settings.ario_verify_url)

    # Populate lifecycle_store from the plugin's anchor results in a
    # background thread (so the registration daemon hydration doesn't
    # block the lifespan). No Arweave I/O happens here — the plugin
    # already uploaded.
    threading.Thread(
        target=_startup_anchor_lifecycle,
        args=(settings, app.state.model_info, app.state.lifecycle_store),
        daemon=True,
    ).start()

    yield

    # Shutdown
    provider.shutdown()


app = FastAPI(title="Verifiable AI Decision Records", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

app.include_router(ui_router)


def _hydrate_record_envelope_from_verified_prediction(store, decision_id: str, verified_result):
    """Background task: wait for VerifiedModel's anchor daemon to settle,
    then hydrate the demo's RecordStore envelope with the real Arweave
    TX. Phase 2.B replaces the legacy ``_anchor_record`` background
    task — VerifiedModel handles the upload itself; this just bridges
    the result back into the demo's UI store. Phase 2.D refactors
    RecordStore into a thin UI cache so this bridge goes away.
    """
    try:
        finished = verified_result.wait_for_anchor(timeout=60.0)
        if not finished:
            logger.warning(
                f"Prediction anchor for {decision_id} did not complete "
                f"in 60s; RecordStore entry stays unhydrated."
            )
            return
        if verified_result.proof_status != "anchored" or not verified_result.tx_id:
            # Could be 'failed' or 'disabled'. UI surfaces "anchoring..."
            # state until something else hydrates this.
            return
        envelope = store.get_by_id(decision_id)
        if envelope is None:
            return
        envelope["arweave_tx_id"] = verified_result.tx_id
        envelope["arweave_url"] = (
            f"https://turbo-gateway.com/{verified_result.tx_id}"
        )
        # Turbo receipt isn't directly accessible from VerifiedPrediction;
        # leave as None. UI doesn't strictly require it.
        store.update(decision_id, envelope)
        logger.info(
            f"Hydrated RecordStore envelope for decision {decision_id}: "
            f"tx={verified_result.tx_id}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Failed to hydrate RecordStore envelope for {decision_id}: {e}"
        )


def _run_prediction(app_state, features: list[float]) -> tuple[dict, object]:
    """Core prediction flow: VerifiedModel anchored predict + legacy
    RecordStore population for UI compatibility.

    Phase 2.B: VerifiedModel.predict() produces the cryptographic proof
    + anchors via plugin daemon thread. The legacy proof + RecordStore
    pattern is retained for UI display compatibility (Phase 2.D
    refactors RecordStore into a thin cache).

    Returns ``(envelope, verified_result)``. ``verified_result`` is a
    VerifiedPrediction the caller can pass to the hydration background
    task so the envelope's TX gets filled in once the daemon settles.
    """
    settings = app_state.settings
    model_info = app_state.model_info
    sklearn_model = model_info["model"]
    verified_model = model_info["verified_model"]

    with tracer.start_as_current_span("predict") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        span_id = format(span.get_span_context().span_id, "016x")

        start = time.time()
        input_data = dict(zip(FEATURE_NAMES, features))

        # UI prediction: use the raw sklearn estimator for class +
        # probability display. VerifiedModel's pyfunc wrapper would
        # only return class predictions; the demo's predict() returns
        # the friendlier {class, class_index, probabilities, features_used}
        # shape the templates render.
        ui_prediction = predict(sklearn_model, features)
        latency_ms = (time.time() - start) * 1000

        # Anchored proof: VerifiedModel.predict signs a pure-commitment
        # envelope, writes ario/predictions/<decision_id>/payload.json
        # on the model's source run, mirrors fields as trace tags, and
        # spawns a daemon thread to upload the envelope to Arweave.
        # OTel context flows through metadata so the signed proof
        # correlates with the demo's existing OpenTelemetry instrumentation.
        verified_result = verified_model.predict(
            input_data,
            metadata={
                "otel_trace_id": trace_id,
                "otel_span_id": span_id,
                "service_name": settings.otel_service_name,
            },
        )

        # RecordStore display cache. Phase 2.D dropped the legacy
        # local proof (proof_engine.create_proof) — the verifiable
        # artifact is the plugin's anchored envelope on Arweave,
        # fetched by /verify/{decision_id}. The cache holds only
        # what the UI needs to display: the prediction, latency,
        # trace correlation, and the eventual TX (hydrated by
        # background task once VerifiedPrediction's daemon settles).
        record = {
            "decision_id": verified_result.decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "span_id": span_id,
            "service_name": settings.otel_service_name,
            "mlflow_run_id": model_info["run_id"],
            "model_name": model_info["model_name"],
            "model_version": model_info["model_version"],
            "artifact_uri": model_info["artifact_uri"],
            "input_hash": hash_data(canonical_json(input_data)),
            "output_hash": hash_data(canonical_json(ui_prediction)),
            "prediction": ui_prediction,
            "latency_ms": round(latency_ms, 2),
            "human_override": False,
        }

        envelope = {
            "record": record,
            # arweave_tx_id starts None and gets hydrated by the
            # background task from verified_result. UI surfaces
            # "anchoring..." until then.
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }

        app_state.store.append(envelope)

        return envelope, verified_result


# --- API Endpoints ---

FEATURE_DEFAULTS: dict[str, float] = {
    "annual_income": 78000,
    "credit_utilization": 0.18,
    "debt_to_income_ratio": 0.22,
    "months_employed": 72,
    "credit_score": 745,
}


@app.post("/predict")
def api_predict(request: Request, body: dict, background_tasks: BackgroundTasks):
    features = [
        float(body.get(name, FEATURE_DEFAULTS[name])) for name in FEATURE_NAMES
    ]
    envelope, verified_result = _run_prediction(request.app.state, features)
    decision_id = envelope["record"]["decision_id"]
    # VerifiedModel handles the Arweave upload via its own daemon
    # thread. We just bridge the result back into the demo's
    # RecordStore so the UI shows the real TX once the daemon settles.
    background_tasks.add_task(
        _hydrate_record_envelope_from_verified_prediction,
        request.app.state.store,
        decision_id,
        verified_result,
    )
    return envelope


@app.post("/predict-form")
def form_predict(
    request: Request,
    background_tasks: BackgroundTasks,
    annual_income: float = Form(FEATURE_DEFAULTS["annual_income"]),
    credit_utilization: float = Form(FEATURE_DEFAULTS["credit_utilization"]),
    debt_to_income_ratio: float = Form(FEATURE_DEFAULTS["debt_to_income_ratio"]),
    months_employed: float = Form(FEATURE_DEFAULTS["months_employed"]),
    credit_score: float = Form(FEATURE_DEFAULTS["credit_score"]),
):
    form_values = {
        "annual_income": annual_income,
        "credit_utilization": credit_utilization,
        "debt_to_income_ratio": debt_to_income_ratio,
        "months_employed": months_employed,
        "credit_score": credit_score,
    }
    features = [float(form_values[name]) for name in FEATURE_NAMES]
    envelope, verified_result = _run_prediction(request.app.state, features)
    decision_id = envelope["record"]["decision_id"]
    # VerifiedModel handles the Arweave upload via its own daemon
    # thread. Bridge the result back into the demo's RecordStore.
    background_tasks.add_task(
        _hydrate_record_envelope_from_verified_prediction,
        request.app.state.store,
        decision_id,
        verified_result,
    )
    return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)


@app.post("/api/train")
def api_train(request: Request, body: dict, background_tasks: BackgroundTasks):
    """Train a new model version. Phase 2.A: anchoring is handled by
    the plugin's headline API (anchor() + ArioMlflowClient) — no longer
    by the demo's hand-rolled proof + background-upload pipeline.

    The lifecycle_store is still populated for UI display compatibility;
    Phase 2.D refactors it into a UI-only cache populated from MLflow tags.
    """
    import random
    settings = request.app.state.settings
    max_iter = int(body.get("max_iter", 200))
    random_state = int(body.get("random_state", random.randint(1, 10000)))

    # Train, anchor, and register via the plugin. Anchoring of the
    # training event happens synchronously inside the run; registration
    # anchoring spawns a daemon thread (visible via
    # ArioMlflowClient.anchor_status / wait_for_anchor).
    info = train_and_register_with_params(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        proof_engine=request.app.state.proof_engine,
        arweave=request.app.state.anchor,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Populate lifecycle_store cache entries from the plugin's anchor
    # results. No legacy proof, no duplicate Arweave upload — the plugin
    # already uploaded the real training+registration proofs.
    plugin_anchor = info.get("training_anchor_result") or {}
    training_record = _build_training_cache_record(info)
    training_envelope = {
        "record": training_record,
        "arweave_tx_id": plugin_anchor.get("tx_id"),
        "arweave_url": plugin_anchor.get("url"),
        "turbo_receipt": plugin_anchor.get("receipt"),
    }
    request.app.state.lifecycle_store.append(training_envelope)

    # Registration cache entry. The plugin's ArioMlflowClient kicked off
    # the real registration anchor in a daemon thread; arweave_tx_id
    # starts None and the bridge task hydrates it when the daemon
    # settles.
    registration_record = _build_registration_cache_record(
        model_name=info["model_name"],
        model_version=info["model_version"],
        run_id=info["run_id"],
        artifact_hash=training_record.get("artifact_hash"),
        training_tx=plugin_anchor.get("tx_id"),
    )
    registration_envelope = {
        "record": registration_record,
        "arweave_tx_id": None,
        "arweave_url": None,
        "turbo_receipt": None,
    }
    request.app.state.lifecycle_store.append(registration_envelope)

    # Background task: wait for ArioMlflowClient's registration anchor
    # daemon to complete, then read the resulting tags into the
    # lifecycle_store entry so the UI converges to the real registration_tx.
    ario_client = info.get("ario_client")
    if ario_client is not None:
        background_tasks.add_task(
            _hydrate_registration_envelope_from_plugin,
            request.app.state.lifecycle_store,
            ario_client,
            info["model_name"],
            info["model_version"],
            registration_record["event_id"],
        )

    # Auto-switch to the newly trained model
    new_model_info = load_model(settings.mlflow_tracking_uri, settings.mlflow_model_name)
    request.app.state.model_info = new_model_info
    logger.info(f"Switched active model to v{info['model_version']}")

    return {
        "run_id": info["run_id"],
        "model_name": info["model_name"],
        "model_version": info["model_version"],
        "accuracy": info["accuracy"],
        "training_event_id": training_record["event_id"],
        "registration_event_id": registration_record["event_id"],
        # Surface the plugin's training TX so callers can verify directly.
        "training_tx": plugin_anchor.get("tx_id"),
        "training_payload_hash": info.get("training_payload_hash"),
    }


def _hydrate_registration_envelope_from_plugin(
    lifecycle_store, ario_client, model_name: str, model_version: str, event_id: str,
):
    """Wait for ArioMlflowClient's registration anchor daemon to settle,
    then update the demo's lifecycle_store entry with the real Arweave
    TX. Bridges Phase 2.A's plugin-anchored registration with the
    legacy old-shape lifecycle_store the UI still reads from. Phase 2.D
    deletes both this helper and the legacy lifecycle_store shape.
    """
    try:
        # Wait up to 60s for the daemon. If it fails or times out, the
        # entry stays at arweave_tx_id=None and the UI shows
        # "anchoring..." until something else hydrates it.
        finished = ario_client.wait_for_anchor("registration", model_name, model_version, timeout=60.0)
        if not finished:
            logger.warning(
                f"Registration anchor for {model_name}/v{model_version} did not "
                f"complete in 60s; lifecycle_store entry stays unhydrated."
            )
            return
        # Read back the model version's tags to find the registration_tx.
        mv = ario_client.get_model_version(model_name, model_version)
        tags = (mv.tags or {})
        tx_id = tags.get("ario.registration_tx")
        arweave_url = tags.get("ario.arweave_url")
        if not tx_id:
            return
        envelope = lifecycle_store.get_by_event_id(event_id)
        if envelope is None:
            return
        envelope["arweave_tx_id"] = tx_id
        envelope["arweave_url"] = arweave_url
        # Turbo receipt isn't accessible from the model version tags
        # directly; leave as None. UI doesn't strictly require it.
        lifecycle_store.update(event_id, envelope)
        logger.info(
            f"Hydrated lifecycle_store registration envelope for "
            f"{model_name}/v{model_version}: tx={tx_id}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Failed to hydrate registration envelope for {model_name}/v{model_version}: {e}"
        )


@app.post("/api/activate/{model_name}/{version}")
def activate_model(request: Request, model_name: str, version: str):
    """Switch the active model to a specific version."""
    import mlflow
    settings = request.app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        return JSONResponse({"error": f"Could not load model: {e}"}, status_code=404)

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    mv = next((v for v in versions if str(v.version) == str(version)), None)

    request.app.state.model_info = {
        "model": model,
        "model_name": model_name,
        "model_version": str(version),
        "run_id": mv.run_id if mv else "unknown",
        "artifact_uri": model_uri,
    }
    logger.info(f"Activated model {model_name}/v{version}")

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse("/", status_code=303)

    return {"activated": True, "model_name": model_name, "model_version": str(version)}


@app.get("/decisions")
def list_decisions(request: Request):
    return request.app.state.store.list_all()


@app.get("/decisions/{decision_id}")
def get_decision(request: Request, decision_id: str):
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)
    # Attach a live turbo_status so the polling UI can update the badge
    # as the proof progresses from Uploading → Confirmed → Permanent without
    # requiring a page reload.
    envelope = dict(envelope)
    if envelope.get("arweave_tx_id"):
        envelope["turbo_status"] = request.app.state.anchor.check_status(envelope["arweave_tx_id"])
    return envelope


@app.post("/verify/{decision_id}")
def verify_decision(request: Request, decision_id: str):
    """Verify a prediction via the plugin's verify_proof_by_tx.

    Looks up the prediction's Arweave TX from the demo's RecordStore
    entry (hydrated by Phase 2.B's background task with the plugin's
    actual prediction TX) and calls ``verify_proof_by_tx`` which fetches
    the pure-commitment envelope from Arweave and runs the four-check
    verification flow in one call.

    Returns the four-check result alongside the legacy-shape fields so
    callers reading either format keep working. The browser path still
    redirects to the decision detail page.
    """
    import mlflow
    from ario_mlflow.verify import verify_proof_by_tx

    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)

    tx_id = envelope.get("arweave_tx_id")
    plugin_result = None
    external_result = None
    ario_result = None

    if tx_id:
        settings = request.app.state.settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow_client = mlflow.tracking.MlflowClient()

        plugin_result = verify_proof_by_tx(
            tx_id,
            anchor=request.app.state.anchor,
            proof_engine=request.app.state.proof_engine,
            mlflow_client=mlflow_client,
            ario_client=request.app.state.ario_verify,
        )

        if plugin_result.get("proof_found"):
            # Legacy-shape mappings for back-compat with existing API
            # consumers. The four-check structure is the canonical
            # result; these are derived projections.
            anchored = plugin_result.get("anchored_bytes", {}) or {}
            ario_block = plugin_result.get("ario_attestation", {}) or {}
            external_result = {
                "arweave_data_found": True,
                "arweave_record_hash": anchored.get("computed_hash"),
                "arweave_matches_original": bool(anchored.get("ok") is True),
            }
            if ario_block.get("ok") is not None:
                ario_result = {
                    "attestation_level": ario_block.get("attestation_level"),
                    "attested_by": ario_block.get("attested_by"),
                    "attested_at": ario_block.get("attested_at"),
                    "report_url": ario_block.get("report_url"),
                }
        else:
            external_result = {"arweave_data_found": False}

    result = {
        "decision_id": decision_id,
        # Legacy projection fields for back-compat
        "external_verification": external_result,
        "ario_verification": ario_result,
        # New four-check structure (Phase 3 UI consumes this) — now also
        # carries proof_found at the top level.
        "plugin_full_verify": plugin_result,
    }

    # If called from browser, redirect to detail page with verification results
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(f"/ui/decisions/{decision_id}?verify=true", status_code=303)

    return result


@app.get("/lifecycle")
def list_lifecycle(request: Request):
    return request.app.state.lifecycle_store.list_all()


@app.get("/lifecycle/{event_id}")
def get_lifecycle_event(request: Request, event_id: str):
    envelope = request.app.state.lifecycle_store.get_by_event_id(event_id)
    if not envelope:
        return JSONResponse({"error": "Lifecycle event not found"}, status_code=404)
    envelope = dict(envelope)
    if envelope.get("arweave_tx_id"):
        envelope["turbo_status"] = request.app.state.anchor.check_status(envelope["arweave_tx_id"])
    return envelope


# Phase 2.D removed:
# - ``compute_chain_integrity`` + ``/api/chain-integrity`` endpoint
# - ``/tamper/{decision_id}`` endpoint
# Phase 3 removed:
# - ``/api/export/{decision_id}`` endpoint — replaced by direct
#   ar.io gateway link in the ``View Proof ↗`` button on
#   templates/decision_detail.html. Arweave is the source of truth
#   for the proof; a server-rendered JSON export is no longer needed.
# Phase 3 reintroduces tamper UX with two buttons per page paired to
# the verification rows.


# Tamper routes mutate live MLflow state and are intended for the
# public demo only. They register only when ``demo_mode`` is True
# (the default; override with VAIDR_DEMO_MODE=false in production).
#
# Module-level lock so two concurrent /demo/reset requests don't race
# on the wipe + re-init. Other endpoints intentionally don't check this
# lock — accept that requests during the ~5-10s reset window may briefly
# see inconsistent state.
_RESET_LOCK = asyncio.Lock()

if get_settings().demo_mode:

    @app.post("/demo/reset")
    async def demo_reset_route(request: Request):
        """Wipe all seeded demo data and re-train a fresh v1.

        Sales / pre-sales workflow: pre-seed the demo before a customer
        call, then wipe afterward so the next call starts clean. Anchored
        proofs already on Arweave are not affected (they remain permanent
        on the network).
        """
        async with _RESET_LOCK:
            try:
                # reset_demo_state is synchronous and CPU/IO-bound (model
                # auto-train + filesystem wipes). Run in a thread so the
                # event loop doesn't stall while v1 trains.
                new_version = await asyncio.to_thread(reset_demo_state, request.app)
            except Exception as e:  # noqa: BLE001
                logger.exception("Demo reset failed")
                return JSONResponse({"error": str(e)}, status_code=500)
            return {"reset": True, "new_version": new_version}

    @app.post("/tamper/saved/{event_type}/{event_id}")
    def tamper_saved_route(request: Request, event_type: str, event_id: str,
                           background_tasks: BackgroundTasks):
        if event_type not in ("decision", "training", "registration"):
            return JSONResponse({"error": "unknown event_type"}, status_code=400)
        settings = request.app.state.settings
        try:
            tamper_mod.tamper_saved(
                event_type, event_id,
                lifecycle_store=request.app.state.lifecycle_store,
                record_store=request.app.state.store,
                tracking_uri=settings.mlflow_tracking_uri,
            )
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        background_tasks.add_task(_scheduled_revert, request.app, event_type, event_id)
        return {"tampered": True, "kind": "saved", "event_id": event_id,
                "ttl_seconds": tamper_mod.TAMPER_TTL_SECONDS}

    @app.post("/tamper/live/{event_type}/{event_id}")
    def tamper_live_route(request: Request, event_type: str, event_id: str,
                          background_tasks: BackgroundTasks):
        if event_type not in ("decision", "training", "registration"):
            return JSONResponse({"error": "unknown event_type"}, status_code=400)
        settings = request.app.state.settings
        try:
            tamper_mod.tamper_live(
                event_type, event_id,
                lifecycle_store=request.app.state.lifecycle_store,
                record_store=request.app.state.store,
                tracking_uri=settings.mlflow_tracking_uri,
            )
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        background_tasks.add_task(_scheduled_revert, request.app, event_type, event_id)
        return {"tampered": True, "kind": "live", "event_id": event_id,
                "ttl_seconds": tamper_mod.TAMPER_TTL_SECONDS}

    @app.post("/tamper/reset/{event_type}/{event_id}")
    def tamper_reset_route(request: Request, event_type: str, event_id: str):
        if event_type not in ("decision", "training", "registration"):
            return JSONResponse({"error": "unknown event_type"}, status_code=400)
        settings = request.app.state.settings
        reverted = tamper_mod.reset(
            event_type, event_id,
            lifecycle_store=request.app.state.lifecycle_store,
            record_store=request.app.state.store,
            tracking_uri=settings.mlflow_tracking_uri,
        )
        return {"reset": True, "reverted_count": reverted, "event_id": event_id}


def _scheduled_revert(app, event_type, event_id):
    """Wrapper for BackgroundTasks — runs reset after the TTL sleep."""
    import time as _time
    _time.sleep(tamper_mod.TAMPER_TTL_SECONDS)
    settings = app.state.settings
    try:
        tamper_mod.reset(
            event_type, event_id,
            lifecycle_store=app.state.lifecycle_store,
            record_store=app.state.store,
            tracking_uri=settings.mlflow_tracking_uri,
        )
    except Exception as e:
        logger.warning(f"Auto-revert raised: {e}")
