import logging
import os
import threading
import time
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
from app.decision_record import build_decision_record
from app.lifecycle import build_training_record, build_registration_record
from app.model import load_model, predict, train_and_register_with_params, FEATURE_NAMES
from app.ui import router as ui_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _startup_anchor_lifecycle(settings, model_info, proof_engine, lifecycle_store, anchor):
    """Populate lifecycle_store from a freshly auto-trained model's plugin
    anchor results.

    Per the redesign (2026-04-28): this function NO LONGER uploads to
    Arweave. The plugin's ``anchor()`` and ``ArioMlflowClient`` already
    uploaded the canonical new-shape envelope inside
    ``train_and_register_with_params``. This helper just stamps the
    lifecycle_store with the resulting TX so the UI's verify path can
    fetch the on-chain envelope by ``arweave_tx_id``.

    For pre-existing models loaded from an existing ``mlruns/`` (no
    ``training_anchor_result`` in ``model_info``), this is a no-op —
    per the redesign, pre-existing models are not retro-anchored.

    The legacy ``_anchor_lifecycle_record`` upload path was deleted in
    Phase 2.C because it was generating LEGACY-shape duplicates on
    Arweave alongside the plugin's NEW-shape proofs, breaking
    verification (verify path fetches by ``arweave_tx_id`` which used
    to point at the legacy upload).
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

    # Training entry — stamps the plugin's NEW-shape Arweave TX. We
    # still build the legacy ``record`` shape for UI display fields
    # (event_id, run_id, params, metrics, etc.); the local
    # ``record_hash`` / ``signature`` fields are kept for the demo's
    # local chain-integrity panel but are NOT what gets verified
    # against Arweave (verify path fetches the new-shape envelope from
    # Arweave directly).
    if not lifecycle_store.get_by_run_id(run_id):
        record = build_training_record(
            settings.mlflow_tracking_uri, run_id, model_name, model_version,
        )
        last = lifecycle_store.list_all()
        previous_hash = last[-1]["record_hash"] if last else "GENESIS"
        proof = proof_engine.create_proof(record, previous_hash)
        envelope = {
            **proof,
            "arweave_tx_id": training_anchor.get("tx_id"),
            "arweave_url": training_anchor.get("url"),
            "turbo_receipt": training_anchor.get("receipt"),
        }
        lifecycle_store.append(envelope)
        logger.info(
            f"Tracked training {run_id} in lifecycle_store: "
            f"tx={training_anchor.get('tx_id')}"
        )

    # Registration entry — the plugin's ArioMlflowClient kicked off the
    # registration anchor in a daemon thread; we don't have its TX yet.
    # Schedule the same bridge task /api/train uses to hydrate the entry
    # once the daemon settles.
    if not lifecycle_store.get_by_model_version(model_name, model_version):
        registration_record = build_registration_record(
            settings.mlflow_tracking_uri, model_name, model_version,
            training_anchor.get("tx_id"),
        )
        last = lifecycle_store.list_all()
        previous_hash = last[-1]["record_hash"] if last else "GENESIS"
        registration_proof = proof_engine.create_proof(registration_record, previous_hash)
        registration_envelope = {
            **registration_proof,
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

    # Anchor training and registration in background thread
    threading.Thread(
        target=_startup_anchor_lifecycle,
        args=(settings, app.state.model_info, app.state.proof_engine,
              app.state.lifecycle_store, app.state.anchor),
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

        # Legacy RecordStore population for UI compatibility. Use
        # VerifiedPrediction's decision_id so UI's verify-this-decision
        # flow can correlate to the real Arweave TX once the daemon
        # completes. Phase 2.D refactors this away.
        record = build_decision_record(
            input_data=input_data,
            prediction=ui_prediction,
            model_name=model_info["model_name"],
            model_version=model_info["model_version"],
            mlflow_run_id=model_info["run_id"],
            artifact_uri=model_info["artifact_uri"],
            trace_id=trace_id,
            span_id=span_id,
            latency_ms=latency_ms,
            service_name=settings.otel_service_name,
        )
        # Override the auto-generated decision_id so UI ↔ plugin link up.
        record["decision_id"] = verified_result.decision_id

        last = app_state.store.get_last()
        previous_hash = last["record_hash"] if last else "GENESIS"

        # Legacy proof for UI's verify_local check. Plugin's anchored
        # proof is the real verifiable artifact; this is signed but
        # not uploaded to Arweave (no double-anchoring).
        proof = app_state.proof_engine.create_proof(record, previous_hash)

        envelope = {
            **proof,
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

    # Populate the demo's lifecycle_store with old-shape envelopes for
    # UI compatibility. Phase 2.D replaces this with a thin cache
    # adapter that reads directly from MLflow tags + Arweave. The
    # background-task duplicate-anchor pipeline is GONE — the plugin
    # already uploaded the real training+registration proofs.
    training_record = build_training_record(
        settings.mlflow_tracking_uri, info["run_id"],
        info["model_name"], info["model_version"],
    )
    last = request.app.state.lifecycle_store.list_all()
    previous_hash = last[-1]["record_hash"] if last else "GENESIS"
    training_proof = request.app.state.proof_engine.create_proof(training_record, previous_hash)
    # Hydrate with the plugin's actual Arweave result so the UI shows
    # the real TX, not None.
    plugin_anchor = info.get("training_anchor_result") or {}
    training_envelope = {
        **training_proof,
        "arweave_tx_id": plugin_anchor.get("tx_id"),
        "arweave_url": plugin_anchor.get("url"),
        "turbo_receipt": plugin_anchor.get("receipt"),
    }
    request.app.state.lifecycle_store.append(training_envelope)

    # Registration record in the legacy store. The plugin's
    # ArioMlflowClient kicked off the real registration anchor in a
    # daemon thread; we don't wait for it here. A lazy refresh path
    # (Phase 2.D) will populate the registration_tx once the daemon
    # completes.
    registration_record = build_registration_record(
        settings.mlflow_tracking_uri, info["model_name"], info["model_version"],
        training_envelope.get("arweave_tx_id"),
    )
    last = request.app.state.lifecycle_store.list_all()
    previous_hash = last[-1]["record_hash"]
    registration_proof = request.app.state.proof_engine.create_proof(registration_record, previous_hash)
    registration_envelope = {
        **registration_proof,
        "arweave_tx_id": None,  # filled by ArioMlflowClient daemon; UI surfaces "anchoring"
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
    """Phase 2.C: verify a prediction via the plugin's full_verify.

    Looks up the prediction's Arweave TX from the demo's RecordStore
    entry (hydrated by Phase 2.B's background task with the plugin's
    actual prediction TX), fetches the pure-commitment envelope from
    Arweave, and runs the four-check verification flow.

    Returns the four-check result alongside the legacy-shape fields so
    callers reading either format keep working. The browser path still
    redirects to the decision detail page.
    """
    import mlflow
    from ario_mlflow.verify import full_verify

    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)

    # Legacy local-envelope check, still surfaced as
    # ``local_verification`` for back-compat with any external API
    # consumers that read it. UI doesn't depend on this path.
    local_result = request.app.state.proof_engine.verify_local(envelope)

    tx_id = envelope.get("arweave_tx_id")
    plugin_result = None
    external_result = None
    ario_result = None

    if tx_id:
        plugin_envelope = request.app.state.anchor.fetch_proof(tx_id)
        if plugin_envelope:
            # Inject TX so verify_ario_attestation can route the call.
            plugin_envelope["_tx_id"] = tx_id

            settings = request.app.state.settings
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow_client = mlflow.tracking.MlflowClient()

            plugin_result = full_verify(
                plugin_envelope,
                proof_engine=request.app.state.proof_engine,
                mlflow_client=mlflow_client,
                ario_client=request.app.state.ario_verify,
            )

            # Legacy-shape mappings for back-compat with existing
            # consumers. Phase 2.E removes once consumers migrate.
            anchored = plugin_result.get("anchored_bytes", {}) or {}
            ario_block = plugin_result.get("ario_attestation", {}) or {}
            external_result = {
                "arweave_data_found": True,
                "arweave_record_hash": anchored.get("computed_hash"),
                "arweave_matches_original": bool(anchored.get("ok") is True),
                "local_tampered": not local_result["overall"],
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
        # Legacy fields (backwards compatibility)
        "local_verification": local_result,
        "external_verification": external_result,
        "ario_verification": ario_result,
        # New four-check structure (Phase 3 UI consumes this)
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


@app.get("/api/export/{decision_id}")
def export_decision(request: Request, decision_id: str):
    """Download a decision record as a JSON file."""
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)
    import json
    content = json.dumps(envelope, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="decision-{decision_id[:8]}.json"'},
    )


def compute_chain_integrity(records: list) -> dict:
    """Pure function: evaluate link + content integrity for a list of envelopes.

    Two concepts are checked independently because a tamper of any single
    record breaks only one of them at a time:

    1. **Link integrity** — each record's ``previous_hash`` matches the
       prior record's ``record_hash``. Breaks if a record is deleted or
       reordered.
    2. **Content integrity** — each record's ``record`` field still hashes
       to its own ``record_hash``. Breaks when a field inside the record
       is modified after signing (what the ``/tamper`` button does).

    The legacy ``intact`` / ``broken_at`` fields are preserved for any
    old clients and reflect whichever check fails first (link, then
    content).
    """
    from ario_mlflow.proof import canonical_json, hash_data

    if not records:
        return {
            "total": 0,
            "link_intact": True,
            "content_intact": True,
            "broken_link_at": None,
            "changed_records": [],
            "intact": True,
            "broken_at": None,
        }

    broken_link_at = None
    for i, rec in enumerate(records):
        expected = records[i - 1]["record_hash"] if i > 0 else "GENESIS"
        if rec.get("previous_hash") != expected:
            broken_link_at = i
            break

    changed_records = []
    for i, rec in enumerate(records):
        record_field = rec.get("record") or {}
        computed = hash_data(canonical_json(record_field))
        if computed != rec.get("record_hash"):
            changed_records.append({
                "index": i,
                "decision_id": record_field.get("decision_id"),
                "reason": "content hash mismatch",
            })

    link_intact = broken_link_at is None
    content_intact = not changed_records
    intact = link_intact and content_intact
    # Legacy broken_at: first broken-link index if any, else first changed record.
    broken_at = broken_link_at if broken_link_at is not None else (
        changed_records[0]["index"] if changed_records else None
    )

    return {
        "total": len(records),
        "link_intact": link_intact,
        "content_intact": content_intact,
        "broken_link_at": broken_link_at,
        "changed_records": changed_records,
        "intact": intact,
        "broken_at": broken_at,
    }


@app.get("/api/chain-integrity")
def chain_integrity(request: Request):
    """Verify link + content integrity across all stored decision records."""
    return compute_chain_integrity(request.app.state.store.list_all())


@app.post("/tamper/{decision_id}")
def tamper_decision(request: Request, decision_id: str):
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)

    # Tamper: modify output_hash in the record
    original_hash = envelope["record"]["output_hash"]
    envelope["record"]["output_hash"] = "TAMPERED_" + original_hash[:50]
    envelope["tampered"] = True

    # Re-run local verification so any UI that keys off `last_verification`
    # (the predictions-page stat cards, the detail page's verification section)
    # immediately reflects the tamper. The Arweave side of last_verification,
    # if previously checked, is still accurate and kept — the permanent copy
    # IS still on the network and matches the ORIGINAL record, even though
    # the local copy no longer does.
    local = request.app.state.proof_engine.verify_local(envelope)
    previous = envelope.get("last_verification") or {}
    envelope["last_verification"] = {
        **previous,
        "hash_valid": local["hash_valid"],
        "signature_valid": local["signature_valid"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    request.app.state.store.update(decision_id, envelope)

    result = {
        "decision_id": decision_id,
        "tampered": True,
        "original_output_hash": original_hash,
        "tampered_output_hash": envelope["record"]["output_hash"],
        "message": "Record tampered locally. Local verification will fail. Arweave record is unaffected.",
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)

    return result
