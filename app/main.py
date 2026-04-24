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


def _anchor_lifecycle_record(lifecycle_store, anchor, event_id: str, proof: dict):
    """Background: upload lifecycle proof to Arweave and update stored record."""
    try:
        anchor_result = anchor.upload_proof(proof)
        if anchor_result:
            envelope = lifecycle_store.get_by_event_id(event_id)
            if envelope:
                envelope["arweave_tx_id"] = anchor_result["tx_id"]
                envelope["arweave_url"] = anchor_result["url"]
                envelope["turbo_receipt"] = anchor_result["receipt"]
                lifecycle_store.update(event_id, envelope)
                logger.info(f"Anchored lifecycle event {event_id}: tx={anchor_result['tx_id']}")
    except Exception as e:
        logger.error(f"Background lifecycle anchoring failed for {event_id}: {e}")


def _startup_anchor_lifecycle(settings, model_info, proof_engine, lifecycle_store, anchor):
    """Anchor training run and model registration if not already done."""
    run_id = model_info["run_id"]
    model_name = model_info["model_name"]
    model_version = model_info["model_version"]

    # Check if training record already exists
    existing_training = lifecycle_store.get_by_run_id(run_id)
    training_tx = None

    if not existing_training:
        logger.info(f"Anchoring training run {run_id}...")
        record = build_training_record(
            settings.mlflow_tracking_uri, run_id, model_name, model_version,
        )
        last = lifecycle_store.list_all()
        previous_hash = last[-1]["record_hash"] if last else "GENESIS"
        proof = proof_engine.create_proof(record, previous_hash)
        envelope = {
            **proof,
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }
        lifecycle_store.append(envelope)

        if anchor.enabled:
            _anchor_lifecycle_record(lifecycle_store, anchor, record["event_id"], proof)
        training_tx = envelope.get("arweave_tx_id")
    else:
        training_tx = existing_training.get("arweave_tx_id")
        logger.info(f"Training run {run_id} already anchored.")

    # Check if registration record already exists
    existing_reg = lifecycle_store.get_by_model_version(model_name, model_version)
    if not existing_reg:
        logger.info(f"Anchoring model registration {model_name}/v{model_version}...")
        record = build_registration_record(
            settings.mlflow_tracking_uri, model_name, model_version, training_tx,
        )
        last = lifecycle_store.list_all()
        previous_hash = last[-1]["record_hash"] if last else "GENESIS"
        proof = proof_engine.create_proof(record, previous_hash)
        envelope = {
            **proof,
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }
        lifecycle_store.append(envelope)

        if anchor.enabled:
            _anchor_lifecycle_record(lifecycle_store, anchor, record["event_id"], proof)
    else:
        logger.info(f"Model registration {model_name}/v{model_version} already anchored.")


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

    # MLflow model
    logger.info("Loading MLflow model...")
    app.state.model_info = load_model(settings.mlflow_tracking_uri, settings.mlflow_model_name)
    logger.info(f"Model loaded: {settings.mlflow_model_name}/v{app.state.model_info['model_version']}")

    # Arweave anchor
    app.state.anchor = ArweaveAnchor(settings.arweave_wallet_path, settings.ario_gateway_host)

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


def _anchor_record(store, anchor, decision_id: str, proof: dict):
    """Background task: upload proof to Arweave and update stored record."""
    try:
        anchor_result = anchor.upload_proof(proof)
        if anchor_result:
            envelope = store.get_by_id(decision_id)
            if envelope:
                envelope["arweave_tx_id"] = anchor_result["tx_id"]
                envelope["arweave_url"] = anchor_result["url"]
                envelope["turbo_receipt"] = anchor_result["receipt"]
                store.update(decision_id, envelope)
                logger.info(f"Anchored decision {decision_id}: tx={anchor_result['tx_id']}")
    except Exception as e:
        logger.error(f"Background anchoring failed for {decision_id}: {e}")


def _run_prediction(app_state, features: list[float]) -> tuple[dict, dict]:
    """Core prediction flow: inference -> record -> proof -> store. Anchoring happens async."""
    settings = app_state.settings
    model_info = app_state.model_info

    with tracer.start_as_current_span("predict") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        span_id = format(span.get_span_context().span_id, "016x")

        # Run inference
        start = time.time()
        input_data = dict(zip(FEATURE_NAMES, features))
        prediction = predict(model_info["model"], features)
        latency_ms = (time.time() - start) * 1000

        # Build decision record
        record = build_decision_record(
            input_data=input_data,
            prediction=prediction,
            model_name=model_info["model_name"],
            model_version=model_info["model_version"],
            mlflow_run_id=model_info["run_id"],
            artifact_uri=model_info["artifact_uri"],
            trace_id=trace_id,
            span_id=span_id,
            latency_ms=latency_ms,
            service_name=settings.otel_service_name,
        )

        # Get previous hash for chaining
        last = app_state.store.get_last()
        previous_hash = last["record_hash"] if last else "GENESIS"

        # Create proof
        proof = app_state.proof_engine.create_proof(record, previous_hash)

        # Build local envelope: proof + placeholder anchoring metadata
        envelope = {
            **proof,
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }

        # Store immediately (without Arweave data)
        app_state.store.append(envelope)

        return envelope, proof


# --- API Endpoints ---

@app.post("/predict")
def api_predict(request: Request, body: dict, background_tasks: BackgroundTasks):
    features = [
        float(body.get("annual_income", 78000)),
        float(body.get("credit_utilization", 0.18)),
        float(body.get("debt_to_income_ratio", 0.22)),
        float(body.get("months_employed", 72)),
        float(body.get("credit_score", 745)),
    ]
    envelope, proof = _run_prediction(request.app.state, features)
    decision_id = envelope["record"]["decision_id"]
    if request.app.state.anchor.enabled:
        background_tasks.add_task(
            _anchor_record,
            request.app.state.store,
            request.app.state.anchor,
            decision_id,
            proof,
        )
    return envelope


@app.post("/predict-form")
def form_predict(
    request: Request,
    background_tasks: BackgroundTasks,
    annual_income: float = Form(78000),
    credit_utilization: float = Form(0.18),
    debt_to_income_ratio: float = Form(0.22),
    months_employed: float = Form(72),
    credit_score: float = Form(745),
):
    features = [
        annual_income,
        credit_utilization,
        debt_to_income_ratio,
        months_employed,
        credit_score,
    ]
    envelope, proof = _run_prediction(request.app.state, features)
    decision_id = envelope["record"]["decision_id"]
    if request.app.state.anchor.enabled:
        background_tasks.add_task(
            _anchor_record,
            request.app.state.store,
            request.app.state.anchor,
            decision_id,
            proof,
        )
    return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)


@app.post("/api/train")
def api_train(request: Request, body: dict, background_tasks: BackgroundTasks):
    """Train a new model version, register it, and anchor proofs."""
    import random
    settings = request.app.state.settings
    max_iter = int(body.get("max_iter", 200))
    random_state = int(body.get("random_state", random.randint(1, 10000)))

    # Train and register (synchronous, <1s)
    info = train_and_register_with_params(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Build lifecycle proof records
    training_record = build_training_record(
        settings.mlflow_tracking_uri, info["run_id"],
        info["model_name"], info["model_version"],
    )
    last = request.app.state.lifecycle_store.list_all()
    previous_hash = last[-1]["record_hash"] if last else "GENESIS"
    training_proof = request.app.state.proof_engine.create_proof(training_record, previous_hash)
    training_envelope = {**training_proof, "arweave_tx_id": None, "arweave_url": None, "turbo_receipt": None}
    request.app.state.lifecycle_store.append(training_envelope)

    registration_record = build_registration_record(
        settings.mlflow_tracking_uri, info["model_name"], info["model_version"],
        training_envelope.get("arweave_tx_id"),
    )
    last = request.app.state.lifecycle_store.list_all()
    previous_hash = last[-1]["record_hash"]
    registration_proof = request.app.state.proof_engine.create_proof(registration_record, previous_hash)
    registration_envelope = {**registration_proof, "arweave_tx_id": None, "arweave_url": None, "turbo_receipt": None}
    request.app.state.lifecycle_store.append(registration_envelope)

    # Anchor both in background
    if request.app.state.anchor.enabled:
        background_tasks.add_task(
            _anchor_lifecycle_record,
            request.app.state.lifecycle_store,
            request.app.state.anchor,
            training_record["event_id"],
            training_proof,
        )
        background_tasks.add_task(
            _anchor_lifecycle_record,
            request.app.state.lifecycle_store,
            request.app.state.anchor,
            registration_record["event_id"],
            registration_proof,
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
    }


@app.post("/api/activate/{model_name}/{version}")
def activate_model(request: Request, model_name: str, version: str):
    """Switch the active model to a specific version."""
    import mlflow
    settings = request.app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
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
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)

    # Local verification
    local_result = request.app.state.proof_engine.verify_local(envelope)

    # External verification (fetch from Arweave and compare)
    external_result = None
    if envelope.get("arweave_tx_id"):
        arweave_data = request.app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            external_result = {
                "arweave_data_found": True,
                "arweave_record_hash": arweave_hash,
                "arweave_matches_original": arweave_hash == arweave_data.get("record_hash"),
                "local_tampered": not local_result["overall"],
            }
        else:
            external_result = {"arweave_data_found": False}

    # ar.io Verify — on-demand attestation
    ario_result = None
    if envelope.get("arweave_tx_id") and request.app.state.ario_verify.enabled:
        # Plugin's submit_verification returns a pre-normalized dict.
        ario_result = request.app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])

    result = {
        "decision_id": decision_id,
        "local_verification": local_result,
        "external_verification": external_result,
        "ario_verification": ario_result,
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
