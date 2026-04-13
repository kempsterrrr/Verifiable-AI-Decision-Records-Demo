import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.config import get_settings
from app.storage import RecordStore
from app.proof import ProofEngine
from app.decision_record import build_decision_record, canonical_json, hash_data
from app.model import load_model, predict, FEATURE_NAMES
from app.anchor import ArweaveAnchor
from app.ario_verify import ArioVerifyClient
from app.ui import router as ui_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    yield

    # Shutdown
    provider.shutdown()


app = FastAPI(title="Verifiable AI Decision Records", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

app.include_router(ui_router)


def _run_prediction(app_state, features: list[float]) -> dict:
    """Core prediction flow: inference -> record -> proof -> anchor -> store."""
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

        # Upload proof to Arweave
        anchor_result = app_state.anchor.upload_proof(proof)

        # Build local envelope: proof + anchoring metadata
        envelope = {
            **proof,
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
        }

        if anchor_result:
            envelope["arweave_tx_id"] = anchor_result["tx_id"]
            envelope["arweave_url"] = anchor_result["url"]
            envelope["turbo_receipt"] = anchor_result["receipt"]

        # Store
        app_state.store.append(envelope)

        return envelope


# --- API Endpoints ---

@app.post("/predict")
def api_predict(request: Request, body: dict):
    features = [
        float(body.get("sepal_length", 5.1)),
        float(body.get("sepal_width", 3.5)),
        float(body.get("petal_length", 1.4)),
        float(body.get("petal_width", 0.2)),
    ]
    envelope = _run_prediction(request.app.state, features)
    return envelope


@app.post("/predict-form")
def form_predict(
    request: Request,
    sepal_length: float = Form(5.1),
    sepal_width: float = Form(3.5),
    petal_length: float = Form(1.4),
    petal_width: float = Form(0.2),
):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    envelope = _run_prediction(request.app.state, features)
    decision_id = envelope["record"]["decision_id"]
    return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)


@app.get("/decisions")
def list_decisions(request: Request):
    return request.app.state.store.list_all()


@app.get("/decisions/{decision_id}")
def get_decision(request: Request, decision_id: str):
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)
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
        ario_result = request.app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])
        if ario_result:
            ario_result = request.app.state.ario_verify._normalize_result(ario_result)

    result = {
        "decision_id": decision_id,
        "local_verification": local_result,
        "external_verification": external_result,
        "ario_verification": ario_result,
    }

    # If called from browser form, redirect to detail page
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)

    return result


@app.post("/tamper/{decision_id}")
def tamper_decision(request: Request, decision_id: str):
    envelope = request.app.state.store.get_by_id(decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)

    # Tamper: modify output_hash in the record
    original_hash = envelope["record"]["output_hash"]
    envelope["record"]["output_hash"] = "TAMPERED_" + original_hash[:50]
    envelope["tampered"] = True

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
