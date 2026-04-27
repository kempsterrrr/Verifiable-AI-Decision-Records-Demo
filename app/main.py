import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.config import get_settings
from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient
from app.model import train_and_register_with_params, FEATURE_NAMES
from ario_mlflow import VerifiedModel
from ario_mlflow.client import ArioMlflowClient
from ario_mlflow.model import IntegrityError
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
    app.state.proof_engine = ProofEngine(
        settings.ed25519_private_key_path,
        settings.ed25519_public_key_path,
    )
    app.state.anchor = ArweaveAnchor(settings.arweave_wallet_path, settings.ario_gateway_host)
    app.state.ario_verify = ArioVerifyClient(settings.ario_verify_url)

    # NEW: ArioMlflowClient handles registration/promotion anchoring with chaining.
    app.state.ario_client = ArioMlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        proof_engine=app.state.proof_engine,
        anchor=app.state.anchor,
    )

    # NEW: VerifiedModel handles inference with integrity check + chained proof.
    # Task 7 will route /predict through this; for now it's loaded so it's ready.
    logger.info("Loading verified model...")
    try:
        app.state.verified_model = VerifiedModel(
            f"models:/{settings.mlflow_model_name}/latest",
            proof_engine=app.state.proof_engine,
            anchor=app.state.anchor,
        )
        app.state.model_info = {
            "model_name": app.state.verified_model.model_name,
            "model_version": app.state.verified_model.model_version,
            "run_id": app.state.verified_model.run_id,
        }
        logger.info(
            f"Verified model loaded: "
            f"{app.state.model_info['model_name']}/v{app.state.model_info['model_version']}"
        )
    except IntegrityError as e:
        # Tampered/mismatched artifacts on the resolved version. Don't silently
        # downgrade — this is a security signal — but don't crash the server
        # either, otherwise the user can't even reach /api/train to recover by
        # training a fresh version. Log loudly, surface the failure on app
        # state so the UI can warn, and refuse to serve predictions until a
        # clean model is loaded (api_train will overwrite verified_model).
        logger.error(
            f"VerifiedModel integrity check FAILED at startup: {e}. "
            f"Predictions disabled until a fresh model is trained."
        )
        app.state.verified_model = None
        app.state.model_info = {"integrity_error": str(e)}
    except Exception as e:
        # On a fresh deployment with no model yet, models:/<name>/latest raises.
        logger.warning(f"VerifiedModel load deferred: {e}")
        app.state.verified_model = None
        app.state.model_info = None

    yield

    # Shutdown
    provider.shutdown()


app = FastAPI(title="Verifiable AI Decision Records", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

app.include_router(ui_router)


def _enrich_prediction(verified_model, features: list[float], raw_prediction) -> dict | None:
    """Build a display-friendly prediction dict for the demo UI.

    Plugin records hold only input/output hashes — sufficient for verifiability,
    insufficient for showing "approved with 87% confidence". This wraps
    sklearn-specific knowledge (CLASS_NAMES, predict_proba) so the demo can
    render the badged prediction card. Display-only; the on-chain proof is
    unaffected.
    """
    try:
        from app.model import CLASS_NAMES
        pyfunc = getattr(verified_model, "_model", None)
        if pyfunc is None or not hasattr(raw_prediction, "__getitem__"):
            return None
        class_index = int(raw_prediction[0])
        rich = {
            "class": CLASS_NAMES[class_index] if 0 <= class_index < len(CLASS_NAMES) else str(class_index),
            "class_index": class_index,
            "features_used": dict(zip(FEATURE_NAMES, features)),
            "probabilities": {},
        }
        # predict_proba is sklearn-specific and lives on the underlying
        # estimator, not the pyfunc wrapper. Try a few attribute paths.
        candidates = [pyfunc]
        impl = getattr(pyfunc, "_model_impl", None)
        if impl is not None:
            candidates.append(impl)
            for attr in ("sklearn_model", "python_model", "model"):
                inner = getattr(impl, attr, None)
                if inner is not None:
                    candidates.append(inner)
        for cand in candidates:
            if hasattr(cand, "predict_proba"):
                try:
                    proba = cand.predict_proba([features])[0]
                    rich["probabilities"] = {
                        CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i): float(p)
                        for i, p in enumerate(proba)
                    }
                    break
                except Exception as e:
                    logger.debug(f"predict_proba on {type(cand).__name__} failed: {e}")
        return rich
    except Exception as e:
        logger.warning(f"Could not enrich prediction display: {e}")
        return None


def _run_prediction(app_state, features: list[float]):
    """Run inference via the plugin's VerifiedModel. Returns (envelope, vp).

    The envelope shape is preserved for the existing template renderers and the
    /predict JSON contract — but the underlying state is now plugin-managed:
    record + signing + chained Arweave anchoring all happen inside
    VerifiedModel.predict() on a daemon thread.

    For display, we enrich the record with a sklearn-specific prediction dict
    (class name, probabilities, features_used) and tag the MLflow trace so
    page reloads (which read from the trace, not from this response) show the
    same data.
    """
    input_data = dict(zip(FEATURE_NAMES, features))
    vp = app_state.verified_model.predict(input_data)

    rich_prediction = _enrich_prediction(app_state.verified_model, features, vp.prediction)
    record = dict(vp.record) if vp.record else {}
    if rich_prediction:
        record["prediction"] = rich_prediction
        # Persist on the trace so /ui/decisions/{id} reloads see the same data.
        if vp.trace_id:
            try:
                import json
                import mlflow
                mlflow.set_trace_tag(
                    vp.trace_id, "ario.display_prediction_json", json.dumps(rich_prediction)
                )
            except Exception as e:
                logger.debug(f"Could not tag trace with display prediction: {e}")

    envelope = {
        "decision_id": vp.decision_id,
        "record": record,
        "proof_status": vp.proof_status,
        "tx_id": vp.tx_id,
    }
    return envelope, vp


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
    envelope, _vp = _run_prediction(request.app.state, features)
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
    envelope, _vp = _run_prediction(request.app.state, features)
    return RedirectResponse(f"/ui/decisions/{envelope['decision_id']}", status_code=303)


@app.post("/api/train")
def api_train(request: Request, body: dict, background_tasks: BackgroundTasks):
    """Train a new model version (anchors data + run + registration via plugin)."""
    import random
    settings = request.app.state.settings
    max_iter = int(body.get("max_iter", 200))
    random_state = int(body.get("random_state", random.randint(1, 10000)))

    info = train_and_register_with_params(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Reload VerifiedModel so the runtime predicts with the new version, and so
    # the prediction chain seeds from the new version's registration_tx.
    request.app.state.verified_model = VerifiedModel(
        f"models:/{settings.mlflow_model_name}/{info['model_version']}",
        proof_engine=request.app.state.proof_engine,
        anchor=request.app.state.anchor,
    )
    request.app.state.model_info = {
        "model_name": info["model_name"],
        "model_version": info["model_version"],
        "run_id": info["run_id"],
    }
    logger.info(f"Switched active model to v{info['model_version']}")

    # Look up anchoring tx IDs from MLflow tags so the training-progress UI
    # can poll /lifecycle/by-tx/{txId} for status updates.
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    _client = _mlflow.tracking.MlflowClient()
    training_tx = None
    registration_tx = None
    try:
        run_tags = _client.get_run(info["run_id"]).data.tags
        training_tx = run_tags.get("ario.training_tx")
    except Exception:
        pass
    try:
        mv_list = _client.search_model_versions(
            f"name='{info['model_name']}' and version='{info['model_version']}'"
        )
        if mv_list:
            registration_tx = (mv_list[0].tags or {}).get("ario.registration_tx")
    except Exception:
        pass

    return {
        "run_id": info["run_id"],
        "model_name": info["model_name"],
        "model_version": info["model_version"],
        "accuracy": info["accuracy"],
        "dataset_tx": info["dataset_tx"],
        "training_tx": training_tx,
        "registration_tx": registration_tx,
    }


@app.post("/api/activate/{model_name}/{version}")
def activate_model(request: Request, model_name: str, version: str):
    """Switch the active model to a specific version."""
    import mlflow
    settings = request.app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        request.app.state.verified_model = VerifiedModel(
            f"models:/{model_name}/{version}",
            proof_engine=request.app.state.proof_engine,
            anchor=request.app.state.anchor,
        )
    except Exception as e:
        return JSONResponse({"error": f"Could not load model: {e}"}, status_code=404)

    request.app.state.model_info = {
        "model_name": request.app.state.verified_model.model_name,
        "model_version": request.app.state.verified_model.model_version,
        "run_id": request.app.state.verified_model.run_id,
    }
    logger.info(f"Activated model {model_name}/v{version}")

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse("/", status_code=303)

    return {"activated": True, "model_name": model_name, "model_version": str(version)}


@app.get("/decisions/{decision_id}")
def get_decision(request: Request, decision_id: str):
    from app.ui import _decision_envelope_by_id
    envelope = _decision_envelope_by_id(request.app, decision_id)
    if not envelope:
        return JSONResponse({"error": "Decision not found"}, status_code=404)
    # Attach a live turbo_status so the polling UI can update the badge
    # as the proof progresses from Uploading → Confirmed → Permanent without
    # requiring a page reload.
    envelope = dict(envelope)
    if envelope.get("arweave_tx_id"):
        envelope["turbo_status"] = request.app.state.anchor.check_status(envelope["arweave_tx_id"])
    return envelope


@app.get("/lifecycle/by-tx/{tx_id}")
def get_lifecycle_by_tx(request: Request, tx_id: str):
    """Return turbo receipt status for a lifecycle proof tx.

    Replaces the old /lifecycle/{event_id} endpoint for the JS polling story
    in run_detail.html. Templates pass a known Arweave tx_id so we can
    check status directly without a local event registry.
    """
    turbo_status = request.app.state.anchor.check_status(tx_id)
    # If anchoring is still in progress, check_status may return NOT_FOUND.
    # The JS only needs arweave_tx_id to be truthy to consider the step done,
    # so we return it unconditionally (the tx was submitted, even if not yet
    # indexed by the gateway).
    return {"tx_id": tx_id, "arweave_tx_id": tx_id, "turbo_status": turbo_status}


@app.get("/api/export/{decision_id}")
def export_decision(request: Request, decision_id: str):
    """Download a decision record as a JSON file."""
    from app.ui import _decision_envelope_by_id
    envelope = _decision_envelope_by_id(request.app, decision_id)
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
       is modified after signing.

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
    """Verify link + content integrity across all anchored decision records.

    Data source is now MLflow traces + Arweave-fetched envelopes (plugin-driven).
    Only anchored records (with arweave_tx_id) are included — unanchored records
    cannot be chain-verified as they lack the full proof envelope.
    """
    from app.ui import _list_recent_decisions, _envelope_from_arweave

    # Fetch lightweight trace-tag envelopes to find all decision IDs + tx IDs.
    trace_envelopes = _list_recent_decisions(request.app)

    # For chain integrity we need the full envelope (with record_hash, previous_hash,
    # record) which only exists on Arweave. Unanchored records are skipped.
    full_envelopes = []
    for env in trace_envelopes:
        tx_id = env.get("arweave_tx_id")
        if not tx_id:
            continue
        full_env = _envelope_from_arweave(request.app, tx_id)
        if full_env:
            full_envelopes.append(full_env)

    # Sort by timestamp so the chain check is order-stable.
    def _ts(e):
        return (e.get("record") or {}).get("timestamp") or ""

    full_envelopes.sort(key=_ts)

    return compute_chain_integrity(full_envelopes)
