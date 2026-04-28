import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
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


def _resolve_tracking_uri_to_local_root(tracking_uri: str) -> str | None:
    """Normalize a tracking URI to an absolute on-disk path, or None if the
    URI is a non-file backend (sqlite, http, etc.).

    MLflow accepts several forms for file-backed stores: ``file:///abs``,
    ``file:rel``, ``rel`` (bare path treated as relative), ``/abs``. All of
    these resolve to a directory we can read trace data from. URIs with any
    other scheme aren't file-backed and our direct-file-mutation tamper
    demo can't operate on them.
    """
    import os as _os
    if not tracking_uri:
        return None
    if tracking_uri.startswith("file://"):
        path = tracking_uri[len("file://"):]
    elif tracking_uri.startswith("file:"):
        path = tracking_uri[len("file:"):]
    elif "://" in tracking_uri:
        # sqlite://, http://, postgresql://, etc. — not file-backed.
        return None
    else:
        path = tracking_uri  # bare path, treat as relative-or-absolute filesystem
    if not _os.path.isabs(path):
        path = _os.path.abspath(path)
    return path if _os.path.isdir(path) else None


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

    # Per-process cache of ar.io Verify attestations, keyed by Arweave tx_id.
    # Attestations are stable for a given tx — once finalized, the level and
    # report URL don't change. Cache lifetime is process lifetime; restart
    # busts. Acceptable for a demo; production would persist.
    app.state.ario_verify_cache: dict[str, dict] = {}

    # Per-process cache of verify_prediction results, keyed by decision_id.
    # Entries are authoritative until explicitly evicted by tamper/untamper/
    # live-verify endpoints. Populated by /predict and /predict-form after
    # wait_for_anchor completes on a daemon thread.
    app.state.decision_verify_cache: dict[str, dict] = {}

    # Per-process cache of verify_model_lifecycle results, keyed by
    # (model_name, model_version). Populated on state-change events
    # (train, promote, tamper, startup), NOT on dashboard render.
    # Startup verification runs in a daemon thread so the dashboard is
    # responsive immediately; entries flip from "verifying" to
    # verified/tampered as the daemon completes.
    app.state.model_lifecycle_cache: dict[tuple[str, str], dict] = {}

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

    def _startup_verify_models():
        """Walk all known model versions and verify each. Runs in a
        daemon thread so dashboard rendering isn't blocked. Entries
        flip from 'verifying' to verified/tampered as the daemon
        completes."""
        try:
            client = app.state.ario_client
            versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
            for mv in versions:
                try:
                    _verify_and_cache_model_lifecycle(app, mv.name, str(mv.version))
                except Exception as e:
                    logger.warning(f"Startup verify of {mv.name}/v{mv.version} failed: {e}")
        except Exception as e:
            logger.warning(f"Startup verification walk failed: {e}")

    import threading
    threading.Thread(target=_startup_verify_models, daemon=True).start()

    yield

    # Shutdown
    provider.shutdown()


app = FastAPI(title="Verifiable AI Decision Records", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

app.include_router(ui_router)


def _verify_and_cache_model_lifecycle(app, model_name: str, model_version: str) -> dict:
    """Run verify_model_lifecycle and cache the result on app.state.

    Always wraps in try/except so a verification failure (gateway down,
    network flake) doesn't crash the caller. On error, the cache stores
    a placeholder ``{error, all_intact: None}`` so the dashboard can
    render "verification pending / failed" rather than blow up.

    Uses ``getattr`` to access app.state.model_lifecycle_cache so legacy
    tests that construct a minimal state object without this attribute
    don't crash (they'll just skip the cache write).
    """
    from ario_mlflow.decision_verify import verify_model_lifecycle
    cache = getattr(getattr(app, "state", None), "model_lifecycle_cache", None)
    try:
        result = verify_model_lifecycle(
            model_name=model_name,
            model_version=str(model_version),
            anchor=app.state.anchor,
            ario_client=app.state.ario_client,
            proof_engine=app.state.proof_engine,
        )
        if cache is not None:
            cache[(model_name, str(model_version))] = result
        logger.info(
            f"Verified model lifecycle for {model_name}/v{model_version}: "
            f"all_intact={result.get('all_intact')}"
        )
        return result
    except Exception as e:
        logger.warning(f"verify_model_lifecycle failed for {model_name}/v{model_version}: {e}")
        placeholder = {
            "model_name": model_name,
            "model_version": str(model_version),
            "events": [],
            "all_intact": None,
            "error": str(e),
        }
        if cache is not None:
            cache[(model_name, str(model_version))] = placeholder
        return placeholder


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

    Envelope shape:
    - record:             EXACTLY what was signed (don't mutate — verification
                          re-hashes this and compares to record_hash)
    - record_hash, ...:   the proof commitments
    - display_prediction: demo-only enrichment (class name + probabilities);
                          a sibling, NOT nested under record. The on-chain
                          proof is unaffected.
    """
    if app_state.verified_model is None:
        info = getattr(app_state, "model_info", None) or {}
        detail = "No verified model is loaded. Train a model first via /api/train."
        if info.get("integrity_error"):
            detail = (
                "Predictions disabled: model artifact integrity check failed at startup. "
                "Train a fresh model via /api/train to recover."
            )
        raise HTTPException(status_code=503, detail=detail)
    input_data = dict(zip(FEATURE_NAMES, features, strict=True))
    vp = app_state.verified_model.predict(input_data)

    rich_prediction = _enrich_prediction(app_state.verified_model, features, vp.prediction)
    if rich_prediction and vp.trace_id:
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
        "record": vp.record,                  # canonical signed bytes — DO NOT mutate
        "proof_status": vp.proof_status,
        "tx_id": vp.tx_id,
        "display_prediction": rich_prediction,  # demo-only enrichment
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


def _schedule_prediction_verify(request, vp):
    """Schedule prediction-integrity verification on a daemon thread so the
    response isn't blocked by Arweave latency. The result lands in
    decision_verify_cache when verification completes; the dashboard
    picks it up on next load.
    """
    import threading

    def _verify_prediction_async():
        try:
            vp.wait_for_anchor(timeout=30.0)
            from ario_mlflow.decision_verify import verify_prediction
            result = verify_prediction(
                decision_id=vp.decision_id,
                tracking_uri=request.app.state.settings.mlflow_tracking_uri,
                arweave=request.app.state.anchor,
            )
            request.app.state.decision_verify_cache[vp.decision_id] = {
                "match_input": result.match_input,
                "match_output": result.match_output,
                "overall_match": result.overall_match,
                "error": result.error,
            }
            logger.info(f"Prediction {vp.decision_id} verified: overall_match={result.overall_match}")
        except Exception as e:
            logger.warning(f"Async verify_prediction failed for {vp.decision_id}: {e}")

    threading.Thread(target=_verify_prediction_async, daemon=True).start()


@app.post("/predict")
def api_predict(request: Request, body: dict, background_tasks: BackgroundTasks):
    features = [
        float(body.get(name, FEATURE_DEFAULTS[name])) for name in FEATURE_NAMES
    ]
    envelope, vp = _run_prediction(request.app.state, features)
    _schedule_prediction_verify(request, vp)
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
    envelope, vp = _run_prediction(request.app.state, features)
    _schedule_prediction_verify(request, vp)
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

    # Verify and cache lifecycle for the newly trained model version.
    _verify_and_cache_model_lifecycle(request.app, info["model_name"], info["model_version"])

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
    except Exception as e:
        logger.warning(f"Could not fetch training_tx for run {info['run_id']}: {e}")
    try:
        from app.ui import _escape_mlflow_filter_value
        name_q = _escape_mlflow_filter_value(info["model_name"])
        ver_q = _escape_mlflow_filter_value(info["model_version"])
        mv_list = _client.search_model_versions(
            f"name='{name_q}' and version='{ver_q}'"
        )
        if mv_list:
            registration_tx = (mv_list[0].tags or {}).get("ario.registration_tx")
    except Exception as e:
        logger.warning(
            f"Could not fetch registration_tx for "
            f"{info['model_name']}/v{info['model_version']}: {e}"
        )

    return {
        "run_id": info["run_id"],
        "model_name": info["model_name"],
        "model_version": info["model_version"],
        "accuracy": info["accuracy"],
        "dataset_tx": info["dataset_tx"],
        "training_tx": training_tx,
        "registration_tx": registration_tx,
    }


@app.post("/api/promote/{model_name}/{version}")
def api_promote(request: Request, model_name: str, version: str):
    """Promote a model version to Production and anchor the stage transition.

    Triggers ArioMlflowClient.transition_model_version_stage which:
    - Calls MLflow's stage transition synchronously.
    - Spawns a daemon thread to sign + upload the promotion proof.
    - On success, writes ario.promotion_tx onto the model version.

    The endpoint waits up to 30s for the anchor to settle so the response
    can include the resulting tx (or surface the failure cleanly).
    """
    ario_client = request.app.state.ario_client
    try:
        ario_client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        logger.warning(f"Stage transition failed for {model_name}/v{version}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not transition stage: {e}",
        )

    # Wait for the background anchor to complete so we can return the tx_id.
    # The thread does a single Turbo upload — 30s is generous for that.
    ario_client.wait_for_anchor("promotion", model_name, str(version), timeout=30.0)
    status = ario_client.anchor_status("promotion", model_name, str(version))

    # Update lifecycle cache for the promoted version.
    _verify_and_cache_model_lifecycle(request.app, model_name, version)

    return {
        "model_name": model_name,
        "model_version": str(version),
        "stage": "Production",
        "promotion_tx": status.get("tx_id"),
        "anchor_status": status.get("status"),
        "anchor_error": status.get("error"),
    }


@app.post("/api/tamper/training/{run_id}")
def api_tamper_training(request: Request, run_id: str):
    """Demo-only: corrupt the run's model.pkl in MLflow's artifact store so
    the artifact-integrity verification on Run Detail catches the change.

    Backs up the original to ``model.pkl.tamper_backup`` next to it. Use
    /api/untamper/training/{run_id} to restore.

    Only supported on file-based MLflow tracking stores. Other backends
    (S3-backed artifacts, etc.) would need a backend-specific implementation.
    """
    settings = request.app.state.settings
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        # Resolve to the local artifact path. download_artifacts on a
        # file-store returns the actual path inside mlruns/, not a copy.
        local_model_dir = _mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not locate run model artifacts: {e}")

    target = os.path.join(local_model_dir, "model.pkl")
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"model.pkl not found in run {run_id}")
    backup = target + ".tamper_backup"
    if os.path.exists(backup):
        raise HTTPException(status_code=409, detail="Already tampered — call /api/untamper first.")

    # Atomic-ish: copy original to backup, then mutate. We append a single
    # byte rather than full overwrite so the file is still a "file" but the
    # hash differs.
    import shutil
    shutil.copyfile(target, backup)
    with open(target, "ab") as f:
        f.write(b"\x00")

    # Invalidate any cached lifecycle for model versions backed by this run.
    cache = getattr(getattr(request.app, "state", None), "model_lifecycle_cache", None)
    client = getattr(getattr(request.app, "state", None), "ario_client", None)
    if cache is not None:
        try:
            if client is not None:
                try:
                    mvs = client.search_model_versions(f"run_id='{run_id}'")
                except Exception:
                    # Fallback: walk all versions and filter Python-side if the
                    # run_id filter isn't supported on this MLflow version.
                    import mlflow as _mlflow2
                    _mlflow2.set_tracking_uri(settings.mlflow_tracking_uri)
                    _c2 = _mlflow2.tracking.MlflowClient()
                    all_mvs = _c2.search_model_versions(f"name='{settings.mlflow_model_name}'")
                    mvs = [mv for mv in all_mvs if mv.run_id == run_id]
                for mv in mvs:
                    cache.pop((mv.name, str(mv.version)), None)
        except Exception as e:
            logger.warning(f"Could not invalidate lifecycle cache for run {run_id}: {e}")

    return {
        "run_id": run_id,
        "tampered_path": target,
        "backup_path": backup,
        "note": "Reload Run Detail to see Artifact Integrity FAIL.",
    }


@app.post("/api/untamper/training/{run_id}")
def api_untamper_training(request: Request, run_id: str):
    """Restore model.pkl from the tamper backup."""
    settings = request.app.state.settings
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        local_model_dir = _mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not locate run model artifacts: {e}")

    target = os.path.join(local_model_dir, "model.pkl")
    backup = target + ".tamper_backup"
    if not os.path.exists(backup):
        raise HTTPException(status_code=404, detail="No tamper backup found — nothing to restore.")

    import shutil
    shutil.move(backup, target)

    # Invalidate any cached lifecycle for model versions backed by this run.
    cache = getattr(getattr(request.app, "state", None), "model_lifecycle_cache", None)
    client = getattr(getattr(request.app, "state", None), "ario_client", None)
    if cache is not None:
        try:
            if client is not None:
                try:
                    mvs = client.search_model_versions(f"run_id='{run_id}'")
                except Exception:
                    # Fallback: walk all versions and filter Python-side if the
                    # run_id filter isn't supported on this MLflow version.
                    import mlflow as _mlflow2
                    _mlflow2.set_tracking_uri(settings.mlflow_tracking_uri)
                    _c2 = _mlflow2.tracking.MlflowClient()
                    all_mvs = _c2.search_model_versions(f"name='{settings.mlflow_model_name}'")
                    mvs = [mv for mv in all_mvs if mv.run_id == run_id]
                for mv in mvs:
                    cache.pop((mv.name, str(mv.version)), None)
        except Exception as e:
            logger.warning(f"Could not invalidate lifecycle cache for run {run_id}: {e}")

    return {
        "run_id": run_id,
        "restored_path": target,
        "note": "Reload Run Detail to see Artifact Integrity PASS.",
    }


@app.post("/api/tamper/decision/{decision_id}")
def api_tamper_decision(request: Request, decision_id: str):
    """Demo-only: mutate the trace's recorded input in MLflow's storage so
    the prediction-integrity verification on Decision Detail catches it.

    File-backend only. Backs up the original trace data file, rewrites it
    with a mutated copy that flips one feature value (credit_score + 100).
    """
    settings = request.app.state.settings
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = _mlflow.tracking.MlflowClient()

    # Find the trace.
    from app.ui import _escape_mlflow_filter_value
    try:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=f"tags.`ario.decision_id` = '{_escape_mlflow_filter_value(decision_id)}'",
            max_results=1,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trace lookup failed: {e}")
    if not traces:
        raise HTTPException(status_code=404, detail="No trace for this decision.")

    trace_info = traces[0].info
    trace_id = getattr(trace_info, "trace_id", None) or getattr(trace_info, "request_id", None)
    if not trace_id:
        raise HTTPException(status_code=500, detail="Could not determine trace_id.")

    # Locate the on-disk traces.json. File backend stores under
    # mlruns/<exp_id>/traces/<trace_id>/artifacts/traces.json.
    mlruns_dir = _resolve_tracking_uri_to_local_root(settings.mlflow_tracking_uri)
    if mlruns_dir is None:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Tamper demo only supports file-based tracking stores "
                f"(got {settings.mlflow_tracking_uri!r}). For file backends, "
                f"the URI may be a bare path like 'mlruns', or 'file:///abs/path'."
            ),
        )

    experiment_id = getattr(trace_info, "experiment_id", None)
    if not experiment_id:
        raise HTTPException(status_code=500, detail="Could not determine experiment_id for trace.")
    traces_json = os.path.join(mlruns_dir, str(experiment_id), "traces", trace_id, "artifacts", "traces.json")
    if not os.path.isfile(traces_json):
        raise HTTPException(status_code=404, detail=f"Trace data file not found: {traces_json}")

    backup = traces_json + ".tamper_backup"
    if os.path.exists(backup):
        raise HTTPException(status_code=409, detail="Already tampered — call /api/untamper/decision first.")

    import shutil, json as _json
    shutil.copyfile(traces_json, backup)

    # Read, mutate, write.
    with open(traces_json, "r") as f:
        data = _json.load(f)
    spans = data.get("spans") or []
    if not spans:
        # Restore and bail.
        shutil.move(backup, traces_json)
        raise HTTPException(status_code=500, detail="Trace has no spans to tamper.")

    # The span attribute keys are JSON-encoded strings — parse, mutate, re-encode.
    attrs = spans[0].get("attributes") or {}
    raw_inputs_str = attrs.get("mlflow.spanInputs")
    if not raw_inputs_str:
        shutil.move(backup, traces_json)
        raise HTTPException(status_code=500, detail="Trace span has no recorded inputs.")
    try:
        inputs = _json.loads(raw_inputs_str)
    except _json.JSONDecodeError as e:
        shutil.move(backup, traces_json)
        raise HTTPException(status_code=500, detail=f"Could not parse span inputs: {e}")

    # Mutate one feature so the recomputed hash diverges.
    target = inputs.get("input_data")
    if isinstance(target, dict) and "credit_score" in target:
        try:
            target["credit_score"] = float(target["credit_score"]) + 100.0
        except (TypeError, ValueError):
            target["credit_score"] = 999999.0
    elif isinstance(target, dict):
        # No credit_score? add a marker key.
        target["__tampered__"] = True
    else:
        # Unknown input shape; just stuff a sentinel into the wrapper.
        inputs["__tampered__"] = True

    attrs["mlflow.spanInputs"] = _json.dumps(inputs)

    with open(traces_json, "w") as f:
        _json.dump(data, f)

    # Bust the row-status cache so the dashboard flips immediately on
    # the next render (otherwise stale up to TTL seconds).
    cache = getattr(request.app.state, "decision_verify_cache", None)
    if cache is not None:
        cache.pop(decision_id, None)

    return {
        "decision_id": decision_id,
        "trace_id": trace_id,
        "tampered_path": traces_json,
        "backup_path": backup,
        "note": "Reload Decision Detail to see Prediction Integrity FAIL.",
    }


@app.post("/api/verify/decision/{decision_id}")
def api_reverify_decision(request: Request, decision_id: str):
    """Live re-verification trigger for the demo's "Verify with ar.io (live)"
    button. Busts caches so the round-trip is visibly fresh, then runs the
    same verification chain the page load uses (single source of truth).

    Returns the flat verification dict — the JS handler updates rows in
    place from this response.
    """
    from app.ui import _decision_envelope_by_id, _compute_decision_verification

    envelope = _decision_envelope_by_id(request.app, decision_id)
    if envelope is None:
        raise HTTPException(status_code=404, detail="Decision not found.")

    # Bust caches BEFORE re-verifying so the round-trip is real.
    tx_id = envelope.get("arweave_tx_id")
    if tx_id:
        request.app.state.ario_verify_cache.pop(tx_id, None)
    # decision_verify_cache is initialised in Phase C; pop conditionally
    # so this code is forward-compatible without depending on Phase C
    # ordering.
    decision_cache = getattr(request.app.state, "decision_verify_cache", None)
    if decision_cache is not None:
        decision_cache.pop(decision_id, None)

    verification = _compute_decision_verification(request.app, envelope, decision_id)
    verification["decision_id"] = decision_id
    return verification


@app.post("/api/untamper/decision/{decision_id}")
def api_untamper_decision(request: Request, decision_id: str):
    """Restore the trace data file from the tamper backup."""
    settings = request.app.state.settings
    import mlflow as _mlflow
    _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = _mlflow.tracking.MlflowClient()

    from app.ui import _escape_mlflow_filter_value
    try:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=f"tags.`ario.decision_id` = '{_escape_mlflow_filter_value(decision_id)}'",
            max_results=1,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trace lookup failed: {e}")
    if not traces:
        raise HTTPException(status_code=404, detail="No trace for this decision.")

    trace_info = traces[0].info
    trace_id = getattr(trace_info, "trace_id", None) or getattr(trace_info, "request_id", None)
    experiment_id = getattr(trace_info, "experiment_id", None)
    if not trace_id or not experiment_id:
        raise HTTPException(status_code=500, detail="Could not resolve trace/experiment id.")

    mlruns_dir = _resolve_tracking_uri_to_local_root(settings.mlflow_tracking_uri)
    if mlruns_dir is None:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Tamper demo only supports file-based tracking stores "
                f"(got {settings.mlflow_tracking_uri!r}). For file backends, "
                f"the URI may be a bare path like 'mlruns', or 'file:///abs/path'."
            ),
        )

    traces_json = os.path.join(mlruns_dir, str(experiment_id), "traces", trace_id, "artifacts", "traces.json")
    backup = traces_json + ".tamper_backup"
    if not os.path.exists(backup):
        raise HTTPException(status_code=404, detail="No tamper backup found.")
    import shutil
    shutil.move(backup, traces_json)

    # Bust the row-status cache so the dashboard flips immediately on
    # the next render (otherwise stale up to TTL seconds).
    cache = getattr(request.app.state, "decision_verify_cache", None)
    if cache is not None:
        cache.pop(decision_id, None)

    return {"decision_id": decision_id, "restored_path": traces_json}


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
    """Roll up cached verification state into a per-model-version chain status.

    Reads from app.state.model_lifecycle_cache and decision_verify_cache.
    Does NOT fetch from Arweave on this path — caches are populated by
    state-change events (train, promote, tamper) and the startup
    verification daemon. Cache misses surface as 'verification_pending'.
    """
    from app.ui import _list_recent_decisions, _evaluate_chain_status

    decisions = _list_recent_decisions(request.app, max_results=5000, compute_status=False)

    # Group decisions by (model_name, model_version).
    chains_by_version: dict = {}
    for env in decisions:
        rec = env.get("record") or {}
        key = (rec.get("model_name"), rec.get("model_version"))
        if not key[0] or not key[1]:
            continue
        chains_by_version.setdefault(key, []).append(env)

    chains = []
    all_intact = True
    any_pending = False
    total = 0

    for (name, version), envs in chains_by_version.items():
        envs.sort(key=lambda e: (e.get("record") or {}).get("timestamp") or "")
        lifecycle = request.app.state.model_lifecycle_cache.get((name, str(version)))
        decision_results = [
            request.app.state.decision_verify_cache.get(
                (env.get("record") or {}).get("decision_id")
            )
            for env in envs
        ]

        chain_status = _evaluate_chain_status(envs, lifecycle, decision_results)
        if chain_status["intact"] is False:
            all_intact = False
        if chain_status["intact"] is None:
            any_pending = True
        total += len(envs)

        chains.append({
            "model_name": name,
            "model_version": str(version),
            "decision_count": len(envs),
            "root_tx": lifecycle.get("registration_tx") if lifecycle else None,
            "intact": chain_status["intact"],
            "broken_reason": chain_status.get("broken_reason"),
            "broken_at": chain_status.get("broken_at"),
            "broken_decision_id": chain_status.get("broken_decision_id"),
        })

    return {
        "total": total,
        "chain_count": len(chains),
        "all_intact": all_intact if not any_pending else None,
        "any_pending": any_pending,
        "chains": chains,
    }
