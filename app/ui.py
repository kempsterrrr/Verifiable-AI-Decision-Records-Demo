import os
from datetime import datetime, timezone

import mlflow
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ario_mlflow.proof import canonical_json, hash_data

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _common_context(app):
    """Shared template context for all pages (status bar, model info)."""
    return {
        "model_info": app.state.model_info,
        "arweave_enabled": app.state.anchor.enabled if app.state.anchor else False,
        "ario_verify_enabled": app.state.ario_verify.enabled if app.state.ario_verify else False,
    }


def _verify_envelope(app, envelope):
    """Run three-level verification on any proof envelope. Returns result dict."""
    local = app.state.proof_engine.verify_local(envelope)
    result = {
        "hash_valid": local["hash_valid"],
        "signature_valid": local["signature_valid"],
        "permanent_copy_found": False,
        "hash_match": False,
        "attestation_level": None,
        "report_url": None,
        "attested_by": None,
        "attested_at": None,
    }

    if envelope.get("arweave_tx_id"):
        arweave_data = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            result["permanent_copy_found"] = True
            result["hash_match"] = arweave_hash == arweave_data.get("record_hash")

        if app.state.ario_verify.enabled:
            # Plugin's submit_verification returns a pre-normalized dict with
            # attestation_level / report_url / attested_by / attested_at.
            normalized = app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])
            if normalized:
                result["attestation_level"] = normalized.get("attestation_level")
                result["report_url"] = normalized.get("report_url")
                result["attested_by"] = normalized.get("attested_by")
                result["attested_at"] = normalized.get("attested_at")

    return result, local


@router.get("/ui/predictions")
def predictions_redirect():
    """Permanent redirect from the old URL. Bookmarks keep working."""
    return RedirectResponse("/ui/decisions", status_code=301)


@router.get("/ui/decisions", response_class=HTMLResponse)
def decisions(request: Request):
    app = request.app
    records = app.state.store.list_all()
    model_info = app.state.model_info

    # Lifecycle status for provenance card
    training_env = app.state.lifecycle_store.get_by_run_id(model_info["run_id"])
    registration_env = app.state.lifecycle_store.get_by_model_version(
        model_info["model_name"], model_info["model_version"]
    )

    training_status = "none"
    if training_env:
        tv = training_env.get("last_verification")
        if tv and tv.get("hash_valid") and tv.get("permanent_copy_found") and tv.get("hash_match"):
            training_status = "verified"
        elif training_env.get("arweave_tx_id"):
            training_status = "anchored"
        else:
            training_status = "local"

    registration_status = "none"
    if registration_env:
        rv = registration_env.get("last_verification")
        if rv and rv.get("hash_valid") and rv.get("permanent_copy_found") and rv.get("hash_match"):
            registration_status = "verified"
        elif registration_env.get("arweave_tx_id"):
            registration_status = "anchored"
        else:
            registration_status = "local"

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            **_common_context(app),
            "records": records,
            "training_status": training_status,
            "registration_status": registration_status,
        },
    )


@router.get("/", response_class=HTMLResponse)
def model_registry(request: Request):
    app = request.app
    settings = app.state.settings
    model_name = settings.mlflow_model_name
    active_version = app.state.model_info["model_version"]

    mlflow.set_tracking_uri(os.path.abspath(settings.mlflow_tracking_uri))
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    version_data = []

    for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
        # Get run metrics
        accuracy = None
        created = None
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                accuracy = run.data.metrics.get("accuracy")
                created = run.info.start_time
            except Exception:
                pass

        # Check lifecycle anchoring status
        training_env = app.state.lifecycle_store.get_by_run_id(mv.run_id) if mv.run_id else None
        reg_env = app.state.lifecycle_store.get_by_model_version(model_name, str(mv.version))

        def _status(env):
            if not env:
                return "none"
            v = env.get("last_verification")
            if v and v.get("hash_valid") and v.get("permanent_copy_found") and v.get("hash_match"):
                return "verified"
            if env.get("arweave_tx_id"):
                return "anchored"
            return "local"

        version_data.append({
            "version": str(mv.version),
            "run_id": mv.run_id or "",
            "accuracy": accuracy,
            "stage": mv.current_stage if hasattr(mv, "current_stage") else "None",
            "training_status": _status(training_env),
            "registration_status": _status(reg_env),
            "is_active": str(mv.version) == str(active_version),
            "created": datetime.fromtimestamp(created / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if created else "",
        })

    return templates.TemplateResponse(
        request,
        "model_registry.html",
        {
            **_common_context(app),
            "model_name": model_name,
            "versions": version_data,
            "active_version": active_version,
        },
    )


@router.get("/ui/registry")
def registry_redirect():
    return RedirectResponse("/", status_code=301)


@router.get("/ui/who-this-is-for", response_class=HTMLResponse)
def who_this_is_for(request: Request):
    """Four-persona framing page so visitors find a doorway matched to their context."""
    app = request.app
    return templates.TemplateResponse(
        request,
        "who_this_is_for.html",
        _common_context(app),
    )


@router.get("/ui/decisions/{decision_id}", response_class=HTMLResponse)
def decision_detail(request: Request, decision_id: str, verify: bool = False):
    app = request.app
    envelope = app.state.store.get_by_id(decision_id)

    if not envelope:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    # Local verification (always — instant, no network)
    local = app.state.proof_engine.verify_local(envelope)

    # Full verification (on-demand — user-triggered via ?verify=true)
    if verify and envelope.get("arweave_tx_id"):
        result = {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "hash_valid": local["hash_valid"],
            "signature_valid": local["signature_valid"],
            "permanent_copy_found": False,
            "hash_match": False,
            "attestation_level": None,
            "report_url": None,
            "pdf_url": None,
            "attested_by": None,
            "attested_at": None,
        }

        # Fetch from ar.io gateway and compare
        arweave_data = app.state.anchor.fetch_proof(envelope["arweave_tx_id"])
        if arweave_data:
            arweave_hash = hash_data(canonical_json(arweave_data.get("record", {})))
            result["permanent_copy_found"] = True
            result["hash_match"] = arweave_hash == arweave_data.get("record_hash")

        # ar.io Verify attestation. Plugin's submit_verification returns a
        # pre-normalized dict — no second _normalize_result call needed.
        if app.state.ario_verify.enabled:
            normalized = app.state.ario_verify.submit_verification(envelope["arweave_tx_id"])
            if normalized:
                result["attestation_level"] = normalized.get("attestation_level")
                result["report_url"] = normalized.get("report_url")
                result["pdf_url"] = normalized.get("pdf_url")
                result["attested_by"] = normalized.get("attested_by")
                result["attested_at"] = normalized.get("attested_at")

        # Persist results on the envelope
        envelope["last_verification"] = result
        app.state.store.update(decision_id, envelope)

    # Check Turbo status for anchored records (fast, single HTTP call)
    turbo_status = None
    if envelope.get("arweave_tx_id"):
        turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])

    return templates.TemplateResponse(
        request,
        "decision_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            "local_verification": local,
            "turbo_status": turbo_status,
        },
    )


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, verify: bool = False):
    app = request.app

    envelope = app.state.lifecycle_store.get_by_run_id(run_id)
    if not envelope:
        return HTMLResponse("<h1>Training run not found</h1>", status_code=404)

    local = app.state.proof_engine.verify_local(envelope)

    if verify and envelope.get("arweave_tx_id"):
        result, _ = _verify_envelope(app, envelope)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        envelope["last_verification"] = result
        app.state.lifecycle_store.update(envelope["record"]["event_id"], envelope)

    turbo_status = None
    if envelope.get("arweave_tx_id"):
        turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])

    # Fetch the live MLflow tags directly from the tracking store so evaluators
    # can confirm the ario.* tags are really on the run (not synthesised by the
    # demo UI). This is the closest thing to "View in MLflow UI" we can offer
    # without running a second server alongside uvicorn on Railway.
    mlflow_tags: dict[str, str] = {}
    try:
        import mlflow as _mlflow
        settings = app.state.settings
        _mlflow.set_tracking_uri(os.path.abspath(settings.mlflow_tracking_uri))
        client = _mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        mlflow_tags = dict(run.data.tags)
    except Exception:
        mlflow_tags = {}

    ario_tags = {k: v for k, v in sorted(mlflow_tags.items()) if k.startswith("ario.")}
    tracking_dir = os.path.abspath(app.state.settings.mlflow_tracking_uri)

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            "local_verification": local,
            "turbo_status": turbo_status,
            "mlflow_ario_tags": ario_tags,
            "mlflow_tracking_dir": tracking_dir,
        },
    )


@router.get("/ui/models/{model_name}/{version}", response_class=HTMLResponse)
def model_chain(request: Request, model_name: str, version: str, verify: bool = False):
    app = request.app

    # Get lifecycle records
    lifecycle_records = app.state.lifecycle_store.list_all()
    training_env = None
    registration_env = None

    for rec in lifecycle_records:
        r = rec.get("record", {})
        if r.get("event_type") == "training_complete" and r.get("model_name") == model_name and str(r.get("model_version")) == str(version):
            training_env = rec
        elif r.get("event_type") == "model_registered" and r.get("model_name") == model_name and str(r.get("model_version")) == str(version):
            registration_env = rec

    # Local verification for each
    training_local = app.state.proof_engine.verify_local(training_env) if training_env else None
    registration_local = app.state.proof_engine.verify_local(registration_env) if registration_env else None

    # Full verification (on-demand)
    training_verify = None
    registration_verify = None
    if verify:
        if training_env:
            training_verify, _ = _verify_envelope(app, training_env)
            training_env["last_verification"] = training_verify
            app.state.lifecycle_store.update(training_env["record"]["event_id"], training_env)
        if registration_env:
            registration_verify, _ = _verify_envelope(app, registration_env)
            registration_env["last_verification"] = registration_verify
            app.state.lifecycle_store.update(registration_env["record"]["event_id"], registration_env)

    # Prediction summary
    predictions = app.state.store.list_all()
    model_predictions = [
        p for p in predictions
        if p.get("record", {}).get("model_name") == model_name
        and str(p.get("record", {}).get("model_version")) == str(version)
    ]
    anchored_count = sum(1 for p in model_predictions if p.get("arweave_tx_id"))
    verified_count = sum(
        1 for p in model_predictions
        if p.get("last_verification", {}).get("hash_valid")
        and p.get("last_verification", {}).get("permanent_copy_found")
    )

    # Turbo status for each
    training_turbo = None
    registration_turbo = None
    if training_env and training_env.get("arweave_tx_id"):
        training_turbo = app.state.anchor.check_status(training_env["arweave_tx_id"])
    if registration_env and registration_env.get("arweave_tx_id"):
        registration_turbo = app.state.anchor.check_status(registration_env["arweave_tx_id"])

    return templates.TemplateResponse(
        request,
        "model_chain.html",
        {
            **_common_context(app),
            "model_name": model_name,
            "version": version,
            "training": training_env,
            "training_local": training_local,
            "training_turbo": training_turbo,
            "registration": registration_env,
            "registration_local": registration_local,
            "registration_turbo": registration_turbo,
            "prediction_count": len(model_predictions),
            "anchored_count": anchored_count,
            "verified_count": verified_count,
        },
    )
