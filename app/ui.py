import logging
import os
from datetime import datetime, timezone

import mlflow
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ario_mlflow.proof import canonical_json, hash_data
from ario_mlflow.verify import full_verify as _plugin_full_verify

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _common_context(app):
    """Shared template context for all pages (status bar, model info)."""
    return {
        "model_info": app.state.model_info,
        "arweave_enabled": app.state.anchor.enabled if app.state.anchor else False,
        "ario_verify_enabled": app.state.ario_verify.enabled if app.state.ario_verify else False,
    }


def _is_fully_verified(verification: dict | None) -> bool:
    """Full-verification gate shared by lifecycle status and aggregates.

    Prefers the plugin's event-aware ``overall`` verdict when present
    (it correctly handles "source_of_truth not applicable for
    predictions" and "ar.io below threshold"). Falls back to the
    explicit four-field check for legacy entries written before
    ``overall`` was persisted.

    Returns ``True`` only when every applicable check passed. Missing
    or pending checks (``None``) mean "not yet verified", not "failed".
    """
    if not verification:
        return False
    overall = verification.get("overall")
    if overall is True:
        return True
    if overall is False:
        return False
    # Legacy fallback: every required field must be explicitly True.
    return bool(
        verification.get("signature_valid") is True
        and verification.get("permanent_copy_found") is True
        and verification.get("hash_match") is True
    )


def _verify_envelope(app, envelope):
    """Verify via the plugin's full_verify against the pure-commitment
    envelope on Arweave.

    The lifecycle_store / RecordStore envelope (passed in) is the demo's
    local display cache; the actual proof being verified is the plugin's
    pure-commitment envelope on Arweave at ``envelope["arweave_tx_id"]``.
    This helper:

    1. Fetches the plugin envelope from Arweave by TX.
    2. Runs ``ario_mlflow.verify.full_verify`` on it (signature +
       anchored bytes from MLflow + live MLflow re-derivation +
       ar.io Verify attestation).
    3. Maps the four-check result to legacy field names so existing
       templates keep working, plus surfaces the plugin's
       ``source_of_truth`` and ``overall`` verdicts directly so the UI
       can render the live-MLflow-re-derivation check (the demo's
       headline tamper-detection signal) and treat predictions
       (which have no re-derivable source) distinctly from
       training/registration.

    Pending semantics: if the TX is set but Arweave hasn't returned the
    envelope yet, every check field is ``None`` — templates show
    "Not checked"/"Pending", not "FAIL". A ``False`` value in this
    dict only ever means "the check ran and failed".
    """
    # Default skeleton — None means "not yet checked" (pending). A
    # False value only appears once a check has actually run and
    # returned a negative result.
    result = {
        "signature_valid": None,
        "permanent_copy_found": False,
        "hash_match": None,
        "source_of_truth_ok": None,
        "source_of_truth_reason": None,
        "overall": None,
        "attestation_level": None,
        "report_url": None,
        "attested_by": None,
        "attested_at": None,
        "plugin_full_verify": None,
    }

    tx_id = envelope.get("arweave_tx_id")
    if not tx_id:
        return result

    plugin_envelope = app.state.anchor.fetch_proof(tx_id)
    if not plugin_envelope:
        # Pending: TX exists but the gateway hasn't indexed it yet, or
        # a transient fetch error. Leave check fields as None so the
        # UI shows "Pending", not "FAIL".
        return result

    # Inject TX so verify_ario_attestation can call ar.io Verify.
    plugin_envelope["_tx_id"] = tx_id

    # Plugin's full_verify needs the demo's MlflowClient for check 2
    # (download payload.json) and check 3 (re-derive from live state).
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    full_result = _plugin_full_verify(
        plugin_envelope,
        proof_engine=app.state.proof_engine,
        mlflow_client=mlflow_client,
        ario_client=app.state.ario_verify,
    )

    # Map the four-check structure to legacy field names + surface the
    # source_of_truth and overall verdicts the templates need to render
    # the headline tamper-detection lesson honestly.
    sig = full_result.get("signature", {}) or {}
    anchored = full_result.get("anchored_bytes", {}) or {}
    sot = full_result.get("source_of_truth", {}) or {}
    ario = full_result.get("ario_attestation", {}) or {}

    result["signature_valid"] = sig.get("ok")
    result["permanent_copy_found"] = anchored.get("payload_bytes") is not None
    result["hash_match"] = anchored.get("ok")
    # source_of_truth.ok is None for predictions (no re-derivable
    # MLflow state) and True/False for training/registration. The
    # UI treats None as "Not applicable" for predictions.
    result["source_of_truth_ok"] = sot.get("ok")
    result["source_of_truth_reason"] = sot.get("reason")
    result["overall"] = full_result.get("overall")
    result["attestation_level"] = ario.get("attestation_level")
    result["report_url"] = ario.get("report_url")
    result["attested_by"] = ario.get("attested_by")
    result["attested_at"] = ario.get("attested_at")
    result["plugin_full_verify"] = full_result

    return result


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
        if _is_fully_verified(training_env.get("last_verification")):
            training_status = "verified"
        elif training_env.get("arweave_tx_id"):
            training_status = "anchored"
        else:
            training_status = "local"

    registration_status = "none"
    if registration_env:
        if _is_fully_verified(registration_env.get("last_verification")):
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

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
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
            if _is_fully_verified(env.get("last_verification")):
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

    # Phase 2.C: verify via the plugin's full_verify against the on-chain
    # envelope — same pattern as the run_detail / model_chain paths and
    # /verify/{decision_id} (app/main.py). Fetches the new pure-commitment
    # envelope from Arweave and runs signature + anchored-bytes + ar.io
    # checks. Source-of-truth check is None for predictions per the
    # plugin's design (predictions don't have re-derivable MLflow state
    # beyond the anchored payload itself).
    #
    # The legacy ``proof_engine.verify_local(envelope)`` "always-on local
    # cache integrity" check was removed in Phase 2.C. It was the
    # mechanism behind today's single-button tamper detection (modify
    # local cache → local hash differs). Per redesign Part 8 + the
    # design principle "MLflow is the system of record, the cache is
    # just a UI display", local-cache integrity is not part of the trust
    # model. Phase 3 reintroduces tamper UX with four buttons paired to
    # the four real checks (signature / anchored bytes / live MLflow /
    # ar.io). Tracked as task #42.
    if verify and envelope.get("arweave_tx_id"):
        result = _verify_envelope(app, envelope)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        # plugin_full_verify carries raw payload_bytes; not JSON-serializable.
        # Mapped legacy fields are sufficient for cached display.
        persistable = {k: v for k, v in result.items() if k != "plugin_full_verify"}
        envelope["last_verification"] = persistable
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
            # local_verification kept as None for template back-compat; the
            # template's "fall back to local" branches now show "Not
            # checked" instead. Removed entirely once template is
            # fully migrated in Phase 3.
            "local_verification": None,
            "turbo_status": turbo_status,
        },
    )


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, verify: bool = False):
    app = request.app

    envelope = app.state.lifecycle_store.get_by_run_id(run_id)
    if not envelope:
        return HTMLResponse("<h1>Training run not found</h1>", status_code=404)

    if verify and envelope.get("arweave_tx_id"):
        result = _verify_envelope(app, envelope)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        # ``plugin_full_verify`` carries raw ``payload_bytes`` which
        # aren't JSON-serializable. Drop it from the persisted form;
        # the mapped legacy fields above are sufficient for cached
        # display. A re-verify recomputes the full structure on demand.
        persistable = {k: v for k, v in result.items() if k != "plugin_full_verify"}
        envelope["last_verification"] = persistable
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
        _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = _mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        mlflow_tags = dict(run.data.tags)
    except Exception as e:
        # Log and degrade to an empty tag set so the page still renders,
        # but don't let a tracking-store outage masquerade as "tagless run".
        logger.warning(
            "MLflow live-tag lookup failed for run %s: %s", run_id, e
        )
        mlflow_tags = {}

    ario_tags = {k: v for k, v in sorted(mlflow_tags.items()) if k.startswith("ario.")}

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            "turbo_status": turbo_status,
            "mlflow_ario_tags": ario_tags,
            # Pass the configured URI verbatim — evaluators run the
            # suggested command from the repo root, where relative
            # tracking URIs like "mlruns" resolve. Exposing the server's
            # absolute path (e.g. "/app/mlruns" on Railway) leaks
            # deployment detail and isn't useful to the reader.
            "mlflow_tracking_uri": app.state.settings.mlflow_tracking_uri,
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

    # Full verification (on-demand)
    training_verify = None
    registration_verify = None
    if verify:
        if training_env:
            training_verify = _verify_envelope(app, training_env)
            # Strip non-JSON-serializable raw bytes from plugin_full_verify
            # before persisting (see run_detail comment).
            training_env["last_verification"] = {
                k: v for k, v in training_verify.items() if k != "plugin_full_verify"
            }
            app.state.lifecycle_store.update(training_env["record"]["event_id"], training_env)
        if registration_env:
            registration_verify = _verify_envelope(app, registration_env)
            registration_env["last_verification"] = {
                k: v for k, v in registration_verify.items() if k != "plugin_full_verify"
            }
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
        if _is_fully_verified(p.get("last_verification"))
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
            "training_turbo": training_turbo,
            "registration": registration_env,
            "registration_turbo": registration_turbo,
            "prediction_count": len(model_predictions),
            "anchored_count": anchored_count,
            "verified_count": verified_count,
        },
    )
