import json
import logging
import os
from datetime import datetime, timezone

import mlflow
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ario_mlflow.proof import canonical_json, hash_data
from ario_mlflow.verify import verify_envelope
from ario_mlflow.anchoring import verify_run_artifact_integrity

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _common_context(app):
    """Shared template context for all pages (status bar, model info)."""
    return {
        "model_info": app.state.model_info or {},
        "arweave_enabled": app.state.anchor.enabled if app.state.anchor else False,
        "ario_verify_enabled": app.state.ario_verify.enabled if app.state.ario_verify else False,
    }


def _is_fully_verified(verification: dict | None) -> bool:
    """Full-verification gate shared by lifecycle status and aggregates.

    Treats a record as verified only when the local hash matches, the
    signature is valid, the permanent copy was found on Arweave, and
    the on-chain hash matches our local record hash. Missing any one
    (including still-propagating ``permanent_copy_found``) means the
    record is not yet verified — not that it failed.
    """
    if not verification:
        return False
    return bool(
        verification.get("hash_valid")
        and verification.get("signature_valid")
        and verification.get("permanent_copy_found")
        and verification.get("hash_match")
    )


def _envelope_from_arweave(app, tx_id: str | None) -> dict | None:
    """Materialize a template-friendly envelope by fetching the proof from Arweave.

    Returns None when no tx is anchored yet. The fetched envelope already has
    record, record_hash, signature, public_key (signed by the plugin). We add
    arweave_tx_id and arweave_url so existing templates keep working.
    """
    if not tx_id or not app.state.anchor or not app.state.anchor.enabled:
        return None
    proof = app.state.anchor.fetch_proof(tx_id)
    if not proof:
        return None
    return {
        **proof,
        "arweave_tx_id": tx_id,
        "arweave_url": f"https://{app.state.settings.ario_gateway_host}/raw/{tx_id}",
    }


def _escape_mlflow_filter_value(value: str) -> str:
    """Escape backslashes and single quotes in MLflow filter string values.

    MLflow filter expressions are SQL-like, so user-derived values
    (decision_id, model_name, version) interpolated into filter_string can
    break the parse or alter matching if they contain quotes. Backslash is
    escaped first to avoid double-escaping the escape itself.
    """
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _envelope_from_trace(trace_info) -> dict:
    """Build a partial envelope from an MLflow TraceInfo for DISPLAY ONLY.

    The trace tags carry pointers (decision_id, arweave_tx, record_hash) and
    display data (display_prediction_json, receipt_json), but they CANNOT
    reproduce the exact bytes that were signed (latency_ms and other signed
    fields aren't fully reconstructable from tags). So this envelope is
    marked ``locally_verifiable=False`` — callers must not run verify_local
    on it. To verify, fetch the canonical proof from Arweave by tx_id.
    """
    tags = dict(getattr(trace_info, "tags", {}) or {})
    request_time_ms = getattr(trace_info, "request_time", None) or getattr(trace_info, "timestamp_ms", None)
    timestamp_iso: str | None = None
    if request_time_ms is not None:
        timestamp_iso = datetime.fromtimestamp(
            request_time_ms / 1000.0, tz=timezone.utc
        ).isoformat()

    rich_prediction: dict | None = None
    raw_pred_json = tags.get("ario.display_prediction_json")
    if raw_pred_json:
        try:
            rich_prediction = json.loads(raw_pred_json)
        except (ValueError, TypeError):
            rich_prediction = None

    turbo_receipt: dict | None = None
    raw_receipt_json = tags.get("ario.receipt_json")
    if raw_receipt_json:
        try:
            turbo_receipt = json.loads(raw_receipt_json)
        except (ValueError, TypeError):
            turbo_receipt = None

    mlflow_trace_id = getattr(trace_info, "trace_id", None) or getattr(trace_info, "request_id", None)
    model_uri = tags.get("ario.model_uri")
    if not model_uri and tags.get("ario.model_name") and tags.get("ario.model_version"):
        model_uri = f"models:/{tags['ario.model_name']}/{tags['ario.model_version']}"

    # Note: this record is structurally incomplete (missing latency_ms,
    # artifact_verified, etc.) and MUST NOT be fed to verify_local. It's
    # populated for display only.
    record = {
        "decision_id": tags.get("ario.decision_id"),
        "event_type": "prediction",
        "timestamp": timestamp_iso,
        "model_name": tags.get("ario.model_name"),
        "model_version": tags.get("ario.model_version"),
        "run_id": tags.get("ario.run_id") or tags.get("mlflow.run_id"),
        "model_uri": model_uri,
        "input_hash": tags.get("ario.input_hash"),
        "output_hash": tags.get("ario.output_hash"),
    }
    return {
        "record": record,
        "record_hash": tags.get("ario.record_hash"),
        "public_key": tags.get("ario.public_key"),
        "arweave_tx_id": tags.get("ario.arweave_tx"),
        "arweave_url": tags.get("ario.arweave_url"),
        "proof_status": tags.get("ario.proof_status"),
        "turbo_receipt": turbo_receipt,
        "display_prediction": rich_prediction,
        "display_trace_id": mlflow_trace_id,
        # Tells decision_detail not to compute hash_valid against this envelope.
        "locally_verifiable": False,
    }


def _compute_row_status(app, envelope: dict) -> str:
    """Compute the dashboard row's badge status from cached verification state.

    Returns one of: 'verified', 'tampered', 'anchored', 'anchoring', 'local'.

    Pure cache-read function — never triggers an Arweave fetch on render.
    Cache misses return 'anchored' (yellow). Status flips to
    'verified'/'tampered' once the cache is populated by /predict,
    the live-verify button, or the tamper/untamper hooks.

    State machine:
        no arweave_tx_id → 'anchoring' if Arweave is enabled, else 'local'
        decision_verify_cache[decision_id].overall_match is False → 'tampered'
        ario_verify_cache has attestation_level for this tx
            AND prediction_integrity passed → 'verified'
        cache miss → 'anchored'  (row is "not yet verified")

    Notes:
    - 'tampered' wins over the cache: an attestation says "the bytes on
      Arweave are intact"; it doesn't speak to whether MLflow's recorded
      inputs match those bytes. Prediction integrity DOES.
    - We do not call ario_verify.submit_verification from here — only
      count cache hits — so the dashboard never hammers ar.io. Rows
      transition anchored→verified after a user has visited the detail
      page (or clicked the live-verify button) for that decision.
    """
    tx_id = envelope.get("arweave_tx_id")
    if not tx_id:
        arweave_enabled = bool(app.state.anchor and app.state.anchor.enabled)
        return "anchoring" if arweave_enabled else "local"

    decision_id = (envelope.get("record") or {}).get("decision_id")
    if not decision_id:
        return "anchored"

    # Prediction integrity: pure cache read, no fetch on miss.
    pi_cache = getattr(app.state, "decision_verify_cache", None)
    pi = pi_cache.get(decision_id) if pi_cache is not None else None

    if pi and pi.get("overall_match") is False:
        return "tampered"

    # Verified requires both prediction integrity passing AND a recorded
    # ar.io attestation in the cache (populated when the user visits
    # detail page or clicks live-verify).
    att_cache = getattr(app.state, "ario_verify_cache", {}) or {}
    attestation = att_cache.get(tx_id)
    if (
        pi
        and pi.get("overall_match") is True
        and attestation
        and attestation.get("attestation_level") in (1, 2, 3)
    ):
        return "verified"

    return "anchored"


def _evaluate_chain_status(envs: list, lifecycle: dict | None, decision_results: list) -> dict:
    """Given a chain's decisions + cached model lifecycle + cached per-decision
    integrity, evaluate the chain's overall status.

    Returns ``{intact, broken_reason, broken_at, broken_decision_id}``.
    ``intact`` is True/False/None; None means "verification pending"
    (cache miss for the lifecycle — the daemon thread or a state-change
    event hasn't populated this entry yet).
    """
    if lifecycle is None:
        return {
            "intact": None,
            "broken_reason": "verification_pending",
            "broken_at": None,
            "broken_decision_id": None,
        }
    if lifecycle.get("error"):
        return {
            "intact": False,
            "broken_reason": "lifecycle_verification_error",
            "broken_at": None,
            "broken_decision_id": None,
        }
    if lifecycle.get("all_intact") is False:
        return {
            "intact": False,
            "broken_reason": "lifecycle_broken",
            "broken_at": None,
            "broken_decision_id": None,
        }
    # Lifecycle intact (or all_intact is None, treat as pending).
    if lifecycle.get("all_intact") is None:
        return {
            "intact": None,
            "broken_reason": "verification_pending",
            "broken_at": None,
            "broken_decision_id": None,
        }

    # Walk decisions, check chain links + per-decision integrity.
    valid_roots = set()
    if lifecycle.get("registration_tx"):
        valid_roots.add(lifecycle["registration_tx"])
    if lifecycle.get("promotion_tx"):
        valid_roots.add(lifecycle["promotion_tx"])

    prior_record_hash = None
    for i, (env, dr) in enumerate(zip(envs, decision_results)):
        decision_id = (env.get("record") or {}).get("decision_id")
        previous_hash = env.get("previous_hash")

        # Check chain link.
        if i == 0:
            if previous_hash not in valid_roots and previous_hash != "GENESIS":
                return {
                    "intact": False,
                    "broken_reason": "orphan_root",
                    "broken_at": i,
                    "broken_decision_id": decision_id,
                }
        else:
            if previous_hash != prior_record_hash:
                return {
                    "intact": False,
                    "broken_reason": "link_mismatch",
                    "broken_at": i,
                    "broken_decision_id": decision_id,
                }

        # Check per-decision integrity (if cached).
        if dr is not None and dr.get("overall_match") is False:
            return {
                "intact": False,
                "broken_reason": "decision_tampered",
                "broken_at": i,
                "broken_decision_id": decision_id,
            }

        prior_record_hash = env.get("record_hash")

    return {
        "intact": True,
        "broken_reason": None,
        "broken_at": None,
        "broken_decision_id": None,
    }


def _list_recent_decisions(app, max_results: int = 50, compute_status: bool = True) -> list[dict]:
    """Return recent decision envelopes by querying MLflow traces.

    Args:
        max_results: Cap on traces returned. Tighten this on the dashboard
            (default 50) to bound the cost of per-row verification.
        compute_status: When True (default), each envelope gets a
            ``row_status`` field populated by :func:`_compute_row_status`.
            Set False from callers like chain-integrity that walk many
            traces and don't need badge state — keeps that path fast.
    """
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        # Search across all experiments (in single-experiment demo, there's just "Default").
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string="tags.`ario.decision_id` != ''",
            max_results=max_results,
            order_by=["timestamp DESC"],
        )
    except Exception as e:
        logger.warning(f"MLflow trace search failed: {e}")
        return []

    envelopes = []
    for trace in traces:
        envelopes.append(_envelope_from_trace(trace.info))

    if compute_status:
        for env in envelopes:
            env["row_status"] = _compute_row_status(app, env)

    return envelopes


def _decision_envelope_by_id(app, decision_id: str) -> dict | None:
    """Look up a single decision's envelope by decision_id.

    Tries trace search first; if the trace's arweave_tx is present, fetches the
    canonical envelope from Arweave for full verification. Falls back to the
    trace-tag envelope when the prediction isn't anchored yet.
    """
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=f"tags.`ario.decision_id` = '{_escape_mlflow_filter_value(decision_id)}'",
            max_results=1,
        )
    except Exception as e:
        logger.warning(f"Trace lookup for decision_id={decision_id} failed: {e}")
        return None

    if not traces:
        return None

    trace_info = traces[0].info
    trace_envelope = _envelope_from_trace(trace_info)
    tags = dict(getattr(trace_info, "tags", {}) or {})
    tx_id = tags.get("ario.arweave_tx")
    if tx_id:
        from_chain = _envelope_from_arweave(app, tx_id)
        if from_chain:
            # Canonical proof from Arweave: record stays untouched (locally
            # verifiable). Carry display-only fields from the trace as
            # envelope-level siblings.
            from_chain = dict(from_chain)
            from_chain["locally_verifiable"] = True
            if trace_envelope.get("turbo_receipt") and not from_chain.get("turbo_receipt"):
                from_chain["turbo_receipt"] = trace_envelope["turbo_receipt"]
            if trace_envelope.get("display_prediction"):
                from_chain["display_prediction"] = trace_envelope["display_prediction"]
            if trace_envelope.get("display_trace_id"):
                from_chain["display_trace_id"] = trace_envelope["display_trace_id"]
            return from_chain
    return trace_envelope


@router.get("/ui/predictions")
def predictions_redirect():
    """Permanent redirect from the old URL. Bookmarks keep working."""
    return RedirectResponse("/ui/decisions", status_code=301)


@router.get("/ui/decisions", response_class=HTMLResponse)
def decisions(request: Request):
    app = request.app
    records = _list_recent_decisions(app)
    model_info = app.state.model_info or {}

    training_status = "none"
    registration_status = "none"
    if model_info.get("model_name") and model_info.get("model_version"):
        chain = app.state.ario_client.lifecycle_for_model(
            model_info["model_name"], version=model_info["model_version"]
        )
        for event in chain:
            if event["event_type"] == "training_complete" and event.get("tx_id"):
                training_status = "anchored"
            elif event["event_type"] == "model_registered" and event.get("tx_id"):
                registration_status = "anchored"

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
    active_version = (app.state.model_info or {}).get("model_version", "")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    version_data = []

    for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
        accuracy = None
        created = None
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                accuracy = run.data.metrics.get("accuracy")
                created = run.info.start_time
            except Exception:
                pass

        # Read chain status from the plugin instead of LifecycleStore.
        chain = app.state.ario_client.lifecycle_for_model(model_name, version=str(mv.version))
        training_status = "none"
        registration_status = "none"
        for event in chain:
            if event["event_type"] == "training_complete" and event.get("tx_id"):
                training_status = "anchored"
            elif event["event_type"] == "model_registered" and event.get("tx_id"):
                registration_status = "anchored"

        version_data.append({
            "version": str(mv.version),
            "run_id": mv.run_id or "",
            "accuracy": accuracy,
            "stage": mv.current_stage if hasattr(mv, "current_stage") else "None",
            "training_status": training_status,
            "registration_status": registration_status,
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


def _compute_decision_verification(app, envelope: dict, decision_id: str) -> dict:
    """Compute the full verification context for a decision.

    Single source of truth for what 'verify a decision' means: used by
    both the page-load route AND the live-verify endpoint, so the two
    cannot drift. Reads from app.state.proof_engine, .anchor,
    .ario_verify, .ario_verify_cache, and .settings.

    Returns a flat dict with these keys (each value may be None):
        local_verification:   {hash_valid, signature_valid, overall} | None
        anchored_on_arweave:  bool
        attestation:          {attestation_level, report_url, pdf_url,
                              attested_by, attested_at} | None
        prediction_integrity: {match_input, match_output, overall_match,
                              anchored_input_hash, recomputed_input_hash,
                              anchored_output_hash, recomputed_output_hash,
                              error} | None
        turbo_status:         dict | None  (from anchor.check_status)
        verified_at:          ISO-8601 string of the moment this ran
    """
    verified_at = datetime.now(timezone.utc).isoformat()

    # Local checks (always cheap; trace-tag envelopes are flagged
    # locally_verifiable=False and skip this).
    local = None
    if (
        envelope.get("locally_verifiable")
        and envelope.get("signature")
        and envelope.get("public_key")
    ):
        local = app.state.proof_engine.verify_local(envelope)

    # Arweave reachability — already implied by locally_verifiable=True
    # (we only set that flag when the envelope came from a successful
    # _envelope_from_arweave fetch).
    anchored_on_arweave = bool(
        envelope.get("locally_verifiable") and envelope.get("arweave_tx_id")
    )

    # ar.io Verify attestation, cached per tx_id.
    attestation = None
    tx_id = envelope.get("arweave_tx_id")
    if tx_id and app.state.ario_verify.enabled:
        cache = app.state.ario_verify_cache
        if tx_id in cache:
            attestation = cache[tx_id]
        else:
            try:
                normalized = app.state.ario_verify.submit_verification(tx_id)
                if normalized:
                    attestation = {
                        "attestation_level": normalized.get("attestation_level"),
                        "report_url": normalized.get("report_url"),
                        "pdf_url": normalized.get("pdf_url"),
                        "attested_by": normalized.get("attested_by"),
                        "attested_at": normalized.get("attested_at"),
                    }
                    cache[tx_id] = attestation
            except Exception as e:
                logger.warning(f"ar.io Verify failed for {tx_id}: {e}")

    # Prediction integrity — re-derive from MLflow span data, compare to
    # anchored proof. The meaningful MLflow ↔ Arweave check.
    prediction_integrity = None
    if envelope.get("arweave_tx_id"):
        try:
            from ario_mlflow.decision_verify import verify_prediction
            result = verify_prediction(
                decision_id=decision_id,
                tracking_uri=app.state.settings.mlflow_tracking_uri,
                arweave=app.state.anchor,
            )
            prediction_integrity = {
                "match_input": result.match_input,
                "match_output": result.match_output,
                "overall_match": result.overall_match,
                "anchored_input_hash": result.anchored_input_hash,
                "recomputed_input_hash": result.recomputed_input_hash,
                "anchored_output_hash": result.anchored_output_hash,
                "recomputed_output_hash": result.recomputed_output_hash,
                "error": result.error,
            }
        except Exception as e:
            logger.warning(f"verify_prediction failed for {decision_id}: {e}")
            prediction_integrity = {"error": str(e)}

    turbo_status = None
    if envelope.get("arweave_tx_id"):
        try:
            turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])
        except Exception as e:
            logger.debug(f"check_status failed for {envelope['arweave_tx_id']}: {e}")

    return {
        "local_verification": local,
        "anchored_on_arweave": anchored_on_arweave,
        "attestation": attestation,
        "prediction_integrity": prediction_integrity,
        "turbo_status": turbo_status,
        "verified_at": verified_at,
    }


@router.get("/ui/decisions/{decision_id}", response_class=HTMLResponse)
def decision_detail(request: Request, decision_id: str):
    app = request.app
    envelope = _decision_envelope_by_id(app, decision_id)
    if not envelope:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    verification = _compute_decision_verification(app, envelope, decision_id)

    return templates.TemplateResponse(
        request,
        "decision_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            **verification,  # local_verification, anchored_on_arweave, attestation, prediction_integrity, turbo_status, verified_at
        },
    )


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, verify: bool = False):
    app = request.app

    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        return HTMLResponse("<h1>Training run not found</h1>", status_code=404)

    mlflow_tags = dict(run.data.tags)
    ario_tags = {k: v for k, v in sorted(mlflow_tags.items()) if k.startswith("ario.")}

    training_tx = mlflow_tags.get("ario.training_tx")
    envelope = _envelope_from_arweave(app, training_tx) if training_tx else None
    if envelope is None:
        # Offline/no-anchor fallback — the page still renders the MLflow tags so
        # evaluators can see ario.* lifecycle metadata even when there's no
        # Arweave proof to fetch (e.g., dev mode without a wallet).
        envelope = {
            "record": {
                "event_type": "training_complete",
                "run_id": run_id,
                "model_name": mlflow_tags.get("ario.model_name"),
                "model_version": mlflow_tags.get("ario.model_version"),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                # Artifact integrity fields — sourced from tags where available.
                "artifact_hash": mlflow_tags.get("ario.artifact_hash"),
                "artifact_checksums": None,
                "git_commit": mlflow_tags.get("ario.git_commit"),
                # event_id used by the JS polling loop in run_detail.html.
                "event_id": mlflow_tags.get("ario.event_id"),
            },
            "record_hash": mlflow_tags.get("ario.record_hash"),
            "public_key": mlflow_tags.get("ario.public_key"),
            "arweave_tx_id": training_tx,
            "arweave_url": None,
            "turbo_receipt": None,
        }

    local = None
    if envelope.get("signature") and envelope.get("public_key"):
        local = app.state.proof_engine.verify_local(envelope)

    if verify and envelope.get("arweave_tx_id"):
        result = verify_envelope(envelope, app.state.proof_engine, app.state.anchor, app.state.ario_verify)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        envelope["last_verification"] = result

    turbo_status = None
    if envelope.get("arweave_tx_id"):
        turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])

    artifact_integrity = verify_run_artifact_integrity(
        run_id,
        (envelope.get("record") or {}).get("artifact_hash"),
    )

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            "local_verification": local,
            "turbo_status": turbo_status,
            "mlflow_ario_tags": ario_tags,
            "mlflow_tracking_uri": app.state.settings.mlflow_tracking_uri,
            "artifact_integrity": artifact_integrity,
        },
    )


@router.get("/ui/models/{model_name}/{version}", response_class=HTMLResponse)
def model_chain(request: Request, model_name: str, version: str, verify: bool = False):
    app = request.app

    chain = app.state.ario_client.lifecycle_for_model(model_name, version=version)
    # Index by event_type for the existing template's variable shape.
    by_type = {e["event_type"]: e for e in chain}

    training_event = by_type.get("training_complete")
    registration_event = by_type.get("model_registered")
    dataset_event = by_type.get("dataset_anchored")
    promotion_event = by_type.get("stage_transition")

    gateway_host = app.state.settings.ario_gateway_host
    if dataset_event and dataset_event.get("tx_id"):
        dataset_event = {
            **dataset_event,
            "arweave_url": f"https://{gateway_host}/raw/{dataset_event['tx_id']}",
        }
    if promotion_event and promotion_event.get("tx_id"):
        promotion_event = {
            **promotion_event,
            "arweave_url": f"https://{gateway_host}/raw/{promotion_event['tx_id']}",
        }

    # Materialize template-shape envelopes by fetching from Arweave.
    training_env = _envelope_from_arweave(app, training_event["tx_id"]) if training_event else None
    registration_env = _envelope_from_arweave(app, registration_event["tx_id"]) if registration_event else None

    training_local = app.state.proof_engine.verify_local(training_env) if training_env else None
    registration_local = app.state.proof_engine.verify_local(registration_env) if registration_env else None

    training_verify = None
    registration_verify = None
    if verify:
        if training_env:
            training_verify = verify_envelope(training_env, app.state.proof_engine, app.state.anchor, app.state.ario_verify)
            training_env["last_verification"] = training_verify
        if registration_env:
            registration_verify = verify_envelope(registration_env, app.state.proof_engine, app.state.anchor, app.state.ario_verify)
            registration_env["last_verification"] = registration_verify

    # Prediction summary — query traces tagged with this model.
    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string=(
                f"tags.`ario.model_name` = '{_escape_mlflow_filter_value(model_name)}' "
                f"and tags.`ario.model_version` = '{_escape_mlflow_filter_value(version)}'"
            ),
            max_results=200,
        )
    except Exception as e:
        logger.warning(f"Trace search failed for {model_name}/v{version}: {e}")
        traces = []

    prediction_count = len(traces)
    anchored_count = sum(
        1 for t in traces
        if (getattr(t.info, "tags", {}) or {}).get("ario.arweave_tx")
    )
    # Verified count is on-demand only; without iterating Arweave we can't compute.
    verified_count = 0

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
            "prediction_count": prediction_count,
            "anchored_count": anchored_count,
            "verified_count": verified_count,
            # NEW context keys for templates that want to surface dataset/promotion links.
            "dataset_event": dataset_event,
            "promotion_event": promotion_event,
        },
    )
