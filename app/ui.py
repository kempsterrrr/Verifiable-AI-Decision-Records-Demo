import logging
import os
from datetime import datetime, timezone

import mlflow
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from ario_mlflow.proof import canonical_json, hash_data
from ario_mlflow.verify import verify_proof_by_tx as _plugin_verify_proof_by_tx

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _common_context(app):
    """Shared template context for all pages (status bar, model info)."""
    settings = app.state.settings
    return {
        "model_info": app.state.model_info,
        "arweave_enabled": app.state.anchor.enabled if app.state.anchor else False,
        "ario_verify_enabled": app.state.ario_verify.enabled if app.state.ario_verify else False,
        # Surfaced so base.html can conditionally render the demo-admin
        # nav link and so other templates can gate demo-only UI.
        "demo_mode": settings.demo_mode,
        # The currently-active production model name + version. Templates
        # use this to badge any reference to a model version with
        # "Active" (matches) or "(replaced by vN)" (older). Coerced to
        # str so template equality doesn't trip on int-vs-str mismatches.
        "active_model_name": str(app.state.model_info.get("model_name", "")),
        "active_model_version": str(app.state.model_info.get("model_version", "")),
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
    """Verify via the plugin's ``verify_proof_by_tx`` against the
    pure-commitment envelope on Arweave.

    The lifecycle_store / RecordStore envelope (passed in) is the demo's
    local display cache; the actual proof being verified is the plugin's
    pure-commitment envelope on Arweave at ``envelope["arweave_tx_id"]``.

    ``verify_proof_by_tx`` collapses the previous fetch + full_verify
    pair into one call and returns ``proof_found`` so the demo's
    "Proof Found" UI row can distinguish "envelope retrieved from
    Arweave" from "envelope was missing" — what that row is supposed
    to express.

    Pending vs failure semantics:
    - No ``arweave_tx_id`` → ``proof_found=None`` (anchoring may still
      be in progress); other check fields stay at their defaults.
    - TX set but gateway returns no envelope → ``proof_found=False``
      and other checks remain ``None`` (the plugin doesn't run them
      when the fetch fails).
    - TX set and gateway returns the envelope → ``proof_found=True``
      and all four checks run.
    """
    # Default skeleton — None means "not yet checked" (pending). A
    # False value only appears once a check has actually run and
    # returned a negative result.
    result = {
        "proof_found": None,
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

    settings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    full_result = _plugin_verify_proof_by_tx(
        tx_id,
        anchor=app.state.anchor,
        proof_engine=app.state.proof_engine,
        mlflow_client=mlflow_client,
        ario_client=app.state.ario_verify,
    )

    result["proof_found"] = full_result.get("proof_found")
    result["plugin_full_verify"] = full_result

    # Map the four-check structure to legacy field names + surface the
    # source_of_truth and overall verdicts the templates need to render
    # the headline tamper-detection lesson honestly. When proof_found
    # is False, every sub-check returns ok=None and these mappings just
    # propagate the pending state.
    sig = full_result.get("signature", {}) or {}
    anchored = full_result.get("anchored_bytes", {}) or {}
    sot = full_result.get("source_of_truth", {}) or {}
    ario = full_result.get("ario_attestation", {}) or {}

    result["signature_valid"] = sig.get("ok")
    result["permanent_copy_found"] = anchored.get("payload_bytes") is not None
    result["hash_match"] = anchored.get("ok")
    # source_of_truth.ok is True/False for predictions, training, and
    # registration — the plugin re-derives canonical bytes from the
    # live trace tag (predictions) or run state (training/registration)
    # and compares to the anchored payload. None only appears when the
    # refetch can't complete (legacy event, missing trace tag, etc.).
    result["source_of_truth_ok"] = sot.get("ok")
    result["source_of_truth_reason"] = sot.get("reason")
    result["overall"] = full_result.get("overall")
    result["attestation_level"] = ario.get("attestation_level")
    result["report_url"] = ario.get("report_url")
    result["attested_by"] = ario.get("attested_by")
    result["attested_at"] = ario.get("attested_at")

    return result


@router.get("/")
def root_redirect():
    """Phase A nav swap: home routes into the verification chain at the
    input side, mirroring the new left-to-right reading order
    Datasets → Runs → Models → Decisions → Lineage."""
    return RedirectResponse("/ui/datasets", status_code=302)


@router.get("/ui/predictions")
def predictions_redirect():
    """Permanent redirect from the old URL. Bookmarks keep working."""
    return RedirectResponse("/ui/decisions", status_code=301)


@router.get("/ui/decisions", response_class=HTMLResponse)
def decisions(request: Request):
    app = request.app
    records = app.state.store.list_all()
    model_info = app.state.model_info
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False

    # Lifecycle status for provenance card
    training_env = app.state.lifecycle_store.get_by_run_id(model_info["run_id"])
    registration_env = app.state.lifecycle_store.get_by_model_version(
        model_info["model_name"], model_info["model_version"]
    )
    training_status = _envelope_status(training_env, arweave_enabled=arweave_enabled)
    registration_status = _envelope_status(registration_env, arweave_enabled=arweave_enabled)

    # Pre-compute the canonical status for each record so the template
    # doesn't have to. Aggregate counts for the filter chip strip and
    # attach a per-record status for the row badge.
    status_counts = {"verified": 0, "pending": 0, "anchoring": 0, "tampered": 0, "none": 0}
    for env in records:
        env["_status"] = _envelope_status(env, arweave_enabled=arweave_enabled)
        status_counts[env["_status"]] = status_counts.get(env["_status"], 0) + 1

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            **_common_context(app),
            "records": records,
            "status_counts": status_counts,
            "training_status": training_status,
            "registration_status": registration_status,
        },
    )


@router.get("/ui/models", response_class=HTMLResponse)
def model_registry(request: Request):
    app = request.app
    settings = app.state.settings
    model_name = settings.mlflow_model_name
    active_version = app.state.model_info["model_version"]
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False

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

        # Check lifecycle anchoring status using the canonical 5-state enum.
        training_env = app.state.lifecycle_store.get_by_run_id(mv.run_id) if mv.run_id else None
        reg_env = app.state.lifecycle_store.get_by_model_version(model_name, str(mv.version))

        version_data.append({
            "version": str(mv.version),
            "run_id": mv.run_id or "",
            "accuracy": accuracy,
            "stage": mv.current_stage if hasattr(mv, "current_stage") else "None",
            "training_status": _envelope_status(training_env, arweave_enabled=arweave_enabled),
            "registration_status": _envelope_status(reg_env, arweave_enabled=arweave_enabled),
            "is_active": str(mv.version) == str(active_version),
            "created": datetime.fromtimestamp(created / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if created else "",
        })

    return templates.TemplateResponse(
        request,
        "models_list.html",
        {
            **_common_context(app),
            "model_name": model_name,
            "versions": version_data,
            "active_version": active_version,
        },
    )


@router.get("/ui/registry")
def registry_redirect():
    return RedirectResponse("/ui/models", status_code=301)


def _build_chain_context(
    app, model_name: str, version: str, verify: bool = False
) -> dict:
    """Resolve all chain entities (datasets → run → registration →
    decisions) for a given model version. Shared by the Lineage page
    and the slim model detail page so they render the same data shape.

    When ``verify=True``, runs the four-check verification on every
    envelope in the chain and persists the result back to lifecycle
    store / record store. Also surfaces the training event's canonical
    bytes + signed commitment for the "How verification works" panel.
    """
    lifecycle_records = app.state.lifecycle_store.list_all()
    training_env = None
    registration_env = None

    for rec in lifecycle_records:
        r = rec.get("record", {})
        if (r.get("event_type") == "training_complete"
                and r.get("model_name") == model_name
                and str(r.get("model_version")) == str(version)):
            training_env = rec
        elif (r.get("event_type") == "model_registered"
                and r.get("model_name") == model_name
                and str(r.get("model_version")) == str(version)):
            registration_env = rec

    dataset_anchored_envs: list[dict] = []
    if training_env:
        target_run_id = training_env["record"]["run_id"]
        for rec in lifecycle_records:
            r = rec.get("record", {})
            if (r.get("event_type") == "dataset_anchored"
                    and r.get("source_run_id") == target_run_id):
                dataset_anchored_envs.append(rec)

    canonical_bytes_json = None
    signed_commitment_json = None

    if verify:
        if training_env:
            training_verify = _verify_envelope(app, training_env)
            training_env["last_verification"] = {
                k: v for k, v in training_verify.items() if k != "plugin_full_verify"
            }
            app.state.lifecycle_store.update(training_env["record"]["event_id"], training_env)

            try:
                import json as _json
                full = training_verify.get("plugin_full_verify") or {}
                anchored = full.get("anchored_bytes") or {}
                payload_bytes = anchored.get("payload_bytes")
                if payload_bytes:
                    try:
                        canonical_bytes_json = _json.dumps(
                            _json.loads(payload_bytes), indent=2
                        )
                    except Exception:
                        canonical_bytes_json = (
                            payload_bytes.decode("utf-8")
                            if isinstance(payload_bytes, (bytes, bytearray))
                            else str(payload_bytes)
                        )
                tx_id = training_env.get("arweave_tx_id")
                plugin_envelope = app.state.anchor.fetch_proof(tx_id) if tx_id else None
                if plugin_envelope:
                    signed_commitment_json = _json.dumps(plugin_envelope, indent=2)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "chain proof-viewer hydration failed for %s/v%s: %s",
                    model_name, version, e,
                )

        if registration_env:
            registration_verify = _verify_envelope(app, registration_env)
            registration_env["last_verification"] = {
                k: v for k, v in registration_verify.items() if k != "plugin_full_verify"
            }
            app.state.lifecycle_store.update(registration_env["record"]["event_id"], registration_env)

        for ds_env in dataset_anchored_envs:
            ds_verify = _verify_envelope(app, ds_env)
            ds_env["last_verification"] = {
                k: v for k, v in ds_verify.items() if k != "plugin_full_verify"
            }
            app.state.lifecycle_store.update(ds_env["record"]["event_id"], ds_env)

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

    training_turbo = None
    registration_turbo = None
    if training_env and training_env.get("arweave_tx_id"):
        training_turbo = app.state.anchor.check_status(training_env["arweave_tx_id"])
    if registration_env and registration_env.get("arweave_tx_id"):
        registration_turbo = app.state.anchor.check_status(registration_env["arweave_tx_id"])

    return {
        "training": training_env,
        "training_turbo": training_turbo,
        "registration": registration_env,
        "registration_turbo": registration_turbo,
        "dataset_anchored": dataset_anchored_envs,
        "prediction_count": len(model_predictions),
        "anchored_count": anchored_count,
        "verified_count": verified_count,
        "canonical_bytes_json": canonical_bytes_json,
        "signed_commitment_json": signed_commitment_json,
    }


def _describe_failed_checks(v: dict | None) -> list[str]:
    """Return human-readable descriptions of which verify checks failed.

    Used by the tamper page to label *what* about a link is broken
    (e.g., "Source-of-truth: live data diverges from anchored payload"),
    so the user can read the cryptographic propagation directly off the
    chain instead of just seeing a binary red/green flag.

    Returns an empty list when:
      - The envelope has no last_verification yet
      - All four checks pass
      - All cryptographic checks pass and only ar.io attestation is
        still propagating (the "Anchored, awaiting ar.io" sub-state)
    """
    if not v:
        return []
    overall = v.get("overall")
    if overall is True:
        return []

    sig_failed = v.get("signature_valid") is False
    hash_failed = v.get("hash_match") is False
    sot_failed = v.get("source_of_truth_ok") is False
    if not (sig_failed or hash_failed or sot_failed):
        # Crypto checks all pass; overall=False means ar.io attestation
        # is still propagating. Don't surface as a failure.
        return []

    failures: list[str] = []
    if sig_failed:
        failures.append("Signature invalid")
    if hash_failed:
        failures.append("Anchored bytes hash mismatch")
    if sot_failed:
        failures.append("Live data diverges from anchored payload")
    return failures


def _envelope_status(env: dict | None, arweave_enabled: bool = False) -> str:
    """Map a lifecycle/record envelope to the canonical 5-state status
    enum used everywhere in the UI:

    - ``verified``  — every check passed (or the legacy 4-field check)
    - ``pending``   — anchored on chain, not yet (or only partially)
                      re-verified. Includes the "Pending ar.io confirmation"
                      sub-state where cryptographic checks pass but ar.io's
                      attestation is still propagating.
    - ``anchoring`` — Arweave anchoring is enabled but the envelope has
                      no TX yet (in flight, not yet on chain)
    - ``tampered``  — a verify pass ran and a real check failed
    - ``none``      — no TX at all (legacy local-only or anchoring disabled)

    Templates render via the ``_status_badge`` macro so labels and colors
    stay in sync; nothing else should branch on these strings.
    """
    if not env:
        return "none"

    v = env.get("last_verification")
    if v:
        overall = v.get("overall")
        if overall is True:
            return "verified"
        if overall is False:
            # Distinguish "real fail" from "Pending ar.io confirmation".
            # Real fail: at least one check that actually ran returned
            # False. None means "not checked / not applicable" — for v1
            # dataset events hash_match and source_of_truth_ok are
            # intentionally None (deferred). Don't read those as failures.
            sig_failed = v.get("signature_valid") is False
            hash_failed = v.get("hash_match") is False
            sot_failed = v.get("source_of_truth_ok") is False
            if sig_failed or hash_failed or sot_failed:
                return "tampered"
            # All checks that ran passed; overall=False means ar.io
            # attestation is still propagating (level < 2).
            return "pending"
        # Legacy fallback for entries written before ``overall`` was
        # persisted: every required field must be explicitly True.
        if (
            v.get("signature_valid") is True
            and v.get("permanent_copy_found")
            and v.get("hash_match") is True
        ):
            return "verified"

    if env.get("arweave_tx_id"):
        return "pending"
    if arweave_enabled:
        return "anchoring"
    return "none"


@router.get("/ui/datasets", response_class=HTMLResponse)
def datasets_list(request: Request):
    """Datasets list — first stop in the verification chain.

    Aggregates dataset_anchored lifecycle entries by digest so each
    unique training dataset shows once with the count of runs that
    consumed it. The digest is the dataset's identity in the signed
    canonical payload, so it's also the URL key for detail pages.
    """
    app = request.app
    lifecycle = app.state.lifecycle_store.list_all()
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False

    # Group dataset_anchored events by digest. The latest entry's
    # arweave fields win for the "Anchor" column display — if a dataset
    # was re-anchored on a later run, we surface the most recent TX.
    grouped: dict[str, dict] = {}
    for env in lifecycle:
        rec = env.get("record", {}) or {}
        if rec.get("event_type") != "dataset_anchored":
            continue
        digest = rec.get("digest")
        if not digest:
            continue
        existing = grouped.get(digest)
        if existing is None:
            grouped[digest] = {
                "digest": digest,
                "name": rec.get("name", "—"),
                "source": rec.get("source", ""),
                "source_type": rec.get("source_type", ""),
                "schema_hash": rec.get("schema_hash", ""),
                "run_ids": set(),
                "latest_env": env,
                "latest_ts": rec.get("timestamp", ""),
            }
            existing = grouped[digest]
        run_id = rec.get("source_run_id")
        if run_id:
            existing["run_ids"].add(run_id)
        ts = rec.get("timestamp", "")
        if ts and ts > existing["latest_ts"]:
            existing["latest_env"] = env
            existing["latest_ts"] = ts

    rows = []
    for d in grouped.values():
        env = d["latest_env"]
        rows.append({
            "digest": d["digest"],
            "name": d["name"],
            "source": d["source"],
            "source_type": d["source_type"],
            "schema_hash": d["schema_hash"],
            "used_by_count": len(d["run_ids"]),
            "arweave_tx_id": env.get("arweave_tx_id"),
            "arweave_url": env.get("arweave_url"),
            "status": _envelope_status(env, arweave_enabled=arweave_enabled),
            "timestamp": d["latest_ts"],
        })
    rows.sort(key=lambda r: r["timestamp"], reverse=True)

    return templates.TemplateResponse(
        request,
        "datasets_list.html",
        {
            **_common_context(app),
            "datasets": rows,
        },
    )


@router.get("/ui/datasets/{digest}", response_class=HTMLResponse)
def dataset_detail(request: Request, digest: str, verify: bool = False):
    """Dataset detail — identity card, ar.io anchor link, runs that
    consumed this dataset, and (when ``verify=true``) the same
    plugin-driven four-check verification used by every other detail
    page. Phase B adds the verify card; Phase C will add the chain
    section and tamper card."""
    app = request.app
    lifecycle = app.state.lifecycle_store.list_all()
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False

    matches = [
        env for env in lifecycle
        if (env.get("record") or {}).get("event_type") == "dataset_anchored"
        and (env.get("record") or {}).get("digest") == digest
    ]
    if not matches:
        return HTMLResponse("<h1>Dataset not found</h1>", status_code=404)

    matches.sort(
        key=lambda e: (e.get("record") or {}).get("timestamp", ""),
        reverse=True,
    )
    primary = matches[0]
    rec = primary.get("record") or {}

    # Verify on demand. Same primitive as run_detail / decision_detail.
    # Persist the result so the next page render shows the latest
    # verdict without re-fetching from the gateway.
    if verify and primary.get("arweave_tx_id"):
        result = _verify_envelope(app, primary)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        persistable = {k: v for k, v in result.items() if k != "plugin_full_verify"}
        primary["last_verification"] = persistable
        app.state.lifecycle_store.update(primary["record"]["event_id"], primary)

    run_ids = sorted({
        (e.get("record") or {}).get("source_run_id")
        for e in matches
        if (e.get("record") or {}).get("source_run_id")
    })

    # Resolve each run's training envelope so we can show the model it
    # produced — gives the detail page an outbound link into the chain.
    used_by_runs = []
    for run_id in run_ids:
        run_env = app.state.lifecycle_store.get_by_run_id(run_id)
        if run_env:
            r = run_env.get("record") or {}
            used_by_runs.append({
                "run_id": run_id,
                "model_name": r.get("model_name", ""),
                "model_version": r.get("model_version", ""),
                "timestamp": r.get("timestamp", ""),
                "status": _envelope_status(run_env, arweave_enabled=arweave_enabled),
            })
        else:
            used_by_runs.append({
                "run_id": run_id,
                "model_name": "",
                "model_version": "",
                "timestamp": "",
                "status": "none",
            })

    # Recompute status post-verify so the badge in the editorial header
    # reflects the just-completed verification without a second
    # round-trip.
    return templates.TemplateResponse(
        request,
        "dataset_detail.html",
        {
            **_common_context(app),
            "dataset": {
                "digest": digest,
                "name": rec.get("name", "—"),
                "source": rec.get("source", ""),
                "source_type": rec.get("source_type", ""),
                "schema_hash": rec.get("schema_hash", ""),
                "payload_hash": rec.get("payload_hash", ""),
                "timestamp": rec.get("timestamp", ""),
                "arweave_tx_id": primary.get("arweave_tx_id"),
                "arweave_url": primary.get("arweave_url"),
                "status": _envelope_status(primary, arweave_enabled=arweave_enabled),
            },
            # Pass the lifecycle envelope at top level so the shared
            # _verify_card macro can read ``envelope.last_verification``
            # and ``envelope.arweave_tx_id`` the same way the other
            # detail pages do.
            "envelope": primary,
            "used_by_runs": used_by_runs,
        },
    )


@router.get("/ui/runs", response_class=HTMLResponse)
def runs_list(request: Request, dataset: str | None = None):
    """Training runs list — second stop in the verification chain.

    Reads training_complete events from lifecycle_store (one per run).
    Hosts the "Train & anchor" form (moved here from the Models page —
    this is where a new run is born). Each row links to the existing
    /ui/runs/{run_id} detail page.

    ``?dataset=<digest>`` preselects that dataset in the train hero's
    selector — used by the "Train a model with this dataset" CTA on
    each dataset detail page.
    """
    from app.model import _parse_synthetic_source, DEFAULT_DATASET_NAME

    app = request.app
    settings = app.state.settings
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
    lifecycle = app.state.lifecycle_store.list_all()

    rows = []
    # Group dataset_anchored events by digest for the train hero's
    # selector. Latest timestamp wins for display (mirrors /ui/datasets'
    # grouping). Standalone-seeded entries (no source_run_id) carry the
    # n_samples/seed columns; legacy per-run entries don't, so fall back
    # to re-parsing source. The dropdown shows the dataset's name plus
    # sample-count hint so the user can tell variants apart at a glance.
    dataset_by_digest: dict[str, dict] = {}
    for env in lifecycle:
        rec = env.get("record") or {}
        if rec.get("event_type") != "training_complete":
            continue
        metrics = rec.get("metrics", {}) or {}
        rows.append({
            "run_id": rec.get("run_id", ""),
            "model_name": rec.get("model_name", ""),
            "model_version": rec.get("model_version", ""),
            "accuracy": metrics.get("accuracy"),
            "dataset_count": len(rec.get("dataset_inputs", []) or []),
            "timestamp": rec.get("timestamp", ""),
            "arweave_tx_id": env.get("arweave_tx_id"),
            "arweave_url": env.get("arweave_url"),
            "status": _envelope_status(env, arweave_enabled=arweave_enabled),
        })
    for env in lifecycle:
        rec = env.get("record") or {}
        if rec.get("event_type") != "dataset_anchored":
            continue
        digest = rec.get("digest")
        if not digest:
            continue
        existing = dataset_by_digest.get(digest)
        ts = rec.get("timestamp", "")
        if existing is None or ts > existing.get("timestamp", ""):
            n = rec.get("n_samples")
            s = rec.get("seed")
            if n is None or s is None:
                parsed = _parse_synthetic_source(rec.get("source") or "")
                if parsed:
                    n, s = parsed
            dataset_by_digest[digest] = {
                "digest": digest,
                "name": rec.get("name") or "—",
                "n_samples": n,
                "seed": s,
                "timestamp": ts,
            }
    # Sort: the default seeded dataset first (so the dropdown's default
    # selection is the canonical one demos walk through), then others by
    # ``timestamp`` ascending (seeded variants before user-created ones).
    def _sort_key(d):
        is_default = 0 if d["name"] == DEFAULT_DATASET_NAME else 1
        return (is_default, d["timestamp"])
    available_datasets = sorted(dataset_by_digest.values(), key=_sort_key)
    # Only honour the query param when it actually matches a known
    # digest — otherwise fall through to the dropdown's first option.
    preselected_dataset_id = dataset if dataset in dataset_by_digest else None

    rows.sort(key=lambda r: r["timestamp"], reverse=True)

    return templates.TemplateResponse(
        request,
        "runs_list.html",
        {
            **_common_context(app),
            "runs": rows,
            "model_name": settings.mlflow_model_name,
            "available_datasets": available_datasets,
            "preselected_dataset_id": preselected_dataset_id,
        },
    )


@router.get("/ui/lineage", response_class=HTMLResponse)
def lineage(request: Request, chain: str | None = None, verify: bool = False):
    """Lineage — focused-chain viewer.

    Renders one connected vertical chain at a time
    (Dataset(s) → Run → Model → Decisions) with a chip picker above
    to swap between chains. ``?chain=<name>/<version>`` selects the
    chain; defaults to the active production model. ``?verify=true``
    runs full verification on every envelope in the selected chain.

    Phase C: replaces the four-column directory shipped in Phase A.
    """
    app = request.app
    settings = app.state.settings

    # Discover available chains — one per registered model version.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    available_chains = []
    try:
        versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
        active_version = str(app.state.model_info.get("model_version", ""))
        for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
            available_chains.append({
                "model_name": settings.mlflow_model_name,
                "model_version": str(mv.version),
                "key": f"{settings.mlflow_model_name}/{mv.version}",
                "is_active": str(mv.version) == active_version,
            })
    except Exception as e:
        logger.warning("Lineage: model registry lookup failed: %s", e)

    # Resolve the selected chain. Falls back to the most recent (first
    # in the sorted list) when none specified or when the requested
    # chain isn't registered.
    selected_chain = None
    if chain:
        parts = chain.rsplit("/", 1)
        if len(parts) == 2:
            for c in available_chains:
                if c["model_name"] == parts[0] and c["model_version"] == parts[1]:
                    selected_chain = c
                    break
    if selected_chain is None and available_chains:
        # Prefer active version if no explicit selection.
        for c in available_chains:
            if c["is_active"]:
                selected_chain = c
                break
        if selected_chain is None:
            selected_chain = available_chains[0]

    # Build chain context for the selected chain.
    chain_context = {}
    if selected_chain:
        chain_context = _build_chain_context(
            app,
            selected_chain["model_name"],
            selected_chain["model_version"],
            verify=verify,
        )

    # System-wide stats for the strip above the picker.
    lifecycle = app.state.lifecycle_store.list_all()
    decisions_all = app.state.store.list_all()

    dataset_total = len({
        (env.get("record") or {}).get("digest")
        for env in lifecycle
        if (env.get("record") or {}).get("event_type") == "dataset_anchored"
        and (env.get("record") or {}).get("digest")
    })
    run_total = sum(
        1 for env in lifecycle
        if (env.get("record") or {}).get("event_type") == "training_complete"
    )
    model_total = len(available_chains)
    decision_total = len(decisions_all)
    decision_verified = sum(
        1 for env in decisions_all
        if _is_fully_verified(env.get("last_verification"))
    )

    return templates.TemplateResponse(
        request,
        "lineage.html",
        {
            **_common_context(app),
            "available_chains": available_chains,
            "selected_chain": selected_chain,
            "stats": {
                "dataset_total": dataset_total,
                "run_total": run_total,
                "model_total": model_total,
                "decision_total": decision_total,
                "decision_verified": decision_verified,
                "verified_pct": (
                    round(decision_verified / decision_total * 100)
                    if decision_total else 0
                ),
            },
            **chain_context,
        },
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
    canonical_bytes_json = None
    signed_commitment_json = None
    if verify and envelope.get("arweave_tx_id"):
        result = _verify_envelope(app, envelope)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        # plugin_full_verify carries raw payload_bytes; not JSON-serializable.
        # Mapped legacy fields are sufficient for cached display.
        persistable = {k: v for k, v in result.items() if k != "plugin_full_verify"}
        envelope["last_verification"] = persistable
        app.state.store.update(decision_id, envelope)

        # Phase 3: surface canonical bytes + signed envelope as
        # pretty-printed JSON for the "How verification works" viewer.
        # The plugin's full_verify already fetched both during
        # _verify_envelope, so just re-format from its outputs.
        try:
            import json as _json
            full = result.get("plugin_full_verify") or {}
            anchored = full.get("anchored_bytes") or {}
            payload_bytes = anchored.get("payload_bytes")
            if payload_bytes:
                # Bytes from MLflow are already JCS-canonicalized; pretty-
                # print the parsed JSON so the viewer is human-readable.
                try:
                    canonical_bytes_json = _json.dumps(
                        _json.loads(payload_bytes), indent=2
                    )
                except Exception:
                    canonical_bytes_json = (
                        payload_bytes.decode("utf-8")
                        if isinstance(payload_bytes, (bytes, bytearray))
                        else str(payload_bytes)
                    )
            # Re-fetch the on-chain envelope and pretty-print it. The
            # plugin's verify path mutates a local copy with `_tx_id`;
            # fetch fresh from the gateway for clean display.
            tx_id = envelope.get("arweave_tx_id")
            plugin_envelope = app.state.anchor.fetch_proof(tx_id) if tx_id else None
            if plugin_envelope:
                signed_commitment_json = _json.dumps(plugin_envelope, indent=2)
        except Exception as e:  # noqa: BLE001
            # Display-only: never block the page render on a viewer hiccup,
            # but log so verification regressions are diagnosable.
            logger.warning(
                "decision_detail proof-viewer hydration failed for %s: %s",
                envelope.get("record", {}).get("decision_id"), e,
            )

    # Check Turbo status for anchored records (fast, single HTTP call)
    turbo_status = None
    if envelope.get("arweave_tx_id"):
        turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])

    # Single-step upstream context for the page's "comes from" strip:
    # the datasets the source training run consumed. Pulled from the
    # training event's inlined dataset_inputs (signed canonical
    # payload), so an auditor sees the same provenance the proof
    # commits to.
    source_run_id = envelope.get("record", {}).get("mlflow_run_id")
    trained_on_datasets = []
    if source_run_id:
        training_env = app.state.lifecycle_store.get_by_run_id(source_run_id)
        if training_env:
            for di in (training_env.get("record", {}).get("dataset_inputs") or []):
                if di.get("digest"):
                    trained_on_datasets.append({
                        "name": di.get("name", ""),
                        "digest": di.get("digest", ""),
                    })

    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
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
            "canonical_bytes_json": canonical_bytes_json,
            "signed_commitment_json": signed_commitment_json,
            "trained_on_datasets": trained_on_datasets,
            "status": _envelope_status(envelope, arweave_enabled=arweave_enabled),
        },
    )


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str, verify: bool = False):
    app = request.app

    envelope = app.state.lifecycle_store.get_by_run_id(run_id)
    if not envelope:
        return HTMLResponse("<h1>Training run not found</h1>", status_code=404)

    canonical_bytes_json = None
    signed_commitment_json = None
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

        # Phase 3: surface canonical bytes + signed envelope as
        # pretty-printed JSON for the "How verification works" viewer
        # (mirrors the decision_detail plumbing).
        try:
            import json as _json
            full = result.get("plugin_full_verify") or {}
            anchored = full.get("anchored_bytes") or {}
            payload_bytes = anchored.get("payload_bytes")
            if payload_bytes:
                try:
                    canonical_bytes_json = _json.dumps(
                        _json.loads(payload_bytes), indent=2
                    )
                except Exception:
                    canonical_bytes_json = (
                        payload_bytes.decode("utf-8")
                        if isinstance(payload_bytes, (bytes, bytearray))
                        else str(payload_bytes)
                    )
            tx_id = envelope.get("arweave_tx_id")
            plugin_envelope = app.state.anchor.fetch_proof(tx_id) if tx_id else None
            if plugin_envelope:
                signed_commitment_json = _json.dumps(plugin_envelope, indent=2)
        except Exception as e:  # noqa: BLE001
            # Display-only: never block the page render on a viewer hiccup,
            # but log so verification regressions are diagnosable.
            logger.warning(
                "run_detail proof-viewer hydration failed for run %s: %s",
                envelope.get("record", {}).get("run_id"), e,
            )

    turbo_status = None
    if envelope.get("arweave_tx_id"):
        turbo_status = app.state.anchor.check_status(envelope["arweave_tx_id"])

    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            **_common_context(app),
            "envelope": envelope,
            "turbo_status": turbo_status,
            "status": _envelope_status(envelope, arweave_enabled=arweave_enabled),
            "canonical_bytes_json": canonical_bytes_json,
            "signed_commitment_json": signed_commitment_json,
        },
    )


@router.get("/ui/models/{model_name}/{version}", response_class=HTMLResponse)
def model_detail(request: Request, model_name: str, version: str, verify: bool = False):
    """Model version detail (slim).

    Phase C: replaces the multi-card chain visualization with a focused
    metadata + verify view. The chain experience now lives on the
    Lineage page; a "View chain in Lineage" CTA bridges to it.
    Tamper UX stays here because it operates on this version's data.
    """
    app = request.app

    chain_context = _build_chain_context(app, model_name, version, verify=verify)
    training_env = chain_context.get("training")
    registration_env = chain_context.get("registration")

    if training_env is None and registration_env is None:
        return HTMLResponse("<h1>Model version not found</h1>", status_code=404)

    # Whether this version is the currently active production model.
    active_version = str(app.state.model_info.get("model_version", ""))
    is_active = str(version) == active_version

    # Surface the dataset_inputs from the training event so the slim
    # detail page can show what this version was trained on without
    # dropping back to the full chain visualization (which now lives
    # on Lineage).
    training_dataset_inputs = []
    if training_env:
        training_dataset_inputs = (
            training_env.get("record", {}).get("dataset_inputs") or []
        )

    # Model-version status surfaces in the editorial header. Anchor on
    # the registration envelope (the model-version-specific proof). Fall
    # back to training when registration is missing so legacy data still
    # renders a meaningful badge.
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
    status = _envelope_status(
        registration_env or training_env,
        arweave_enabled=arweave_enabled,
    )

    return templates.TemplateResponse(
        request,
        "model_detail.html",
        {
            **_common_context(app),
            "model_name": model_name,
            "version": version,
            "is_active": is_active,
            "training_dataset_inputs": training_dataset_inputs,
            "status": status,
            **chain_context,
        },
    )


def _decision_verdict(env: dict, arweave_enabled: bool) -> str:
    """Map a decision envelope's status to the Reports view's three
    audit-reader verdict states:

    - ``verified`` — all four checks passed.
    - ``issues``   — a verify pass found a real check failing
                     (canonical ``tampered`` state).
    - ``pending``  — anything else (not yet re-derived, ar.io
                     attestation still propagating, anchoring in
                     flight, or no anchor at all). The audit-reader
                     view collapses these into one "not yet
                     determined" bucket since none of them are an
                     audit-reader verdict.
    """
    status = _envelope_status(env, arweave_enabled=arweave_enabled)
    if status == "verified":
        return "verified"
    if status == "tampered":
        return "issues"
    return "pending"


def _build_audit_checks(v: dict | None, signer_key: str | None) -> list[dict]:
    """Build the four plain-language audit-check entries the Reports
    detail page's "Has anything changed?" section renders.

    Mirrors the four checks in ``_verify_envelope``'s return shape but
    rewrites each into a single audit-reader sentence with a ✓ / ✗
    icon. The Decisions detail page shows the same data as labelled
    table rows; this view collapses them into prose for a non-technical
    reader. Each entry: ``label`` (slug for CSS), ``status``
    ("pass" / "fail" / "pending"), ``statement`` (the full sentence).

    Returns an empty list when ``v`` is ``None`` — the template
    interprets that as "verification hasn't run yet" and shows the
    Pending banner instead of an empty checks panel.
    """
    if not v:
        return []
    checks: list[dict] = []

    # 1. proof retrieval from ar.io
    pcf = v.get("permanent_copy_found")
    if pcf is True:
        checks.append({
            "label": "found",
            "status": "pass",
            "statement": "We retrieved the proof from ar.io — the proof was found and is readable.",
        })
    elif pcf is False:
        checks.append({
            "label": "found",
            "status": "fail",
            "statement": "We tried to retrieve the proof from ar.io — but the proof was not found at the expected address.",
        })

    # 2. signature
    sv = v.get("signature_valid")
    key_hint = f" by recognized key {signer_key[:8]}…" if signer_key else ""
    if sv is True:
        checks.append({
            "label": "signature",
            "status": "pass",
            "statement": f"We verified the cryptographic signature — valid, signed{key_hint}.",
        })
    elif sv is False:
        checks.append({
            "label": "signature",
            "status": "fail",
            "statement": "We verified the cryptographic signature — invalid. The proof appears to have been modified after signing.",
        })

    # 3. anchored-bytes hash
    hm = v.get("hash_match")
    if hm is True:
        checks.append({
            "label": "hash",
            "status": "pass",
            "statement": "We re-hashed the decision proof from its source — the hash matches what was anchored.",
        })
    elif hm is False:
        checks.append({
            "label": "hash",
            "status": "fail",
            "statement": "We re-hashed the decision proof from its source — the hash does not match what was anchored.",
        })

    # 4. ar.io attestation maturity
    attestation_level = v.get("attestation_level")
    if isinstance(attestation_level, int):
        level_labels = {1: "Submitted", 2: "Confirmed", 3: "Permanent"}
        level_word = level_labels.get(attestation_level, f"Level {attestation_level}")
        if attestation_level >= 2:
            checks.append({
                "label": "attestation",
                "status": "pass",
                "statement": f"We checked ar.io attestation maturity — Level {attestation_level} ({level_word}).",
            })
        else:
            checks.append({
                "label": "attestation",
                "status": "pending",
                "statement": f"We checked ar.io attestation maturity — still propagating (Level {attestation_level}, {level_word}). Try again in a few minutes.",
            })

    return checks


def _failing_line_for_banner(checks: list[dict]) -> str | None:
    """Pick a single plain-language line for the Issues-Found banner
    sub-text. Prefers signature failures (most alarming — points at
    impersonation), then hash, then proof-retrieval. Returns ``None``
    if no checks failed."""
    priority = ["signature", "hash", "found"]
    fails = {c["label"]: c["statement"] for c in checks if c["status"] == "fail"}
    for label in priority:
        if label in fails:
            return fails[label]
    return next(iter(fails.values()), None)


def _format_decision_outcome(prediction: dict) -> str:
    """Human-readable outcome string for the Reports list / detail.

    "Approved (0.87)" / "Denied (0.31)" — the score of the chosen
    class so a reader sees how confident the classifier was.
    Falls back to em-dash when prediction shape is unexpected.
    """
    cls = (prediction or {}).get("class")
    probabilities = (prediction or {}).get("probabilities") or {}
    if cls == "approve":
        label = "Approved"
    elif cls == "deny":
        label = "Denied"
    else:
        return "—"
    score = probabilities.get(cls)
    if isinstance(score, (int, float)):
        return f"{label} ({score:.2f})"
    return label


@router.get("/ui/reports", response_class=HTMLResponse)
def reports_list(request: Request):
    """Reports — audit-reader view of every anchored decision.

    Headline summary (green / red / neutral) + table sorted issues
    first, then most-recent verified, then pending. Reuses the
    existing five-state envelope status — only the *presentation*
    is audit-reader shaped. No new verification path; the same
    ``last_verification`` the Decisions detail page persists is
    what the verdict reads.

    Distinct from ``/ui/decisions/`` (the technical workflow page):
    same underlying records, different reader, different framing.
    """
    app = request.app
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
    records = app.state.store.list_all()

    # Cache training-run lookups so a workload with N decisions
    # against the same model version doesn't trigger N lifecycle-
    # store scans for the same dataset name.
    dataset_name_by_run_id: dict[str, str] = {}
    def _dataset_name_for_run(run_id: str | None) -> str:
        if not run_id:
            return "—"
        if run_id in dataset_name_by_run_id:
            return dataset_name_by_run_id[run_id]
        training_env = app.state.lifecycle_store.get_by_run_id(run_id)
        name = "—"
        if training_env:
            inputs = (training_env.get("record") or {}).get("dataset_inputs") or []
            if inputs:
                name = inputs[0].get("name") or "—"
        dataset_name_by_run_id[run_id] = name
        return name

    rows = []
    counts = {"verified": 0, "issues": 0, "pending": 0, "total": 0}
    for env in records:
        rec = env.get("record") or {}
        verdict = _decision_verdict(env, arweave_enabled)
        counts[verdict] += 1
        counts["total"] += 1
        rows.append({
            "decision_id": rec.get("decision_id", ""),
            "timestamp": rec.get("timestamp", ""),
            "outcome": _format_decision_outcome(rec.get("prediction") or {}),
            "model_name": rec.get("model_name", ""),
            "model_version": rec.get("model_version", ""),
            "dataset_name": _dataset_name_for_run(rec.get("mlflow_run_id")),
            "verdict": verdict,
        })

    # Stable two-pass sort:
    #   pass 1: most recent first within whatever bucket they end up in
    #   pass 2: verdict priority — issues at the top, then verified,
    #           then pending. Stable so the timestamp ordering survives.
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    verdict_priority = {"issues": 0, "verified": 1, "pending": 2}
    rows.sort(key=lambda r: verdict_priority[r["verdict"]])

    return templates.TemplateResponse(
        request,
        "reports_list.html",
        {
            **_common_context(app),
            "rows": rows,
            "counts": counts,
        },
    )


@router.get("/ui/reports/{decision_id}", response_class=HTMLResponse)
def reports_detail(request: Request, decision_id: str, verify: bool = False):
    """Per-decision audit report — the shareable artifact.

    Renders a verdict banner + three plain-language Q&A sections answering
    *which model made this decision?*, *what data was the model trained
    on?*, and *has anything changed since this decision was anchored?*.
    Each section sources its data from the same MLflow run / lifecycle
    envelopes the Decisions detail page reads; nothing new is fetched
    or computed.

    On ``?verify=true`` re-runs the plugin's verification (same path the
    Decisions detail page uses) and persists ``last_verification`` on the
    envelope so subsequent loads show the fresh verdict without
    re-hitting the gateway. Otherwise reads cached ``last_verification``
    from disk; pending verifications surface as the pending banner.

    Phases 4 onward layer the accordions (independent-verify commands,
    raw evidence) and shareability (print CSS, copy-permalink) on top
    of the same data this route already gathers.
    """
    import json as _json

    app = request.app
    arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
    envelope = app.state.store.get_by_id(decision_id)
    if not envelope:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    # Re-verify on demand. Same persistence pattern as decision_detail
    # — once the verify pass completes, ``last_verification`` is the
    # canonical state both pages read from.
    if verify and envelope.get("arweave_tx_id"):
        result = _verify_envelope(app, envelope)
        result["verified_at"] = datetime.now(timezone.utc).isoformat()
        persistable = {k: v for k, v in result.items() if k != "plugin_full_verify"}
        envelope["last_verification"] = persistable
        app.state.store.update(decision_id, envelope)

    rec = envelope.get("record") or {}
    verdict = _decision_verdict(envelope, arweave_enabled)

    # Resolve training run + registration + dataset via the lifecycle
    # store. Each of these is the upstream entity the audit report's
    # Q&A sections reference (e.g. "registered on [date] by [signer]").
    source_run_id = rec.get("mlflow_run_id")
    training_env = (
        app.state.lifecycle_store.get_by_run_id(source_run_id) if source_run_id else None
    )
    training_rec = (training_env or {}).get("record") or {}

    reg_env = app.state.lifecycle_store.get_by_model_version(
        rec.get("model_name"), rec.get("model_version"),
    ) if rec.get("model_name") and rec.get("model_version") else None
    reg_rec = (reg_env or {}).get("record") or {}

    # Dataset: take the first dataset_input from the training event
    # (mirrors the Decisions detail page's "trained on" strip) and
    # look up its standalone dataset_anchored envelope for the n_samples
    # / anchor TX fields.
    dataset_input = (training_rec.get("dataset_inputs") or [{}])[0] if training_rec else {}
    dataset_digest = dataset_input.get("digest")
    dataset_env = None
    if dataset_digest:
        for env in app.state.lifecycle_store.list_all():
            r = env.get("record") or {}
            if r.get("event_type") == "dataset_anchored" and r.get("digest") == dataset_digest:
                dataset_env = env
                break
    dataset_rec = (dataset_env or {}).get("record") or {}

    # Signer key — pulled from the persisted signed commitment JSON the
    # plugin writes at anchor time. Truncated to 8 hex chars for the
    # plain-language signature sentence ("signed by recognized key
    # abc12345…"). When a verified key registry ships (paused plan, see
    # docs/plans/active/2026-05-05-verification-correctness-piece-b-c.md
    # Piece C), this becomes the org-bound identity instead of a raw
    # hex fingerprint.
    signer_key: str | None = None
    sc_json = envelope.get("signed_commitment_json")
    if sc_json:
        try:
            signer_key = (_json.loads(sc_json) or {}).get("public_key")
        except (ValueError, TypeError):
            signer_key = None

    checks = _build_audit_checks(envelope.get("last_verification"), signer_key)
    failing_line = _failing_line_for_banner(checks) if verdict == "issues" else None

    # Verified-at timestamp from the cached verification.
    last_v = envelope.get("last_verification") or {}
    verified_at = last_v.get("verified_at")

    return templates.TemplateResponse(
        request,
        "reports_detail.html",
        {
            **_common_context(app),
            "decision_id": decision_id,
            "envelope": envelope,
            "rec": rec,
            "verdict": verdict,
            "verified_at": verified_at,
            "signer_key": signer_key,
            "failing_line": failing_line,
            "checks": checks,
            "outcome": _format_decision_outcome(rec.get("prediction") or {}),
            "training_rec": training_rec,
            "training_env": training_env,
            "reg_env": reg_env,
            "reg_rec": reg_rec,
            "dataset_rec": dataset_rec,
            "dataset_env": dataset_env,
        },
    )


# Demo administration page — sales / pre-sales workflow only. Registers
# only when ``demo_mode`` is True (the default; override with
# VAIDR_DEMO_MODE=false in production). The page hosts the "Reset demo
# data" button which calls ``POST /demo/reset`` (see ``app/main.py``).
if get_settings().demo_mode:

    @router.get("/demo/admin", response_class=HTMLResponse)
    def demo_admin(request: Request):
        app = request.app
        return templates.TemplateResponse(
            request,
            "demo_admin.html",
            _common_context(app),
        )

    @router.get("/demo/tamper", response_class=HTMLResponse)
    def tamper_page(request: Request, chain: str | None = None, verify: bool = False):
        """Demo-only tamper page. Single connected chain at a time with
        tamper buttons inline at each link.

        Phase E: lifted out of the per-detail pages so the verification
        chain reads as evidence, not as something users would intentionally
        break. Same chip-picker UX as Lineage; same tamper/reset endpoints
        the per-detail pages used to call. Re-verifies the chain after
        each tamper via ``?verify=true`` so the badge state flips
        immediately on the reloaded page.
        """
        app = request.app
        settings = app.state.settings

        # Discover available chains — same logic as the lineage handler.
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        available_chains = []
        try:
            versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
            active_version = str(app.state.model_info.get("model_version", ""))
            for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
                available_chains.append({
                    "model_name": settings.mlflow_model_name,
                    "model_version": str(mv.version),
                    "key": f"{settings.mlflow_model_name}/{mv.version}",
                    "is_active": str(mv.version) == active_version,
                })
        except Exception as e:
            logger.warning("Tamper page: model registry lookup failed: %s", e)

        # Resolve selected chain (default: active).
        selected_chain = None
        if chain:
            parts = chain.rsplit("/", 1)
            if len(parts) == 2:
                for c in available_chains:
                    if c["model_name"] == parts[0] and c["model_version"] == parts[1]:
                        selected_chain = c
                        break
        if selected_chain is None and available_chains:
            for c in available_chains:
                if c["is_active"]:
                    selected_chain = c
                    break
            if selected_chain is None:
                selected_chain = available_chains[0]

        chain_context = {}
        if selected_chain:
            chain_context = _build_chain_context(
                app,
                selected_chain["model_name"],
                selected_chain["model_version"],
                verify=verify,
            )

        # Recent decisions for the selected chain so we can render
        # decision tamper buttons inline. Cap to keep the section
        # readable when a chain has many decisions.
        arweave_enabled = app.state.anchor.enabled if app.state.anchor else False
        decision_envs = app.state.store.list_all()
        chain_decisions_raw = sorted(
            (
                d for d in decision_envs
                if d.get("record", {}).get("model_name") == (selected_chain or {}).get("model_name")
                and str(d.get("record", {}).get("model_version", "")) == (selected_chain or {}).get("model_version", "")
            ),
            key=lambda e: (e.get("record") or {}).get("timestamp", ""),
            reverse=True,
        )[:10]

        # Re-verify decisions when ?verify=true. _build_chain_context
        # only verifies training / registration / dataset events; the
        # tamper page lists decisions too, so a decision tamper would
        # otherwise leave the badge stale. Persist the result so
        # subsequent renders read the fresh state.
        if verify:
            for d in chain_decisions_raw:
                if not d.get("arweave_tx_id"):
                    continue
                result = _verify_envelope(app, d)
                result["verified_at"] = datetime.now(timezone.utc).isoformat()
                persistable = {
                    k: v for k, v in result.items() if k != "plugin_full_verify"
                }
                d["last_verification"] = persistable
                app.state.store.update(d["record"]["decision_id"], d)

        # Active tamper snapshots — used to flag the card the user
        # actually clicked tamper on (the "Mutated" indicator), even
        # when the verifier flag flips on a different downstream link.
        from app import tamper as tamper_mod

        def _is_mutated(et: str, eid: str) -> bool:
            return any(
                (et, str(eid), kind) in tamper_mod._snapshots
                for kind in ("saved", "live")
            )

        # Pre-compute per-envelope status + failure descriptions so the
        # template can render the canonical 5-state badge and a "what
        # failed" hint without inline branching. Distinguishes Pending
        # ar.io confirmation (yellow) from genuinely Tampered (red).
        training_run_id = (
            chain_context.get("training", {}).get("record", {}).get("run_id")
            if chain_context.get("training") else None
        )

        def _wrap(env, *, mutated_event_type=None, mutated_event_id=None):
            return {
                "envelope": env,
                "status": _envelope_status(env, arweave_enabled=arweave_enabled),
                "failures": _describe_failed_checks(env.get("last_verification")),
                "is_mutated": (
                    _is_mutated(mutated_event_type, mutated_event_id)
                    if mutated_event_type and mutated_event_id else False
                ),
            }

        training_view = (
            _wrap(
                chain_context["training"],
                mutated_event_type="training",
                mutated_event_id=training_run_id,
            ) if chain_context.get("training") else None
        )
        registration_view = (
            _wrap(
                chain_context["registration"],
                mutated_event_type="registration",
                mutated_event_id=chain_context["registration"]["record"]["event_id"],
            ) if chain_context.get("registration") else None
        )
        # Dataset tamper is keyed on run_id (the dataset_meta tamper
        # mutates the dataset registry tied to the training run), so all
        # dataset views in the same chain share the same mutated flag.
        dataset_views = [
            _wrap(
                env,
                mutated_event_type="dataset",
                mutated_event_id=training_run_id,
            )
            for env in chain_context.get("dataset_anchored", []) or []
        ]
        decision_views = [
            {
                **_wrap(d, mutated_event_type="decision", mutated_event_id=d["record"]["decision_id"]),
                "decision_id": d["record"]["decision_id"],
            }
            for d in chain_decisions_raw
        ]

        return templates.TemplateResponse(
            request,
            "tamper.html",
            {
                **_common_context(app),
                "available_chains": available_chains,
                "selected_chain": selected_chain,
                "training_view": training_view,
                "registration_view": registration_view,
                "dataset_views": dataset_views,
                "decision_views": decision_views,
                "prediction_count": chain_context.get("prediction_count", 0),
                "verified_count": chain_context.get("verified_count", 0),
            },
        )
