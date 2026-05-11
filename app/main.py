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
from app.model import (
    load_model,
    predict,
    train_and_register_with_params,
    anchor_synthetic_dataset,
    seed_default_datasets,
    _parse_synthetic_source,
    DEFAULT_DATASETS,
    DEFAULT_DATASET_NAME,
    FEATURE_NAMES,
)
from app.ui import router as ui_router
from app import tamper as tamper_mod
from app.reset import reset_demo_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _proof_display_json(payload: dict | None, envelope: dict | None) -> tuple[str | None, str | None]:
    """Pretty-print the canonical payload + signed envelope for display.

    Returns ``(canonical_bytes_json, signed_commitment_json)``. Either
    can be ``None`` when the corresponding dict is missing — the always-on
    verification panel renders only what's available.

    Persisted on the lifecycle/record envelope at anchor time so the
    panel renders without a gateway round-trip per page view. Phase E
    swap from the older lazy-on-?verify=true fetch path.
    """
    import json as _json
    canonical_bytes_json = None
    signed_commitment_json = None
    if payload is not None:
        try:
            canonical_bytes_json = _json.dumps(payload, indent=2, sort_keys=False)
        except (TypeError, ValueError):
            canonical_bytes_json = None
    if envelope is not None:
        try:
            signed_commitment_json = _json.dumps(envelope, indent=2, sort_keys=False)
        except (TypeError, ValueError):
            signed_commitment_json = None
    return canonical_bytes_json, signed_commitment_json


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

    # Merge each auto-anchored dataset's TX (and Arweave URL) into the
    # corresponding dataset_inputs entry by name. Templates render the
    # TX as a "View on ar.io" link without needing a separate lookup.
    # The TX is navigation only — chain integrity comes from the
    # inlined digest + schema_hash that are part of the signed
    # canonical payload (see standalone-dataset-anchoring plan).
    anchors_by_name: dict[str, dict] = {}
    for da in (model_info.get("training_dataset_anchors") or []):
        name = da.get("dataset_name")
        if not name:
            continue
        ar = da.get("anchor_result") or {}
        anchors_by_name[name] = {
            "anchor_tx": ar.get("tx_id"),
            "anchor_url": ar.get("url"),
        }
    enriched_dataset_inputs = []
    for di in payload.get("dataset_inputs", []) or []:
        name = di.get("name")
        merged = dict(di)
        if name and name in anchors_by_name:
            merged.update(anchors_by_name[name])
        enriched_dataset_inputs.append(merged)

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
        # Surfaced from the canonical anchored payload (inlined
        # metadata) plus per-dataset anchor_tx / anchor_url merged in
        # from training_dataset_anchors. Empty list when proof was
        # anchored under the legacy escape hatch.
        "dataset_inputs": enriched_dataset_inputs,
    }


def _build_standalone_dataset_envelope(anchor_result: dict) -> dict:
    """Wrap a standalone ``ario_mlflow.anchor(dataset=ds)`` result as a
    lifecycle_store envelope.

    Same shape as the per-training-run dataset_anchored envelopes built
    by ``_build_dataset_anchored_records``, except ``source_run_id`` is
    ``None`` (no associated training run) and ``n_samples`` / ``seed``
    are surfaced from the synthetic-source query string so the UI and
    /api/train can recover the generator params without re-parsing the
    canonical payload's source field.
    """
    payload = anchor_result.get("payload") or {}
    ar = anchor_result.get("anchor_result") or {}
    parsed = _parse_synthetic_source(payload.get("source") or "")
    n_samples, seed = (parsed if parsed else (None, None))
    record = {
        "event_id": str(uuid.uuid4()),
        "event_type": "dataset_anchored",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name":        payload.get("name"),
        "source":      payload.get("source"),
        "source_type": payload.get("source_type"),
        "digest":      payload.get("digest"),
        "schema_hash": payload.get("schema_hash"),
        # No run yet — this dataset was seeded or created standalone.
        # The "Used by" table on the dataset detail page picks up
        # training runs by digest, not by source_run_id, so this stays
        # accurate after a training run consumes the dataset.
        "source_run_id": None,
        "payload_hash": anchor_result.get("payload_hash"),
        # Synthetic generator params, surfaced so /api/train can
        # regenerate the same train subset deterministically.
        "n_samples": n_samples,
        "seed": seed,
    }
    canonical_bytes_json, signed_commitment_json = _proof_display_json(
        anchor_result.get("payload"),
        anchor_result.get("envelope"),
    )
    return {
        "record": record,
        "arweave_tx_id": ar.get("tx_id"),
        "arweave_url": ar.get("url"),
        "turbo_receipt": ar.get("receipt"),
        "canonical_bytes_json": canonical_bytes_json,
        "signed_commitment_json": signed_commitment_json,
    }


def _find_dataset_spec_by_digest(lifecycle_store, digest: str) -> dict | None:
    """Look up a dataset_anchored event by digest and return the
    ``{"name", "n_samples", "seed"}`` spec needed by
    ``train_and_register_with_params``. Returns ``None`` if no event
    matches or its source string doesn't carry synthetic params.
    """
    for env in lifecycle_store.list_all():
        rec = env.get("record") or {}
        if rec.get("event_type") != "dataset_anchored":
            continue
        if rec.get("digest") != digest:
            continue
        # Prefer the explicit n_samples/seed columns when the standalone
        # envelope wrote them; fall back to re-parsing source for older
        # per-run dataset_anchored entries.
        n = rec.get("n_samples")
        s = rec.get("seed")
        if n is None or s is None:
            parsed = _parse_synthetic_source(rec.get("source") or "")
            if not parsed:
                return None
            n, s = parsed
        return {"name": rec.get("name"), "n_samples": int(n), "seed": int(s)}
    return None


def _build_dataset_anchored_records(model_info: dict) -> list[dict]:
    """Build one lifecycle_store envelope per auto-anchored dataset.

    Each entry mirrors the training/registration envelope shape so the
    UI can render it as its own chain node with its own verification
    badge: a ``record`` sub-dict with the dataset's identity, plus
    ``arweave_tx_id`` / ``arweave_url`` / ``turbo_receipt`` from the
    plugin's per-dataset anchor result. Returns one dict per dataset
    with ``source_run_id`` set so the model_chain page can filter
    these entries to the relevant training run.
    """
    out = []
    for da in (model_info.get("training_dataset_anchors") or []):
        ds_payload = da.get("payload") or {}
        anchor_result = da.get("anchor_result") or {}
        record = {
            "event_id": str(uuid.uuid4()),
            "event_type": "dataset_anchored",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name":        ds_payload.get("name"),
            "source":      ds_payload.get("source"),
            "source_type": ds_payload.get("source_type"),
            "digest":      ds_payload.get("digest"),
            "schema_hash": ds_payload.get("schema_hash"),
            "source_run_id": model_info["run_id"],
            "payload_hash": da.get("payload_hash"),
        }
        # Phase E: persist display JSON for the always-on verification
        # panel. Dataset events have NO MLflow artifact for the canonical
        # bytes (the plugin's _anchor_dataset_event signs and uploads
        # only — no log_artifact), so eager persist here is the *only*
        # way to render the canonical bytes side without a fresh
        # Arweave fetch on every detail-page render.
        canonical_bytes_json, signed_commitment_json = _proof_display_json(
            da.get("payload"),
            da.get("envelope"),
        )
        out.append({
            "record": record,
            "arweave_tx_id": anchor_result.get("tx_id"),
            "arweave_url": anchor_result.get("url"),
            "turbo_receipt": anchor_result.get("receipt"),
            "canonical_bytes_json": canonical_bytes_json,
            "signed_commitment_json": signed_commitment_json,
        })
    return out


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
        canonical_bytes_json, signed_commitment_json = _proof_display_json(
            model_info.get("training_payload"),
            model_info.get("training_envelope"),
        )
        envelope = {
            "record": record,
            "arweave_tx_id": training_anchor.get("tx_id"),
            "arweave_url": training_anchor.get("url"),
            "turbo_receipt": training_anchor.get("receipt"),
            # Phase E: persist display JSON so the always-on verification
            # panel renders the canonical bytes ↔ signed commitment
            # without a gateway round-trip per page view.
            "canonical_bytes_json": canonical_bytes_json,
            "signed_commitment_json": signed_commitment_json,
        }
        lifecycle_store.append(envelope)
        logger.info(
            f"Tracked training {run_id} in lifecycle_store: "
            f"tx={training_anchor.get('tx_id')}"
        )

        # Per-dataset standalone anchors as their own lifecycle entries.
        # Each gets a chain-card on the model lineage page with its own
        # verification badge — same shape as training/registration.
        for ds_envelope in _build_dataset_anchored_records(model_info):
            lifecycle_store.append(ds_envelope)
        logger.info(
            f"Tracked {len(model_info.get('training_dataset_anchors') or [])} "
            f"dataset_anchored entries for run {run_id}."
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


def _ensure_default_datasets_seeded(lifecycle_store, proof_engine, arweave) -> int:
    """Idempotently anchor DEFAULT_DATASETS standalone into the
    lifecycle_store.

    Skipped when any DEFAULT_DATASETS name already has a dataset_anchored
    entry — defaults seed as a unit, so a partial pre-existing set
    (e.g. one was deleted) won't trigger a re-seed. Returns the number
    of new entries written.
    """
    existing_names = {
        (env.get("record") or {}).get("name")
        for env in lifecycle_store.list_all()
        if (env.get("record") or {}).get("event_type") == "dataset_anchored"
    }
    if any(spec["name"] in existing_names for spec in DEFAULT_DATASETS):
        return 0
    results = seed_default_datasets(proof_engine=proof_engine, arweave=arweave)
    for result in results:
        lifecycle_store.append(_build_standalone_dataset_envelope(result))
    if results:
        logger.info(
            f"Seeded {len(results)} default datasets into lifecycle_store."
        )
    return len(results)


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
    #
    # Pass wallet_path only when the file actually exists. The plugin's
    # ArweaveAnchor treats a non-empty path as caller intent and raises
    # WalletLoadError if the file is missing or malformed (so production
    # operators get a loud signal). For the demo, the default path
    # (``keys/arweave_wallet.json``) is "use this if present, otherwise
    # auto-generate" — we coerce that to None when the file is absent
    # so a fresh local checkout boots without a wallet.
    arweave_wallet = (
        settings.arweave_wallet_path
        if os.path.exists(settings.arweave_wallet_path)
        else None
    )
    app.state.anchor = ArweaveAnchor(arweave_wallet, settings.ario_gateway_host)

    # Seed default datasets standalone before load_model. On first boot
    # this populates the Datasets list before any training run exists;
    # idempotent on subsequent boots. Each default gets its own
    # signed proof + Arweave TX with no associated source_run_id.
    _ensure_default_datasets_seeded(
        app.state.lifecycle_store,
        app.state.proof_engine,
        app.state.anchor,
    )

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


def _hydrate_record_envelope_from_verified_prediction(store, decision_id: str, verified_result, tracking_uri: str | None = None):
    """Background task: wait for VerifiedModel's anchor daemon to settle,
    then hydrate the demo's RecordStore envelope with the real Arweave
    TX. Phase 2.B replaces the legacy ``_anchor_record`` background
    task — VerifiedModel handles the upload itself; this just bridges
    the result back into the demo's UI store. Phase 2.D refactors
    RecordStore into a thin UI cache so this bridge goes away.

    Phase E: also downloads ``ario/predictions/<id>/proof.json`` from
    MLflow once the daemon has written it, so the always-on
    verification panel can render the signed commitment without a
    per-page-render gateway fetch.
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

        # Phase E: pull the signed envelope from MLflow now that the
        # plugin's anchor daemon has finished writing the proof
        # artifact. Best-effort — a failure here just leaves
        # signed_commitment_json missing; canonical bytes (set
        # synchronously when the envelope was created) still render.
        if tracking_uri and not envelope.get("signed_commitment_json"):
            try:
                import json as _json
                import tempfile as _tempfile
                import mlflow as _mlflow
                from mlflow.tracking import MlflowClient as _MlflowClient
                _mlflow.set_tracking_uri(tracking_uri)
                client = _MlflowClient()
                run_id = (envelope.get("record") or {}).get("mlflow_run_id")
                if run_id:
                    with _tempfile.TemporaryDirectory() as tmp:
                        proof_path = client.download_artifacts(
                            run_id,
                            f"ario/predictions/{decision_id}/proof.json",
                            tmp,
                        )
                        with open(proof_path) as f:
                            envelope_dict = _json.load(f)
                        _, sc_json = _proof_display_json(None, envelope_dict)
                        envelope["signed_commitment_json"] = sc_json
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Could not load prediction signed envelope for "
                    f"decision {decision_id}: {e}"
                )

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

        # Phase E: pretty-print the canonical payload immediately —
        # ``verified_result.record`` is the dict the plugin signed.
        # signed_commitment_json gets backfilled by the hydration
        # background task once the plugin's anchor daemon writes the
        # proof artifact to MLflow.
        canonical_bytes_json, _ = _proof_display_json(
            getattr(verified_result, "record", None), None,
        )
        envelope = {
            "record": record,
            # arweave_tx_id starts None and gets hydrated by the
            # background task from verified_result. UI surfaces
            # "anchoring..." until then.
            "arweave_tx_id": None,
            "arweave_url": None,
            "turbo_receipt": None,
            "canonical_bytes_json": canonical_bytes_json,
            "signed_commitment_json": None,
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
        request.app.state.settings.mlflow_tracking_uri,
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
        request.app.state.settings.mlflow_tracking_uri,
    )
    return RedirectResponse(f"/ui/decisions/{decision_id}", status_code=303)


@app.post("/api/datasets")
def api_create_dataset(request: Request, body: dict):
    """Create + anchor a new synthetic dataset standalone (no training run).

    Accepts ``{name, n_samples, random_state}``. Generates the
    deterministic train subset, wraps it as an MLflow Dataset, anchors
    it via ``ario_mlflow.anchor(dataset=ds)``, and writes a
    ``dataset_anchored`` lifecycle_store envelope. Returns the dataset's
    digest plus its detail-page URL so the UI can redirect.
    """
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    try:
        n_samples = int(body.get("n_samples"))
        seed = int(body.get("random_state"))
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": "n_samples and random_state must be integers"},
            status_code=400,
        )
    if n_samples < 50 or n_samples > 50000:
        return JSONResponse(
            {"error": "n_samples must be between 50 and 50000"},
            status_code=400,
        )

    anchor_result = anchor_synthetic_dataset(
        name, n_samples, seed,
        proof_engine=request.app.state.proof_engine,
        arweave=request.app.state.anchor,
    )
    envelope = _build_standalone_dataset_envelope(anchor_result)
    request.app.state.lifecycle_store.append(envelope)
    digest = envelope["record"]["digest"]
    return {
        "dataset_id": digest,
        "name": name,
        "n_samples": n_samples,
        "random_state": seed,
        "arweave_tx_id": envelope.get("arweave_tx_id"),
        "redirect_url": f"/ui/datasets/{digest}",
    }


@app.post("/api/train")
def api_train(request: Request, body: dict, background_tasks: BackgroundTasks):
    """Train a new model version against a chosen dataset.

    Phase 2.A: anchoring is handled by the plugin's headline API
    (anchor() + ArioMlflowClient) — no longer by the demo's hand-rolled
    proof + background-upload pipeline.

    Now requires ``dataset_id`` (a dataset's content digest) in the
    body. The dataset must already exist in the lifecycle_store —
    either seeded at boot, created via ``POST /api/datasets``, or
    auto-anchored by a previous training run. The route looks up the
    dataset's synthetic-generator params from the store and threads
    them through ``train_and_register_with_params`` as a
    ``dataset_spec``, so the training run's regenerated train subset
    has the same digest as the one anchored on Arweave.
    """
    import random
    settings = request.app.state.settings

    dataset_id = (body.get("dataset_id") or "").strip()
    if not dataset_id:
        return JSONResponse(
            {"error": "dataset_id is required"}, status_code=400,
        )
    dataset_spec = _find_dataset_spec_by_digest(
        request.app.state.lifecycle_store, dataset_id,
    )
    if dataset_spec is None:
        return JSONResponse(
            {"error": f"unknown dataset_id {dataset_id!r}"}, status_code=404,
        )

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
        dataset_spec=dataset_spec,
    )

    # Populate lifecycle_store cache entries from the plugin's anchor
    # results. No legacy proof, no duplicate Arweave upload — the plugin
    # already uploaded the real training+registration proofs.
    plugin_anchor = info.get("training_anchor_result") or {}
    training_record = _build_training_cache_record(info)
    canonical_bytes_json, signed_commitment_json = _proof_display_json(
        info.get("training_payload"),
        info.get("training_envelope"),
    )
    training_envelope = {
        "record": training_record,
        "arweave_tx_id": plugin_anchor.get("tx_id"),
        "arweave_url": plugin_anchor.get("url"),
        "turbo_receipt": plugin_anchor.get("receipt"),
        "canonical_bytes_json": canonical_bytes_json,
        "signed_commitment_json": signed_commitment_json,
    }
    request.app.state.lifecycle_store.append(training_envelope)

    # Per-dataset standalone-anchor lifecycle entries — one chain-card
    # per logged dataset on the model lineage page, each with its own
    # Arweave TX and verification status.
    for ds_envelope in _build_dataset_anchored_records(info):
        request.app.state.lifecycle_store.append(ds_envelope)

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

    # Auto-switch to the newly trained model. Pass proof_engine + arweave so
    # the newly-active VerifiedModel can anchor predictions; without these the
    # post-train model would silently lose inference-time anchoring until the
    # next process boot.
    new_model_info = load_model(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        proof_engine=request.app.state.proof_engine,
        arweave=request.app.state.anchor,
    )
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

        # Phase E: also pull the registration's canonical bytes + signed
        # envelope from MLflow artifacts so the always-on verification
        # panel renders without a per-page-render fetch. The plugin
        # writes both files alongside model registration in
        # ario_mlflow/client.py:391-403. Best-effort: a transient MLflow
        # outage just leaves the JSON missing; the panel falls back to
        # a placeholder.
        try:
            import json as _json
            import tempfile as _tempfile
            source_run_id = (envelope.get("record") or {}).get("source_run_id")
            if source_run_id:
                # ArioMlflowClient extends MlflowClient — download_artifacts
                # is inherited. Files are written by the plugin at
                # ario_mlflow/client.py:391-403 alongside the registration
                # anchor.
                with _tempfile.TemporaryDirectory() as tmp:
                    payload_path = ario_client.download_artifacts(
                        source_run_id, "ario/registration_payload.json", tmp,
                    )
                    proof_path = ario_client.download_artifacts(
                        source_run_id, "ario/registration_proof.json", tmp,
                    )
                    with open(payload_path, "rb") as f:
                        payload_dict = _json.loads(f.read())
                    with open(proof_path) as f:
                        envelope_dict = _json.load(f)
                cb_json, sc_json = _proof_display_json(payload_dict, envelope_dict)
                envelope["canonical_bytes_json"] = cb_json
                envelope["signed_commitment_json"] = sc_json
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not load registration display JSON for "
                f"{model_name}/v{model_version}: {e}"
            )

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
        if event_type not in ("decision", "training", "registration", "dataset"):
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
        if event_type not in ("decision", "training", "registration", "dataset"):
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
        if event_type not in ("decision", "training", "registration", "dataset"):
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
