"""Demo-mode reset helper.

Wipes the MLflow tracking store, the local RecordStore + LifecycleStore
files, and re-initialises in-memory state so the next request hits a
freshly auto-trained v1. Anchored proofs already on Arweave are not
affected — they remain permanent on the network.

Designed for the sales / pre-sales workflow: pre-seed the demo before
a customer call, then wipe afterward so the next call starts clean.
"""

import json
import logging
import os
import shutil

from app.config import get_settings
from app.lifecycle_store import LifecycleStore
from app.model import load_model
from app.storage import RecordStore

logger = logging.getLogger(__name__)


def reset_demo_state(app) -> str:
    """Wipe demo state and re-initialise. Returns the new model version string.

    Mirrors the lifespan handler's initialisation order
    (``app/main.py`` ``lifespan``):

    1. Delete the MLflow tracking store + local cache files.
    2. Re-instantiate the stores against fresh empty files.
    3. Re-run ``load_model`` (auto-trains a fresh v1 since
       ``mlruns/`` is now empty).
    4. Re-run ``_startup_anchor_lifecycle`` synchronously so the new
       v1 has a lifecycle entry before the response returns.
    5. Swap the new state onto ``app.state``.

    Synchronous on purpose: the user clicked "Reset" and is waiting on
    the response. Background-threaded init would let the homepage
    render before v1 is registered.
    """
    settings = get_settings()

    # 1. Wipe MLflow tracking store. ignore_errors so a missing dir
    #    (e.g. first-ever reset on a fresh deploy) doesn't crash.
    #    Don't recreate the directory: MLflow's FileStore only seeds
    #    the default experiment "0" inside ``__init__`` when the root
    #    directory itself is missing. If we mkdir an empty mlruns/, the
    #    next ``mlflow.start_run()`` raises "Could not find experiment
    #    with ID 0" because the seed step is skipped.
    shutil.rmtree(settings.mlflow_tracking_uri, ignore_errors=True)

    # MLflow caches per-URI ``FileStore`` instances inside a process-level
    # ``lru_cache``; the cached store retains in-memory experiment IDs +
    # paths that no longer exist after the wipe. Without invalidating
    # both the tracking and the model-registry caches, the next
    # ``mlflow.start_run()`` raises ``Invalid parent directory '.../.trash'``.
    # Clearing both caches forces MLflow to rebuild against the empty
    # tree and create a fresh default experiment.
    try:
        import mlflow
        from mlflow.tracking._tracking_service.utils import _tracking_store_registry
        from mlflow.tracking._model_registry.utils import _get_store_registry

        _tracking_store_registry._get_store_with_resolved_uri.cache_clear()
        _get_store_registry()._get_store_with_resolved_uri.cache_clear()
        # Also clear any active run / experiment so the fluent API
        # rebuilds against the fresh store.
        mlflow.end_run()
    except Exception as e:  # noqa: BLE001
        # Caches are private API; if MLflow's internals shift, log
        # but don't crash the reset — the auto-train will surface a
        # clearer error if it fails.
        logger.warning(f"Reset: MLflow cache invalidation failed (non-fatal): {e}")

    # 2. Wipe local cache files. Both stores expect a JSON list on disk;
    #    write empty lists explicitly so the stores load cleanly on
    #    re-instantiation (``RecordStore.__init__`` only writes the
    #    seed file when it doesn't exist).
    for path in (settings.records_file, settings.lifecycle_file):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([], f)

    # 3. Fresh stores against the now-empty files.
    new_store = RecordStore(settings.records_file)
    new_lifecycle_store = LifecycleStore(settings.lifecycle_file)

    # 4. Re-load (auto-trains v1 because mlruns/ is empty). Threads
    #    through the same proof_engine + arweave anchor the lifespan
    #    handler used so signing key + wallet stay consistent.
    #
    #    Order matters: load_model must run *before* any standalone
    #    dataset seeding. On a second consecutive /demo/reset (with
    #    MLflow state already loaded in this process), calling
    #    ``mlflow.data.from_pandas`` ahead of the first ``start_run``
    #    caches a FileStore reference that bypasses the experiment-0
    #    seeding the FileStore would otherwise do against the just-
    #    wiped mlruns/. Result: ``start_run`` later raises ``Could not
    #    find experiment with ID 0``. Auto-training first lets MLflow's
    #    FileStore seed experiment 0 cleanly; the standalone seeding
    #    in step 6 below then runs against an already-warm store.
    logger.info("Reset: re-loading MLflow model (will auto-train v1)...")
    new_model_info = load_model(
        settings.mlflow_tracking_uri,
        settings.mlflow_model_name,
        proof_engine=app.state.proof_engine,
        arweave=app.state.anchor,
    )

    # 5. Synchronously populate the lifecycle_store from the plugin's
    #    anchor results. The lifespan handler runs this in a daemon
    #    thread to avoid blocking startup; here we run it inline so
    #    the homepage renders the new v1 immediately on redirect.
    from app.main import _startup_anchor_lifecycle
    _startup_anchor_lifecycle(settings, new_model_info, new_lifecycle_store)

    # 6. Re-seed default datasets standalone. Calls ``seed_default_datasets``
    #    directly (not the ``_ensure_default_datasets_seeded`` idempotency
    #    wrapper) — the lifecycle store at this point already has the
    #    auto-anchored dataset_anchored event the training run produced
    #    for the default variant, which would trip the wrapper's
    #    "any-default-name-present → skip" guard. Skip the variants
    #    that are already covered by a standalone-shaped (source_run_id=None)
    #    entry so re-reset doesn't pile up duplicates beyond the
    #    documented "one standalone + one auto-anchored per trained
    #    digest" caveat.
    from app.main import _build_standalone_dataset_envelope
    from app.model import seed_default_datasets, DEFAULT_DATASETS
    standalone_names_present = {
        (env.get("record") or {}).get("name")
        for env in new_lifecycle_store.list_all()
        if (env.get("record") or {}).get("event_type") == "dataset_anchored"
        and (env.get("record") or {}).get("source_run_id") is None
    }
    if not standalone_names_present.issuperset(d["name"] for d in DEFAULT_DATASETS):
        anchor_results = seed_default_datasets(
            proof_engine=app.state.proof_engine,
            arweave=app.state.anchor,
        )
        appended = 0
        for result in anchor_results:
            env = _build_standalone_dataset_envelope(result)
            name = (env.get("record") or {}).get("name")
            if name in standalone_names_present:
                continue
            new_lifecycle_store.append(env)
            appended += 1
        logger.info(
            f"Reset: re-seeded {appended} standalone dataset entries."
        )

    app.state.store = new_store
    app.state.lifecycle_store = new_lifecycle_store
    app.state.model_info = new_model_info

    new_version = new_model_info["model_version"]
    logger.info(f"Reset complete: fresh model v{new_version}")
    return new_version
