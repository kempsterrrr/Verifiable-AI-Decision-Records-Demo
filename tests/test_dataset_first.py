"""Tests for the dataset-first lifecycle changes.

These cover behaviour added when the demo was inverted so datasets are
first-class entities (seeded standalone at boot, creatable via the
``Create dataset`` form on ``/ui/datasets``) instead of side-effects of
training runs.

Bootstrap pattern mirrors ``tests/test_demo_reset.py``: each test reloads
``app.main`` + ``app.ui`` with isolated paths and ``VAIDR_DEMO_MODE=true``
so demo-only routes register correctly and lifespan runs.
"""
import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient


DEFAULT_DATASET_NAMES = [
    "Credit scoring - small",
    "Credit scoring - default",
    "Credit scoring - large",
]


def _reload_app(monkeypatch, tmp_path):
    """Reload app.main + app.ui under isolated paths with Arweave disabled."""
    monkeypatch.setenv("VAIDR_DEMO_MODE", "true")
    monkeypatch.setenv("VAIDR_RECORDS_FILE", str(tmp_path / "records.json"))
    monkeypatch.setenv("VAIDR_LIFECYCLE_FILE", str(tmp_path / "lifecycle.json"))
    monkeypatch.setenv("VAIDR_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    # Arweave disabled — upload is best-effort; lifecycle entries still
    # get written and have ``arweave_tx_id == None``.
    monkeypatch.setenv("VAIDR_ARWEAVE_WALLET_PATH", "")

    from app.config import get_settings
    get_settings.cache_clear()

    import app.ui
    importlib.reload(app.ui)
    import app.main
    importlib.reload(app.main)
    return app.main.app


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = _reload_app(monkeypatch, tmp_path)
    with TestClient(app) as c:
        c._tmp_path = tmp_path
        yield c


def _dataset_events(client):
    """All ``dataset_anchored`` envelopes currently in the lifecycle store."""
    return [
        env for env in client.get("/lifecycle").json()
        if (env.get("record") or {}).get("event_type") == "dataset_anchored"
    ]


def _datasets_grouped_by_digest(client):
    out: dict[str, list[dict]] = {}
    for env in _dataset_events(client):
        rec = env.get("record") or {}
        out.setdefault(rec["digest"], []).append(rec)
    return out


# ── Lifespan seeding ──────────────────────────────────────────────────


def test_lifespan_seeds_three_default_datasets(client):
    """After boot, the three DEFAULT_DATASETS appear as standalone
    dataset_anchored events (source_run_id=None)."""
    standalone = [
        rec for rec in (env.get("record") for env in _dataset_events(client))
        if rec.get("source_run_id") is None
    ]
    names = sorted(r["name"] for r in standalone)
    assert names == sorted(DEFAULT_DATASET_NAMES), (
        f"expected seeded names {sorted(DEFAULT_DATASET_NAMES)!r}, got {names!r}"
    )


def test_seeded_datasets_have_distinct_digests(client):
    """Each seeded dataset has its own content digest — n_samples/seed
    differ across variants so digests can't collide."""
    standalone = {
        rec["name"]: rec["digest"]
        for env in _dataset_events(client)
        if (rec := env.get("record") or {}).get("source_run_id") is None
        and rec.get("name") in DEFAULT_DATASET_NAMES
    }
    assert len(set(standalone.values())) == len(DEFAULT_DATASET_NAMES), (
        f"digests collided across seeded datasets: {standalone!r}"
    )


def test_seeded_dataset_envelope_carries_synthetic_params(client):
    """``n_samples`` and ``seed`` are surfaced on the standalone envelope
    so /api/train can recover the generator params without reparsing
    the source string."""
    rec = next(
        env.get("record") for env in _dataset_events(client)
        if (env.get("record") or {}).get("source_run_id") is None
        and (env.get("record") or {}).get("name") == "Credit scoring - default"
    )
    assert rec.get("n_samples") == 800
    assert rec.get("seed") == 42


def test_lifespan_seeding_is_idempotent_on_reboot(tmp_path, monkeypatch):
    """Booting twice against the same data directory only seeds once."""
    # First boot.
    app1 = _reload_app(monkeypatch, tmp_path)
    with TestClient(app1) as c:
        first_count = len([
            env for env in c.get("/lifecycle").json()
            if (env.get("record") or {}).get("event_type") == "dataset_anchored"
            and (env.get("record") or {}).get("source_run_id") is None
        ])
    # Second boot reuses the same lifecycle.json — idempotency check
    # should short-circuit before re-seeding.
    app2 = _reload_app(monkeypatch, tmp_path)
    with TestClient(app2) as c:
        second_count = len([
            env for env in c.get("/lifecycle").json()
            if (env.get("record") or {}).get("event_type") == "dataset_anchored"
            and (env.get("record") or {}).get("source_run_id") is None
        ])
    assert first_count == second_count == 3, (
        f"re-seeded on reboot: first={first_count}, second={second_count}"
    )


# ── POST /api/datasets ────────────────────────────────────────────────


def test_post_datasets_creates_standalone_envelope(client):
    """Happy path: returns digest + redirect_url, writes a new
    dataset_anchored envelope with source_run_id=None.

    Doesn't assert on overall count delta because ``_startup_anchor_lifecycle``
    runs in a daemon thread (per ``app/main.py``) — it may write the
    v1 training's auto-anchored ``dataset_anchored`` event at any
    point between test setup and the post-POST snapshot, racing the
    explicit POST we're checking here. The check below identifies the
    newly-created envelope by name, which is unambiguous and stable.
    """
    r = client.post("/api/datasets", json={
        "name": "Test dataset", "n_samples": 500, "random_state": 99,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "Test dataset"
    assert body["n_samples"] == 500
    assert body["random_state"] == 99
    assert body["dataset_id"]  # digest string
    assert body["redirect_url"] == f"/ui/datasets/{body['dataset_id']}"

    after = _dataset_events(client)
    new = next(
        env for env in after
        if (env.get("record") or {}).get("name") == "Test dataset"
    )
    assert (new.get("record") or {}).get("source_run_id") is None
    assert (new.get("record") or {}).get("n_samples") == 500
    assert (new.get("record") or {}).get("seed") == 99
    assert new.get("record", {}).get("digest") == body["dataset_id"]


def test_post_datasets_400_on_missing_name(client):
    r = client.post("/api/datasets", json={"n_samples": 500, "random_state": 0})
    assert r.status_code == 400
    assert "name" in r.json().get("error", "")


def test_post_datasets_400_on_n_samples_too_small(client):
    r = client.post("/api/datasets", json={
        "name": "too small", "n_samples": 10, "random_state": 0,
    })
    assert r.status_code == 400
    assert "n_samples" in r.json().get("error", "")


def test_post_datasets_400_on_n_samples_non_int(client):
    r = client.post("/api/datasets", json={
        "name": "bad", "n_samples": "five", "random_state": 0,
    })
    assert r.status_code == 400


# ── POST /api/train ───────────────────────────────────────────────────


def test_post_train_400_when_dataset_id_missing(client):
    r = client.post("/api/train", json={"max_iter": 50})
    assert r.status_code == 400
    assert "dataset_id" in r.json().get("error", "")


def test_post_train_404_on_unknown_dataset_id(client):
    r = client.post("/api/train", json={
        "dataset_id": "doesnotexist", "max_iter": 50,
    })
    assert r.status_code == 404
    assert "unknown dataset_id" in r.json().get("error", "")


def test_post_train_with_seeded_dataset_succeeds(client):
    """Training against a seeded standalone dataset writes a
    training_complete event whose dataset digest matches the input."""
    grouped = _datasets_grouped_by_digest(client)
    # Pick the small variant — fastest to fit.
    small_digest = next(
        digest for digest, recs in grouped.items()
        if any(r["name"] == "Credit scoring - small" for r in recs)
    )
    r = client.post("/api/train", json={
        "dataset_id": small_digest, "max_iter": 50,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"]
    assert body["model_version"]

    # Find the resulting training_complete event and confirm its
    # dataset_input matches the digest we trained against.
    events = client.get("/lifecycle").json()
    tc = next(
        env for env in events
        if (env.get("record") or {}).get("run_id") == body["run_id"]
        and (env.get("record") or {}).get("event_type") == "training_complete"
    )
    inputs = (tc.get("record") or {}).get("dataset_inputs", [])
    assert any(di.get("digest") == small_digest for di in inputs), (
        f"trained run's dataset_inputs ({inputs}) don't reference "
        f"the digest we asked for ({small_digest})"
    )


def test_train_against_seeded_dataset_produces_duplicate_event(client):
    """Documents the known caveat: training auto-anchors the dataset
    (plugin behaviour, no dedup yet — see ROADMAP "Cross-run dataset
    reuse / dedup"). Both the standalone-seed event and the
    training-auto-anchor event share the same digest, so the UI's
    grouping-by-digest collapses them visually.

    The lifecycle store ends up with at least two ``dataset_anchored``
    entries for the trained digest: one standalone-seeded (source_run_id
    None), one from the training run's auto-anchor (source_run_id set).
    """
    grouped = _datasets_grouped_by_digest(client)
    small_digest = next(
        digest for digest, recs in grouped.items()
        if any(r["name"] == "Credit scoring - small" for r in recs)
    )
    # Before: just the seeded standalone entry.
    before_small = [r for r in _dataset_events(client)
                    if (r.get("record") or {}).get("digest") == small_digest]
    assert len(before_small) == 1, (
        f"expected exactly one seeded entry for 'small' pre-train, "
        f"got {len(before_small)}"
    )

    client.post("/api/train", json={"dataset_id": small_digest, "max_iter": 50})

    # After: standalone seed + training-time auto-anchor → both share
    # the same digest.
    after_small = [r for r in _dataset_events(client)
                   if (r.get("record") or {}).get("digest") == small_digest]
    assert len(after_small) == 2, (
        f"expected 2 events sharing digest {small_digest} after train "
        f"(seeded + auto-anchored), got {len(after_small)}"
    )
    source_run_ids = {
        (r.get("record") or {}).get("source_run_id") for r in after_small
    }
    # One None (standalone seed) + one with the training run's id.
    assert None in source_run_ids
    assert any(srid for srid in source_run_ids if srid)


# ── /demo/reset ───────────────────────────────────────────────────────


def test_demo_reset_re_seeds_defaults_and_auto_trains(client):
    """After /demo/reset: lifecycle re-seeds the 3 defaults *and*
    auto-trains v1 against the default dataset (the existing reset
    sales-workflow guarantee, now extended)."""
    # Add some noise — a user-created dataset and a training run — so we
    # can confirm reset wipes pre-existing state cleanly.
    client.post("/api/datasets", json={
        "name": "transient", "n_samples": 200, "random_state": 1,
    })
    grouped = _datasets_grouped_by_digest(client)
    transient_digest = next(
        d for d, recs in grouped.items()
        if any(r["name"] == "transient" for r in recs)
    )
    client.post("/api/train", json={"dataset_id": transient_digest, "max_iter": 50})

    r = client.post("/demo/reset")
    assert r.status_code == 200, r.text
    assert r.json()["new_version"] == "1"

    # Standalone seeded defaults present again.
    standalone_names = sorted(
        (env.get("record") or {}).get("name")
        for env in _dataset_events(client)
        if (env.get("record") or {}).get("source_run_id") is None
    )
    assert standalone_names == sorted(DEFAULT_DATASET_NAMES), (
        f"reset failed to re-seed defaults; got {standalone_names!r}"
    )

    # Transient user-created dataset is gone.
    all_names = {
        (env.get("record") or {}).get("name")
        for env in _dataset_events(client)
    }
    assert "transient" not in all_names

    # v1 auto-trained against the default dataset.
    events = client.get("/lifecycle").json()
    tcs = [
        env for env in events
        if (env.get("record") or {}).get("event_type") == "training_complete"
    ]
    assert len(tcs) >= 1
    last_tc_inputs = (tcs[-1].get("record") or {}).get("dataset_inputs", [])
    assert any(
        di.get("name") == "Credit scoring - default" for di in last_tc_inputs
    ), f"reset v1 didn't train against the default; inputs={last_tc_inputs!r}"
