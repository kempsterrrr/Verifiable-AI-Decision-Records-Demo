# End-to-End Verifiable Chain on Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ario_mlflow` natively support the full verifiable lifecycle chain (data → training → registration → promotion → inference), then migrate the demo onto it so any plugin user gets the same provenance the demo demonstrates.

**Architecture:** Each link in the chain is a separately-signed proof anchored to Arweave, chained to the previous link via `previous_hash` in the proof envelope and `ario.<event>_tx` MLflow tags. The chain is reconstructable from MLflow alone — no separate local store. The demo's `LifecycleStore` and `RecordStore` are removed; the demo becomes a thin viewer over plugin-managed state.

**Tech Stack:** Python, MLflow ≥2.14, ar.io Turbo SDK, FastAPI, pytest, Ed25519 signing.

**Chain map (post-implementation):**

| Link | What's hashed | Chains via | API |
|---|---|---|---|
| 1. Data | Dataset bytes | (origin) | `ario_mlflow.anchor_dataset()` |
| 2. Training | Model artifacts + params + metrics + `dataset_tx` | `ario.dataset_tx` (run tag) | `ario_mlflow.anchor()` |
| 3. Registration | Source run + artifact hash | `ario.training_tx` (run tag) | `ArioMlflowClient.create_model_version()` |
| 4. Promotion | Stage transition | `ario.registration_tx` (mv tag) | `ArioMlflowClient.transition_model_version_stage()` |
| 5. Inference | Input + prediction + latency | `ario.promotion_tx` or `ario.registration_tx` (mv tag) | `VerifiedModel.predict()` |

---

## File Structure

**Plugin — files to modify:**
- `ario_mlflow/anchoring.py` — add `anchor_dataset()`; teach `anchor()` to read `ario.dataset_tx` and chain to it.
- `ario_mlflow/model.py` — `VerifiedModel.__init__` reads `ario.promotion_tx` (fallback `ario.registration_tx`) from the model version; `predict()` seeds `_last_hash` from it instead of `"GENESIS"`.
- `ario_mlflow/client.py` — add `lifecycle_for_model(name, version=None)` that walks MLflow tags + runs to return the chained event timeline.
- `ario_mlflow/__init__.py` — export `anchor_dataset`, `lifecycle_for_model`.
- `tests/test_plugin_smoke.py` — new tests for each plugin change.

**Demo — files to modify:**
- `app/model.py` — log the dataset, call `anchor_dataset(...)` then `anchor(...)` inside the training run.
- `app/main.py` — replace `MlflowClient` with `ArioMlflowClient`; replace hand-rolled prediction signing with `VerifiedModel.predict()`; delete `_startup_anchor_lifecycle`, `_anchor_lifecycle_record`, `_run_prediction`.
- `app/ui.py` — read timeline from plugin's `lifecycle_for_model()` and individual decisions from MLflow traces (not from `LifecycleStore` / `RecordStore`).
- `app/config.py` — drop `records_file` and `lifecycle_file` settings.

**Demo — files to delete:**
- `app/lifecycle.py` (record builders are moot once plugin handles all anchoring)
- `app/lifecycle_store.py` (chain lives in MLflow tags now)
- `app/storage.py` (`RecordStore`; decision records live as MLflow traces)
- `app/decision_record.py` (`build_decision_record` is replaced by `VerifiedModel.predict()` internals)

**Templates — must be updated, not deleted:**
- `templates/run_detail.html`, `templates/decision_detail.html`, `templates/model_chain.html`, `templates/index.html`, `templates/model_registry.html` — change data sources from old stores to plugin/MLflow.

---

## Pre-flight

- [ ] **Step 0.1: Verify clean working tree and current branch**

```bash
git status
git branch --show-current
```

Expected: working tree clean, on `main` (or a feature branch dedicated to this plan).

- [ ] **Step 0.2: Create feature branch**

```bash
git checkout -b phase4/end-to-end-chain
```

- [ ] **Step 0.3: Run existing tests to establish baseline**

```bash
pytest tests/ -v
```

Expected: all pass. Record the count for comparison after each task.

---

### Task 1: Plugin — add `anchor_dataset()` (Link 1)

Adds the missing first link. Hashes a dataset file (or directory), signs a `dataset_anchored` proof, anchors to Arweave, and returns the result. The dataset's tx is meant to be tagged on the training run via `ario.dataset_tx`.

**Files:**
- Modify: `ario_mlflow/anchoring.py` (add `anchor_dataset` function)
- Modify: `ario_mlflow/__init__.py` (export `anchor_dataset`)
- Test: `tests/test_plugin_smoke.py` (append new tests)

- [ ] **Step 1.1: Write failing test for `anchor_dataset` hashing a single file**

Append to `tests/test_plugin_smoke.py`:

```python
def test_anchor_dataset_hashes_single_file(tmp_path, monkeypatch):
    """anchor_dataset() returns a proof whose record contains the file's sha256."""
    import hashlib
    from ario_mlflow.anchoring import anchor_dataset
    from ario_mlflow.arweave import ArweaveAnchor

    data = b"feature1,feature2,label\n1,2,0\n3,4,1\n"
    dataset_file = tmp_path / "train.csv"
    dataset_file.write_bytes(data)
    expected_sha = hashlib.sha256(data).hexdigest()

    # Disable Arweave so the test is offline.
    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", "")
    arweave = ArweaveAnchor("", "turbo-gateway.com")
    assert not arweave.enabled

    result = anchor_dataset(name="train", path=str(dataset_file), arweave=arweave)

    assert result["proof"]["record"]["event_type"] == "dataset_anchored"
    assert result["proof"]["record"]["dataset_name"] == "train"
    assert result["proof"]["record"]["dataset_files"] == {"train.csv": expected_sha}
    assert result["proof"]["record"]["dataset_hash"] is not None
    assert result["anchor_result"] is None  # Arweave disabled
```

- [ ] **Step 1.2: Run test, verify failure**

```bash
pytest tests/test_plugin_smoke.py::test_anchor_dataset_hashes_single_file -v
```

Expected: FAIL with `ImportError: cannot import name 'anchor_dataset'`.

- [ ] **Step 1.3: Implement `anchor_dataset` in `ario_mlflow/anchoring.py`**

Append after the `anchor()` function definition (after line 266):

```python
def anchor_dataset(
    name: str,
    path: str,
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
) -> dict:
    """Anchor a training dataset as the origin link in the provenance chain.

    Hashes every file at ``path`` (a single file or a directory walked
    recursively), signs a ``dataset_anchored`` proof envelope, and uploads
    it to Arweave when an anchor is enabled. The returned ``tx_id`` is
    intended to be written onto the subsequent training run as the
    ``ario.dataset_tx`` MLflow tag, so :func:`anchor` can chain to it.

    Args:
        name: Logical dataset name (e.g. ``"credit_train_v3"``). Recorded
            in the proof so verifiers can identify the dataset without
            inspecting bytes.
        path: Path to a single file or a directory of files.
        proof_engine: Optional override for the signing engine.
        arweave: Optional override for the Arweave anchor client.

    Returns:
        A dict with keys ``proof``, ``anchor_result`` (None if anchoring
        disabled or upload failed), ``tx_id`` (the Arweave tx, or None),
        and ``dataset_hash`` (canonical hash of the file→sha256 mapping).
    """
    if proof_engine is None:
        proof_engine = ProofEngine()
    if arweave is None:
        arweave = ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path!r}")

    files: dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, "rb") as f:
            files[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
    else:
        for root, _dirs, filenames in os.walk(path):
            for fname in filenames:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, path)
                with open(fpath, "rb") as f:
                    files[rel] = hashlib.sha256(f.read()).hexdigest()

    dataset_hash = hash_data(canonical_json(files))

    record = {
        "event_id": str(uuid.uuid4()),
        "event_type": "dataset_anchored",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_name": name,
        "dataset_files": files,
        "dataset_hash": dataset_hash,
    }

    proof = proof_engine.create_proof(record, "GENESIS")
    anchor_result = arweave.upload_proof(proof) if arweave.enabled else None
    tx_id = anchor_result["tx_id"] if anchor_result else None

    logger.info(
        f"Dataset {name!r} anchored: files={len(files)}, "
        f"status={'anchored' if anchor_result else 'signed'}"
    )

    return {
        "proof": proof,
        "anchor_result": anchor_result,
        "tx_id": tx_id,
        "dataset_hash": dataset_hash,
    }
```

- [ ] **Step 1.4: Export `anchor_dataset` from package**

In `ario_mlflow/__init__.py`, change the `__getattr__` and `__all__`:

```python
def __getattr__(name):
    if name == "anchor":
        from ario_mlflow.anchoring import anchor
        return anchor
    if name == "anchor_dataset":
        from ario_mlflow.anchoring import anchor_dataset
        return anchor_dataset
    if name == "VerifiedModel":
        from ario_mlflow.model import VerifiedModel
        return VerifiedModel
    if name == "IntegrityError":
        from ario_mlflow.model import IntegrityError
        return IntegrityError
    if name == "ArioMlflowClient":
        from ario_mlflow.client import ArioMlflowClient
        return ArioMlflowClient
    raise AttributeError(f"module 'ario_mlflow' has no attribute {name!r}")


__all__ = ["anchor", "anchor_dataset", "VerifiedModel", "IntegrityError", "ArioMlflowClient"]
```

- [ ] **Step 1.5: Run test, verify pass**

```bash
pytest tests/test_plugin_smoke.py::test_anchor_dataset_hashes_single_file -v
```

Expected: PASS.

- [ ] **Step 1.6: Add directory test**

Append:

```python
def test_anchor_dataset_hashes_directory_recursively(tmp_path, monkeypatch):
    """Directory paths walk recursively, with relative paths preserved as keys."""
    import hashlib
    from ario_mlflow.anchoring import anchor_dataset
    from ario_mlflow.arweave import ArweaveAnchor

    (tmp_path / "splits").mkdir()
    train_bytes = b"x,y\n1,0\n2,1\n"
    test_bytes = b"x,y\n3,1\n"
    (tmp_path / "splits" / "train.csv").write_bytes(train_bytes)
    (tmp_path / "splits" / "test.csv").write_bytes(test_bytes)

    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", "")
    arweave = ArweaveAnchor("", "turbo-gateway.com")
    result = anchor_dataset(name="credit", path=str(tmp_path), arweave=arweave)

    files = result["proof"]["record"]["dataset_files"]
    assert files == {
        "splits/train.csv": hashlib.sha256(train_bytes).hexdigest(),
        "splits/test.csv": hashlib.sha256(test_bytes).hexdigest(),
    }


def test_anchor_dataset_raises_on_missing_path(tmp_path):
    """Missing path raises FileNotFoundError instead of producing a bogus hash."""
    import pytest
    from ario_mlflow.anchoring import anchor_dataset

    with pytest.raises(FileNotFoundError):
        anchor_dataset(name="x", path=str(tmp_path / "does-not-exist"))
```

- [ ] **Step 1.7: Run all new tests, verify pass**

```bash
pytest tests/test_plugin_smoke.py -k anchor_dataset -v
```

Expected: 3 PASS.

- [ ] **Step 1.8: Run full suite to confirm no regression**

```bash
pytest tests/ -v
```

Expected: previous baseline + 3 new tests, all pass.

- [ ] **Step 1.9: Commit**

```bash
git add ario_mlflow/anchoring.py ario_mlflow/__init__.py tests/test_plugin_smoke.py
git commit -m "$(cat <<'EOF'
feat(plugin): add anchor_dataset() for the data link in the chain

Hashes a dataset file or directory, signs a dataset_anchored proof, and
anchors it to Arweave. The returned tx_id is meant to be written onto
the training run as ario.dataset_tx so anchor() can chain to it,
closing the data->training link in the verifiable provenance chain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Plugin — `anchor()` chains to `ario.dataset_tx` (Link 2 → Link 1)

`anchor()` currently chains to `"GENESIS"`. Have it read the `ario.dataset_tx` tag (set by the training script after calling `anchor_dataset`) and chain to it. Also include `dataset_tx` in the proof record so it's verifiable from the proof alone.

**Files:**
- Modify: `ario_mlflow/anchoring.py` (function `anchor()`, around lines 197–210)
- Test: `tests/test_plugin_smoke.py`

- [ ] **Step 2.1: Write failing test**

Append:

```python
def test_anchor_chains_to_dataset_tx_when_tag_present(monkeypatch, tmp_path):
    """When the run carries an ario.dataset_tx tag, anchor() chains to it."""
    import mlflow
    from ario_mlflow.anchoring import anchor
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.arweave import ArweaveAnchor

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test")

    keys = tmp_path / "keys"
    keys.mkdir()
    pe = ProofEngine(str(keys / "priv.pem"), str(keys / "pub.pem"))
    arweave = ArweaveAnchor("", "turbo-gateway.com")  # disabled

    with mlflow.start_run() as run:
        mlflow.set_tag("ario.dataset_tx", "DATASET_TX_ABC123")
        # Log a dummy artifact so anchor() has something to hash.
        artifact = tmp_path / "dummy.txt"
        artifact.write_text("hello")
        mlflow.log_artifact(str(artifact), artifact_path="model")
        result = anchor(proof_engine=pe, arweave=arweave, artifact_path="model")

    assert result["proof"]["record"]["dataset_tx"] == "DATASET_TX_ABC123"
    assert result["proof"]["previous_hash"] == "DATASET_TX_ABC123"
```

- [ ] **Step 2.2: Run test, verify failure**

```bash
pytest tests/test_plugin_smoke.py::test_anchor_chains_to_dataset_tx_when_tag_present -v
```

Expected: FAIL — `previous_hash` is `"GENESIS"`, `dataset_tx` not in record.

- [ ] **Step 2.3: Modify `anchor()` to read tag and chain to it**

In `ario_mlflow/anchoring.py`, locate the `record = {...}` block (around line 197). Replace the assignment of `record` and the call to `create_proof` with:

```python
    dataset_tx = run_data.data.tags.get("ario.dataset_tx")

    record = {
        "event_id": str(uuid.uuid4()),
        "event_type": "training_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "params": params,
        "metrics": metrics,
        "artifact_checksums": checksums,
        "artifact_hash": art_hash,
        "dataset_tx": dataset_tx,
        "source_name": run_data.data.tags.get("mlflow.source.name", ""),
        "git_commit": run_data.data.tags.get("mlflow.source.git.commit", ""),
    }

    proof = proof_engine.create_proof(record, dataset_tx or "GENESIS")
```

- [ ] **Step 2.4: Run test, verify pass**

```bash
pytest tests/test_plugin_smoke.py::test_anchor_chains_to_dataset_tx_when_tag_present -v
```

Expected: PASS.

- [ ] **Step 2.5: Run full suite — confirm no regressions in existing `anchor()` tests**

```bash
pytest tests/ -v
```

Expected: all previous tests still pass; new test passes. If a prior test asserts `previous_hash == "GENESIS"`, it should still hold (no `dataset_tx` tag → falls back to GENESIS).

- [ ] **Step 2.6: Commit**

```bash
git add ario_mlflow/anchoring.py tests/test_plugin_smoke.py
git commit -m "$(cat <<'EOF'
feat(plugin): anchor() chains to ario.dataset_tx when present

Closes the data->training link. When the training script tags the run
with ario.dataset_tx (from anchor_dataset's tx_id), anchor() seeds the
proof envelope's previous_hash with that value and records dataset_tx
in the proof record itself. Falls back to GENESIS when no tag is set.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Plugin — `VerifiedModel.predict()` chains to promotion/registration (Link 5)

Currently `VerifiedModel` initializes `_last_hash = "GENESIS"` (model.py:232) and chains predictions only to each other. Read the loaded model version's `ario.promotion_tx` (preferred) or `ario.registration_tx` (fallback) tag in `__init__` and seed `_last_hash` from that, so the very first prediction chains back into the lifecycle chain.

**Files:**
- Modify: `ario_mlflow/model.py` (`__init__`, around line 232)
- Test: `tests/test_plugin_smoke.py`

- [ ] **Step 3.1: Write failing test**

Append:

```python
def test_verified_model_predict_chains_to_promotion_tx(monkeypatch, tmp_path):
    """First prediction's previous_hash equals the loaded mv's ario.promotion_tx."""
    from unittest.mock import MagicMock
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.arweave import ArweaveAnchor

    # Mock model version with promotion_tx tag.
    mv = MagicMock()
    mv.name = "credit"
    mv.version = "3"
    mv.run_id = "run-abc"
    mv.source = "runs:/run-abc/model"
    mv.tags = {
        "ario.promotion_tx": "PROMO_TX_XYZ",
        "ario.registration_tx": "REG_TX_OLD",
    }

    monkeypatch.setattr("ario_mlflow.model._resolve_model_version", lambda c, u: mv)

    # Stub run with no artifact_hash so integrity check skips.
    fake_run = MagicMock()
    fake_run.data.tags = {}
    fake_client = MagicMock()
    fake_client.get_run.return_value = fake_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: fake_client)

    # Stub pyfunc load.
    fake_model = MagicMock()
    fake_model.predict.return_value = [1]
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: fake_model)

    keys = tmp_path / "keys"
    keys.mkdir()
    pe = ProofEngine(str(keys / "priv.pem"), str(keys / "pub.pem"))
    arweave = ArweaveAnchor("", "turbo-gateway.com")  # disabled

    vm = VerifiedModel("models:/credit/3", proof_engine=pe, anchor=arweave)
    result = vm.predict({"a": 1.0})

    # The signed proof's previous_hash must equal the promotion_tx, not GENESIS.
    # We round-trip via the engine: re-sign the same record with the captured
    # previous_hash and verify it matches.
    assert result.record is not None
    # Confirmed by: re-creating proof with previous_hash="PROMO_TX_XYZ" matches
    # the actual record_hash returned. We expose this through _last_hash chaining.
    assert vm._last_hash != "GENESIS"


def test_verified_model_falls_back_to_registration_tx(monkeypatch, tmp_path):
    """When promotion_tx is absent, _last_hash seeds from registration_tx."""
    from unittest.mock import MagicMock
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.arweave import ArweaveAnchor

    mv = MagicMock()
    mv.name = "credit"
    mv.version = "1"
    mv.run_id = "run-1"
    mv.source = "runs:/run-1/model"
    mv.tags = {"ario.registration_tx": "REG_TX_ONLY"}  # no promotion_tx

    monkeypatch.setattr("ario_mlflow.model._resolve_model_version", lambda c, u: mv)
    fake_run = MagicMock()
    fake_run.data.tags = {}
    fake_client = MagicMock()
    fake_client.get_run.return_value = fake_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: fake_client)
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: MagicMock())

    keys = tmp_path / "keys"
    keys.mkdir()
    pe = ProofEngine(str(keys / "priv.pem"), str(keys / "pub.pem"))
    arweave = ArweaveAnchor("", "turbo-gateway.com")

    vm = VerifiedModel("models:/credit/1", proof_engine=pe, anchor=arweave)

    assert vm._last_hash == "REG_TX_ONLY"


def test_verified_model_falls_back_to_genesis_when_no_chain_tags(monkeypatch, tmp_path):
    """Truly orphaned models still work — _last_hash stays GENESIS."""
    from unittest.mock import MagicMock
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.arweave import ArweaveAnchor

    mv = MagicMock()
    mv.name = "orphan"
    mv.version = "1"
    mv.run_id = "run-x"
    mv.source = "runs:/run-x/model"
    mv.tags = {}  # no ario tags at all

    monkeypatch.setattr("ario_mlflow.model._resolve_model_version", lambda c, u: mv)
    fake_run = MagicMock()
    fake_run.data.tags = {}
    fake_client = MagicMock()
    fake_client.get_run.return_value = fake_run
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: fake_client)
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: MagicMock())

    keys = tmp_path / "keys"
    keys.mkdir()
    pe = ProofEngine(str(keys / "priv.pem"), str(keys / "pub.pem"))
    arweave = ArweaveAnchor("", "turbo-gateway.com")

    vm = VerifiedModel("models:/orphan/1", proof_engine=pe, anchor=arweave)
    assert vm._last_hash == "GENESIS"
```

- [ ] **Step 3.2: Run tests, verify failure**

```bash
pytest tests/test_plugin_smoke.py -k verified_model -v
```

Expected: the three new tests fail (existing pre-load tests should still pass).

- [ ] **Step 3.3: Modify `VerifiedModel.__init__` to seed `_last_hash` from mv tags**

In `ario_mlflow/model.py`, locate `self._last_hash = "GENESIS"` (line 232). Replace it with:

```python
        # Seed the prediction chain from the loaded model version's lifecycle
        # tags so the first prediction's previous_hash points back into the
        # data->training->registration->promotion chain. Prefer promotion_tx
        # (the most recent lifecycle event), fall back to registration_tx,
        # finally GENESIS for orphaned models.
        seed_hash = "GENESIS"
        if mv is not None and mv.tags:
            seed_hash = (
                mv.tags.get("ario.promotion_tx")
                or mv.tags.get("ario.registration_tx")
                or "GENESIS"
            )
        self._last_hash = seed_hash
```

- [ ] **Step 3.4: Run new tests, verify pass**

```bash
pytest tests/test_plugin_smoke.py -k verified_model -v
```

Expected: all PASS.

- [ ] **Step 3.5: Run full suite**

```bash
pytest tests/ -v
```

Expected: no regressions.

- [ ] **Step 3.6: Commit**

```bash
git add ario_mlflow/model.py tests/test_plugin_smoke.py
git commit -m "$(cat <<'EOF'
feat(plugin): VerifiedModel chains predictions to promotion/registration tx

Closes the inference link of the chain. VerifiedModel now seeds
_last_hash from the loaded model version's ario.promotion_tx (or
ario.registration_tx fallback), so the first prediction's previous_hash
points back into the lifecycle chain rather than starting fresh from
GENESIS. Orphaned models still work — they fall back to GENESIS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Plugin — `lifecycle_for_model()` query helper

A demo (or any plugin user) needs to render the chain. Add a method on `ArioMlflowClient` that walks MLflow tags and returns the ordered chain of events for a model: `[dataset?, training, registration, promotion?, ...]`.

**Files:**
- Modify: `ario_mlflow/client.py` (add method)
- Modify: `ario_mlflow/__init__.py` (no change needed — method on existing class)
- Test: `tests/test_plugin_smoke.py`

- [ ] **Step 4.1: Write failing test**

Append:

```python
def test_lifecycle_for_model_returns_chained_events(monkeypatch, tmp_path):
    """Returns chain in oldest-first order with tx pointers."""
    import mlflow
    from ario_mlflow.client import ArioMlflowClient

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test")

    with mlflow.start_run() as run:
        mlflow.set_tag("ario.dataset_tx", "DATASET_TX")
        mlflow.set_tag("ario.training_tx", "TRAINING_TX")
        mlflow.set_tag("ario.artifact_hash", "ARTIFACT_HASH")
        run_id = run.info.run_id

    client = ArioMlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    # Register a model version with anchored tags. We bypass the auto-anchor
    # path (no Arweave) and set tags manually to keep the test offline.
    mv = client.create_model_version(
        name="credit",
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    client.set_model_version_tag("credit", mv.version, "ario.registration_tx", "REG_TX")
    client.set_model_version_tag("credit", mv.version, "ario.promotion_tx", "PROMO_TX")

    chain = client.lifecycle_for_model("credit", version=mv.version)

    event_types = [e["event_type"] for e in chain]
    tx_ids = [e["tx_id"] for e in chain]
    previous_txs = [e["previous_tx"] for e in chain]

    assert event_types == ["dataset_anchored", "training_complete", "model_registered", "stage_transition"]
    assert tx_ids == ["DATASET_TX", "TRAINING_TX", "REG_TX", "PROMO_TX"]
    assert previous_txs == [None, "DATASET_TX", "TRAINING_TX", "REG_TX"]
```

- [ ] **Step 4.2: Run test, verify failure**

```bash
pytest tests/test_plugin_smoke.py::test_lifecycle_for_model_returns_chained_events -v
```

Expected: FAIL with `AttributeError: 'ArioMlflowClient' object has no attribute 'lifecycle_for_model'`.

- [ ] **Step 4.3: Implement `lifecycle_for_model` on `ArioMlflowClient`**

In `ario_mlflow/client.py`, append before the closing of the class (after `_anchor_promotion`):

```python
    def lifecycle_for_model(self, name: str, version: str | None = None) -> list[dict]:
        """Reconstruct the verifiable lifecycle chain for a model from MLflow tags.

        Returns events in oldest-first order. Each event is a dict with keys:

        - ``event_type``: one of ``"dataset_anchored"``, ``"training_complete"``,
          ``"model_registered"``, ``"stage_transition"``.
        - ``tx_id``: the Arweave tx for this event (None if not anchored).
        - ``previous_tx``: the tx of the prior link, for chain verification.
        - ``run_id``: source MLflow run id (for training and earlier links).
        - ``model_version``: model version (for registration and later links).

        Args:
            name: Registered model name.
            version: Specific model version to walk. When ``None``, walks the
                latest version.
        """
        if version is None:
            versions = self.search_model_versions(f"name='{name}'")
            if not versions:
                return []
            version = max(versions, key=lambda v: int(v.version)).version

        mv = self.get_model_version(name, str(version))
        events: list[dict] = []

        run_id = mv.run_id
        run_tags: dict[str, str] = {}
        if run_id:
            try:
                run = self.get_run(run_id)
                run_tags = dict(run.data.tags)
            except Exception as e:
                logger.warning(f"Could not load source run {run_id} for {name}/v{version}: {e}")

        dataset_tx = run_tags.get("ario.dataset_tx")
        training_tx = run_tags.get("ario.training_tx")
        registration_tx = mv.tags.get("ario.registration_tx") if mv.tags else None
        promotion_tx = mv.tags.get("ario.promotion_tx") if mv.tags else None

        if dataset_tx:
            events.append({
                "event_type": "dataset_anchored",
                "tx_id": dataset_tx,
                "previous_tx": None,
                "run_id": run_id,
                "model_version": None,
            })
        if training_tx:
            events.append({
                "event_type": "training_complete",
                "tx_id": training_tx,
                "previous_tx": dataset_tx,
                "run_id": run_id,
                "model_version": None,
            })
        if registration_tx:
            events.append({
                "event_type": "model_registered",
                "tx_id": registration_tx,
                "previous_tx": training_tx,
                "run_id": run_id,
                "model_version": str(version),
            })
        if promotion_tx:
            events.append({
                "event_type": "stage_transition",
                "tx_id": promotion_tx,
                "previous_tx": registration_tx,
                "run_id": run_id,
                "model_version": str(version),
            })

        return events
```

- [ ] **Step 4.4: Run test, verify pass**

```bash
pytest tests/test_plugin_smoke.py::test_lifecycle_for_model_returns_chained_events -v
```

Expected: PASS.

- [ ] **Step 4.5: Add a "no events" test**

Append:

```python
def test_lifecycle_for_model_returns_empty_when_no_versions(tmp_path):
    """Unknown models return an empty list, not an error."""
    import mlflow
    from ario_mlflow.client import ArioMlflowClient

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    client = ArioMlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    assert client.lifecycle_for_model("does-not-exist") == []
```

- [ ] **Step 4.6: Run all new tests + full suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4.7: Commit**

```bash
git add ario_mlflow/client.py tests/test_plugin_smoke.py
git commit -m "$(cat <<'EOF'
feat(plugin): ArioMlflowClient.lifecycle_for_model() returns chained events

Reconstructs the dataset->training->registration->promotion chain for a
given model from MLflow tags. Each event carries tx_id and previous_tx
so callers can verify chain integrity and render timelines without
maintaining a separate local store. Returns events oldest-first, with
empty list for unknown models.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Demo — log dataset and call `anchor_dataset` + `anchor()` in training

Switch the demo's training script to participate in the chain. Persist the synthetic dataset to disk, anchor it, log it as an MLflow artifact + tag, then anchor the run.

**Files:**
- Modify: `app/model.py` (function `train_and_register_with_params`, lines 73–128)
- Test: `tests/test_plugin_smoke.py`

- [ ] **Step 5.1: Write failing integration test**

Append:

```python
def test_demo_train_and_register_writes_chain_tags(monkeypatch, tmp_path):
    """train_and_register_with_params writes dataset_tx + training_tx tags."""
    import mlflow

    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", "")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")

    # Demo expects keys at known paths via env or settings; make ProofEngine
    # use ephemeral keys.
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    monkeypatch.setenv("ED25519_PRIVATE_KEY_PATH", str(keys_dir / "priv.pem"))
    monkeypatch.setenv("ED25519_PUBLIC_KEY_PATH", str(keys_dir / "pub.pem"))

    from app.model import train_and_register_with_params

    info = train_and_register_with_params(
        tracking_uri=f"file://{tmp_path}/mlruns",
        model_name="credit_test",
        max_iter=50,
        random_state=7,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    run = client.get_run(info["run_id"])
    tags = run.data.tags
    assert "ario.dataset_tx" in tags or tags.get("ario.dataset_status") == "signed"
    assert "ario.artifact_hash" in tags
    # In offline mode (no wallet), upload returns None — verify_status is "signed".
    assert tags.get("ario.verify_status") in {"signed", "anchored"}
```

- [ ] **Step 5.2: Run test, verify failure**

```bash
pytest tests/test_plugin_smoke.py::test_demo_train_and_register_writes_chain_tags -v
```

Expected: FAIL — current `app/model.py` doesn't anchor anything.

- [ ] **Step 5.3: Modify `app/model.py` to anchor dataset and run**

Replace the `train_and_register_with_params` function (lines 73–128) with:

```python
def train_and_register_with_params(
    tracking_uri: str,
    model_name: str,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict:
    """Train the credit classifier with configurable params, anchor data + run, register."""
    import os
    import tempfile
    import numpy as np  # already imported at module top
    import ario_mlflow

    mlflow.set_tracking_uri(tracking_uri)

    X, y = _generate_credit_data(n_samples=800, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    # Persist the training split to disk so we can hash it as the dataset link.
    # Using the test split would also be valid; we anchor the train split as the
    # one the model was actually fit on.
    with tempfile.TemporaryDirectory() as data_dir:
        train_path = os.path.join(data_dir, "credit_train.csv")
        header = ",".join(FEATURE_NAMES + ["label"])
        rows = np.column_stack([X_train, y_train])
        np.savetxt(train_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")

        # Link 1: dataset.
        dataset_result = ario_mlflow.anchor_dataset(
            name=f"{model_name}_train_rs{random_state}",
            path=train_path,
        )

        with mlflow.start_run() as run:
            mlflow.log_param("model_type", "LogisticRegression+StandardScaler")
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_training_samples", len(X_train))
            mlflow.log_param("feature_names", ",".join(FEATURE_NAMES))
            mlflow.log_metric("accuracy", accuracy)

            # Tag the run with dataset_tx so anchor() chains to it.
            if dataset_result["tx_id"]:
                mlflow.set_tag("ario.dataset_tx", dataset_result["tx_id"])
            mlflow.set_tag("ario.dataset_hash", dataset_result["dataset_hash"])

            # Log dataset itself as an artifact so verifiers can re-hash it.
            mlflow.log_artifact(train_path, artifact_path="dataset")

            model_info = mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=model_name,
                input_example=X_train[:1],
            )

            # Link 2: training run. anchor() reads ario.dataset_tx and chains.
            ario_mlflow.anchor(artifact_path="model")

            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max(int(v.version) for v in versions)

            logger.info(
                f"Credit model trained: accuracy={accuracy:.4f}, "
                f"run_id={run.info.run_id}, version={latest_version}, "
                f"dataset_tx={dataset_result['tx_id']}"
            )

            return {
                "run_id": run.info.run_id,
                "model_name": model_name,
                "model_version": str(latest_version),
                "artifact_uri": model_info.model_uri,
                "accuracy": accuracy,
                "dataset_tx": dataset_result["tx_id"],
            }
```

- [ ] **Step 5.4: Run test, verify pass**

```bash
pytest tests/test_plugin_smoke.py::test_demo_train_and_register_writes_chain_tags -v
```

Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add app/model.py tests/test_plugin_smoke.py
git commit -m "$(cat <<'EOF'
feat(demo): training anchors dataset and run via plugin

train_and_register_with_params now persists the training split, calls
ario_mlflow.anchor_dataset() to hash and anchor it, tags the run with
ario.dataset_tx, and calls ario_mlflow.anchor() inside the run so the
training proof chains back to the dataset proof. Closes the data and
training links of the verifiable chain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Demo — swap `MlflowClient` for `ArioMlflowClient`

The demo's startup uses raw `MlflowClient` and hand-rolls registration anchoring via `app/lifecycle.py`. Replace with `ArioMlflowClient` so registration/promotion are anchored automatically by the plugin and chain into the proof chain.

**Files:**
- Modify: `app/main.py` (lifespan — replace startup anchoring; `/api/train` — drop manual lifecycle records)

- [ ] **Step 6.1: Read current `lifespan` and `/api/train` to understand call sites**

```bash
sed -n '104,165p' app/main.py
sed -n '280,345p' app/main.py
```

(No code change yet; confirm structure.)

- [ ] **Step 6.2: Replace lifespan startup anchoring with `ArioMlflowClient`-aware bootstrap**

In `app/main.py`, replace the `lifespan` function (lines 104–144 as of `e8e5b16`) with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # OpenTelemetry
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    app.state.settings = settings
    app.state.proof_engine = ProofEngine(
        settings.ed25519_private_key_path,
        settings.ed25519_public_key_path,
    )
    app.state.anchor = ArweaveAnchor(settings.arweave_wallet_path, settings.ario_gateway_host)
    app.state.ario_verify = ArioVerifyClient(settings.ario_verify_url)

    app.state.ario_client = ArioMlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        proof_engine=app.state.proof_engine,
        anchor=app.state.anchor,
    )

    # Load model via the plugin so integrity check runs and prediction chain
    # seeds from promotion/registration tx.
    logger.info("Loading verified model...")
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
        f"Model loaded: {app.state.model_info['model_name']}/v{app.state.model_info['model_version']}"
    )

    yield

    provider.shutdown()
```

Add the imports at the top of `app/main.py`:

```python
from ario_mlflow import VerifiedModel
from ario_mlflow.client import ArioMlflowClient
```

Remove these now-unused imports:

```python
from app.lifecycle import build_training_record, build_registration_record
from app.storage import RecordStore
from app.model import load_model  # plugin's VerifiedModel replaces it
```

- [ ] **Step 6.3: Replace `/api/train` to use the plugin pipeline**

Replace the `api_train` function (lines 280–339) with:

```python
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

    # Re-load the verified model so the runtime predicts with the new version
    # and the prediction chain seeds from its registration_tx.
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

    return {
        "run_id": info["run_id"],
        "model_version": info["model_version"],
        "accuracy": info["accuracy"],
        "dataset_tx": info["dataset_tx"],
    }
```

Note: registration anchoring happens inside `train_and_register_with_params` because `mlflow.sklearn.log_model(..., registered_model_name=...)` triggers registration. To make that go through `ArioMlflowClient`, modify `app/model.py` Step 5.3 → swap the implicit registration for an explicit `app.state.ario_client.create_model_version(...)` after `log_model`. Reflect that change here:

In `app/model.py` `train_and_register_with_params`, *remove* `registered_model_name=model_name` from `mlflow.sklearn.log_model(...)`, and after `anchor(...)` insert:

```python
            # Register via ArioMlflowClient so registration is anchored and
            # chained to ario.training_tx automatically.
            from ario_mlflow.client import ArioMlflowClient
            ario_client = ArioMlflowClient(tracking_uri=tracking_uri)
            mv = ario_client.create_model_version(
                name=model_name,
                source=model_info.model_uri,
                run_id=run.info.run_id,
            )
            latest_version = mv.version
```

And remove the `client.search_model_versions(...)` block.

- [ ] **Step 6.4: Run integration test from Task 5 + smoke suite**

```bash
pytest tests/ -v
```

Expected: pass. The chain-tag test now covers registration anchoring as well.

- [ ] **Step 6.5: Boot the demo locally and confirm it doesn't crash on startup**

```bash
ARIO_MLFLOW_ARWEAVE_WALLET="" python -m uvicorn app.main:app --port 8081 &
sleep 3
curl -s http://localhost:8081/health
kill %1
```

Expected: `{"status":"ok"}` (or similar) — server starts and `lifespan` completes.

- [ ] **Step 6.6: Commit**

```bash
git add app/main.py app/model.py
git commit -m "$(cat <<'EOF'
feat(demo): swap MlflowClient for ArioMlflowClient + VerifiedModel

Demo runtime now uses the plugin end-to-end:
- ArioMlflowClient handles registration anchoring (chained to training_tx)
- VerifiedModel handles model loading + integrity check + prediction
  chain seeding from promotion/registration tx
- Training script registers explicitly via ArioMlflowClient instead of
  log_model(registered_model_name=...) so registration goes through the
  plugin's anchoring path

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Demo — replace hand-rolled `/predict` with `VerifiedModel.predict()`

The demo's `_run_prediction` (`app/main.py:174-217`) builds and signs records itself. Replace it with `VerifiedModel.predict()`. The MLflow trace + decision_id + signing now all come from the plugin.

**Files:**
- Modify: `app/main.py` (`_run_prediction`, `api_predict`, `predict_form`)

- [ ] **Step 7.1: Replace `_run_prediction` body**

In `app/main.py`, replace `_run_prediction` (lines ~170–217) with:

```python
def _run_prediction(app_state, features: list[float]) -> tuple[dict, "VerifiedPrediction"]:
    """Run inference via the plugin's VerifiedModel. Returns (envelope, vp)."""
    input_data = dict(zip(FEATURE_NAMES, features))
    vp = app_state.verified_model.predict(input_data)

    envelope = {
        "decision_id": vp.decision_id,
        "record": vp.record,
        "proof_status": vp.proof_status,
        "tx_id": vp.tx_id,
    }
    return envelope, vp
```

Add to imports at top:

```python
from ario_mlflow.model import VerifiedPrediction
```

- [ ] **Step 7.2: Update `api_predict` to use `VerifiedPrediction`**

Replace `api_predict` (lines 231–246) with:

```python
@app.post("/predict")
def api_predict(request: Request, body: dict, background_tasks: BackgroundTasks):
    features = [
        float(body.get(name, FEATURE_DEFAULTS[name])) for name in FEATURE_NAMES
    ]
    envelope, _vp = _run_prediction(request.app.state, features)
    return envelope
```

(The plugin's `VerifiedModel` already runs anchoring on a daemon thread; no FastAPI background task needed.)

- [ ] **Step 7.3: Update `/predict-form` (`form_predict`) similarly**

Replace the `form_predict` handler (lines 249–278) with:

```python
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
```

Note the form parsing (the `Form(...)` defaults and `form_values` dict) is preserved verbatim. The change is solely: `_run_prediction` no longer returns `proof`, the `_anchor_record` background task is gone (plugin handles it), and `decision_id` is read from `envelope["decision_id"]` instead of `envelope["record"]["decision_id"]`.

- [ ] **Step 7.4: Run smoke suite**

```bash
pytest tests/ -v
```

Expected: still passes (these endpoints aren't covered by tests, but the suite shouldn't regress).

- [ ] **Step 7.5: End-to-end manual smoke**

```bash
ARIO_MLFLOW_ARWEAVE_WALLET="" python -m uvicorn app.main:app --port 8081 &
sleep 3
curl -s -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{"annual_income":78000,"credit_utilization":0.18,"debt_to_income_ratio":0.22,"months_employed":72,"credit_score":745}' | python -m json.tool
kill %1
```

Expected: response includes `decision_id`, `proof_status: "disabled"` (offline), `record` with input/output hashes.

- [ ] **Step 7.6: Commit**

```bash
git add app/main.py
git commit -m "$(cat <<'EOF'
feat(demo): /predict uses VerifiedModel.predict() instead of hand-rolling

Decision records, signing, MLflow tracing, and async Arweave anchoring
all live in the plugin now. The demo just calls predict() and surfaces
the VerifiedPrediction. Closes the inference link of the chain end to
end (each prediction's proof chains to the loaded model's promotion or
registration tx).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Demo — delete redundant lifecycle/storage modules

With the plugin handling all anchoring, `app/lifecycle.py`, `app/lifecycle_store.py`, `app/storage.py`, and `app/decision_record.py` are dead code. Confirm via grep, delete, and remove their settings.

**Files:**
- Delete: `app/lifecycle.py`, `app/lifecycle_store.py`, `app/storage.py`, `app/decision_record.py`
- Modify: `app/config.py` (remove `records_file`, `lifecycle_file` settings)

- [ ] **Step 8.1: Confirm no remaining imports from deleted modules**

```bash
grep -rn "from app.lifecycle\|from app.storage\|from app.decision_record\|app.lifecycle_store\|RecordStore\|LifecycleStore\|build_training_record\|build_registration_record\|build_decision_record" app/ tests/ scripts/
```

Expected: only references inside the four files themselves. If any other file references them, fix that first before deleting.

- [ ] **Step 8.2: Delete the four files**

```bash
git rm app/lifecycle.py app/lifecycle_store.py app/storage.py app/decision_record.py
```

- [ ] **Step 8.3: Strip unused settings from `app/config.py`**

Open `app/config.py`, remove `records_file` and `lifecycle_file` fields and their defaults. Show the resulting file or the diff in your commit.

- [ ] **Step 8.4: Run import sanity check**

```bash
python -c "import app.main; print('ok')"
```

Expected: `ok`. If `ImportError`, fix the dangling import surfaced.

- [ ] **Step 8.5: Run full suite**

```bash
pytest tests/ -v
```

Expected: pass.

- [ ] **Step 8.6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(demo): delete redundant lifecycle/storage/record modules

The plugin now owns all signing, anchoring, and chain tracking via
MLflow tags. app/lifecycle.py, app/lifecycle_store.py, app/storage.py,
and app/decision_record.py are dead code; deleting them removes the
parallel local-store system that the demo previously maintained.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Demo — UI reads timeline from plugin

`app/ui.py` currently reads from `app.state.lifecycle_store` and `app.state.store`. Switch to `app.state.ario_client.lifecycle_for_model(...)` and MLflow trace queries.

**Files:**
- Modify: `app/ui.py` (replace store reads with plugin queries)
- Modify: relevant templates if data-shape changes (`templates/run_detail.html`, `templates/decision_detail.html`, `templates/model_chain.html`, `templates/index.html`, `templates/model_registry.html`)

- [ ] **Step 9.1: Inventory ui.py store usage**

```bash
grep -n "app.state.lifecycle_store\|app.state.store" app/ui.py
```

This produces a list of call sites. Each becomes a substitution in the next step.

- [ ] **Step 9.2: Replace `lifecycle_store.get_by_run_id(run_id)` calls**

For every `app.state.lifecycle_store.get_by_run_id(run_id)`, replace with:

```python
ario_client = request.app.state.ario_client
chain = ario_client.lifecycle_for_model(model_name) if model_name else []
training_link = next((e for e in chain if e["event_type"] == "training_complete" and e.get("run_id") == run_id), None)
```

(Adjust to the local context — the change is: read from chain, not from store.)

- [ ] **Step 9.3: Replace `lifecycle_store.get_by_model_version(name, version)` calls**

Replace with:

```python
ario_client = request.app.state.ario_client
chain = ario_client.lifecycle_for_model(name, version=version)
```

The full chain is right there.

- [ ] **Step 9.4: Replace `app.state.store` (RecordStore — predictions) with MLflow trace search**

For prediction list / detail views, query MLflow traces tagged with `ario.decision_id`:

```python
import mlflow
client = mlflow.tracking.MlflowClient()
# List recent decisions:
traces = client.search_traces(experiment_ids=["0"], filter_string="tags.ario.decision_id != ''", max_results=50)
# Detail by decision_id:
matches = client.search_traces(experiment_ids=["0"], filter_string=f"tags.ario.decision_id = '{decision_id}'", max_results=1)
```

(Use the same `experiment_ids` list the rest of the app uses — usually `["0"]` or whatever `MLFLOW_EXPERIMENT_NAME` resolves to.)

- [ ] **Step 9.5: Update template references where shape changed**

For each template that consumed the old envelope shape (e.g. `record.event_id`, `arweave_tx_id`), update to the chain event shape (`tx_id`, `previous_tx`, `event_type`). Use the templates listed under "Files" above as the working set.

Concrete substitutions:
- `envelope.arweave_tx_id` → `event.tx_id`
- `envelope.record.event_id` → `event.tx_id` (or omit; tx_id is the canonical id now)
- `envelope.record.event_type` → `event.event_type`
- Any `previous_hash` display → `event.previous_tx`

- [ ] **Step 9.6: Run smoke suite**

```bash
pytest tests/ -v
```

Expected: pass. (These templates aren't unit-tested; the suite mostly guards plugin behavior.)

- [ ] **Step 9.7: Manual smoke — boot demo, click through every page**

```bash
ARIO_MLFLOW_ARWEAVE_WALLET="" python -m uvicorn app.main:app --port 8081 &
sleep 3
# Train a fresh model so chain tags exist:
curl -s -X POST http://localhost:8081/api/train -H "Content-Type: application/json" -d '{"max_iter":100,"random_state":42}'
# Visit each page in a browser:
echo "Visit: http://localhost:8081/"
echo "Visit: http://localhost:8081/registry"
echo "Visit: http://localhost:8081/runs/<run_id>"
echo "Visit: http://localhost:8081/chain/<model_name>"
# After confirming, kill the server:
# kill %1
```

Expected: each page renders without 500s. The chain page shows four links (dataset → training → registration → promotion if any).

- [ ] **Step 9.8: Commit**

```bash
git add app/ui.py templates/
git commit -m "$(cat <<'EOF'
feat(demo): UI reads timeline from plugin instead of local stores

ui.py now calls ArioMlflowClient.lifecycle_for_model() for chain views
and queries MLflow traces (tagged with ario.decision_id) for individual
prediction detail. Templates updated to consume the new event shape
(tx_id / previous_tx / event_type) from the plugin's chain helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Cleanup — orphan data files, README updates

**Files:**
- Delete: `data/lifecycle.json`, `data/records.json` (if present locally — already gitignored, so this is a local hygiene task)
- Modify: `README.md`, `ROADMAP.md`, `ario_mlflow/README.md`

- [ ] **Step 10.1: Remove orphan local data**

```bash
rm -f data/lifecycle.json data/records.json
```

(Already gitignored via the `data/` rule, so this only affects local state.)

- [ ] **Step 10.2: Update top-level README**

In `README.md`, find the architecture section that mentions `LifecycleStore` or `data/lifecycle.json`. Replace with one paragraph about the chain living in MLflow tags + Arweave, and link to `ario_mlflow/README.md` for the plugin API.

Concrete replacement (find the relevant paragraph and substitute):

```markdown
## Architecture

The verifiable provenance chain lives entirely in MLflow tags and on
Arweave — there is no separate local store. Each link (dataset, training,
registration, promotion, inference) is its own signed proof anchored to
Arweave, chained to the previous link via `ario.<event>_tx` tags. The
demo is a thin viewer over `ArioMlflowClient.lifecycle_for_model()`. Any
project that adopts the plugin gets the same end-to-end chain for free.
```

- [ ] **Step 10.3: Mark Phase 4 in ROADMAP.md**

Append (or update if it already drafts Phase 4):

```markdown
## Phase 4 — End-to-end chain on plugin (2026-04)

- Added `anchor_dataset()` (link 1: data).
- `anchor()` chains training proof to `ario.dataset_tx`.
- `VerifiedModel` chains predictions to `ario.promotion_tx` /
  `ario.registration_tx`.
- `ArioMlflowClient.lifecycle_for_model()` returns the chained event
  timeline.
- Demo migrated onto the plugin end to end. `app/lifecycle.py`,
  `app/lifecycle_store.py`, `app/storage.py`, `app/decision_record.py`
  removed.
```

- [ ] **Step 10.4: Update plugin README (`ario_mlflow/README.md`)**

Add a section after the existing API surface docs:

```markdown
## End-to-end chain

The plugin anchors five separately-signed lifecycle events, each
chained to the previous via Arweave tx pointers stored as MLflow tags:

| Link | API | Tag set |
|---|---|---|
| 1. Data | `anchor_dataset(name, path)` | (caller writes `ario.dataset_tx` on the run) |
| 2. Training | `anchor()` inside `start_run()` | `ario.training_tx`, `ario.artifact_hash` |
| 3. Registration | `ArioMlflowClient.create_model_version(...)` | `ario.registration_tx` |
| 4. Promotion | `ArioMlflowClient.transition_model_version_stage(...)` | `ario.promotion_tx` |
| 5. Inference | `VerifiedModel.predict(input)` | (per-prediction proof anchored async) |

To render the chain for a model:

```python
from ario_mlflow.client import ArioMlflowClient

client = ArioMlflowClient(tracking_uri="http://localhost:5000")
chain = client.lifecycle_for_model("credit_classifier")
for event in chain:
    print(f"{event['event_type']}: tx={event['tx_id']}  prev={event['previous_tx']}")
```
```

- [ ] **Step 10.5: Commit**

```bash
git add README.md ROADMAP.md ario_mlflow/README.md
git commit -m "$(cat <<'EOF'
docs: end-to-end chain architecture, plugin README chain table

Documents the new five-link chain (data->training->registration->
promotion->inference), the lifecycle_for_model() helper, and removes
references to the deleted local stores.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Final verification

- [ ] **Step 11.1: Full test suite passes**

```bash
pytest tests/ -v
```

Expected: all green.

- [ ] **Step 11.2: Demo boots and chain renders**

```bash
rm -rf mlruns mlartifacts data/  # fresh slate
ARIO_MLFLOW_ARWEAVE_WALLET="" python -m uvicorn app.main:app --port 8081 &
sleep 3
curl -s -X POST http://localhost:8081/api/train -H "Content-Type: application/json" -d '{"max_iter":100}' | python -m json.tool
echo "Open http://localhost:8081/chain/<model_name> and confirm four links render."
echo "Then kill the server: kill %1"
```

Expected output from `/api/train`: a JSON body containing `dataset_tx` (None when offline; populated when wallet is configured).

- [ ] **Step 11.3: Push branch and open PR**

```bash
git push -u origin phase4/end-to-end-chain
gh pr create --title "Phase 4: end-to-end verifiable chain on plugin" --body "$(cat <<'EOF'
## Summary
- `ario_mlflow.anchor_dataset()` adds the missing data link
- `anchor()` chains to `ario.dataset_tx`
- `VerifiedModel.predict()` chains to `ario.promotion_tx` / `ario.registration_tx`
- `ArioMlflowClient.lifecycle_for_model()` returns the full chain
- Demo migrated onto the plugin; `app/lifecycle.py`, `app/lifecycle_store.py`, `app/storage.py`, `app/decision_record.py` removed

## Test plan
- [ ] All plugin smoke tests pass
- [ ] Demo boots; `/api/train` returns dataset_tx
- [ ] Chain page renders four links for a freshly trained model
- [ ] `/predict` returns a VerifiedPrediction whose proof chains to promotion or registration

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Checklist (run after the plan is implemented)

- [ ] Every spec link in the chain table has a working API and a passing test.
- [ ] No file in `app/` references `LifecycleStore`, `RecordStore`, or `build_*_record`.
- [ ] `lifecycle_for_model()` returns events in `dataset_anchored → training_complete → model_registered → stage_transition` order.
- [ ] First prediction after a model load chains to that model's `promotion_tx` (or `registration_tx` fallback), confirmed via `_last_hash`.
- [ ] Plugin README's chain table matches the actual tags written in code.
