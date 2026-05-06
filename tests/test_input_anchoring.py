"""Tests for input-side anchoring (Piece A of the input-side anchoring plan).

Covers the plugin's reading of MLflow's ``run.inputs.dataset_inputs``,
serialization into the canonical training payload (with the schema
hashed via the project's existing JCS canonicalization), the
fail-closed ``anchor()`` check when no dataset inputs are logged, and
the deterministic-ordering rule for multi-input runs.

These tests stub MLflow rather than using a live tracking store. The
demo-fixture-based regression tests for the verify path live separately
in ``test_tamper_endpoints.py``.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data


# --------------------------------------------------------------------------- #
# Test stubs                                                                  #
# --------------------------------------------------------------------------- #

class _FakeDataset:
    """Minimal stand-in for an MLflow Dataset proto."""
    def __init__(self, *, name, source, source_type, digest, schema):
        self.name = name
        self.source = source
        self.source_type = source_type
        self.digest = digest
        self.schema = schema


class _FakeInputTag:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class _FakeDatasetInput:
    def __init__(self, *, dataset, tags=()):
        self.dataset = dataset
        self.tags = list(tags)


class _FakeRunInputs:
    def __init__(self, dataset_inputs):
        self.dataset_inputs = list(dataset_inputs)


class _FakeRunData:
    def __init__(self, *, params=None, metrics=None, tags=None):
        self.params = params or {}
        self.metrics = metrics or {}
        self.tags = tags or {}


class _FakeRun:
    def __init__(self, *, params=None, metrics=None, tags=None, dataset_inputs=()):
        self.data = _FakeRunData(params=params, metrics=metrics, tags=tags)
        self.inputs = _FakeRunInputs(dataset_inputs)


class _FakeRunInfo:
    def __init__(self, run_id):
        self.run_id = run_id


class _FakeActiveRun:
    def __init__(self, run_id):
        self.info = _FakeRunInfo(run_id)


class _FakeMlflowClient:
    def __init__(self, run):
        self._run = run

    def get_run(self, rid):
        return self._run

    def set_tag(self, rid, key, value):
        pass

    def search_model_versions(self, query):
        return []


class _FakeAnchor:
    enabled = False
    wallet_mode = "user-configured"

    def upload_proof(self, env, *a, **kw):
        return None


def _make_dataset_input(name, source, digest, *, schema_json=None,
                       source_type="local", context="training"):
    """Build a _FakeDatasetInput with sane defaults for tests."""
    return _FakeDatasetInput(
        dataset=_FakeDataset(
            name=name,
            source=source,
            source_type=source_type,
            digest=digest,
            schema=schema_json or '{"mlflow_colspec": []}',
        ),
        tags=[_FakeInputTag("mlflow.data.context", context)] if context else [],
    )


def _patch_anchor_env(monkeypatch, run, *, anchor=None):
    """Wire up the MLflow stubs anchor() needs.

    Reuses _FakeAnchor by default. Exposes nothing — tests that need to
    inspect anchor()'s outputs use the return value of anchor() itself.
    """
    import ario_mlflow.anchoring as anchoring

    monkeypatch.setattr(
        anchoring.mlflow, "active_run",
        lambda: _FakeActiveRun(run.data.tags.get("_run_id_for_test", "run-test")),
    )
    monkeypatch.setattr(
        anchoring.mlflow.tracking, "MlflowClient",
        lambda: _FakeMlflowClient(run),
    )
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(anchoring.mlflow, "get_active_trace_id", lambda: None)
    monkeypatch.setattr(anchoring.mlflow, "get_tracking_uri", lambda: "file:./mlruns")
    monkeypatch.setattr(
        anchoring, "artifact_checksums",
        lambda run_id, artifact_path="model": {},
    )
    return anchor or _FakeAnchor()


# --------------------------------------------------------------------------- #
# Fail-closed behaviour                                                       #
# --------------------------------------------------------------------------- #

def test_anchor_raises_when_no_dataset_inputs_logged(tmp_path, monkeypatch):
    """anchor() refuses to mint a training proof when the run has no
    logged dataset inputs. Closes the input-side honesty gap by
    construction — a chain without a dataset reference can't be valid."""
    import ario_mlflow.anchoring as anchoring

    run = _FakeRun(dataset_inputs=())  # empty
    fake_anchor = _patch_anchor_env(monkeypatch, run)

    with pytest.raises(ValueError, match="dataset"):
        anchoring.anchor(
            proof_engine=ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
            arweave=fake_anchor,
        )


def test_anchor_succeeds_with_empty_inputs_when_escape_hatch_set(tmp_path, monkeypatch):
    """Documented escape hatch for the rare legitimate case (research,
    GPAI workflows with no single dataset). Caller opts in explicitly."""
    import ario_mlflow.anchoring as anchoring

    run = _FakeRun(dataset_inputs=())
    fake_anchor = _patch_anchor_env(monkeypatch, run)

    result = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        allow_empty_dataset_inputs=True,
    )

    assert result["payload"]["dataset_inputs"] == []


# --------------------------------------------------------------------------- #
# Inclusion + serialization                                                   #
# --------------------------------------------------------------------------- #

def test_anchor_includes_dataset_inputs_when_run_has_log_input(tmp_path, monkeypatch):
    """anchor() reads run.inputs.dataset_inputs and folds the dataset
    identity fields into the canonical payload's new dataset_inputs list."""
    import ario_mlflow.anchoring as anchoring

    di = _make_dataset_input(
        name="train_q1",
        source='{"uri": "s3://b/train.csv"}',
        digest="abc123",
        schema_json='{"mlflow_colspec":[{"type":"long","name":"a"}]}',
    )
    run = _FakeRun(dataset_inputs=[di])
    fake_anchor = _patch_anchor_env(monkeypatch, run)

    result = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    payload = result["payload"]
    assert "dataset_inputs" in payload
    assert len(payload["dataset_inputs"]) == 1
    entry = payload["dataset_inputs"][0]
    assert entry["name"] == "train_q1"
    assert entry["source"] == '{"uri": "s3://b/train.csv"}'
    assert entry["digest"] == "abc123"
    assert entry["context"] == "training"


def test_anchor_serializes_schema_as_jcs_hash_not_plaintext(tmp_path, monkeypatch):
    """Schema is fingerprinted, not anchored verbatim. Column names stay
    in MLflow but never enter the proof — privacy by design.

    The hash is computed over the JCS-canonicalized schema, not over
    MLflow's raw schema string, so the same logical schema produces the
    same hash regardless of MLflow's whitespace or key ordering."""
    import ario_mlflow.anchoring as anchoring

    schema_json = '{"mlflow_colspec":[{"type":"long","name":"a","required":true}]}'
    di = _make_dataset_input(
        name="ds", source="s.csv", digest="d1", schema_json=schema_json,
    )
    run = _FakeRun(dataset_inputs=[di])
    fake_anchor = _patch_anchor_env(monkeypatch, run)

    result = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )
    entry = result["payload"]["dataset_inputs"][0]

    # Privacy: plaintext schema must NOT appear in the canonical bytes.
    assert "schema" not in entry, (
        "schema column names must not be in the canonical payload — "
        "use schema_hash instead"
    )
    assert b'"name":"a"' not in result["payload_bytes"]

    # Hash computed over JCS-canonical bytes of the parsed schema.
    expected = hash_data(canonical_json(json.loads(schema_json)))
    assert entry["schema_hash"] == expected


def test_anchor_dataset_inputs_serialization_is_deterministic(tmp_path, monkeypatch):
    """Multiple datasets logged in different orders produce the SAME
    canonical bytes. Sort key (name, source, context, digest) breaks
    every reasonable tie."""
    import ario_mlflow.anchoring as anchoring

    a = _make_dataset_input(name="alpha", source="s1.csv", digest="d1")
    b = _make_dataset_input(name="beta",  source="s2.csv", digest="d2")

    # Order 1: alpha, beta
    run_1 = _FakeRun(dataset_inputs=[a, b])
    fa1 = _patch_anchor_env(monkeypatch, run_1)
    bytes_1 = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "priv1"), str(tmp_path / "pub1")),
        arweave=fa1,
    )["payload_bytes"]

    # Order 2: beta, alpha (reversed log order)
    run_2 = _FakeRun(dataset_inputs=[b, a])
    fa2 = _patch_anchor_env(monkeypatch, run_2)
    bytes_2 = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "priv2"), str(tmp_path / "pub2")),
        arweave=fa2,
    )["payload_bytes"]

    assert bytes_1 == bytes_2, "dataset_inputs ordering must be canonical"


# --------------------------------------------------------------------------- #
# Verify-side re-derivation (Piece A Task A2)                                  #
# --------------------------------------------------------------------------- #

def _build_anchored_training_payload(dataset_inputs):
    """Build a canonical training payload as if anchor() produced it,
    using the same _serialize_dataset_inputs helper. Returns
    (payload_dict, payload_bytes)."""
    from ario_mlflow.anchoring import _serialize_dataset_inputs
    fresh = _serialize_dataset_inputs(_FakeRun(dataset_inputs=dataset_inputs))
    payload = {
        "event_type": "training_complete",
        "run_id": "run-x",
        "params": {},
        "metrics": {},
        "artifact_checksums": {},
        "source_name": "",
        "git_commit": "",
        "dataset_inputs": fresh,
    }
    return payload, canonical_json(payload)


def _patch_verify_env(monkeypatch):
    """Stub the artifact_checksums + _logged_model_paths the refetcher
    pulls in via the anchoring module."""
    monkeypatch.setattr(
        "ario_mlflow.anchoring.artifact_checksums",
        lambda run_id, artifact_path="model": {},
    )
    monkeypatch.setattr(
        "ario_mlflow.anchoring._logged_model_paths",
        lambda run: [],
    )


def test_verify_source_of_truth_passes_when_dataset_inputs_match(monkeypatch):
    """Anchor a training proof with dataset inputs; verifier re-fetches
    the same inputs from MLflow → rebuilt canonical bytes equal anchored
    bytes → SoT passes."""
    from ario_mlflow.verify import verify_source_of_truth

    di = _make_dataset_input(name="ds", source="s.csv", digest="abc")
    payload, payload_bytes = _build_anchored_training_payload([di])

    # Live MLflow returns the SAME inputs.
    fake_client = _FakeMlflowClient(_FakeRun(dataset_inputs=[di]))
    _patch_verify_env(monkeypatch)

    envelope = {"event_type": "training_complete"}
    result = verify_source_of_truth(envelope, payload_bytes, fake_client)
    assert result["ok"] is True, result


def test_verify_source_of_truth_fails_when_dataset_input_digest_mutated(monkeypatch):
    """Mutate the dataset's digest in MLflow after anchoring → rebuilt
    canonical bytes diverge → SoT correctly flips to FAIL.

    This is the core tamper-detection path for input-side anchoring.
    The C-piece tamper button writes to the on-disk dataset meta.yaml;
    this unit test exercises the same outcome by stubbing different
    digests at anchor time vs verify time."""
    from ario_mlflow.verify import verify_source_of_truth

    anchored_di = _make_dataset_input(name="ds", source="s.csv", digest="ORIGINAL")
    payload, payload_bytes = _build_anchored_training_payload([anchored_di])

    # Live MLflow now returns a different digest (tamper occurred).
    tampered_di = _make_dataset_input(name="ds", source="s.csv", digest="TAMPERED")
    fake_client = _FakeMlflowClient(_FakeRun(dataset_inputs=[tampered_di]))
    _patch_verify_env(monkeypatch)

    envelope = {"event_type": "training_complete"}
    result = verify_source_of_truth(envelope, payload_bytes, fake_client)
    assert result["ok"] is False, (
        "source-of-truth must FAIL when a dataset's digest changes in "
        f"MLflow after anchoring; got {result}"
    )


def test_verify_source_of_truth_fails_when_dataset_input_added_after_anchor(monkeypatch):
    """Add a fraudulent extra dataset input to MLflow after anchoring →
    rebuilt canonical bytes have an extra entry → SoT FAILs."""
    from ario_mlflow.verify import verify_source_of_truth

    original = _make_dataset_input(name="train", source="t.csv", digest="t1")
    payload, payload_bytes = _build_anchored_training_payload([original])

    # Live MLflow now has an additional fraudulent input.
    fraud = _make_dataset_input(name="forged", source="f.csv", digest="ff")
    fake_client = _FakeMlflowClient(
        _FakeRun(dataset_inputs=[original, fraud])
    )
    _patch_verify_env(monkeypatch)

    envelope = {"event_type": "training_complete"}
    result = verify_source_of_truth(envelope, payload_bytes, fake_client)
    assert result["ok"] is False, (
        "source-of-truth must FAIL when a dataset_input is added after "
        f"anchoring; got {result}"
    )


# --------------------------------------------------------------------------- #
# (Original Task A1 tests continue below)                                     #
# --------------------------------------------------------------------------- #

def test_anchor_schema_hash_stable_across_calls(tmp_path, monkeypatch):
    """Same logical schema → same schema_hash. Belt-and-braces test
    against any future change to MLflow's schema serialization (ours is
    JCS, theirs may not be)."""
    import ario_mlflow.anchoring as anchoring

    # Two strings that parse to the SAME JSON object but with different
    # whitespace/key ordering. JCS canonicalization should normalize them.
    schema_a = '{"mlflow_colspec":[{"type":"long","name":"a"}]}'
    schema_b = '{ "mlflow_colspec" : [ { "name" : "a" , "type" : "long" } ] }'

    di_a = _make_dataset_input(name="ds", source="s.csv", digest="d", schema_json=schema_a)
    di_b = _make_dataset_input(name="ds", source="s.csv", digest="d", schema_json=schema_b)

    run_a = _FakeRun(dataset_inputs=[di_a])
    fa_a = _patch_anchor_env(monkeypatch, run_a)
    hash_a = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "p1"), str(tmp_path / "v1")),
        arweave=fa_a,
    )["payload"]["dataset_inputs"][0]["schema_hash"]

    run_b = _FakeRun(dataset_inputs=[di_b])
    fa_b = _patch_anchor_env(monkeypatch, run_b)
    hash_b = anchoring.anchor(
        proof_engine=ProofEngine(str(tmp_path / "p2"), str(tmp_path / "v2")),
        arweave=fa_b,
    )["payload"]["dataset_inputs"][0]["schema_hash"]

    assert hash_a == hash_b
