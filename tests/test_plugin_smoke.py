"""Smoke tests for the ario-mlflow plugin.

Covers CodeRabbit PR #3 fixes and the S1 CLI write-back behaviours. No network
or MLflow server required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient
from ario_mlflow.report import generate_verification_html


# --- proof engine ---------------------------------------------------------


def test_canonical_json_and_hash_are_deterministic():
    a = {"b": 2, "a": 1, "c": [3, 1, 2]}
    b = {"a": 1, "c": [3, 1, 2], "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert hash_data(canonical_json(a)) == hash_data(canonical_json(b))


def test_proof_engine_roundtrip_with_auto_generated_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIO_MLFLOW_KEYS_DIR", str(tmp_path))
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    proof = engine.create_proof({"foo": "bar", "timestamp": "2026-04-21T00:00:00Z"}, "GENESIS")
    result = engine.verify_local(proof)
    assert result["hash_valid"] is True
    assert result["signature_valid"] is True
    assert result["overall"] is True


def test_proof_engine_rejects_tampered_record(tmp_path):
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    proof = engine.create_proof({"foo": "bar", "timestamp": "2026-04-21T00:00:00Z"}, "GENESIS")
    proof["record"]["foo"] = "mutated"
    result = engine.verify_local(proof)
    assert result["overall"] is False


# --- ArweaveAnchor wallet fallbacks (CodeRabbit #1) -----------------------


def test_arweave_anchor_with_missing_wallet_generates_in_memory(monkeypatch):
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)
    anchor = ArweaveAnchor(wallet_path=None)
    # Either turbo_sdk is installed and we have an enabled in-memory wallet, or
    # it is absent and init silently disables. Both are valid outcomes; crucially
    # we must not crash.
    assert isinstance(anchor.enabled, bool)


def test_arweave_anchor_with_unreadable_wallet_falls_back(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)
    with caplog.at_level("WARNING"):
        anchor = ArweaveAnchor(wallet_path=str(bad))
    # Must not raise. Warning must name the invalid wallet, and we must fall
    # through to the auto-generated wallet path.
    assert isinstance(anchor.enabled, bool)
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Invalid Arweave wallet" in m for m in warnings), warnings


def test_arweave_anchor_with_structurally_invalid_jwk_falls_back(tmp_path, monkeypatch, caplog):
    """Valid JSON but missing RSA fields should fall back, not crash ArweaveSigner."""
    bad = tmp_path / "incomplete.json"
    bad.write_text('{"kty": "RSA"}')  # valid JSON, missing n/e/d/...
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)
    with caplog.at_level("WARNING"):
        anchor = ArweaveAnchor(wallet_path=str(bad))
    assert isinstance(anchor.enabled, bool)
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Invalid Arweave wallet" in m and "RSA JWK" in m for m in warnings), warnings


# --- ArioVerifyClient normalize key rename (S1 / #5) ----------------------


def test_ario_verify_normalize_returns_attestation_level():
    client = ArioVerifyClient.__new__(ArioVerifyClient)
    client.base_url = "https://example.test"
    raw = {
        "verificationId": "v-1",
        "existence": {"status": "found"},
        "level": 3,
        "links": {"dashboard": "/dash/v-1", "pdf": "https://cdn/pdf"},
        "attestation": {"gateway": "gw-1", "attestedAt": "2026-04-21T00:00:00Z"},
    }
    out = client._normalize(raw)
    assert out["attestation_level"] == 3
    assert "level" not in out
    assert out["report_url"] == "https://example.test/dash/v-1"
    assert out["pdf_url"] == "https://cdn/pdf"  # already absolute — not re-prefixed
    assert out["attested_by"] == "gw-1"


# --- HTML report (CodeRabbit #5, #6) --------------------------------------


def _minimal_proof(tx_id: str = "TX123") -> dict:
    return {
        "record": {
            "event_type": "training_complete",
            "run_id": "run-abc",
            "timestamp": "2026-04-21T00:00:00Z",
        },
        "record_hash": "a" * 64,
        "previous_hash": "GENESIS",
        "signature": "b" * 128,
        "public_key": "c" * 64,
    }


def test_report_renders_without_crash():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX123", "url": "https://turbo-gateway.com/TX123", "receipt": None},
        artifact_hash="deadbeef",
    )
    assert "ar.io Verification Report" in html
    assert "TX123" in html


def test_report_curl_example_uses_raw_tx_path():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX123", "url": "https://turbo-gateway.com/TX123", "receipt": None},
        artifact_hash="deadbeef",
    )
    # CodeRabbit #6: fetch is /raw/<tx_id>, not /<tx_id>/raw
    assert "/raw/TX123" in html
    assert "TX123/raw" not in html


def test_report_verify_command_override_and_url_base():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX999", "url": "https://turbo-gateway.com/TX999", "receipt": None},
        artifact_hash="ab",
        cli_verify_cmd="ario-mlflow verify model foo/1",
        verify_base_url="https://custom.example/verify",
    )
    assert "ario-mlflow verify model foo/1" in html
    assert "https://custom.example/verify/TX999" in html
    # Old hardcoded hostname must not appear when overridden.
    assert "vilenarios.com" not in html


def test_report_verify_command_fallback_uses_run_id():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX5", "url": "https://turbo-gateway.com/TX5", "receipt": None},
        artifact_hash="ab",
    )
    assert "ario-mlflow verify run run-abc" in html


def test_report_shows_attestation_level_when_verified():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX1", "url": "https://turbo-gateway.com/TX1", "receipt": None},
        artifact_hash="deadbeef",
        verification={
            "attestation_level": 3,
            "report_url": "https://verify.example/v/1",
            "attested_by": "ar.io operator",
            "attested_at": "2026-04-21T00:00:00Z",
        },
    )
    assert "Level 3" in html
    assert "ar.io operator" in html
    # When verification is present, the "run CLI to verify" nudge block is hidden.
    assert "to verify this proof" not in html


# --- CLI wiring (S1) ------------------------------------------------------


def test_cli_verify_subparser_includes_trace():
    """Exercises the real ario_mlflow.cli.build_parser — not a handcrafted copy."""
    from ario_mlflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["verify", "trace", "trace-xyz"])
    assert args.command == "verify"
    assert args.verify_type == "trace"
    assert args.trace_id == "trace-xyz"


def test_cli_verify_run_and_model_and_audit_parse():
    """Real-parser smoke coverage for the other verify subcommands."""
    from ario_mlflow.cli import build_parser

    parser = build_parser()

    run_args = parser.parse_args(["verify", "run", "run-123"])
    assert run_args.command == "verify" and run_args.verify_type == "run"
    assert run_args.run_id == "run-123"

    model_args = parser.parse_args(["verify", "model", "foo/1"])
    assert model_args.command == "verify" and model_args.verify_type == "model"
    assert model_args.model == "foo/1"

    audit_args = parser.parse_args(["audit", "foo/1"])
    assert audit_args.command == "audit"
    assert audit_args.model == "foo/1"


# --- CLI verification-tag mapping (S1) ------------------------------------


def test_verification_run_tags_maps_all_fields():
    from ario_mlflow.cli import _verification_run_tags

    tags = _verification_run_tags({
        "attestation_level": 3,
        "report_url": "https://r/",
        "attested_by": "gw",
        "attested_at": "2026-04-21T00:00:00Z",
    })
    assert tags == {
        "ario.verify_status": "verified",
        "ario.attestation_level": "3",
        "ario.report_url": "https://r/",
        "ario.attested_by": "gw",
        "ario.attested_at": "2026-04-21T00:00:00Z",
    }


def test_verification_run_tags_skips_when_level_missing():
    from ario_mlflow.cli import _verification_run_tags

    # Attestation not yet granted — don't mark as verified.
    out = _verification_run_tags({"report_url": "https://r/"})
    assert "ario.verify_status" not in out
    assert "ario.attestation_level" not in out
    assert out == {"ario.report_url": "https://r/"}


def test_verification_run_tags_empty_for_none():
    from ario_mlflow.cli import _verification_run_tags

    assert _verification_run_tags(None) == {}


# --- VerifiedModel ordering: integrity must run before load_model (CodeRabbit r2 #4) ---


def test_verified_model_checks_integrity_before_load(monkeypatch):
    """Regression: tampered pyfunc artifacts must be rejected before load_model runs.

    We stub every MLflow surface VerifiedModel touches and record the order that
    artifact_checksums and mlflow.pyfunc.load_model are invoked. On mismatch,
    IntegrityError must fire before load_model is ever called.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel, IntegrityError

    calls: list[str] = []

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": "EXPECTED"}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"

    class _FakeClient:
        def get_model_version(self, name, version):
            return _FakeMV()

        def get_run(self, run_id):
            return _FakeRun()

    def _fake_checksums(run_id, *a, **kw):
        calls.append("artifact_checksums")
        return {"model/foo": "deadbeef"}  # will hash != "EXPECTED"

    def _fake_load_model(uri):
        calls.append("load_model")
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    with pytest.raises(IntegrityError):
        VerifiedModel("models:/foo/1")

    # The key assertion: load_model must NOT have been reached.
    assert "artifact_checksums" in calls, calls
    assert "load_model" not in calls, (
        "Tampered model code would have executed: load_model ran before IntegrityError"
    )


def test_verified_model_loads_only_after_integrity_passes(monkeypatch):
    """Complement: when hashes match, load_model still runs and _artifact_verified is True."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import canonical_json, hash_data

    # Build matching hashes so verification succeeds.
    checksums = {"model/foo": "deadbeef"}
    expected = hash_data(canonical_json(checksums))

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": expected}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"

    class _FakeClient:
        def get_model_version(self, n, v): return _FakeMV()
        def get_run(self, rid): return _FakeRun()

    call_order: list[str] = []

    def _fake_checksums(run_id, *a, **kw):
        call_order.append("integrity")
        return checksums

    def _fake_load_model(uri):
        call_order.append("load")
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    vm = VerifiedModel("models:/foo/1")
    assert vm._artifact_verified is True
    # Integrity ran first, load ran second.
    assert call_order == ["integrity", "load"], call_order


# --- parse_runs_uri helper -------------------------------------------------


def test_parse_runs_uri_extracts_run_and_artifact_path():
    from ario_mlflow.anchoring import parse_runs_uri

    assert parse_runs_uri("runs:/abc123/model") == ("abc123", "model")
    assert parse_runs_uri("runs:/abc123/sklearn-model") == ("abc123", "sklearn-model")
    assert parse_runs_uri("runs:/abc123/nested/path") == ("abc123", "nested/path")
    assert parse_runs_uri("runs:/abc123") == ("abc123", None)
    assert parse_runs_uri("s3://bucket/model") == (None, None)
    assert parse_runs_uri(None) == (None, None)
    assert parse_runs_uri("") == (None, None)


# --- CodeRabbit round 3 regressions ---------------------------------------


def test_verified_model_uses_mv_source_for_load_and_integrity(monkeypatch):
    """Regression for CodeRabbit r3 #3: non-default artifact paths must be respected.

    When a model is registered at e.g. ``sklearn-model``, load_uri must be
    ``mv.source`` (not ``runs:/<id>/model``) and the integrity check must hash
    the ``sklearn-model`` artifact path.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import canonical_json, hash_data

    checksums = {"sklearn-model/data.pkl": "abc"}
    expected = hash_data(canonical_json(checksums))

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": expected}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/sklearn-model"

    class _FakeClient:
        def get_model_version(self, n, v): return _FakeMV()
        def get_run(self, rid): return _FakeRun()

    recorded: dict = {}

    def _fake_checksums(run_id, *, artifact_path="model", **_):
        recorded["artifact_path"] = artifact_path
        return checksums

    def _fake_load_model(uri):
        recorded["load_uri"] = uri
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    vm = VerifiedModel("models:/foo/1")
    assert vm._artifact_verified is True
    assert recorded["artifact_path"] == "sklearn-model"
    assert recorded["load_uri"] == "runs:/run-xyz/sklearn-model"


# --- CodeRabbit round 4 regressions ---------------------------------------


def test_artifact_checksums_raises_on_download_failure(monkeypatch):
    """Regression for CodeRabbit r4 #1: silent {} on failure would be anchored as a bogus hash."""
    import ario_mlflow.anchoring as anchoring

    def _boom(*a, **kw):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(anchoring.mlflow.artifacts, "download_artifacts", _boom)

    with pytest.raises(anchoring.ArtifactAccessError) as excinfo:
        anchoring.artifact_checksums("run-123", artifact_path="model")
    assert "run-123" in str(excinfo.value)
    assert "tracking store unavailable" in str(excinfo.value)


def test_anchor_omits_artifact_hash_when_artifacts_unavailable(monkeypatch, tmp_path):
    """Regression: anchor() must not publish an empty-tree hash as ario.artifact_hash."""
    import ario_mlflow.anchoring as anchoring
    from ario_mlflow.anchoring import anchor, ArtifactAccessError

    def _boom(run_id, artifact_path="model"):
        raise ArtifactAccessError("simulated failure")

    # Stand in for artifact_checksums.
    monkeypatch.setattr(anchoring, "artifact_checksums", _boom)

    # Minimal MLflow stubs for the rest of anchor().
    class _RunData:
        params: dict = {}
        metrics: dict = {}
        tags: dict = {}

    class _RunInfo:
        run_id = "run-xyz"

    class _ActiveRun:
        info = _RunInfo()

    class _FakeRun:
        data = _RunData()

    class _FakeMlflowClient:
        def get_run(self, run_id): return _FakeRun()
        def set_tag(self, run_id, key, value):
            set_tags.setdefault(run_id, {})[key] = value

    set_tags: dict = {}

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    monkeypatch.setattr(anchoring.mlflow, "active_run", lambda: _ActiveRun())
    monkeypatch.setattr(anchoring.mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient())
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", lambda *a, **kw: None)

    # Point ProofEngine / ArweaveAnchor at deterministic stubs.
    result = anchor(
        proof_engine=anchoring.ProofEngine(
            str(tmp_path / "priv"), str(tmp_path / "pub")
        ),
        arweave=_FakeAnchor(),
    )

    # The fatal assertion: no ario.artifact_hash tag was written.
    assert "ario.artifact_hash" not in set_tags.get("run-xyz", {}), set_tags
    assert "ario.artifact_hash" not in result["tags"]
    # Record's artifact_hash is None — not the hash of an empty dict.
    assert result["proof"]["record"]["artifact_hash"] is None


def test_anchor_accepts_custom_artifact_path(monkeypatch, tmp_path):
    """Regression for CodeRabbit r4 #2: anchor() hashes the logged path, not hardcoded 'model'."""
    import ario_mlflow.anchoring as anchoring
    from ario_mlflow.anchoring import anchor

    recorded: dict = {}

    def _capture_path(run_id, artifact_path="model"):
        recorded["artifact_path"] = artifact_path
        return {f"{artifact_path}/data.pkl": "abc123"}

    monkeypatch.setattr(anchoring, "artifact_checksums", _capture_path)

    class _RunData:
        params: dict = {}
        metrics: dict = {}
        tags: dict = {}

    class _RunInfo:
        run_id = "run-abc"

    class _ActiveRun:
        info = _RunInfo()

    class _FakeRun:
        data = _RunData()

    class _FakeMlflowClient:
        def get_run(self, run_id): return _FakeRun()
        def set_tag(self, *a, **kw): pass

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    monkeypatch.setattr(anchoring.mlflow, "active_run", lambda: _ActiveRun())
    monkeypatch.setattr(anchoring.mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient())
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", lambda *a, **kw: None)

    anchor(
        proof_engine=anchoring.ProofEngine(
            str(tmp_path / "priv"), str(tmp_path / "pub")
        ),
        arweave=_FakeAnchor(),
        artifact_path="sklearn-model",
    )

    assert recorded["artifact_path"] == "sklearn-model"


def test_verified_model_resolves_alias_uri(monkeypatch):
    """Regression for CodeRabbit r4 #3: models:/name@alias must use get_model_version_by_alias."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    class _FakeMV:
        name = "fraud-detector"
        version = 7
        run_id = "run-alias"
        source = "runs:/run-alias/model"

    calls: list[str] = []

    class _FakeClient:
        def get_model_version_by_alias(self, name, alias):
            calls.append(f"by_alias:{name}:{alias}")
            return _FakeMV()

        def get_model_version(self, *a, **kw):
            calls.append("by_version_WRONG")
            raise AssertionError("numeric API was used for an alias URI")

        def get_run(self, run_id):
            return type("R", (), {"data": type("D", (), {"tags": {}})()})()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", lambda uri: object())

    vm = VerifiedModel("models:/fraud-detector@champion")
    assert calls == ["by_alias:fraud-detector:champion"]
    assert vm.model_name == "fraud-detector"
    assert vm.model_version == "7"
    assert vm.run_id == "run-alias"


def test_verified_model_resolves_stage_uri(monkeypatch):
    """Regression for CodeRabbit r4 #3: models:/name/Production uses search_model_versions."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    class _FakeMV:
        name = "fraud-detector"
        version = 3
        run_id = "run-stage"
        source = "runs:/run-stage/model"

    calls: list[str] = []

    class _FakeClient:
        def search_model_versions(self, query):
            calls.append(f"search:{query}")
            return [_FakeMV()]

        def get_model_version(self, *a, **kw):
            calls.append("by_version_WRONG")
            raise AssertionError("numeric API was used for a stage URI")

        def get_run(self, run_id):
            return type("R", (), {"data": type("D", (), {"tags": {}})()})()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", lambda uri: object())

    vm = VerifiedModel("models:/fraud-detector/Production")
    assert len(calls) == 1 and "Production" in calls[0]
    assert vm.model_name == "fraud-detector"
    assert vm.model_version == "3"


def test_ario_client_skips_registration_anchor_on_get_run_failure(monkeypatch, caplog):
    """Regression for CodeRabbit r3 #2: don't mint a bad GENESIS proof when the
    source run lookup fails transiently."""
    import ario_mlflow.client as client_module

    captured: dict = {"create_proof_called": False, "upload_called": False}

    class _FailingClient(client_module.ArioMlflowClient):
        def __init__(self):
            # Skip MlflowClient __init__; we override everything we touch.
            self._proof_engine = type(
                "PE", (), {"create_proof": lambda *a, **kw: (captured.__setitem__("create_proof_called", True) or {})}
            )()
            self._anchor = type(
                "A", (), {"enabled": True, "upload_proof": lambda *a, **kw: (captured.__setitem__("upload_called", True) or None)}
            )()

        def get_run(self, run_id):
            raise RuntimeError("tracking store down")

        def set_model_version_tag(self, *a, **kw):
            captured["set_tag_called"] = True

        def log_artifacts(self, *a, **kw):
            captured["log_artifacts_called"] = True

    c = _FailingClient()
    with caplog.at_level("WARNING"):
        c._anchor_registration("fraud", "1", "run-id", "runs:/run-id/model")

    assert captured["create_proof_called"] is False, (
        "Must not mint a registration proof when source-run lookup failed"
    )
    assert captured["upload_called"] is False
    assert not captured.get("set_tag_called")
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Skipping registration anchoring" in m for m in warnings), warnings
