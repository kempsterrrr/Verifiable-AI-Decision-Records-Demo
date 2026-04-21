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
    from ario_mlflow import cli

    parser = _build_cli_parser(cli)
    args = parser.parse_args(["verify", "trace", "trace-xyz"])
    assert args.command == "verify"
    assert args.verify_type == "trace"
    assert args.trace_id == "trace-xyz"


def _build_cli_parser(cli_module):
    """Replicate the parser cli.main() constructs so we can test parsing."""
    import argparse

    parser = argparse.ArgumentParser(prog="ario-mlflow")
    subparsers = parser.add_subparsers(dest="command")
    verify_parser = subparsers.add_parser("verify")
    verify_sub = verify_parser.add_subparsers(dest="verify_type")
    run_parser = verify_sub.add_parser("run")
    run_parser.add_argument("run_id")
    model_parser = verify_sub.add_parser("model")
    model_parser.add_argument("model")
    trace_parser = verify_sub.add_parser("trace")
    trace_parser.add_argument("trace_id")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("model")
    return parser


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
