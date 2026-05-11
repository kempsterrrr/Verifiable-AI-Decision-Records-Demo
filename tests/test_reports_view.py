"""Tests for the audit-reader Reports view (``/ui/reports`` + detail).

The route is a presentation layer over the existing RecordStore +
lifecycle_store + ``_verify_envelope`` primitives — no new
verification semantics. These tests inject envelopes directly into
the RecordStore so the verdict-state combinatorics (verified /
issues / pending / empty) are exercised without depending on a live
Arweave gateway.

Bootstrap pattern mirrors ``tests/test_dataset_first.py``: each test
reloads ``app.main`` + ``app.ui`` with isolated paths,
``VAIDR_DEMO_MODE=true``, Arweave disabled, and ``VAIDR_TAMPER_TTL_SECONDS=0``
to keep tamper-route tests fast.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("VAIDR_DEMO_MODE", "true")
    monkeypatch.setenv("VAIDR_RECORDS_FILE", str(tmp_path / "records.json"))
    monkeypatch.setenv("VAIDR_LIFECYCLE_FILE", str(tmp_path / "lifecycle.json"))
    monkeypatch.setenv("VAIDR_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("VAIDR_ARWEAVE_WALLET_PATH", "")
    monkeypatch.setenv("VAIDR_TAMPER_TTL_SECONDS", "0")

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
        c._app = app
        yield c


# ── Envelope-injection helpers ─────────────────────────────────────────


def _inject_envelope(client, *, decision_id, verdict, model_name=None, model_version=None):
    """Append a minimal decision envelope directly to the RecordStore.

    Sidesteps the full /predict flow so a test can pin the verdict
    state explicitly. The lifespan handler already trained a v1 we
    can hang ``model_name`` / ``model_version`` off so the Reports
    detail route's lookups find a matching training event.
    """
    app = client._app
    mi = app.state.model_info
    model_name = model_name or mi["model_name"]
    model_version = model_version or mi["model_version"]

    record = {
        "decision_id": decision_id,
        "timestamp": "2026-05-11T13:24:32Z",
        "trace_id": "trace-" + decision_id,
        "span_id": "span-" + decision_id,
        "service_name": "verifiable-ai-demo",
        "mlflow_run_id": mi["run_id"],
        "model_name": model_name,
        "model_version": model_version,
        "artifact_uri": mi.get("artifact_uri", ""),
        "input_hash": "input-" + decision_id,
        "output_hash": "output-" + decision_id,
        "prediction": {
            "class": "approve",
            "class_index": 1,
            "probabilities": {"approve": 0.87, "deny": 0.13},
            "features_used": [],
        },
        "latency_ms": 12.0,
        "human_override": False,
    }
    envelope = {
        "record": record,
        "arweave_tx_id": None,
        "arweave_url": None,
        "turbo_receipt": None,
        "canonical_bytes_json": None,
        "signed_commitment_json": None,
    }

    if verdict == "verified":
        envelope["arweave_tx_id"] = "TX-VERIFIED-" + decision_id
        envelope["arweave_url"] = "https://turbo-gateway.com/" + envelope["arweave_tx_id"]
        envelope["signed_commitment_json"] = (
            '{"event_id": "' + decision_id + '", '
            '"event_type": "prediction", '
            '"public_key": "888810666e7c7ef86d64b8386caee5cc"}'
        )
        envelope["last_verification"] = {
            "overall": True,
            "signature_valid": True,
            "permanent_copy_found": True,
            "hash_match": True,
            "source_of_truth_ok": True,
            "attestation_level": 2,
            "verified_at": "2026-05-11T13:30:00Z",
        }
    elif verdict == "issues":
        envelope["arweave_tx_id"] = "TX-TAMPERED-" + decision_id
        envelope["arweave_url"] = "https://turbo-gateway.com/" + envelope["arweave_tx_id"]
        envelope["last_verification"] = {
            "overall": False,
            "signature_valid": True,
            "permanent_copy_found": True,
            "hash_match": False,  # tampered: re-hashed bytes diverged
            "source_of_truth_ok": None,
            "attestation_level": 2,
            "verified_at": "2026-05-11T13:35:00Z",
        }
    elif verdict == "pending":
        # No last_verification, no TX — leaves the route at the
        # "pending" rollup state.
        pass
    else:
        raise ValueError(f"Unknown verdict for test injection: {verdict!r}")

    app.state.store.append(envelope)
    return envelope


# ── Portfolio list (/ui/reports) ──────────────────────────────────────


def test_reports_list_empty_state(client):
    """No decisions in RecordStore → empty state copy + zero in stat."""
    r = client.get("/ui/reports")
    assert r.status_code == 200
    assert "No decisions to audit yet" in r.text
    # Decisions stat in the editorial header reads 0.
    assert ">0<" in r.text


def test_reports_list_all_verified_headline(client):
    """One verified decision → green headline 'All N decisions verified'."""
    _inject_envelope(client, decision_id="dec-verified-1", verdict="verified")
    r = client.get("/ui/reports")
    assert r.status_code == 200
    assert "All 1 decision verified" in r.text
    # The green variant class is applied to the headline div (not just
    # defined in the CSS block, which appears unconditionally).
    assert 'class="audit-headline audit-headline-green"' in r.text
    assert "Verified" in r.text
    # The red variant class is NOT applied to any element.
    assert 'class="audit-headline audit-headline-red"' not in r.text


def test_reports_list_issues_headline(client):
    """A tampered decision → red headline + 'View issues' CTA + issues count."""
    _inject_envelope(client, decision_id="dec-bad-1", verdict="issues")
    _inject_envelope(client, decision_id="dec-good-1", verdict="verified")
    r = client.get("/ui/reports")
    assert r.status_code == 200
    assert "audit-headline-red" in r.text
    assert "1 of 2 decisions have audit issues" in r.text
    assert "View issues" in r.text
    # Issues-found verdict badge surfaces in the row.
    assert "Issues found" in r.text


def test_reports_list_pending_headline(client):
    """A pending (never-verified) decision → neutral 'still pending' headline."""
    _inject_envelope(client, decision_id="dec-pending-1", verdict="pending")
    r = client.get("/ui/reports")
    assert r.status_code == 200
    assert "audit-headline-neutral" in r.text
    assert "still pending" in r.text
    assert "Pending verification" in r.text


def test_reports_list_issues_sort_first(client):
    """Mixed verdicts → issues rows render before verified rows in HTML order."""
    _inject_envelope(client, decision_id="dec-verified-A", verdict="verified")
    _inject_envelope(client, decision_id="dec-issues-B", verdict="issues")
    _inject_envelope(client, decision_id="dec-verified-C", verdict="verified")
    _inject_envelope(client, decision_id="dec-pending-D", verdict="pending")
    r = client.get("/ui/reports")
    assert r.status_code == 200
    html = r.text
    pos_issues = html.find("dec-issues-B")
    pos_verifiedA = html.find("dec-verified-A")
    pos_verifiedC = html.find("dec-verified-C")
    pos_pending = html.find("dec-pending-D")
    assert pos_issues != -1 and pos_verifiedA != -1 and pos_verifiedC != -1 and pos_pending != -1
    # Issues row comes before any verified row.
    assert pos_issues < pos_verifiedA
    assert pos_issues < pos_verifiedC
    # Verified rows come before any pending row.
    assert pos_verifiedA < pos_pending
    assert pos_verifiedC < pos_pending


# ── Per-decision detail (/ui/reports/<id>) ─────────────────────────────


def test_reports_detail_404_on_unknown_id(client):
    r = client.get("/ui/reports/no-such-decision")
    assert r.status_code == 404


def test_reports_detail_renders_three_qa_questions(client):
    """Detail page shows the three plain-language Q&A questions."""
    _inject_envelope(client, decision_id="dec-detail-1", verdict="verified")
    r = client.get("/ui/reports/dec-detail-1")
    assert r.status_code == 200
    assert "Which model made this decision?" in r.text
    assert "What data was the model trained on?" in r.text
    assert "Has anything changed since this decision was anchored?" in r.text


def test_reports_detail_verified_banner(client):
    """Verified envelope renders the green banner + verified copy."""
    _inject_envelope(client, decision_id="dec-vb-1", verdict="verified")
    r = client.get("/ui/reports/dec-vb-1")
    assert r.status_code == 200
    assert "verdict-banner-verified" in r.text
    assert "Verified — this decision's full chain of evidence is intact." in r.text


def test_reports_detail_issues_banner_surfaces_failing_line(client):
    """Tampered envelope (hash_match=False) flips the banner to red and
    surfaces the *hash* plain-language sentence as the sub-line."""
    _inject_envelope(client, decision_id="dec-tampered-1", verdict="issues")
    r = client.get("/ui/reports/dec-tampered-1")
    assert r.status_code == 200
    assert "verdict-banner-issues" in r.text
    assert "Issues found — one or more checks failed." in r.text
    # The four-check list contains the plain-language fail sentence for
    # hash_match. The banner sub-line picks the same string.
    assert "the hash does not match what was anchored" in r.text


def test_reports_detail_pending_banner_when_unverified(client):
    """A never-verified envelope renders the neutral pending banner."""
    _inject_envelope(client, decision_id="dec-pending-2", verdict="pending")
    r = client.get("/ui/reports/dec-pending-2")
    assert r.status_code == 200
    assert "verdict-banner-pending" in r.text
    assert "Pending — verification hasn't completed yet." in r.text


# ── Verify-independently accordion ─────────────────────────────────────


def test_reports_detail_independent_verify_lead_with_plugin(client):
    """Easy-mode accordion shows ``ar-io-mlflow verify trace <id>``."""
    _inject_envelope(client, decision_id="dec-iv-1", verdict="verified")
    r = client.get("/ui/reports/dec-iv-1")
    assert r.status_code == 200
    assert "ar-io-mlflow verify trace dec-iv-1" in r.text
    assert "Easy mode" in r.text


def test_reports_detail_independent_verify_advanced_path_present(client):
    """Nested 'Verify without trusting the plugin (advanced)' details
    contains the curl + nacl/jcs flow with the decision's TX inlined."""
    _inject_envelope(client, decision_id="dec-iv-2", verdict="verified")
    r = client.get("/ui/reports/dec-iv-2")
    assert r.status_code == 200
    assert "Verify without trusting the plugin (advanced)" in r.text
    assert "curl -sS https://arweave.net/TX-VERIFIED-dec-iv-2" in r.text
    assert "from nacl.signing import VerifyKey" in r.text
    assert "jcs.canonicalize" in r.text


def test_reports_detail_no_raw_evidence_accordion(client):
    """Raw evidence section is deliberately absent — audit-reader page
    doesn't duplicate the technical detail page's JSON viewer."""
    _inject_envelope(client, decision_id="dec-noraw-1", verdict="verified")
    r = client.get("/ui/reports/dec-noraw-1")
    assert r.status_code == 200
    assert "Raw evidence" not in r.text


# ── Print CSS ──────────────────────────────────────────────────────────


def test_reports_detail_print_css_present(client):
    """Print stylesheet block exists and strips topbar/footer."""
    _inject_envelope(client, decision_id="dec-print-1", verdict="verified")
    r = client.get("/ui/reports/dec-print-1")
    assert r.status_code == 200
    assert "@media print" in r.text
    assert ".topbar" in r.text  # part of the print-hide selectors


# ── Nav ────────────────────────────────────────────────────────────────


def test_reports_nav_appears_in_topbar(client):
    """Reports nav link is present on every page after Lineage."""
    r = client.get("/ui/datasets")
    assert r.status_code == 200
    assert 'href="/ui/reports"' in r.text
    # Same on /ui/reports itself.
    r = client.get("/ui/reports")
    assert r.status_code == 200
    assert 'href="/ui/reports"' in r.text
