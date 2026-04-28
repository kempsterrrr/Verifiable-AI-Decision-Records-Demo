"""CLI: ario-mlflow verify and audit commands.

Updated for the pure-commitment redesign \u2014 runs the four-check
verification flow (signature / anchored bytes / live MLflow / ar.io
Verify Level 3) instead of the legacy three-level helpers.
"""

import argparse
import json
import os
import sys
import tempfile

import mlflow

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import (
    ArioVerifyClient,
    verify_signature,
    verify_anchored_bytes,
    verify_source_of_truth,
    verify_ario_attestation,
    _compute_overall_ok,
)
from ario_mlflow.report import generate_verification_html


def _get_components():
    proof_engine = ProofEngine()
    anchor = ArweaveAnchor(
        os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
        os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
    )
    ario = ArioVerifyClient()
    return proof_engine, anchor, ario


def _print_check(label: str, result: dict, value_when_ok: str | None = None):
    """Print one check line. Status: \u2713 / \u2717 / ? (not applicable)."""
    check = "\033[32m\u2713\033[0m"
    cross = "\033[31m\u2717\033[0m"
    pending = "\033[33m?\033[0m"

    ok = result.get("ok")
    if ok is True:
        symbol = check
        suffix = value_when_ok or "passed"
    elif ok is False:
        symbol = cross
        suffix = result.get("reason", "FAILED")
    else:
        symbol = pending
        suffix = result.get("reason", "not applicable")
    print(f"  {label:<22} {symbol} {suffix}")


def _print_four_checks(
    sig: dict,
    bytes_check: dict,
    sot: dict,
    ario_attestation: dict,
):
    """Print the four-check verification panel for a single envelope.

    Always surfaces the ar.io Verify attestation level when present \u2014
    even on a "below threshold" failure \u2014 so the user can see how the
    TX is maturing. (Per ROADMAP "Receipts vs. attestation as a
    two-stage verify UX", attestation level is fundamentally a maturity
    gradient, not a binary pass/fail.)
    """
    _print_check("Cryptographic", sig, "signature valid")
    _print_check("Anchored bytes", bytes_check, "intact")
    _print_check("Source of truth", sot, "live MLflow matches anchored bytes")

    # ar.io Verify \u2014 show the maturity level whenever the API returned
    # something, regardless of whether it passed the threshold. Helps
    # users see "Level 1, growing" vs. "TX missing" at a glance.
    check = "\033[32m\u2713\033[0m"
    cross = "\033[31m\u2717\033[0m"
    pending = "\033[33m?\033[0m"
    ok = ario_attestation.get("ok")
    level = ario_attestation.get("attestation_level")
    attester = ario_attestation.get("attested_by") or "unknown"
    threshold = ario_attestation.get("min_attestation_level")

    if ok is True and level is not None:
        print(f"  {'ar.io Verify':<22} {check} Level {level} by {attester}")
        if ario_attestation.get("report_url"):
            print(f"  {'':>22} report: {ario_attestation['report_url']}")
    elif ok is False and ario_attestation.get("reason") == "attestation_level_below_threshold":
        # Show the actual level + the threshold so the user knows the
        # TX is maturing, just hasn't reached the bar yet.
        print(
            f"  {'ar.io Verify':<22} {cross} Level {level} (below threshold "
            f"{threshold}) by {attester}"
        )
        print(
            f"  {'':>22} TX is indexed but not yet at the configured "
            f"attestation bar. Re-run later to check progression."
        )
        if ario_attestation.get("report_url"):
            print(f"  {'':>22} report: {ario_attestation['report_url']}")
    elif ok is False:
        print(f"  {'ar.io Verify':<22} {cross} {ario_attestation.get('reason', 'FAILED')}")
    else:
        # ok is None: not applicable / not checked
        print(f"  {'ar.io Verify':<22} {pending} {ario_attestation.get('reason', 'not checked')}")


def _verify_envelope_for_tx(
    tx_id: str,
    proof_engine: ProofEngine,
    anchor: ArweaveAnchor,
    ario_client: ArioVerifyClient,
    mlflow_client,
) -> tuple[dict, bool]:
    """Fetch an envelope from Arweave and run all four checks.

    Returns ``(combined_result, overall_ok)``. ``combined_result`` has
    keys ``signature`` / ``anchored_bytes`` / ``source_of_truth`` /
    ``ario_attestation`` for callers that want to programmatically
    inspect; the printed output goes to stdout.
    """
    envelope = anchor.fetch_proof(tx_id)
    if not envelope:
        print(f"  Could not fetch envelope from Arweave for TX {tx_id}.")
        return {}, False

    sig = verify_signature(envelope, proof_engine)
    bytes_check = verify_anchored_bytes(envelope, mlflow_client)
    sot = (
        verify_source_of_truth(envelope, bytes_check.get("payload_bytes") or b"", mlflow_client)
        if bytes_check.get("payload_bytes")
        else {"ok": None, "reason": bytes_check.get("reason", "no_payload_to_compare")}
    )
    # For ar.io Verify, inject the TX ID the caller already knows. The
    # envelope itself doesn't carry it (the TX IS its address).
    envelope_with_tx = dict(envelope)
    envelope_with_tx["_tx_id"] = tx_id
    ario_result = verify_ario_attestation(envelope_with_tx, ario_client)

    _print_four_checks(sig, bytes_check, sot, ario_result)

    # Use the shared overall-ok logic so CLI and full_verify() agree.
    # For training/registration envelopes, ok=None on signature /
    # anchored_bytes / source_of_truth fails overall \u2014 None means
    # "couldn't verify", not "fine."
    overall = _compute_overall_ok(envelope, sig, bytes_check, sot, ario_result)
    overall_ok = bool(overall)  # CLI returns bool; None coerces to False

    return {
        "envelope": envelope,
        "signature": sig,
        "anchored_bytes": bytes_check,
        "source_of_truth": sot,
        "ario_attestation": ario_result,
    }, overall_ok


def _verification_run_tags(verification: dict | None) -> dict[str, str]:
    """Map a normalized ar.io Verify result to MLflow tag key/values."""
    tags: dict[str, str] = {}
    if not verification:
        return tags
    level = verification.get("attestation_level")
    if level is not None:
        tags["ario.verify_status"] = "verified"
        tags["ario.attestation_level"] = str(level)
    if verification.get("report_url"):
        tags["ario.report_url"] = verification["report_url"]
    if verification.get("attested_by"):
        tags["ario.attested_by"] = verification["attested_by"]
    if verification.get("attested_at"):
        tags["ario.attested_at"] = verification["attested_at"]
    return tags


def _regenerate_html(
    run_id: str,
    proof: dict,
    tx_id: str,
    arweave_url: str | None,
    artifact_hash: str | None,
    artifact_verified: bool | None,
    verification: dict | None,
    filename: str,
    cli_verify_cmd: str | None = None,
):
    """Regenerate an ario/<filename> artifact on a run with updated verification."""
    anchor_result = {"tx_id": tx_id, "url": arweave_url or "", "receipt": None}
    html_content = generate_verification_html(
        proof,
        anchor_result,
        artifact_hash=artifact_hash,
        artifact_verified=artifact_verified,
        verification=verification,
        cli_verify_cmd=cli_verify_cmd,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ario_dir = os.path.join(tmpdir, "ario")
        os.makedirs(ario_dir)
        with open(os.path.join(ario_dir, filename), "w") as f:
            f.write(html_content)
        client = mlflow.tracking.MlflowClient()
        client.log_artifacts(run_id, ario_dir, "ario")


def cmd_verify_run(args):
    """Verify a training run's commitment via the four-check flow."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(args.run_id)
    tx_id = run.data.tags.get("ario.training_tx")

    if not tx_id:
        print(f"Run {args.run_id}: no ario.training_tx tag found. Not anchored.")
        return 1

    print(f"Verifying training run {args.run_id}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(tx_id, proof_engine, anchor, ario_client, client)
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    tags = _verification_run_tags(ario)
    if tags:
        for key, value in tags.items():
            client.set_tag(args.run_id, key, value)
        print(f"  -> updated {len(tags)} MLflow tag(s) on run")

    return 0 if ok else 1


def cmd_verify_model(args):
    """Verify a model version's registration commitment via the four-check flow."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    parts = args.model.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else "1"

    mv = client.get_model_version(name, version)
    tx_id = mv.tags.get("ario.registration_tx")

    if not tx_id:
        print(f"Model {name}/v{version}: no ario.registration_tx tag found. Not anchored.")
        return 1

    print(f"Verifying model registration {name}/v{version}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(tx_id, proof_engine, anchor, ario_client, client)
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    tags = _verification_run_tags(ario)
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name, version, key, value)
        print(f"  -> updated {len(tags)} MLflow tag(s) on model version")

    return 0 if ok else 1


def cmd_verify_trace(args):
    """Verify a prediction trace's commitment via the four-check flow.

    Note: predictions don't write a payload.json artifact (per the
    privacy-preserving design — canonical fields are mirrored as trace
    tags). Check 2 (anchored bytes intact) and check 3 (live MLflow
    matches) report as not-applicable. Check 1 (signature) and check 4
    (ar.io Verify) work normally. Auditors with the raw input/output
    can verify input_hash / output_hash directly.
    """
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    try:
        trace = mlflow.get_trace(args.trace_id)
    except Exception as e:
        print(f"Could not load trace {args.trace_id}: {e}")
        return 1

    if trace is None:
        print(f"Trace {args.trace_id} not found.")
        return 1

    tags = getattr(trace.info, "tags", {}) or {}
    # Prefer the new tag name (ario.prediction_tx); accept the legacy
    # (ario.arweave_tx) for backwards compat with any traces anchored
    # under v1 still in MLflow.
    tx_id = tags.get("ario.prediction_tx") or tags.get("ario.arweave_tx")

    if not tx_id:
        print(f"Trace {args.trace_id}: no ario.prediction_tx tag found. Not anchored yet.")
        return 1

    print(f"Verifying trace {args.trace_id}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(tx_id, proof_engine, anchor, ario_client, client)
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    back_tags = _verification_run_tags(ario)
    if back_tags:
        for key, value in back_tags.items():
            try:
                mlflow.set_trace_tag(args.trace_id, key, value)
            except Exception as e:
                print(f"  ! failed to set trace tag {key}: {e}")
        print(f"  -> updated {len(back_tags)} MLflow trace tag(s)")

    return 0 if ok else 1


def cmd_audit(args):
    """Audit the full lineage (training → registration → promotion) for a model version."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    parts = args.model.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else "1"

    print(f"Auditing model lineage: {name}/v{version}")
    print("=" * 50)

    mv = client.get_model_version(name, version)
    all_ok = True

    # 1. Training
    training_tx = None
    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
            training_tx = run.data.tags.get("ario.training_tx")
        except Exception:
            pass

    print(f"\nTraining (run {mv.run_id or 'unknown'}):")
    if training_tx:
        _, ok = _verify_envelope_for_tx(training_tx, proof_engine, anchor, ario_client, client)
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 2. Registration
    registration_tx = mv.tags.get("ario.registration_tx")
    print(f"\nRegistration (v{version}):")
    if registration_tx:
        _, ok = _verify_envelope_for_tx(registration_tx, proof_engine, anchor, ario_client, client)
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 3. Promotion
    promotion_tx = mv.tags.get("ario.promotion_tx")
    print(f"\nPromotion ({mv.current_stage}):")
    if promotion_tx:
        _, ok = _verify_envelope_for_tx(promotion_tx, proof_engine, anchor, ario_client, client)
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 4. Artifact integrity
    artifact_hash = None
    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
            artifact_hash = run.data.tags.get("ario.artifact_hash")
        except Exception:
            pass

    print(f"\nArtifact integrity:")
    if artifact_hash:
        print(f"  Anchored hash: {artifact_hash[:24]}...")
    else:
        print("  No artifact hash recorded.")

    print(f"\n{'=' * 50}")
    check = "\033[32m\u2713\033[0m"
    cross = "\033[31m\u2717\033[0m"
    print(f"Overall: {check + ' All checks passed' if all_ok else cross + ' Issues found'}")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. Exposed so tests can exercise the real wiring."""
    parser = argparse.ArgumentParser(prog="ario-mlflow", description="ar.io MLflow verification CLI")
    subparsers = parser.add_subparsers(dest="command")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify a proof record")
    verify_sub = verify_parser.add_subparsers(dest="verify_type")

    run_parser = verify_sub.add_parser("run", help="Verify a training run")
    run_parser.add_argument("run_id", help="MLflow run ID")

    model_parser = verify_sub.add_parser("model", help="Verify a model registration")
    model_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    trace_parser = verify_sub.add_parser("trace", help="Verify an inference trace")
    trace_parser.add_argument("trace_id", help="MLflow trace ID")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit full model lineage (training → registration → promotion)")
    audit_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "verify":
        if args.verify_type == "run":
            sys.exit(cmd_verify_run(args))
        elif args.verify_type == "model":
            sys.exit(cmd_verify_model(args))
        elif args.verify_type == "trace":
            sys.exit(cmd_verify_trace(args))
        else:
            # Print help for the verify subparser by re-parsing.
            parser.parse_args(["verify", "--help"])
    elif args.command == "audit":
        sys.exit(cmd_audit(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
