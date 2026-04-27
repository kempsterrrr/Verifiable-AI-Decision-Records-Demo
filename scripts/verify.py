#!/usr/bin/env python3
"""CLI tool for independent verification of decision records.

Rewritten in Phase 4 to read from MLflow traces + Arweave instead of the
now-deleted records_file setting.  Each trace that carries an
``ario.arweave_tx`` tag is independently verified by fetching the proof
envelope from Arweave and running ProofEngine.verify_local() on it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow

from app.config import get_settings
from ario_mlflow.proof import ProofEngine
from ario_mlflow.arweave import ArweaveAnchor


def main():
    parser = argparse.ArgumentParser(description="Verify AI decision records")
    parser.add_argument("decision_id", nargs="?", help="Decision ID to verify (omit for --all)")
    parser.add_argument("--all", action="store_true", help="Verify all anchored records")
    parser.add_argument("--max-results", type=int, default=100, help="Max traces to scan (default 100)")
    args = parser.parse_args()

    if not args.decision_id and not args.all:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    proof_engine = ProofEngine(
        settings.ed25519_private_key_path,
        settings.ed25519_public_key_path,
    )
    anchor = ArweaveAnchor(settings.arweave_wallet_path, settings.ario_gateway_host)

    # Discover all experiment IDs so we search across every experiment.
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments] or ["0"]
    except Exception:
        experiment_ids = ["0"]

    # Fetch traces that have been anchored (ario.decision_id tag present).
    try:
        traces = client.search_traces(
            experiment_ids=experiment_ids,
            filter_string="tags.`ario.decision_id` != ''",
            max_results=args.max_results,
        )
    except Exception as exc:
        print(f"Could not query MLflow traces: {exc}")
        sys.exit(1)

    if not traces:
        print("No anchored decisions found.")
        sys.exit(0)

    # Filter to a specific decision ID if requested.
    if args.decision_id:
        traces = [
            t for t in traces
            if dict(getattr(t.info, "tags", {}) or {}).get("ario.decision_id") == args.decision_id
        ]
        if not traces:
            print(f"Decision {args.decision_id} not found in MLflow traces.")
            sys.exit(1)

    # Verify each anchored trace.
    print(f"Verifying {len(traces)} record(s)...\n")
    all_valid = True

    for trace in traces:
        tags = dict(getattr(trace.info, "tags", {}) or {})
        decision_id = tags.get("ario.decision_id", "<unknown>")
        tx_id = tags.get("ario.arweave_tx") or tags.get("ario.decision_tx")

        if not tx_id:
            print(f"[-] {decision_id[:12]}... — SKIPPED (no Arweave tx)")
            print()
            continue

        proof = anchor.fetch_proof(tx_id)
        if not proof:
            print(f"[?] {decision_id[:12]}... — FETCH FAILED")
            print(f"    Arweave:   {tx_id}")
            print()
            all_valid = False
            continue

        result = proof_engine.verify_local(proof)
        status = "VALID" if result.get("overall") else "INVALID"
        symbol = "+" if result.get("overall") else "x"

        print(f"[{symbol}] {decision_id[:12]}... — {status}")
        print(f"    Hash:      {'PASS' if result.get('hash_valid') else 'FAIL'}")
        print(f"    Signature: {'PASS' if result.get('signature_valid') else 'FAIL'}")
        print(f"    Arweave:   {tx_id}")

        if not result.get("hash_valid"):
            stored = result.get("stored_hash", "")
            computed = result.get("computed_hash", "")
            if stored:
                print(f"    Stored:    {stored[:32]}...")
            if computed:
                print(f"    Computed:  {computed[:32]}...")

        print()

        if not result.get("overall"):
            all_valid = False

    if all_valid:
        print("All records verified successfully.")
    else:
        print("Some records failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
