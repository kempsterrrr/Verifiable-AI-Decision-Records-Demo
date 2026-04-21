#!/usr/bin/env python3
"""CLI tool for independent verification of decision records."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from ario_mlflow.proof import ProofEngine, canonical_json, hash_data


def verify_record(envelope: dict, proof_engine: ProofEngine) -> dict:
    """Verify a single record envelope."""
    result = proof_engine.verify_local(envelope)
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify AI decision records")
    parser.add_argument("decision_id", nargs="?", help="Decision ID to verify")
    parser.add_argument("--all", action="store_true", help="Verify all records")
    args = parser.parse_args()

    if not args.decision_id and not args.all:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()
    proof_engine = ProofEngine(settings.ed25519_private_key_path, settings.ed25519_public_key_path)

    # Load records
    if not os.path.exists(settings.records_file):
        print("No records file found.")
        sys.exit(1)

    with open(settings.records_file) as f:
        records = json.load(f)

    if not records:
        print("No records to verify.")
        sys.exit(1)

    # Filter
    if args.all:
        targets = records
    else:
        targets = [r for r in records if r.get("record", {}).get("decision_id") == args.decision_id]
        if not targets:
            print(f"Decision {args.decision_id} not found.")
            sys.exit(1)

    # Verify chain
    print(f"Verifying {len(targets)} record(s)...\n")
    all_valid = True

    for i, envelope in enumerate(targets):
        decision_id = envelope["record"]["decision_id"]
        result = verify_record(envelope, proof_engine)

        status = "VALID" if result["overall"] else "INVALID"
        symbol = "+" if result["overall"] else "x"
        print(f"[{symbol}] {decision_id[:12]}... — {status}")
        print(f"    Hash:      {'PASS' if result['hash_valid'] else 'FAIL'}")
        print(f"    Signature: {'PASS' if result['signature_valid'] else 'FAIL'}")

        if not result["hash_valid"]:
            print(f"    Stored:    {result['stored_hash'][:32]}...")
            print(f"    Computed:  {result['computed_hash'][:32]}...")

        if envelope.get("arweave_tx_id"):
            print(f"    Arweave:   {envelope['arweave_tx_id']}")
        else:
            print(f"    Arweave:   Not anchored")

        # Verify chain link
        if i > 0:
            expected_prev = records[records.index(envelope) - 1]["record_hash"] if envelope in records else "?"
            actual_prev = envelope["previous_hash"]
            chain_ok = actual_prev == expected_prev or actual_prev == "GENESIS"
            print(f"    Chain:     {'LINKED' if chain_ok else 'BROKEN'}")

        print()

        if not result["overall"]:
            all_valid = False

    if all_valid:
        print("All records verified successfully.")
    else:
        print("Some records failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
