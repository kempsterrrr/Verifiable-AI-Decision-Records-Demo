"""CLI: ario-mlflow verify and audit commands."""

import argparse
import json
import os
import sys

import mlflow

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.anchor import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient, verify_record, verify_arweave, verify_ario


def _get_components():
    proof_engine = ProofEngine()
    anchor = ArweaveAnchor(
        os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
        os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
    )
    ario = ArioVerifyClient()
    return proof_engine, anchor, ario


def _print_verification(label: str, local: dict, arweave: dict, ario: dict | None):
    """Print a verification result block."""
    check = "\033[32m\u2713\033[0m"
    cross = "\033[31m\u2717\033[0m"
    pending = "\033[33m?\033[0m"

    hash_ok = local.get("hash_valid", False)
    sig_ok = local.get("signature_valid", False)
    print(f"  Local:    {check if hash_ok else cross} hash {'valid' if hash_ok else 'INVALID'}, "
          f"{check if sig_ok else cross} signature {'valid' if sig_ok else 'INVALID'}")

    if arweave.get("arweave_data_found"):
        match = arweave.get("hash_match", False)
        print(f"  Arweave:  {check if match else cross} permanent copy {'matches' if match else 'MISMATCH'}")
    elif arweave.get("reason") == "no_tx_id":
        print(f"  Arweave:  {pending} not anchored")
    else:
        print(f"  Arweave:  {pending} fetch failed")

    if ario:
        level = ario.get("level")
        attester = ario.get("attested_by", "unknown")
        if level:
            print(f"  ar.io:    {check} attested (Level {level}) by {attester}")
            if ario.get("report_url"):
                print(f"            Report: {ario['report_url']}")
        else:
            print(f"  ar.io:    {pending} attestation pending")
    else:
        print(f"  ar.io:    {pending} not checked")


def cmd_verify_run(args):
    """Verify a training run's proof record."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(args.run_id)
    tx_id = run.data.tags.get("ario.training_tx")

    if not tx_id:
        print(f"Run {args.run_id}: no ario.training_tx tag found. Not anchored.")
        return 1

    print(f"Verifying training run {args.run_id}")
    print(f"  TX: {tx_id}")

    # Fetch the proof from Arweave
    proof_data = anchor.fetch_proof(tx_id)
    if not proof_data:
        print("  Could not fetch proof from Arweave.")
        return 1

    local = proof_engine.verify_local(proof_data)
    arweave_result = {"arweave_data_found": True, "hash_match": local.get("hash_valid", False)}
    ario_result = verify_ario({"arweave_tx_id": tx_id}, ario_client)

    _print_verification("Training Run", local, arweave_result, ario_result)
    return 0 if local.get("overall", False) else 1


def cmd_verify_model(args):
    """Verify a model version's registration proof."""
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

    proof_data = anchor.fetch_proof(tx_id)
    if not proof_data:
        print("  Could not fetch proof from Arweave.")
        return 1

    local = proof_engine.verify_local(proof_data)
    arweave_result = {"arweave_data_found": True, "hash_match": local.get("hash_valid", False)}
    ario_result = verify_ario({"arweave_tx_id": tx_id}, ario_client)

    _print_verification("Registration", local, arweave_result, ario_result)
    return 0 if local.get("overall", False) else 1


def cmd_audit(args):
    """Audit full chain of custody for a model version."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    parts = args.model.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else "1"

    print(f"Auditing chain of custody: {name}/v{version}")
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
        proof_data = anchor.fetch_proof(training_tx)
        if proof_data:
            local = proof_engine.verify_local(proof_data)
            ario_result = verify_ario({"arweave_tx_id": training_tx}, ario_client)
            _print_verification("Training", local, {"arweave_data_found": True, "hash_match": local.get("hash_valid")}, ario_result)
            if not local.get("overall"):
                all_ok = False
        else:
            print("  Could not fetch proof from Arweave.")
            all_ok = False
    else:
        print("  Not anchored.")

    # 2. Registration
    registration_tx = mv.tags.get("ario.registration_tx")
    print(f"\nRegistration (v{version}):")
    if registration_tx:
        proof_data = anchor.fetch_proof(registration_tx)
        if proof_data:
            local = proof_engine.verify_local(proof_data)
            ario_result = verify_ario({"arweave_tx_id": registration_tx}, ario_client)
            _print_verification("Registration", local, {"arweave_data_found": True, "hash_match": local.get("hash_valid")}, ario_result)
            if not local.get("overall"):
                all_ok = False
        else:
            print("  Could not fetch proof from Arweave.")
            all_ok = False
    else:
        print("  Not anchored.")

    # 3. Promotion
    promotion_tx = mv.tags.get("ario.promotion_tx")
    print(f"\nPromotion ({mv.current_stage}):")
    if promotion_tx:
        proof_data = anchor.fetch_proof(promotion_tx)
        if proof_data:
            local = proof_engine.verify_local(proof_data)
            ario_result = verify_ario({"arweave_tx_id": promotion_tx}, ario_client)
            _print_verification("Promotion", local, {"arweave_data_found": True, "hash_match": local.get("hash_valid")}, ario_result)
            if not local.get("overall"):
                all_ok = False
        else:
            print("  Could not fetch proof from Arweave.")
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


def main():
    parser = argparse.ArgumentParser(prog="ario-mlflow", description="ar.io MLflow verification CLI")
    subparsers = parser.add_subparsers(dest="command")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify a proof record")
    verify_sub = verify_parser.add_subparsers(dest="verify_type")

    run_parser = verify_sub.add_parser("run", help="Verify a training run")
    run_parser.add_argument("run_id", help="MLflow run ID")

    model_parser = verify_sub.add_parser("model", help="Verify a model registration")
    model_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit full chain of custody")
    audit_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    args = parser.parse_args()

    if args.command == "verify":
        if args.verify_type == "run":
            sys.exit(cmd_verify_run(args))
        elif args.verify_type == "model":
            sys.exit(cmd_verify_model(args))
        else:
            verify_parser.print_help()
    elif args.command == "audit":
        sys.exit(cmd_audit(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
