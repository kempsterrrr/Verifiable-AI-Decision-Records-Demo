# Architecture

## Two-part repository

This repo contains two distinct deliverables:

- **`ario_mlflow/`** — the **product**. A standalone MLflow plugin that any MLflow user can adopt with `pip install -e .` (eventually `pip install ario-mlflow`). Provides verifiable provenance for training, registration, promotion, and inference events.
- **`app/`** — a **sales-facing demo**. A FastAPI + Jinja2 web app that uses the plugin to demonstrate the verification flow visually. Hosted on Railway, used by sales / pre-sales to show prospective adopters what the plugin enables.

Code that's a real verification capability belongs in the plugin. Code that exists to *demonstrate* (UI rendering, tamper buttons, admin reset) lives in the demo.

## Core design

**Arweave is a witness; MLflow is the system of record.** The plugin's job is not to put MLflow data on Arweave — it's to commit to a hash that lets anyone verify "what's in MLflow now matches what was anchored at time X."

### Pure-commitment proofs (~500 bytes on Arweave)

Each lifecycle event produces a small signed envelope that goes on Arweave:

```json
{
  "event_id": "uuid",
  "event_type": "training_complete | model_registered | prediction",
  "subject": {"type": "mlflow_run", "run_id": "..."},
  "payload_hash": "SHA-256 of canonical bytes",
  "previous_hash": "prior event's payload_hash (or GENESIS)",
  "signed_at": "ISO-8601",
  "public_key": "Ed25519 public key",
  "signature": "Ed25519 signature over canonical(envelope - signature)"
}
```

The envelope is bounded 400–700 bytes. No source data goes on chain.

### Canonical bytes preserved in MLflow

The bytes that were hashed live in MLflow as `ario/payload.json` artifacts:

- `ario/payload.json` for training and registration
- `ario/predictions/<decision_id>/payload.json` for inferences

Verifiers download the artifact, re-hash, and compare to the envelope's `payload_hash`. They can also re-derive the canonical bytes from a separate MLflow surface (run params/metrics for training, the `ario.payload_json` trace tag for predictions) to detect post-anchoring tampering.

This is the AgentSystems Notary pattern — canonical bytes in caller's existing system of record, commitments on a public chain.

### RFC-8785 (JCS) canonicalization

Both the canonical payload and the signed envelope are serialised with RFC 8785 JSON Canonicalization Scheme. Any RFC-8785 verifier in any language reproduces the same bytes — interoperable with Notary, Sigstore, etc.

### Per-event-type chain semantics

Events form a DAG, not a strict line. Each event type chains independently:

- **Training proofs** chain via the registered model's `ario.last_training_hash` tag.
- **Registration proofs** chain to the source training run's `ario.training_tx` tag.
- **Prediction proofs** chain to the model version's `ario.registration_tx` tag (read at `VerifiedModel` init; never written at predict time).

This sidesteps MLflow's lack of a CAS primitive — the high-frequency busy case (predictions) never writes the chain head, eliminating races.

## Verification — three independent checks

Every verification surface (plugin CLI, demo UI, third-party verifier) runs the same three core checks plus an optional fourth:

| Check | What it proves |
|---|---|
| **Proof Found** | The pure-commitment envelope was retrieved from ar.io for the given TX ID |
| **Record Matches** | `ario/payload.json` in MLflow re-hashes to the envelope's `payload_hash`, AND re-deriving the canonical bytes from a separate MLflow surface produces the same bytes |
| **Signature Confirmed** | The envelope's signature verifies against the embedded public key |
| **ar.io attestation** *(optional)* | An ar.io gateway operator independently verified the on-chain proof |

Each returns one of three states: PASS, FAIL, or Pending. The first three checks are decisive for "is this proof intact"; ar.io attestation is an independent third-party witness.

## The evidence chain

Each event accumulates layers of evidence from independent parties:

1. **Commitment + Ed25519 signature** by the AI system — attests to the event
2. **Canonical payload in MLflow** (`ario/payload.json`) — the source bytes that were hashed
3. **Turbo receipt** with millisecond timestamp — independent service attests when the proof was submitted
4. **Arweave block** — network consensus confirms permanent storage
5. **ar.io Verify** (on-demand, gateway operator's signature) — independent verification of the anchored data

## Plugin API surface

Three integration points for MLflow users:

- **`ario_mlflow.anchor()`** inside `mlflow.start_run()` — training provenance
- **`ario_mlflow.ArioMlflowClient`** — registration + promotion (drop-in for `mlflow.tracking.MlflowClient`)
- **`ario_mlflow.VerifiedModel`** — inference (load-time integrity check + per-prediction anchoring)

Plus the CLI: `ario-mlflow verify run|model|trace <id>` and `ario-mlflow audit <model>/<version>`.

See `ario_mlflow/README.md` for usage examples.

## Demo architecture

The demo is a thin presentation layer:

- **FastAPI routes** in `app/main.py` and `app/ui.py` — predict, verify, decision/run/model detail pages
- **Jinja2 templates** in `templates/` — three-row verify cards, tamper buttons, model lineage view, demo admin page
- **Demo-only state** in `app/storage.py` (RecordStore — UI cache for decisions) and `app/lifecycle_store.py` (LifecycleStore — UI cache for training/registration events). Both are JSON files; not the source of truth (MLflow is).
- **Demo-only features** in `app/tamper.py` (tamper backend for the four tamper buttons), `app/reset.py` (Reset demo data feature), `app/main.py` `/tamper/*` and `/demo/*` endpoints (gated behind `demo_mode`).

The demo wraps the plugin — it does not extend or duplicate plugin behavior. Verification semantics are owned entirely by the plugin's `verify_signature`, `verify_anchored_bytes`, `verify_source_of_truth`, `verify_ario_attestation`, and `full_verify` functions.

## What gets anchored

A pure-commitment envelope per lifecycle event:

- **Training** — params, metrics, artifact checksums (re-hashed model artifacts)
- **Registration** — chains to training; re-hashes artifacts at registration time to catch model swaps
- **Prediction** — input hash, output hash, OTel trace IDs, model lineage; never the raw input/output values

Source data stays in the caller's MLflow. Arweave only holds the ~500-byte commitment per event.

## Why this design

- **Privacy** — no PII or business data leaves MLflow. Public Arweave only stores commitments.
- **Cost** — 500 bytes/event is sustainable for high-volume inference; full payloads on chain wouldn't be.
- **Interoperability** — JCS + standard SHA-256 + Ed25519 means any verifier in any language can independently check a proof. No "call our API to verify" lock-in.
- **Auditor independence** — `ario-mlflow verify` reproduces the entire check using only MLflow + an Arweave gateway. No dependency on the demo's internals.
