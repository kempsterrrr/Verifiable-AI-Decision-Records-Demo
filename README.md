# Verifiable AI Decision Records

Tamper-evident AI audit trail anchored to Arweave.

## What This Demonstrates

Every AI prediction creates a **decision record** that is:

1. **Traced** — OpenTelemetry captures runtime context (trace ID, span ID)
2. **Linked to model lineage** — MLflow records which model version produced the output
3. **Hashed & signed** — SHA-256 hash chain + Ed25519 digital signature
4. **Anchored to Arweave** — Immutable permanent storage via ArDrive Turbo SDK
5. **Receipted** — Turbo upload receipt with millisecond timestamp and signed attestation
6. **Independently verifiable** — ar.io Verify produces on-demand attestations

If someone tampers with the local record, the **Arweave-anchored copy** remains intact and verifiable.

## Architecture

```
User Input
  |
  v
FastAPI /predict
  |---> MLflow (model lineage: run_id, version, artifact_uri)
  |---> OpenTelemetry (trace_id, span_id)
  |---> Inference (sklearn LogisticRegression)
  |
  v
Decision Record (canonical JSON)
  |---> SHA-256 hash + hash chain (previous_hash)
  |---> Ed25519 signature
  |
  v
Proof
  |---> Turbo SDK upload to Arweave (returns signed receipt with ms timestamp)
  |
  v
Local storage: proof + anchoring metadata (Arweave TX, Turbo receipt)
Arweave: proof only (record, hash, chain link, signature, public key)

  ... later, on demand ...

/verify endpoint
  |---> Local verification (re-hash, check signature)
  |---> External verification (fetch from Arweave, compare)
  |---> ar.io Verify attestation (independent third-party check)
```

### Recording vs Verification

Recording and verification are separate concerns:

**Recording** happens at prediction time — the system creates a proof, anchors it to Arweave, and stores the Turbo receipt. This is fast (~7s) and produces an immutable audit trail.

**Verification** happens later, on demand — when an auditor, operator, or regulator needs to confirm records are still intact. The `/verify/{id}` endpoint re-hashes the record, checks the Ed25519 signature, fetches the proof from Arweave for comparison, and requests an ar.io Verify attestation.

### What Gets Anchored

The Arweave anchor contains only what's needed for independent verification:

```json
{
  "record": { "decision_id": "...", "prediction": {...}, ... },
  "record_hash": "SHA-256 of canonical JSON",
  "previous_hash": "prior record's hash (or GENESIS)",
  "signature": "Ed25519 signature",
  "public_key": "Ed25519 public key"
}
```

An auditor can verify any proof with standard cryptographic tools — no dependency on ar.io, MLflow, or any external service.

### The Evidence Chain

Each prediction creates multiple layers of evidence from independent parties:

1. **Proof** (Ed25519 signature) — the AI system attests to the decision
2. **Turbo receipt** (Turbo's signature + ms timestamp) — independent service attests when the proof was submitted
3. **Arweave block** — network consensus confirms permanent storage
4. **ar.io Verify** (on-demand, gateway operator's signature) — independent verification of the anchored data

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (optional — auto-trains on first prediction)

```bash
python scripts/train_model.py
```

### 3. Start the App

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Dashboard

Navigate to [http://localhost:8000](http://localhost:8000)

## Arweave Setup (Optional)

To enable permanent anchoring:

1. Generate an Arweave wallet at [arweave.app](https://arweave.app)
2. Save the wallet JSON as `keys/arweave_wallet.json`
3. Fund with credits via [ardrive.io](https://ardrive.io)

Without a wallet, the app runs in **local proof mode** — hashing, signing, and verification all work locally.

## Demo Walkthrough

### 1. Make a Prediction

Submit the form with iris flower measurements. Default values are pre-filled.

### 2. View the Decision Record

Click a decision ID to see the full record:
- **Prediction** — class, probabilities with visual bars, features used
- **Model metadata** — MLflow run ID, version, artifact URI
- **Trace context** — OpenTelemetry trace/span IDs
- **Proof layer** — record hash, chain link, Ed25519 signature
- **Arweave anchoring** — transaction ID, gateway URL
- **Turbo upload receipt** — millisecond timestamp, wallet owner, signed receipt

### 3. Verify a Record

Click **Re-Verify** to run on-demand verification:
- **Local verification** — re-hashes the record and checks the Ed25519 signature
- **External verification** — fetches the proof from Arweave and compares
- **ar.io Verify** — requests an independent attestation from the ar.io gateway

### 4. Tamper with a Record

Click **Tamper** to modify the local record's output hash, then **Re-Verify** — local verification FAILS but the Arweave copy is still intact, proving the local record was modified after anchoring.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run prediction, create decision record (JSON body) |
| `/predict-form` | POST | Same, from HTML form (redirects to detail) |
| `/decisions` | GET | List all decision records |
| `/decisions/{id}` | GET | Get a single decision record |
| `/verify/{id}` | POST | Verify a decision (local + Arweave + ar.io Verify) |
| `/tamper/{id}` | POST | Tamper with a record (demo only) |

### Example: Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Example: Verify a Decision

```bash
curl -X POST http://localhost:8000/verify/<decision_id>
```

## CLI Verification

Verify records independently from the command line:

```bash
# Verify all records
python scripts/verify.py --all

# Verify a specific record
python scripts/verify.py <decision_id>
```

## Configuration

Environment variables (prefix: `VAIDR_`):

| Variable | Default | Description |
|---|---|---|
| `VAIDR_ARWEAVE_WALLET_PATH` | `keys/arweave_wallet.json` | Arweave JWK wallet |
| `VAIDR_MLFLOW_TRACKING_URI` | `mlruns` | MLflow tracking directory |
| `VAIDR_MLFLOW_MODEL_NAME` | `iris-classifier` | Registered model name |
| `VAIDR_ARIO_GATEWAY_HOST` | `turbo-gateway.com` | ar.io gateway hostname |
| `VAIDR_ARIO_VERIFY_URL` | `https://vilenarios.com/local/verify` | ar.io Verify service URL |
| `VAIDR_RECORDS_FILE` | `data/records.json` | Local record storage path |

## How It Works

### MLflow — Model Lineage
Every prediction is tied to a specific model version. MLflow captures the run ID, model version, and artifact URI, creating an auditable link between the model and its decisions.

### OpenTelemetry — Runtime Trace
Each prediction creates a distributed trace. The trace ID and span ID are embedded in the decision record, allowing correlation with infrastructure monitoring.

### Proof Layer — Integrity
Decision records are serialized to deterministic canonical JSON (sorted keys, compact separators, floats normalized to 6 decimal places), then:
- **SHA-256 hashed** — any change to the record changes the hash
- **Hash-chained** — each record links to the previous record's hash
- **Ed25519 signed** — cryptographic proof of record origin

### ArDrive Turbo SDK — Anchoring
The proof is uploaded to Arweave permanent storage via the Turbo SDK. The upload returns a signed receipt with a millisecond-precision timestamp — an independent attestation of when the proof was submitted. Once confirmed on Arweave, the data is immutable and publicly accessible.

### ar.io Verify — Independent Attestation
When verification is requested, ar.io Verify independently fetches the Arweave data, recomputes hashes, checks signatures where available, and produces a signed attestation. Verification levels:
- **Level 1** — Data found on the network, verification in progress
- **Level 2** — Data hash confirmed, signature not yet available
- **Level 3** — Digital signature verified, full authenticity confirmed

### Auditor Verification
An auditor can independently verify any proof with standard cryptographic tools:
1. Fetch the proof from Arweave using the TX ID
2. Recompute `SHA-256(canonical_json(record))` and compare to `record_hash`
3. Verify the Ed25519 signature against the `public_key`
4. Check hash chain links across records
5. No dependency on ar.io, MLflow, or any external service required
