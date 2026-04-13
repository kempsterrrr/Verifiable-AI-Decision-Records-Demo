# Verifiable AI Decision Records

Tamper-evident AI audit trail anchored to Arweave.

## What This Demonstrates

Every AI prediction creates a **decision record** that is:

1. **Traced** — OpenTelemetry captures runtime context (trace ID, span ID)
2. **Linked to model lineage** — MLflow records which model version produced the output
3. **Hashed & signed** — SHA-256 hash chain + Ed25519 digital signature
4. **Anchored to Arweave** — Immutable permanent storage via ArDrive Turbo SDK
5. **Receipted** — Turbo upload receipt with millisecond timestamp and signed attestation
6. **Independently verifiable** — ar.io Gateway + ar.io Verify produces attestations

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
Clean Proof (5 fields: record, record_hash, previous_hash, signature, public_key)
  |---> Turbo SDK upload to Arweave (returns signed receipt with ms timestamp)
  |---> ar.io Verify attestation
  |
  v
Local storage: proof + operational metadata (Arweave TX, Turbo receipt, ar.io verify status)
Arweave: clean proof only (no metadata, no null fields)
```

### What Gets Anchored vs What's Stored Locally

The **Arweave anchor** contains only the self-contained, independently verifiable proof:

```json
{
  "record": { "decision_id": "...", "prediction": {...}, ... },
  "record_hash": "SHA-256 of canonical JSON",
  "previous_hash": "prior record's hash (or GENESIS)",
  "signature": "Ed25519 signature",
  "public_key": "Ed25519 public key"
}
```

**Local storage** wraps the proof with operational metadata:

- `arweave_tx_id` / `arweave_url` — where the proof is anchored
- `turbo_receipt` — the full signed upload receipt from Turbo (timestamp, signature, owner wallet)
- `ario_verify_*` — ar.io Verify attestation status

This separation ensures the anchored artifact is clean for auditors while local records retain full operational context.

### The Evidence Chain

Each prediction creates four layers of evidence from independent parties:

1. **Proof** (Ed25519 signature) — the AI system attests to the decision
2. **Turbo receipt** (Turbo's signature + ms timestamp) — independent service attests when the proof was submitted
3. **Arweave block** — network consensus confirms permanent storage
4. **ar.io Verify** (gateway operator's signature) — independent verification of the anchored data

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
- **ar.io verification** — level, attestation
- **Local verification** — hash match, signature valid, overall pass/fail
- **External verification** — Arweave data fetched and compared, tamper detection

### 3. Tamper with a Record

Click **Tamper** to modify the local record's output hash.

### 4. Verify After Tampering

Click **Verify** — local verification now **FAILS** because:
- The record hash no longer matches the canonical JSON
- The overall result is INVALID

If the record was anchored to Arweave, the external verification shows the **Arweave copy is STILL VALID** — proving the local record was modified after anchoring.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run prediction, create decision record (JSON body) |
| `/predict-form` | POST | Same, from HTML form (redirects to detail) |
| `/decisions` | GET | List all decision records |
| `/decisions/{id}` | GET | Get a single decision record |
| `/verify/{id}` | POST | Verify a decision (local + external) |
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

### Example: Tamper and Re-Verify

```bash
# Tamper
curl -X POST http://localhost:8000/tamper/<decision_id>

# Verify (will show hash_valid=false, overall=false)
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
The clean proof (record + hash + chain link + signature + public key) is uploaded to Arweave permanent storage. The upload returns a signed receipt with a millisecond-precision timestamp — an independent attestation of when the proof was submitted. Once confirmed on Arweave, the data is immutable and publicly accessible.

### ar.io Verify — Independent Validation
ar.io Verify independently fetches the Arweave data, recomputes hashes, verifies signatures, and produces a signed attestation — proving the record exists and is authentic without trusting the original system. Attestations include verification levels (1: pending, 2: partially verified, 3: fully verified) and downloadable PDF certificates.

### Auditor Verification
An auditor can independently verify any proof with standard cryptographic tools:
1. Fetch the proof from Arweave using the TX ID
2. Recompute `SHA-256(canonical_json(record))` and compare to `record_hash`
3. Verify the Ed25519 signature against the `public_key`
4. Check hash chain links across records
5. No dependency on ar.io, MLflow, or any external service required
