# Verifiable AI Decision Records

Tamper-evident AI audit trail anchored to Arweave, covering the full MLflow lifecycle.

## What This Demonstrates

This project provides **verifiable provenance for the entire ML lifecycle** — from training through to production predictions. Every lifecycle event creates a signed proof record anchored to Arweave via ar.io:

1. **Training provenance** — params, metrics, and artifact hashes are captured and anchored when a model is trained
2. **Registration provenance** — model registration events are signed and anchored with a link back to the training proof
3. **Prediction records** — every inference creates a decision record with full model lineage
4. **Chain of custody** — an unbroken, verifiable chain from training → registration → predictions

Each record is:

- **Traced** — OpenTelemetry captures runtime context (trace ID, span ID)
- **Linked to model lineage** — MLflow records which model version produced the output
- **Hashed & signed** — SHA-256 hash chain + Ed25519 digital signature
- **Anchored to Arweave** — Immutable permanent storage via ar.io Turbo
- **Independently verifiable** — ar.io Verify produces on-demand attestations

If someone tampers with a local record, the **Arweave-anchored copy** remains intact and verifiable.

## Architecture

```
Startup (background thread)
  |---> MLflow: read training run (params, metrics, artifact checksums)
  |---> Build training proof → sign → anchor to Arweave
  |---> MLflow: read model registration metadata
  |---> Build registration proof → sign → anchor to Arweave
  |---> Chain: training TX → registration TX

User Input
  |
  v
FastAPI /predict (returns instantly)
  |---> MLflow (model lineage: run_id, version, artifact_uri)
  |---> OpenTelemetry (trace_id, span_id)
  |---> Inference (sklearn LogisticRegression)
  |
  v
Decision Record (canonical JSON)
  |---> SHA-256 hash + hash chain (previous_hash)
  |---> Ed25519 signature
  |---> Store locally (instant)
  |
  v (background thread)
Proof upload
  |---> ar.io Turbo upload to Arweave
  |---> TX ID written back to stored record
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

### Async Anchoring

Predictions return instantly (~4ms). Arweave uploads happen in a background thread and typically complete within 1-2 seconds. The UI auto-polls and updates when the TX ID arrives.

### What Gets Anchored

Small JSON proof records (~1-5 KB each) — not model binaries. Each contains:

```json
{
  "record": { "event_type": "...", "model_name": "...", ... },
  "record_hash": "SHA-256 of canonical JSON",
  "previous_hash": "prior record's hash (or GENESIS)",
  "signature": "Ed25519 signature",
  "public_key": "Ed25519 public key"
}
```

### The Evidence Chain

Each event creates multiple layers of evidence from independent parties:

1. **Proof** (Ed25519 signature) — the AI system attests to the event
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

### 1. Train a Model

The landing page (`/`) shows the Models page. Click **Train & Anchor** to train a new model. Watch the four-step progress: training → registering → anchoring training proof → anchoring registration proof. After completion, the app automatically redirects to the chain of custody page.

### 2. View the Chain of Custody

The chain of custody page shows the proof chain forming in real time: Training Run → Model Registration → Predictions. Each node shows its verification status and Arweave transaction ID.

### 3. Make a Prediction

Navigate to **Predictions** and submit the form with iris flower measurements. The response is instant — the detail page shows "Anchoring..." with a pulsing indicator, then auto-updates when the Arweave upload completes (~1-2s).

### 4. View the Decision Record

Click a decision ID to see the full record:
- **Prediction** — class, probabilities with visual bars, features used
- **ar.io Verification** — three-level verification status (hash, signature, permanent copy, attestation)
- **Model lineage** — MLflow run ID, version, artifact URI, with link to chain of custody
- **Proof layer** — record hash, chain link, Ed25519 signature
- **Arweave anchoring** — transaction ID, status (Anchoring → Anchored → Confirmed → Permanent)
- **Turbo upload receipt** — millisecond timestamp, wallet owner, signed receipt

### 5. Verify a Record

Click **Verify with ar.io** to run on-demand verification:
- **Local** — re-hashes the record and checks the Ed25519 signature
- **Arweave** — fetches the proof from an ar.io gateway and compares hashes
- **ar.io Verify** — requests an independent attestation from the ar.io gateway operator

### 6. Tamper with a Record

Click **Tamper** to modify the local record's output hash, then **Verify with ar.io** — local verification FAILS but the Arweave copy is still intact, proving the local record was modified after anchoring.

## Pages

| Page | URL | Description |
|---|---|---|
| Models (landing page) | `/` | Model versions, train new models, activate versions |
| Predictions | `/ui/predictions` | Prediction records, stats, prediction form, model provenance card, version filter |
| Decision detail | `/ui/decisions/{id}` | Full decision record with three-level verification |
| Chain of custody | `/ui/models/{name}/{version}` | Training → Registration → Predictions chain |
| Training run detail | `/ui/runs/{run_id}` | Training params, metrics, artifact hashes, verification |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run prediction, create decision record (JSON body) |
| `/predict-form` | POST | Same, from HTML form (redirects to detail) |
| `/decisions` | GET | List all decision records |
| `/decisions/{id}` | GET | Get a single decision record |
| `/verify/{id}` | POST | Verify a decision (local + Arweave + ar.io Verify) |
| `/tamper/{id}` | POST | Tamper with a record (demo only) |
| `/api/activate/{name}/{version}` | POST | Switch the active model to a specific version |
| `/lifecycle` | GET | List all lifecycle records (training, registration) |
| `/lifecycle/{event_id}` | GET | Get a single lifecycle record |

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
| `VAIDR_LIFECYCLE_FILE` | `data/lifecycle.json` | Lifecycle record storage path |

---

## MLflow Plugin (`ario-mlflow`)

The `ario_mlflow/` package is a standalone MLflow plugin that any MLflow user can adopt. It provides the same verifiable provenance as the demo, integrated into the native MLflow workflow.

### Install

```bash
pip install -e .
```

### Usage

**Training — one explicit `anchor()` call:**

```python
import mlflow
from ario_mlflow import anchor

with mlflow.start_run() as run:
    mlflow.log_param("lr", 0.01)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", 0.95)
    anchor()  # signs a proof, uploads to Arweave, writes ario.* tags + artifacts
```

Just importing `ario_mlflow` auto-injects `ario.enabled` / `ario.version` tags on every run via the `RunContextProvider`. `anchor()` adds the rich proof layer and must be called inside the run.

**Registration / promotion — one client swap:**

```python
from ario_mlflow import ArioMlflowClient

client = ArioMlflowClient()
client.create_model_version("fraud-detector", source=uri, run_id=run_id)
# → Re-hashes artifacts, anchors registration proof, writes ario.registration_tx
#   and ario.artifact_verified on the model version.

client.transition_model_version_stage("fraud-detector", "1", "Production")
# → Anchors promotion proof, writes ario.promotion_tx tag.
```

**Inference — `VerifiedModel` wrapper:**

```python
from ario_mlflow import VerifiedModel

model = VerifiedModel("models:/fraud-detector/Production")
# Model artifacts are re-hashed at load time and compared to ario.artifact_hash.
# A mismatch raises IntegrityError (refuses to serve).

result = model.predict({"feature1": 1.0, "feature2": 2.0})
# result.prediction   — immediate
# result.decision_id  — immediate
# result.proof_status — "anchoring" → "anchored"
```

Every `predict()` creates an MLflow trace (visible in the Traces tab) and anchors
a signed proof for that specific prediction in the background.

**Reject tampered artifacts at load time:**

```python
from ario_mlflow import VerifiedModel, IntegrityError

try:
    model = VerifiedModel("models:/fraud-detector/Production")
except IntegrityError as e:
    # Artifact hash does not match ario.artifact_hash — do not serve this model
    alert_secops(e)
    raise
```

**Compliance / Audit — CLI:**

```bash
# Verify a training run
ario-mlflow verify run <run_id>

# Verify a model registration
ario-mlflow verify model fraud-detector/3

# Verify an individual inference trace
ario-mlflow verify trace <trace_id>

# Full chain of custody audit
ario-mlflow audit fraud-detector/3
```

All three `verify` commands call ar.io Verify, then write the attestation back to MLflow:

- `verify run` → sets `ario.verify_status`, `ario.attestation_level`, `ario.report_url`, `ario.attested_by`, `ario.attested_at` on the run and refreshes `ario/verification.html`.
- `verify model` → sets the same tags on the model version and refreshes `ario/registration_verification.html` on the source run.
- `verify trace` → sets the same tags on the trace.

Re-run any `verify` command to pick up newer attestation levels (1 → 2 → 3) as the proof propagates.

### Plugin configuration

| Variable | Required | Description |
|---|---|---|
| `ARIO_MLFLOW_ARWEAVE_WALLET` | No | Path to Arweave JWK wallet. If unset, a wallet is auto-generated at `~/.ario-mlflow/wallets/default.json` on first use. |
| `ARIO_MLFLOW_SIGNING_KEY` | No | Base64 Ed25519 seed. If unset, a keypair is auto-generated at `~/.ario-mlflow/keys/`. |
| `ARIO_MLFLOW_GATEWAY_HOST` | No | ar.io gateway (default: `turbo-gateway.com`) |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | No | ar.io Verify URL. Verification is skipped if unset. |

### What shows up where in MLflow

| MLflow surface | Who writes it | What you see |
|---|---|---|
| **Runs** tab → Tags + `ario/` artifacts | `anchor()` | `ario.training_tx`, `ario.artifact_hash`, `ario.verify_status`; `ario/proof.json`, `ario/receipt.json`, `ario/verification.html` |
| **Models** tab → Model version tags + `ario/` artifacts on the source run | `ArioMlflowClient` | `ario.registration_tx`, `ario.promotion_tx`, `ario.artifact_verified`; `ario/registration_verification.html` |
| **Traces** tab → Trace tags | `VerifiedModel.predict()` | `ario.decision_id`, `ario.arweave_tx`, `ario.record_hash`, `ario.proof_status`, `ario.artifact_verified` |

After running the CLI `verify` commands, each surface also carries `ario.verify_status = verified`, `ario.attestation_level`, `ario.report_url`, `ario.attested_by`, `ario.attested_at`.

### MLflow UI integration

The plugin writes these tags, all visible in the native MLflow UI:

| Tag | Where | Written by |
|---|---|---|
| `ario.training_tx` | Run | `anchor()` |
| `ario.registration_tx` | Model version | `ArioMlflowClient.create_model_version` |
| `ario.promotion_tx` | Model version | `ArioMlflowClient.transition_model_version_stage` |
| `ario.artifact_hash` | Run | `anchor()` |
| `ario.artifact_verified` | Model version, Trace | `ArioMlflowClient`, `VerifiedModel` |
| `ario.verify_status` | Run, Model version, Trace | `anchor()`, `ArioMlflowClient`, CLI verify |
| `ario.attestation_level` | Run, Model version, Trace | CLI verify |
| `ario.report_url` | Run, Model version, Trace | CLI verify |
| `ario.attested_by` | Run, Model version, Trace | CLI verify |
| `ario.attested_at` | Run, Model version, Trace | CLI verify |
| `ario.arweave_tx` | Trace | `VerifiedModel.predict` |
| `ario.arweave_url` | Run, Model version, Trace | all |

---

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

### ar.io Turbo — Anchoring
The proof is uploaded to Arweave permanent storage via ar.io Turbo. The upload returns a signed receipt with a millisecond-precision timestamp — an independent attestation of when the proof was submitted. Once confirmed on Arweave, the data is immutable and publicly accessible.

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
