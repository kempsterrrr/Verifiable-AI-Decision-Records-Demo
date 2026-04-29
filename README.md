# Verifiable AI Decision Records

Tamper-evident AI audit trail anchored to Arweave, covering the full MLflow lifecycle.

## What This Demonstrates

This project provides **verifiable provenance for the entire ML lifecycle** — from training through to production predictions. Every lifecycle event creates a signed proof record anchored to Arweave via ar.io:

1. **Training provenance** — params, metrics, and artifact hashes are captured and anchored when a model is trained
2. **Registration provenance** — model registration events are signed and anchored with a link back to the training proof
3. **Prediction records** — every inference creates a decision record with full model lineage
4. **Model lineage** — a cryptographically verifiable audit trail from training → registration → predictions

Each event is:

- **Traced** — OpenTelemetry captures runtime context (trace ID, span ID)
- **Linked to model lineage** — MLflow records which model version produced the output
- **Committed** — A pure-commitment envelope (~500 bytes; bounded 400–700: `event_id`, `event_type`, `subject`, `payload_hash`, `previous_hash`, `signed_at`, signature, public key) is anchored to Arweave
- **Reproducible** — The canonical bytes that were hashed live alongside the proof as `ario/payload.json` in MLflow, so an auditor can re-compute the hash and re-derive from current MLflow state
- **Independently verifiable** — ar.io Verify produces on-demand attestations of the on-chain proof

MLflow stays the system of record (canonical bytes); Arweave is the witness (commitment + signature only). No PII or business data leaves your MLflow.

## Architecture

```text
User trains a model
  |
  v
ario_mlflow.anchor() inside mlflow.start_run()
  |---> Build canonical payload from MLflow state (params, metrics, artifact checksums)
  |---> Write canonical bytes to MLflow as ario/payload.json artifact
  |---> Sign a pure-commitment envelope (event_id, payload_hash, previous_hash, ...)
  |---> Upload commitment to Arweave via ar.io Turbo
  |---> Tag the run: ario.training_tx, ario.payload_hash
  |---> Update chain head on the registered model: ario.last_training_hash
  |
  v
ArioMlflowClient.create_model_version(...)
  |---> Re-hash artifacts, compare to ario.artifact_hash (catches model swap)
  |---> Sign + anchor a registration commitment chained to ario.training_tx
  |---> Tag the model version: ario.registration_tx, ario.artifact_verified

User submits a prediction
  |
  v
FastAPI /predict (returns instantly)
  |---> VerifiedModel.predict({...})
  |     |---> Load model (integrity-checked at load time)
  |     |---> Run inference
  |     |---> Build canonical payload (input/output/trace IDs/...)
  |     |---> Write ario/predictions/<id>/payload.json to MLflow
  |     |---> Sign a pure-commitment envelope chained to ario.registration_tx
  |     |---> Upload to Arweave in background thread
  |
  v
RecordStore: display cache (input, output, decision_id, latency, arweave_tx)

  ... later, on demand ...

/verify endpoint (or `ario-mlflow verify ...` CLI)
  |---> Fetch the pure-commitment envelope from Arweave
  |---> Check 1: Ed25519 signature is valid (cryptographic)
  |---> Check 2: download ario/payload.json from MLflow → re-hash → matches envelope
  |---> Check 3: re-derive canonical bytes from current MLflow state → matches anchored
  |---> Check 4: ar.io Verify attestation (independent third-party check)
```

### Async Anchoring

Predictions return instantly (~4ms). Arweave uploads happen in a background thread and typically complete within 1-2 seconds. The UI auto-polls and updates when the TX ID arrives.

### What Gets Anchored

A pure-commitment envelope per lifecycle event — ~500 bytes on Arweave (the plugin's smoke test bounds this at 400–700 bytes), no source data:

```json
{
  "event_id": "uuid",
  "event_type": "training_complete | model_registered | prediction",
  "subject": {"type": "mlflow_run", "run_id": "..."},
  "payload_hash": "SHA-256 of the canonical bytes",
  "previous_hash": "prior event's payload_hash (or GENESIS)",
  "signed_at": "ISO-8601",
  "public_key": "Ed25519 public key",
  "signature": "Ed25519 signature over canonical(envelope - signature)"
}
```

The canonical bytes that were hashed live in MLflow as `ario/payload.json`. The envelope commits to those bytes; verifiers re-hash and compare. This keeps Arweave costs minimal and PII out of public storage.

### The Evidence Chain

Each event creates multiple layers of evidence from independent parties:

1. **Commitment + signature** (Ed25519, by the AI system) — attests to the event
2. **Canonical payload in MLflow** (`ario/payload.json`) — the source bytes that were hashed
3. **Turbo receipt** (Turbo's signature + ms timestamp) — independent service attests when the proof was submitted
4. **Arweave block** — network consensus confirms permanent storage
5. **ar.io Verify** (on-demand, gateway operator's signature) — independent verification of the anchored data

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

The landing page (`/`) shows the Models page. Click **Train & Anchor** to train a new model. Watch the four-step progress: training → registering → anchoring training proof → anchoring registration proof. After completion, the app automatically redirects to the model lineage page.

### 2. View the Model Lineage

The model lineage page shows the proof chain forming in real time: Training Run → Model Registration → Decisions. Each node shows its verification status and Arweave transaction ID. Compliance readers can think of this as a cryptographically verifiable audit trail.

### 3. Make a Prediction

Navigate to **Decisions** and submit the form with applicant features (income, credit score, etc.). The response is instant — the detail page shows "Anchoring..." with a pulsing indicator, then auto-updates when the Arweave upload completes (~1-2s).

### 4. View the Decision Record

Click a decision ID to see the full record:
- **Prediction** — class, probabilities with visual bars, features used
- **ar.io Verification** — four-check panel (signature, anchored bytes intact, source-of-truth re-derivation, ar.io attestation). All four apply to predictions, training, and registration.
- **Model lineage** — MLflow run ID, version, artifact URI, with link to the full lineage view
- **ar.io anchoring** — transaction ID, status (Anchoring → Anchored → Confirmed → Permanent)
- **Upload receipt** — ar.io's timestamp witness: millisecond timestamp, wallet owner, signed receipt

### 5. Verify a Record

Click **Verify with ar.io** to run on-demand verification:
- **Signature** — Ed25519 signature on the on-chain commitment is valid
- **Anchored bytes intact** — `ario/payload.json` in MLflow re-hashes to the envelope's `payload_hash`
- **Source of truth matches** — re-derive canonical bytes from a *separate* live MLflow surface and compare to the anchored payload. The point is to catch MLflow tampering — if either surface was modified after anchoring, the two won't agree.
  - **Training:** re-fetches `run.data.params/metrics/artifact_checksums` from the run.
  - **Registration:** re-derives the artifact-verified state from the source run.
  - **Predictions:** re-fetches the `ario.payload_json` trace tag (mirrored at predict time) and compares to the artifact. If the trace was pruned by an MLflow retention policy, this surfaces as `live_refetch_incomplete` (not a silent pass).
- **ar.io Verify attestation** — independent third-party check by an ar.io gateway operator. **Conditional:** runs only when `VAIDR_ARIO_VERIFY_URL` (or `ARIO_MLFLOW_ARIO_VERIFY_URL` for the plugin CLI) is configured. Otherwise this check is reported as Pending / Not available, and the overall verdict is computed from the remaining three checks.

The first three checks apply uniformly to predictions, training, and registration — feature-equivalent verification across the lifecycle. The demo, the `/verify/{decision_id}` endpoint, and the `ario-mlflow verify run|model|trace` CLI all run the same checks. Each check returns one of three states: PASS, FAIL, or Pending (transient — re-verify later).

### 6. Tamper Demo (deferred to Phase 3)

The single "Tamper" button was removed. It modified a local cache that isn't part of the trust model under the new design (MLflow is the system of record). Phase 3 reintroduces tamper UX with four buttons paired to the four real checks above — modify the proof envelope, overwrite `ario/payload.json` in MLflow, mutate MLflow params/metrics, swap in a fake TX ID. Each tamper triggers exactly one check, teaching what that check actually proves.

## Pages

| Page | URL | Description |
|---|---|---|
| Models (landing page) | `/` | Model versions, train new models, activate versions |
| Decisions | `/ui/decisions` | Decision records, stats, prediction form, model provenance card, version filter (`/ui/predictions` 301-redirects here for bookmarks) |
| Decision detail | `/ui/decisions/{id}` | Full decision record with three-level verification |
| Model lineage | `/ui/models/{name}/{version}` | Training → Registration → Decisions chain |
| Training run detail | `/ui/runs/{run_id}` | Training params, metrics, artifact hashes, verification |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run prediction, create decision record (JSON body) |
| `/predict-form` | POST | Same, from HTML form (redirects to detail) |
| `/decisions` | GET | List all decision records |
| `/decisions/{id}` | GET | Get a single decision record |
| `/verify/{id}` | POST | Verify a decision (full four-check: signature + anchored bytes + source-of-truth + ar.io) |
| `/api/activate/{name}/{version}` | POST | Switch the active model to a specific version |
| `/api/train` | POST | Train a new model version |
| `/lifecycle` | GET | List all lifecycle records (training, registration) |
| `/lifecycle/{event_id}` | GET | Get a single lifecycle record |

### Example: Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"annual_income":78000,"credit_utilization":0.18,"debt_to_income_ratio":0.22,"months_employed":72,"credit_score":745}'
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
| `VAIDR_MLFLOW_MODEL_NAME` | `credit-scorer` | Registered model name |
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

# Full model lineage audit (training → registration → promotion)
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
| `ARIO_MLFLOW_ARWEAVE_WALLET` | No | Path to Arweave JWK wallet. If unset or unreadable, an in-memory wallet is generated for the session (not persisted — set this in production so proofs stay owned by the same address). |
| `ARIO_MLFLOW_SIGNING_KEY` | No | Base64 Ed25519 seed. If unset, a keypair is auto-generated at `~/.ario-mlflow/keys/`. |
| `ARIO_MLFLOW_GATEWAY_HOST` | No | ar.io gateway (default: `turbo-gateway.com`) |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | No | ar.io Verify URL. Verification is skipped if unset. |

### What shows up where in MLflow

| MLflow surface | Who writes it | What you see |
|---|---|---|
| **Runs** tab → Tags + `ario/` artifacts | `anchor()` | `ario.training_tx`, `ario.payload_hash`, `ario.artifact_hash`, `ario.last_training_hash`, `ario.verify_status`; `ario/payload.json`, `ario/proof.json`, `ario/receipt.json`, `ario/verification.html` |
| **Models** tab → Model version tags + `ario/` artifacts on the source run | `ArioMlflowClient` | `ario.registration_tx`, `ario.promotion_tx`, `ario.artifact_verified`; `ario/registration_payload.json`, `ario/registration_verification.html` |
| **Traces** tab → Trace tags | `VerifiedModel.predict()` | `ario.decision_id`, `ario.prediction_tx`, `ario.payload_hash`, `ario.input_hash`, `ario.output_hash`, `ario.proof_status`, `ario.artifact_verified` |

After running the CLI `verify` commands, each surface also carries `ario.verify_status = verified`, `ario.attestation_level`, `ario.report_url`, `ario.attested_by`, `ario.attested_at`.

### MLflow UI integration

The plugin writes these tags, all visible in the native MLflow UI:

| Tag | Where | Written by |
|---|---|---|
| `ario.training_tx` | Run | `anchor()` |
| `ario.payload_hash` | Run, Trace | `anchor()`, `VerifiedModel.predict` |
| `ario.last_training_hash` | Registered model | `anchor()` (chain head) |
| `ario.registration_tx` | Model version | `ArioMlflowClient.create_model_version` |
| `ario.promotion_tx` | Model version | `ArioMlflowClient.transition_model_version_stage` |
| `ario.artifact_hash` | Run | `anchor()` |
| `ario.artifact_verified` | Model version, Trace | `ArioMlflowClient`, `VerifiedModel` |
| `ario.verify_status` | Run, Model version, Trace | `anchor()`, `ArioMlflowClient`, CLI verify |
| `ario.attestation_level` | Run, Model version, Trace | CLI verify |
| `ario.report_url` | Run, Model version, Trace | CLI verify |
| `ario.attested_by` | Run, Model version, Trace | CLI verify |
| `ario.attested_at` | Run, Model version, Trace | CLI verify |
| `ario.prediction_tx` | Trace | `VerifiedModel.predict` |
| `ario.input_hash`, `ario.output_hash` | Trace | `VerifiedModel.predict` |
| `ario.decision_id` | Trace | `VerifiedModel.predict` |
| `ario.proof_status` | Trace | `VerifiedModel.predict` |
| `ario.arweave_url` | Run, Model version, Trace | all |

---

## How It Works

### MLflow — Model Lineage
Every prediction is tied to a specific model version. MLflow captures the run ID, model version, and artifact URI, creating an auditable link between the model and its decisions.

### OpenTelemetry — Runtime Trace
Each prediction creates a distributed trace. The trace ID and span ID are embedded in the decision record, allowing correlation with infrastructure monitoring.

### Proof Layer — Integrity
The plugin builds a canonical payload from MLflow state (training params/metrics/artifact checksums, or prediction input/output) and serializes it to RFC-8785 (JCS) — a deterministic JSON canonicalization that any RFC-8785 verifier in any language can reproduce. Then:
- **Canonical bytes preserved** — written to MLflow as `ario/payload.json` (and as `ario/predictions/<id>/payload.json` for inferences)
- **Hash committed** — SHA-256 of the canonical bytes goes into a small commitment envelope
- **Chained per event type** — each event's `previous_hash` points back to the prior anchor of the same kind (training chain head on the registered model; registration chains to its source training TX; predictions chain to their model version's registration TX)
- **Ed25519 signed** — the envelope (minus signature) is signed; `public_key` travels with it for verification
- **Anchored to Arweave** — only the ~500-byte envelope is uploaded (bounded 400–700 by the plugin's smoke test); no source data leaves your MLflow

### ar.io Turbo — Anchoring
The proof is uploaded to Arweave permanent storage via ar.io Turbo. The upload returns a signed receipt with a millisecond-precision timestamp — an independent attestation of when the proof was submitted. Once confirmed on Arweave, the data is immutable and publicly accessible.

### ar.io Verify — Independent Attestation
When verification is requested, ar.io Verify independently fetches the Arweave data, recomputes hashes, and checks signatures. The three levels describe **how much of the proof has been independently verified**, not network-confirmation depth:

- **Level 1 — Finalized on Arweave.** The record was found in a confirmed block on the Arweave network at a specific block height and timestamp. On Arweave, a confirmed block is permanent storage. Content and signature verification still to come.
- **Level 2 — Content integrity confirmed.** ar.io re-downloaded the raw record and recomputed its SHA-256 fingerprint. The bytes match the gateway's digest, so the content is intact. Cryptographic signature verification still pending.
- **Level 3 — Cryptographically verified.** The digital signature on the record has been independently verified using the original signer's public key (RSA-PSS / Ed25519 / ECDSA, depending on wallet type). This is a mathematical proof, not a trust claim: the record is authentic and attributable to the stated signer.

**Operator attestation.** When an ar.io gateway operator configures a signing wallet, the verification result is also signed with that operator's wallet. This creates an attestation — an independent statement from a known operator on the ar.io network that they personally verified the record. You'll see "Attested by [operator]" in the Verification section when the operator signing this deployment is attesting. The attestation is itself verifiable: it's standard RSA-PSS SHA-256 over the canonical JSON payload, checkable against the operator's public key.

These levels describe integrity of the anchored record, not the correctness of the underlying ML decision. Semantic verification (whether *this model* produced *this output* on *this input*) is a separate problem and is on the roadmap, not in v0.1.

### Auditor Verification
An auditor can independently verify any proof with standard cryptographic tools (no dependency on the demo's internals):

1. **Fetch the commitment** from Arweave via any ar.io gateway:
   ```bash
   curl https://turbo-gateway.com/raw/<tx_id>
   ```
2. **Verify the Ed25519 signature.** Strip the `signature` field, JCS-canonicalize (RFC 8785) the remaining envelope, verify against the embedded `public_key`.
3. **Fetch the canonical bytes** from MLflow:
   ```bash
   mlflow artifacts download -r <run_id> -a ario/payload.json     # training/registration
   mlflow artifacts download -r <run_id> -a ario/predictions/<decision_id>/payload.json  # prediction
   ```
   Compute `SHA-256` of the raw bytes; compare to the envelope's `payload_hash`.
4. **Re-derive the canonical bytes from a separate MLflow surface** and compare to the downloaded `payload.json`. Any mismatch means MLflow was modified after anchoring.
   - **Training:** rebuild from `run.data.params`, `run.data.metrics`, and `artifact_checksums` (recompute by re-hashing the model artifact files).
   - **Registration:** re-derive `artifact_verified` from the source run's `ario.artifact_hash` tag and freshly-recomputed artifact checksums.
   - **Prediction:** read the `ario.payload_json` trace tag (the plugin mirrors the canonical payload onto the trace at predict time) and compare to `ario/predictions/<decision_id>/payload.json`. Don't try to rebuild from raw input/output — predictions commit to *hashes* of input/output, not raw values, so there's nothing to re-derive from app-side data. (To verify the *raw* I/O independently, hash your own copy and compare to the payload's `input_hash`/`output_hash` — but that's a separate caller-side check, not part of MLflow's source-of-truth comparison.)
5. **Walk the chain.** Each envelope's `previous_hash` should be retrievable on Arweave (or `GENESIS`).
6. **Optional:** request an ar.io Verify attestation for an independent third-party check.

`ario-mlflow verify run|model|trace <id>` runs all of the above in one command and writes the result back to MLflow as `ario.verify_status`, `ario.attestation_level`, etc. No dependency on the demo's internals — works against any MLflow + Arweave deployment.
