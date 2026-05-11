# Verifiable AI Decision Records

Tamper-evident verifiable record for the full MLflow lifecycle, anchored to ar.io.

## What's in this repo

This repository contains two parts:

- **`ario_mlflow/`** — the **product**. A standalone MLflow plugin that anchors training, registration, and inference events to ar.io / Arweave. Any MLflow user can adopt it with `pip install -e .` (and eventually `pip install ario-mlflow`). See [Plugin section](#mlflow-plugin-ario-mlflow) below or `ario_mlflow/README.md` for usage.
- **`app/`** — a **sales-facing demonstration**. A FastAPI + Jinja2 web app that uses the plugin to make the verification flow tangible. Hosted on Railway at [verifiable-ai-demo-production.up.railway.app](https://verifiable-ai-demo-production.up.railway.app). The demo is not the product — it's a working showcase used by sales / pre-sales.

For deeper context: [`docs/architecture.md`](docs/architecture.md) covers the system design; [`docs/deployment.md`](docs/deployment.md) covers the demo's production deployment; [`ROADMAP.md`](ROADMAP.md) tracks what's next.

## What This Demonstrates

This project provides **verifiable provenance for the entire ML lifecycle** — from training through to production predictions. Every lifecycle event creates a signed proof record anchored to ar.io:

1. **Training provenance** — params, metrics, and artifact hashes are captured and anchored when a model is trained
2. **Registration provenance** — model registration events are signed and anchored with a link back to the training proof
3. **Prediction records** — every inference creates a decision record with full model lineage
4. **Model lineage** — a cryptographically verifiable record from training → registration → predictions

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
  |---> Fetch the pure-commitment envelope from ar.io
  |---> Check 1 (Proof Found): envelope retrieved from ar.io
  |---> Check 2 ({Event} Record Matches): download ario/payload.json from MLflow → re-hash → matches envelope, and re-derive canonical bytes from current MLflow state → matches anchored
  |---> Check 3 (Signature Confirmed): the signature on the envelope verifies against the embedded public key
  |---> Plus: ar.io Verify attestation (independent third-party check by a gateway operator)
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

The demo's navigation walks the verification chain left to right: **Datasets → Training Runs → Models → Decisions → Lineage**. Each list page leads to a detail page, and every detail page links upstream and downstream so an auditor can step through the chain in either direction.

### 1. Train a Model

Open **Training Runs** (`/ui/runs`). The train form is the expandable card above the runs table. Click **Train & Anchor** to train a new model — training → registering → anchoring training proof → anchoring registration proof. After completion the new run + model version appear in their respective list pages.

### 2. Trace the Chain

Open **Lineage** (`/ui/lineage`). A single connected chain is rendered at a time — Dataset(s) → Training Run → Model Version → Decisions — with verification status on each node. The chip picker at the top switches between chains (one per model version); the URL updates so chains are linkable. Every detail page has a **View chain in Lineage →** CTA so any entity is one click away from its full chain.

### 3. Make a Prediction

Navigate to **Decisions** (`/`) and submit the prediction form (income, credit score, etc.). The response is instant — the new decision's detail page shows "Anchoring..." with a pulsing indicator, then auto-updates when the ar.io upload completes (~1-2s).

### 4. View the Decision Record

Click a decision to see the full record. The detail page is a two-column layout:
- **Left rail** — Identity (features, probabilities, MLflow run/version), ar.io anchor card (TX, signer key, block), upstream context links to the model + run + datasets used.
- **Main column** — **ar.io Verification** three-row verify card (`Proof Found`, `Decision Record Matches`, `Signature Confirmed`) plus the always-on **Anchored proof** viewer that shows the canonical bytes from MLflow side-by-side with the signed commitment on ar.io.

The same shape repeats on every detail page (Dataset, Run, Model, Decision) so an auditor learns one pattern and reads every page the same way. Status badges in the editorial header use the canonical five-state enum: **Verified · Anchored · Anchoring · Tampered · None**.

### 5. Verify a Record

Click **Verify now** on any detail page to run on-demand verification. The three-row verify card shows:

- **Proof Found** — the pure-commitment envelope was fetched from ar.io.
- **Decision Record Matches** *(or `Training Record Matches` / `Registration Record Matches` depending on event type)* — `ario/payload.json` in MLflow re-hashes to the envelope's `payload_hash`, **and** re-deriving the canonical bytes from a *separate* live MLflow surface produces the same bytes. This consolidated check catches MLflow tampering — if either surface was modified after anchoring, the two won't agree.
  - **Training:** re-fetches `run.data.params/metrics/artifact_checksums` from the run.
  - **Registration:** re-derives the artifact-verified state from the source run.
  - **Predictions:** re-fetches the `ario.payload_json` trace tag (mirrored at predict time) and compares to the artifact. If the trace was pruned by an MLflow retention policy, this surfaces as `live_refetch_incomplete` (not a silent pass).
- **Signature Confirmed** — the signature on the envelope verifies against the embedded public key.

Plus an **Attested by** line (operator + timestamp) when an ar.io gateway operator has independently signed an attestation. **Conditional:** ar.io Verify runs only when `VAIDR_ARIO_VERIFY_URL` (or `ARIO_MLFLOW_ARIO_VERIFY_URL` for the plugin CLI) is configured. Otherwise the row reads `Pending` and the overall verdict is computed from the three core checks.

The three core checks apply uniformly to predictions, training, and registration — feature-equivalent verification across the lifecycle. The demo, the `/verify/{decision_id}` endpoint, and the `ario-mlflow verify run|model|trace` CLI all run the same checks. Each check returns one of three states: PASS, FAIL, or Pending (transient — re-verify later).

### 6. Tamper Demo

Open the **tamper** page in the top nav (`/demo/tamper`). Each chain link — Dataset, Training Run, Model Registration, Decision — has one tamper button paired to the real check it breaks: modify the proof envelope, overwrite `ario/payload.json` in MLflow, mutate MLflow params/metrics, swap in a fake TX ID. Each tamper triggers exactly one verify row to fail across the affected links, teaching what each row actually proves. Tampers auto-revert after a TTL so the next demo starts clean.

### 7. Reset for the next session

Sales / pre-sales workflow: pre-seed the demo with example data before a customer call, then wipe everything afterward so the next session starts clean. Open `/demo/admin` to access the reset page. Anchored proofs on Arweave aren't deleted — they remain permanent on the network.

## Pages

| Page | URL | Description |
|---|---|---|
| Decisions (landing) | `/` | Decision records, stats, prediction form, status filter chips (`/ui/predictions` 301-redirects here for bookmarks) |
| Decision detail | `/ui/decisions/{id}` | Full decision record with three-row verify card and anchored-proof viewer |
| Datasets | `/ui/datasets` | Anchored dataset records (one row per `digest`) + "Create dataset" form for new synthetic variants |
| Dataset detail | `/ui/datasets/{digest}` | Dataset record, ar.io anchor, used-by runs, verify card, "Train a model with this dataset" CTA |
| Training Runs | `/ui/runs` | Run list + Train & Anchor form (dataset selector + max-iter + random-state) |
| Training run detail | `/ui/runs/{run_id}` | Params, metrics, artifact hashes, datasets used, verify card |
| Models | `/ui/models` | Registered model versions (one row per version), active flag, verify status |
| Model detail | `/ui/models/{name}/{version}` | Identity, training/registration anchors, activate, verify card |
| Lineage | `/ui/lineage` | Focused-chain viewer with chip picker — Dataset(s) → Run → Model → Decisions |
| Tamper | `/demo/tamper` | Per-chain-link tamper buttons (demo affordance) |
| Admin | `/demo/admin` | One-click reset for sales workflow |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run prediction, create decision record (JSON body) |
| `/predict-form` | POST | Same, from HTML form (redirects to detail) |
| `/decisions` | GET | List all decision records |
| `/decisions/{id}` | GET | Get a single decision record |
| `/verify/{id}` | POST | Verify a decision (Proof Found + Decision Record Matches + Signature Confirmed + ar.io attestation) |
| `/api/activate/{name}/{version}` | POST | Switch the active model to a specific version |
| `/api/datasets` | POST | Create + anchor a new synthetic dataset standalone (body: `{name, n_samples, random_state}`) |
| `/api/train` | POST | Train a new model version against a chosen dataset (body: `{dataset_id, max_iter, random_state}` — `dataset_id` required) |
| `/lifecycle` | GET | List all lifecycle records (datasets, training, registration) |
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

### Example: Create a Dataset, then Train Against It

```bash
# 1. Create a synthetic dataset (returns the new dataset's digest)
DIGEST=$(curl -s -X POST http://localhost:8000/api/datasets \
  -H "Content-Type: application/json" \
  -d '{"name":"My variant","n_samples":1000,"random_state":42}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['dataset_id'])")

# 2. Train a model against it
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"$DIGEST\",\"max_iter\":200}"
```

The demo seeds three default datasets on first boot (Credit scoring -
small / default / large) so a training run can pick from a populated
list without creating one first.

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
| `VAIDR_DEMO_MODE` | `true` | Register `/demo/*` and `/tamper/*` routes; set `false` in production |

### Demo administration (sales workflow)

Sales / pre-sales users can pre-seed the demo with example data before a customer call, then wipe everything afterward so the next call starts clean. Open `/demo/admin` to access the reset page. Resetting wipes all decisions, training runs, and model versions; a fresh v1 is auto-trained on the next request. Anchored proofs on Arweave are not deleted (they remain permanent on the network). The `/demo/admin` page and `/demo/reset` endpoint are only registered when `VAIDR_DEMO_MODE=true` (the default for the public demo on Railway). Production deployments should set `VAIDR_DEMO_MODE=false` to disable.

### Production deployment

The demo is deployed on Railway with a mounted volume at `/app/persistent`. All path-based `VAIDR_*` env vars must point under that path, otherwise data lives on the ephemeral container filesystem and gets wiped on every deploy. See [`docs/deployment.md`](docs/deployment.md) for the full env-var configuration and operational guidance.

---

## MLflow Plugin (`ario-mlflow`)

The `ario_mlflow/` package is a standalone MLflow plugin that any MLflow user can adopt. It provides the same verifiable provenance as the demo, integrated into the native MLflow workflow.

### Install

```bash
pip install -e .
```

### Usage

**Training — log a dataset, then call `anchor()`:**

```python
import mlflow
import mlflow.data
from ario_mlflow import anchor

dataset = mlflow.data.from_pandas(
    train_df, source="s3://bucket/train.csv", name="train_q1_2026",
)

with mlflow.start_run() as run:
    mlflow.log_input(dataset, context="training")    # required (see below)
    mlflow.log_param("lr", 0.01)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", 0.95)
    anchor()
    # → Signs a proof for the dataset, uploads to Arweave (its own TX).
    # → Signs a proof for the training run, uploads to Arweave (linked TX).
    # → Writes ario.* tags + artifacts on the run.
```

Just importing `ario_mlflow` auto-injects `ario.enabled` / `ario.version` tags on every run via the `RunContextProvider`. `anchor()` adds the rich proof layer and must be called inside the run.

**Strict-by-default on dataset inputs.** As of input-side anchoring v1, `anchor()` raises `ValueError` when the run has no logged dataset inputs. The fail-closed default exists so a chain that claims to be auditable can't silently ship without a dataset reference at the head. For workflows that genuinely have no single dataset (research scripts, GPAI training on internet-scale corpora) pass `allow_empty_dataset_inputs=True` to override.

**The dataset's default digest samples rows** (MLflow's `PandasDataset` default — fast for large frames, doesn't catch single-row poisoning). For airtight integrity, supply your own SHA-256:

```python
import hashlib
true_digest = hashlib.sha256(open("train.csv", "rb").read()).hexdigest()
dataset = mlflow.data.from_pandas(
    train_df, source="s3://bucket/train.csv", name="train_q1_2026",
    digest=true_digest,
)
```

The plugin commits to whatever digest the Dataset reports — the strength of the integrity guarantee follows the strength of the digest the caller chose.

**Schema is fingerprinted, not stored verbatim.** Each dataset's `schema_hash` in the canonical bytes is the SHA-256 of the JCS-canonicalized schema JSON — column names never enter the proof. Privacy-preserving for regulated domains where column names can be sensitive (medical, financial). Tamper-detectable: schema mutations in MLflow flip the verifier's source-of-truth check via training's inlined dataset metadata.

**Standalone dataset anchoring (publisher / pre-train pattern).** `anchor()` accepts an optional `dataset=` keyword that anchors a Dataset object on its own — no active MLflow run required. Useful for dataset publishers (anchor once, hand the TX to downstream model trainers) or pre-train workflows where the dataset is finalized before training begins.

```python
from ario_mlflow import anchor
import mlflow.data

dataset = mlflow.data.from_pandas(df, source="...", name="train_q1_2026")
result = anchor(dataset=dataset)
print(result["envelope"]["payload_hash"])
print(result["anchor_result"]["tx_id"])
```

Inside `mlflow.start_run()`, the default training-mode `anchor()` auto-anchors each `mlflow.log_input()` dataset as a standalone event before signing the training proof. Each gets its own Arweave TX, written to the run as a tag (`ario.dataset_anchor_tx.<dataset_name>`) for chain-walking navigation. Auditors fetch the dataset proof independently of the training proof.

**Limits worth knowing.** Two v1 deferrals:

- **Dataset SoT is deferred (v2).** Today the dataset event is verified via signature + ar.io attestation. Live re-derivation against MLflow's dataset registry — to catch post-anchor mutation of dataset metadata at the dataset proof's *own* SoT check — is the v2 follow-up. In v1, that mutation is still caught one link down: training's signed payload inlines the dataset's identity, so the verifier's training Record Matches check flips on tamper. The v2 work will tie the plugin's dataset events more deeply into MLflow's dataset registry API — when it lands, the dataset's own Record Matches check will fire on tamper.
- **Cross-run dataset reuse re-anchors each time.** Two training runs against the same dataset both produce a fresh dataset proof. A registry of pre-anchored datasets (digest → TX) to dedupe is a v3 follow-up.

Both are tracked in [ROADMAP.md](ROADMAP.md).

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
When verification is requested, ar.io Verify independently fetches the anchored data, recomputes hashes, and checks signatures. The user-facing result is `Verified` once the record has reached the configured maturity (proof finalized on the network, content bytes match the gateway's digest, and the signature has been independently verified against the original signer's public key — RSA-PSS / Ed25519 / ECDSA, depending on wallet type). Internal `attestation_level` values (1, 2, 3) describe the maturity gradient programmatically; user-facing copy collapses to **`Verified`** / **`Pending verification`**.

**Operator attestation.** When an ar.io gateway operator configures a signing wallet, the verification result is also signed with that operator's wallet. This creates an attestation — an independent statement from a known operator on the ar.io network that they personally verified the record. You'll see "Attested by [operator]" in the Verification section when the operator signing this deployment is attesting. The attestation is itself verifiable: it's standard RSA-PSS SHA-256 over the canonical JSON payload, checkable against the operator's public key.

This describes integrity of the anchored record, not the correctness of the underlying ML decision. Semantic verification (whether *this model* produced *this output* on *this input*) is a separate problem and is on the roadmap, not in v0.1.

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
