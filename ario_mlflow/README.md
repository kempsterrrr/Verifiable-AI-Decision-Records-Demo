# ario-mlflow

Verifiable provenance for the MLflow lifecycle — training, registration, promotion, inference.
Signed cryptographic proofs are anchored to Arweave via ar.io, so an auditor can verify a model
or decision long after your MLflow server is gone.

> **Status.** Early-shape idea, not a production-ready system. Default behaviors
> prioritize frictionless evaluation over production hardening. See `ROADMAP.md`
> at repo root for what's next.

## Install

```bash
pip install -e .
```

Python 3.10+. Installs MLflow (≥ 2.14), PyNaCl, and the ar.io Turbo SDK.

## Quickstart

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import ario_mlflow

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200).fit(X, y)
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, "model")

    # Signs a proof, hashes the logged artifacts, writes ario.* tags,
    # and (if anchoring is enabled) uploads to Arweave.
    result = ario_mlflow.anchor()
    print(result["tags"]["ario.training_tx"])
```

No wallet configured? The plugin auto-generates one on first run and persists it
to `~/.ario-mlflow/wallet.json` so your signing address stays stable across
sessions. Set `ARIO_MLFLOW_ARWEAVE_WALLET=/path/to/wallet.json` to use your own.

A full runnable example lives in `examples/sklearn-quickstart/`.

## End-to-end chain

The plugin anchors five separately-signed lifecycle events, each
chained to the previous via Arweave tx pointers stored as MLflow tags:

| Link | API | Tag set |
|---|---|---|
| 1. Data | `anchor_dataset(name, path)` | (caller writes `ario.dataset_tx` on the run) |
| 2. Training | `anchor()` inside `start_run()` | `ario.training_tx`, `ario.artifact_hash` |
| 3. Registration | `ArioMlflowClient.create_model_version(...)` | `ario.registration_tx` |
| 4. Promotion | `ArioMlflowClient.transition_model_version_stage(...)` | `ario.promotion_tx` |
| 5. Inference | `VerifiedModel.predict(input)` | (per-prediction proof, anchored async; trace tagged with `ario.decision_id`/`ario.arweave_tx`) |

To render the chain for a model:

```python
from ario_mlflow.client import ArioMlflowClient

client = ArioMlflowClient(tracking_uri="http://localhost:5000")
chain = client.lifecycle_for_model("credit_classifier")
for event in chain:
    print(f"{event['event_type']}: tx={event['tx_id']}  prev={event['previous_tx']}")
```

The chain is reconstructable from MLflow alone — no separate local store
needed.

## The three integration points

### 1. `ario_mlflow.anchor()` — training provenance

Call inside an active `mlflow.start_run()` after logging your model. The plugin
auto-resolves the logged model's `artifact_path` from MLflow's log-model history,
so you rarely need to pass it explicitly.

Returns a dict with `proof`, `anchor_result`, `tags`, `artifact_path`,
`artifact_status` (`"hashed"` / `"no_artifacts"` / `"hash_failed"`), and
`artifact_error`.

### 2. `ario_mlflow.ArioMlflowClient` — registration + promotion

A drop-in replacement for `mlflow.tracking.MlflowClient`. Registration and stage
promotions are anchored automatically in a background thread. Query the outcome
via the client:

```python
from ario_mlflow import ArioMlflowClient

client = ArioMlflowClient()
mv = client.create_model_version("credit-scorer", "runs:/<run_id>/model")

# Block until the async anchor finishes (optional):
client.wait_for_anchor("registration", "credit-scorer", mv.version, timeout=30)

status = client.anchor_status("registration", "credit-scorer", mv.version)
# {"status": "anchored", "tx_id": "...", "error": None, "done": True}
```

### 3. `ario_mlflow.VerifiedModel` — inference

Wraps a registered model with an integrity check that runs **before** the
underlying pyfunc model is loaded (so a tampered artifact never gets a chance
to execute user code):

```python
from ario_mlflow import VerifiedModel

vm = VerifiedModel("models:/credit-scorer/1")  # raises IntegrityError on hash mismatch
# Features, in order: annual_income, credit_utilization, debt_to_income_ratio,
# months_employed, credit_score.
result = vm.predict([78000, 0.18, 0.22, 72, 745])
print(result.decision_id, result.proof_status)  # "anchoring" → "anchored"

# Wait for the background anchor if you want the TX synchronously:
result.wait_for_anchor(timeout=10)
print(result.tx_id, result.anchor_error)
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ARIO_MLFLOW_ARWEAVE_WALLET` | Path to an Arweave JWK wallet file | auto-generates + persists at `~/.ario-mlflow/wallet.json` |
| `ARIO_MLFLOW_GATEWAY_HOST` | ar.io gateway for uploads & fetches | `turbo-gateway.com` |
| `ARIO_MLFLOW_SIGNING_KEY` | Base64-encoded Ed25519 seed | auto-generates at `~/.ario-mlflow/keys/` |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | ar.io Verify REST API base URL | ar.io attestation disabled if unset |

## Tags the plugin writes

On the training run (`anchor()`):

- `ario.enabled`, `ario.version` — via the registered `RunContextProvider`
- `ario.public_key`, `ario.verify_status`, `ario.artifact_hash`
- `ario.training_tx`, `ario.arweave_url` — when the Arweave upload succeeded
- `ario.wallet_mode` — `user-configured` / `persistent` / `ephemeral`

On model versions (`ArioMlflowClient`):

- `ario.artifact_verified` — `true` / `false` from re-hashing at registration
- `ario.registration_tx`, `ario.promotion_tx`, `ario.arweave_url`

After running `ario-mlflow verify …` (training run or model version):

- `ario.verify_status` → `verified`
- `ario.attestation_level` — `1`, `2`, or `3` (see levels section below)
- `ario.report_url` — link to the ar.io Verify dashboard for this proof
- `ario.attested_by`, `ario.attested_at` — gateway operator and timestamp,
  only present when the operator has configured a signing wallet

On `@mlflow.trace` spans emitted by `VerifiedModel.predict()`:

- `ario.decision_id`, `ario.model_name`, `ario.model_version`, `ario.run_id`
- `ario.input_hash`, `ario.output_hash`, `ario.record_hash`
- `ario.proof_status`, `ario.arweave_tx`, `ario.arweave_url`
- `ario.artifact_verified` (when known)

## CLI

```bash
ario-mlflow verify run <run_id>                  # verify training proof
ario-mlflow verify model <name>/<version>        # verify registration proof
ario-mlflow verify trace <trace_id>              # verify an inference proof
ario-mlflow audit <name>/<version>               # full model-lineage audit
```

All `verify` commands check the proof locally (re-hash + Ed25519 signature),
fetch the permanent copy from Arweave, and (if `ARIO_MLFLOW_ARIO_VERIFY_URL` is
set) request an ar.io Verify attestation. Results are written back to the
MLflow tags and the HTML report is regenerated.

## What the attestation levels actually mean

`ario-mlflow verify` reports an ar.io attestation level. The levels describe
**how much of the proof has been independently verified**, not network-
confirmation depth:

- **Level 1 — Finalized on Arweave.** The proof was found in a confirmed block
  on the Arweave network at a specific block height and timestamp. On Arweave,
  a confirmed block means permanent storage.
- **Level 2 — Content integrity confirmed.** ar.io re-downloaded the raw proof
  and recomputed its SHA-256 fingerprint. The bytes match the gateway's digest.
  Cryptographic signature verification still pending.
- **Level 3 — Cryptographically verified.** The digital signature on the proof
  has been independently verified against the original signer's public key.
  This is a mathematical proof, not a trust claim.

**Operator attestation.** When an ar.io gateway operator has configured a
signing wallet, the verification result is itself signed with that operator's
wallet and `ario.attested_by` / `ario.attested_at` are written back to your
MLflow tags. This is an independent statement from a known ar.io operator that
they personally verified the proof — separate from and additional to the level
above. The operator signature is standard RSA-PSS SHA-256 over canonical JSON,
so any third party can verify it with the operator's public key.

These levels and attestations cover integrity and authenticity of the anchored
record. Semantic verification (whether this model produced this decision on
this input) is on the roadmap, not in v0.1.

## Tests

```bash
python -m pytest tests/test_plugin_smoke.py
```

33 smoke tests, no network required.

## Related docs

- Demo app: the repo root `README.md`
- Team roadmap and deferred work: `ROADMAP.md` at repo root
