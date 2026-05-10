# ario-mlflow

Verifiable provenance for the MLflow lifecycle — training, registration, promotion, inference.
Signed cryptographic proofs are anchored to ar.io, so an auditor can verify a model
or decision long after your MLflow server is gone.

> **Status.** Early-shape idea, not a production-ready system. Default behaviors
> prioritize frictionless evaluation over production hardening. See `ROADMAP.md`
> at repo root for what's next.

## Install

```bash
# From source — the only path right now (PyPI publish is on the roadmap)
git clone https://github.com/vilenarios/Verifiable-AI-Decision-Records-Demo.git
cd Verifiable-AI-Decision-Records-Demo
pip install -e .
```

Python 3.10+. Pulls in MLflow, PyNaCl, the ar.io Turbo SDK, and `cryptography`.

### MLflow version compatibility

Tested against MLflow 2.14 through 2.x. MLflow 3.x **mostly works** but has one
known issue: the prediction-side `verify_source_of_truth` check can return
`live_refetch_incomplete` because MLflow 3.x changed how trace artifact
locations are resolved. Training and registration verification are unaffected.
If you're on MLflow 3.x, expect that specific check to be stricter than on 2.x.

## Quickstart

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import ario_mlflow

# Point MLflow at a tracking store. Skip if MLFLOW_TRACKING_URI is
# already set in your env, or if you're happy with the cwd's ./mlruns.
mlflow.set_tracking_uri("file:///tmp/mlruns")

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200).fit(X, y)
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, name="model")

    # Signs a proof, hashes the logged artifacts, writes ario.* tags,
    # and uploads ~500 bytes to Arweave via Turbo (free for small payloads).
    # allow_empty_dataset_inputs=True opts out of dataset anchoring; see
    # "Dataset anchoring" below for the recommended pattern.
    result = ario_mlflow.anchor(allow_empty_dataset_inputs=True)
    print(result["tags"]["ario.training_tx"])
```

No wallet configured? The plugin auto-generates one on first run and persists it
to `~/.ario-mlflow/wallet.json` so your signing address stays stable across
sessions. Set `ARIO_MLFLOW_ARWEAVE_WALLET=/path/to/wallet.json` to use your own.
The auto-generated wallet starts unfunded — that's fine for typical usage
because Turbo's free tier covers small uploads (see "Wallet & cost" below).

A full runnable example lives in `examples/sklearn-quickstart/`.

## The three integration points

### 1. `ario_mlflow.anchor()` — training provenance

Call inside an active `mlflow.start_run()` after logging your model. The plugin
auto-resolves the logged model's `artifact_path` from MLflow's log-model history,
so you rarely need to pass it explicitly.

Returns a dict with `envelope`, `payload`, `payload_bytes`, `payload_hash`,
`previous_hash`, `anchor_result`, `tags`, `artifact_path`, `artifact_status`
(`"hashed"` / `"no_artifacts"` / `"hash_failed"`), and `artifact_error`.

**Failure modes.** `anchor()` is synchronous and runs to completion before the
`with` block exits.

- **Arweave upload fails** (gateway down, wallet unfunded, network): the
  envelope is still signed locally and `ario.verify_status` is set to `signed`;
  `ario.training_tx` is absent. Your MLflow run still succeeds. Re-run later
  to retry the upload.
- **Artifact hashing fails** (artifacts not yet logged, store unreachable):
  raises `ario_mlflow.anchoring.ArtifactAccessError`. Wrap the call if you
  want to log-and-continue.
- **No active run**: raises `RuntimeError`. The function requires an active
  `mlflow.start_run()` block.

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

**Failure modes.** Registration and promotion both return their MLflow
`ModelVersion` immediately; anchoring runs in a daemon thread.

- The MLflow operation always succeeds independently — anchoring failures
  never break `create_model_version()` or `transition_model_version_stage()`.
- `anchor_status()` returns `{"status": ...}` where status is one of
  `anchoring` (in flight), `anchored` (Arweave upload succeeded), `signed`
  (envelope signed but Arweave upload failed), `failed` (anchoring crashed —
  see `error`), or `unknown` (no anchor was ever queued for this key).
- `wait_for_anchor()` returns `False` on timeout. Process exit before the
  daemon completes is fine — the daemon is non-blocking by design.

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

**Failure modes.**

- **Tampered model artifact** — `VerifiedModel(model_uri)` raises
  `ario_mlflow.IntegrityError` *before* the underlying pyfunc model is loaded,
  so a swapped model never gets the chance to execute user code. Catch this
  exception to alert your security operations rather than silently fail open.
- **`predict()` always returns** the model's output even if anchoring later
  fails. Inspect `result.proof_status`: `anchoring` (in flight), `anchored`
  (Arweave upload succeeded), `failed` (see `result.anchor_error`), or
  `disabled` (no wallet / Turbo unavailable).
- **No registered model TX yet** — predictions chain to the model version's
  `ario.registration_tx`. If `ArioMlflowClient`'s registration daemon hasn't
  finished, the first few predictions chain to `GENESIS` (read once at model
  init; the registration TX never gets re-read on per-prediction calls — this
  avoids races).

## Dataset anchoring

Each MLflow dataset can have its own signed Arweave proof, independent of any
specific training run. Useful for:

- **Auditors** who need to prove "this dataset existed at time T, signed by X"
  without depending on a particular model run.
- **Dataset publishers** who anchor once and hand the TX to downstream model
  trainers.
- **Compliance** (e.g. EU AI Act Article 53 GPAI training-data summaries) that
  expects dataset-level artifacts, not fragments inside a model proof.

Two ways to use it:

```python
import mlflow
import ario_mlflow

ds = mlflow.data.from_pandas(df, source="s3://bucket/train_q1.parquet", name="train_q1")

# A) Implicit — auto-anchored inside training (recommended for typical use)
with mlflow.start_run():
    mlflow.log_input(ds, context="training")
    model.fit(...)
    mlflow.sklearn.log_model(model, "model")
    ario_mlflow.anchor()
    # Each logged dataset gets its own Arweave TX automatically;
    # the training proof references each by TX.

# B) Explicit — publisher pattern, no MLflow run needed
result = ario_mlflow.anchor(dataset=ds)
print(result["tx_id"])  # standalone dataset proof, hand off to downstream
```

The standalone-dataset envelope commits to the dataset's name, source URI,
digest, and schema hash — not to its rows. Datasets stay private; the
commitment is portable.

## Wallet & cost

Each anchored event is a ~500-byte signed commitment (bounded 400–700 bytes by
the plugin's smoke test). **Turbo's free tier covers uploads under 105 KiB**,
so typical usage is free — the auto-generated wallet works out of the box
with zero balance, and most teams never need to fund it.

You'd only need to fund the wallet if you're hitting Turbo's per-account
free-tier limits or anchoring larger payloads. To top up:

- Visit [console.ar.io](https://console.ar.io) — credit-card or crypto top-up
  for the wallet address logged by the plugin on first use
  (`wallet: <address>, mode=persistent`).

**For production deployments**, generate a dedicated wallet (don't rely on the
auto-generated one), set `ARIO_MLFLOW_ARWEAVE_WALLET=/path/to/your/wallet.json`,
and treat the wallet like any other production secret. Source data (params,
metrics, artifact bytes) always stays in MLflow — nothing else goes on chain —
so costs are flat regardless of how big your training run was.

## Network requirements

If your environment restricts outbound traffic, allowlist:

| Host | Used for |
|---|---|
| `turbo-gateway.com` | Uploads (Turbo bundler) and proof fetches |
| `arweave.net` *or other ar.io gateways* | Proof fetches (fallback) |
| `turbo.ardrive.io` | TX bundler-status checks |
| Your configured `ARIO_MLFLOW_ARIO_VERIFY_URL` | Optional ar.io Verify attestations |

Override the upload/fetch host with `ARIO_MLFLOW_GATEWAY_HOST` if you want to
route through a specific gateway operator.

## Performance

What blocks vs what runs in the background:

| Call | Behavior |
|---|---|
| `anchor()` | **Synchronous.** Hashes artifacts, signs, uploads to Turbo before returning. Typically a few seconds end-to-end; longer if artifact hashing is large. |
| `ArioMlflowClient.create_model_version()` / `transition_model_version_stage()` | **Returns immediately**, anchors in a daemon thread. Use `wait_for_anchor()` if you need the TX before continuing. |
| `VerifiedModel.__init__` | **Synchronous.** Re-hashes artifacts, compares to `ario.artifact_hash`, raises `IntegrityError` on mismatch. One-time cost per model load. |
| `VerifiedModel.predict()` | **Returns immediately** with the prediction; anchor runs in a daemon thread. No per-prediction latency added by anchoring. |

For high-throughput inference, the predict path is the hot one — predictions
return as soon as the model produces an output. The Arweave upload happens
asynchronously and writes back to the trace tags when it completes.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ARIO_MLFLOW_ARWEAVE_WALLET` | Path to an Arweave JWK wallet file | auto-generates + persists at `~/.ario-mlflow/wallet.json` |
| `ARIO_MLFLOW_GATEWAY_HOST` | ar.io gateway for uploads & fetches | `turbo-gateway.com` |
| `ARIO_MLFLOW_SIGNING_KEY` | Base64-encoded Ed25519 seed | auto-generates at `~/.ario-mlflow/keys/` |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | ar.io Verify REST API base URL — e.g. `https://perma.online/local/verify` (an ar.io operator's Verify endpoint) | ar.io attestation disabled if unset |

## Tags the plugin writes

On the training run (`anchor()`):

- `ario.enabled`, `ario.version` — via the registered `RunContextProvider`
- `ario.public_key`, `ario.verify_status`, `ario.artifact_hash`
- `ario.payload_hash` — SHA-256 of the canonical payload bytes (the same hash committed in the envelope)
- `ario.training_tx`, `ario.arweave_url` — when the Arweave upload succeeded
- `ario.wallet_mode` — `user-configured` / `persistent` / `ephemeral`

On the registered model (chain head, written by `anchor()`):

- `ario.last_training_hash` — pointer to the most recent training proof for this registered model; the next training reads it to set its `previous_hash`

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

- `ario.payload_json` — the full canonical payload (mirror of the
  `ario/predictions/<id>/payload.json` artifact). Read by `verify_source_of_truth`
  as the second MLflow surface for prediction check 3.
- `ario.decision_id`, `ario.model_name`, `ario.model_version`
- `ario.input_hash`, `ario.output_hash`, `ario.payload_hash`
- `ario.proof_status`, `ario.prediction_tx`, `ario.arweave_url`
- `ario.artifact_verified` (when known)

## CLI

```bash
ario-mlflow verify run <run_id>                  # verify training proof
ario-mlflow verify model <name>/<version>        # verify registration proof
ario-mlflow verify trace <trace_id>              # verify an inference proof
ario-mlflow audit <name>/<version>               # full model-lineage audit
```

The CLI reads `MLFLOW_TRACKING_URI` (default `./mlruns`) — export it to point
at the same store you used at training time, otherwise the run lookup will
fail with `Run '<id>' not found`. Set `ARIO_MLFLOW_ARIO_VERIFY_URL` to enable
the optional ar.io attestation row.

All `verify` commands run the same three-row verify flow plus the optional
ar.io attestation:

1. **Proof Found** — fetch the pure-commitment envelope from ar.io for the
   recorded TX ID.
2. **Decision / Training / Registration Record Matches** — download
   `ario/payload.json` from MLflow, re-hash, compare to the envelope's
   `payload_hash`, **and** re-derive canonical bytes from a *separate*
   live MLflow surface and compare to the anchored payload. This catches
   MLflow tampering — if either surface was modified after anchoring, the
   two won't agree.
   - `verify run` (`Training Record Matches`) re-fetches
     `run.data.params/metrics/artifact_checksums`.
   - `verify model` (`Registration Record Matches`) re-derives the
     artifact-verified state from the source run.
   - `verify trace` (`Decision Record Matches`) re-fetches the
     `ario.payload_json` trace tag (mirrored by `VerifiedModel.predict` at
     write time) and compares to the artifact.
3. **Signature Confirmed** — the signature on the envelope verifies
   against the embedded public key.

Plus an `Attested by` line — independent third-party check by an ar.io
gateway operator (when `ARIO_MLFLOW_ARIO_VERIFY_URL` is configured).

Results are written back to the MLflow tags and the HTML report is regenerated.

If an MLflow retention policy has pruned a prediction's trace, row 2 returns
`reason=live_refetch_incomplete` rather than silently passing — the proof
itself (signature + anchored bytes + ar.io) is on permanent storage and
remains verifiable.

## What the ar.io attestation means

`ario-mlflow verify` reports the ar.io attestation as `Verified` or
`Pending verification`. A proof reads `Verified` once an ar.io gateway has:

1. Found it permanently stored on Arweave.
2. Re-downloaded the bytes and matched the SHA-256 against the gateway's own digest.
3. Verified the signature against the original signer's public key.

For programmatic callers, `ario.attestation_level` exposes the same status as
an integer (1, 2, or 3) — useful when you want to distinguish "still
propagating" from "fully verified."

**Operator attestation.** When the ar.io gateway operator has configured a
signing wallet, the verification result is itself signed and
`ario.attested_by` / `ario.attested_at` get written back to your MLflow tags.
That's an independent statement from a known operator, verifiable by any
third party against their public key (standard RSA-PSS SHA-256).

These attestations cover **integrity and authenticity** of the anchored
record. Semantic verification (whether *this model* produced *this output*
on *this input*) is a separate problem and on the roadmap, not in v0.1.

## Tests

```bash
pytest tests/test_plugin_smoke.py tests/test_plugin_verify.py tests/test_input_anchoring.py
```

No network or MLflow server required.

## Related docs

- [`CHANGELOG.md`](../CHANGELOG.md) — release history and known limitations.
- [`docs/architecture.md`](../docs/architecture.md) — system design (pure-commitment proofs, per-event chains, JCS canonicalization).
- Demo app — see the repo-root `README.md`.
- Roadmap and deferred work — `ROADMAP.md` at the repo root.
