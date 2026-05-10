# Changelog

All notable changes to `ario-mlflow` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Active development. See open pull requests at
[github.com/vilenarios/Verifiable-AI-Decision-Records-Demo/pulls](https://github.com/vilenarios/Verifiable-AI-Decision-Records-Demo/pulls)
for in-flight work — currently a plugin safety pass, a network resilience
pass, and this documentation pass.

## [0.1.0] — 2026

Initial alpha release. Not yet published to PyPI; install from source via
`pip install -e .`.

### Added

- **`anchor()`** — training provenance helper that signs a pure-commitment
  envelope over the active MLflow run's params, metrics, and artifact
  checksums, and uploads it to Arweave via Turbo.
- **`ArioMlflowClient`** — drop-in replacement for `MlflowClient` that
  auto-anchors `create_model_version()` and
  `transition_model_version_stage()` in a daemon thread. Exposes
  `anchor_status()` and `wait_for_anchor()` for callers that need the
  outcome.
- **`VerifiedModel`** — inference wrapper with load-time integrity check
  (raises `IntegrityError` on artifact-hash mismatch before user code runs)
  and per-prediction anchoring in a background thread.
- **Standalone dataset anchoring** — `anchor(dataset=ds)` mints an
  independent signed event with its own Arweave TX, no MLflow run required.
  In-training calls also auto-anchor each logged dataset and reference its
  TX in the training proof.
- **CLI** — `ario-mlflow verify run|model|trace <id>` and
  `ario-mlflow audit <name>/<version>` for after-the-fact verification and
  full-lineage audits.
- **MLflow `RunContextProvider` entry point** — importing the package
  auto-tags every run with `ario.enabled` and `ario.version`.
- **OpenTelemetry correlation** — auto-captures `otel_trace_id` /
  `otel_span_id` into the canonical payload when an active span exists, so
  proofs are correlatable with infrastructure tracing.
- **HTML verification report** generated as an MLflow artifact
  (`ario/verification.html`) on each anchored event.

### Known limitations

- MLflow 3.x: prediction-side `verify_source_of_truth` may return
  `live_refetch_incomplete` due to changed trace artifact location handling
  in 3.x. Training and registration verification are unaffected.
- No retry on transient gateway failures, no multi-gateway fallback yet
  (queued in the resilience pass PR).
- A caller-supplied wallet path that points to a missing or malformed file
  silently falls back to an auto-generated wallet (queued: raise
  `WalletLoadError` instead, in the safety pass PR).
