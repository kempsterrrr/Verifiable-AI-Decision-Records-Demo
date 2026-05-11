# Standalone Dataset Anchoring — Implementation Plan

> **STATUS (2026-05-06):** Active. Extends the input-side anchoring work (`2026-05-06-input-side-anchoring.md`) by giving datasets their own signed event with their own Arweave TX, instead of only being anchored *inside* the training proof.
>
> The v1 input-side work (Pieces A and B already shipped on the `input-side-anchoring/v1` branch) is the foundation: training proofs already commit to dataset metadata, the demo already shows it. This plan adds standalone dataset anchoring on top so each dataset has its own independently verifiable proof.
>
> **Execution:** direct main agent, opus 4.7, pause-after-each-piece, regression test for every new behavior. Same cadence as the previous work on this branch.

---

## Why this expansion (the use cases the inlined-only design can't fully serve)

1. **Independent dataset verifiability** — auditor proves "this dataset existed at time T, signed by X" without needing any specific training run.
2. **EU AI Act Article 53 GPAI alignment** — providers publishing training-data summaries need a dataset-level artifact, not a fragment inside a model's training proof.
3. **Cross-organization dataset publishing** — dataset producer anchors once, downstream model trainers reference the public TX. Today they'd have to re-state the dataset metadata in every training proof they create.
4. **Dataset versioning lineage** — `dataset_v1 → v2 → v3` as its own chain (deferred — see Out of scope).

The headline change is moving the dataset from "metadata inlined inside training" to "a first-class event with its own envelope and TX, *referenced* by training."

---

## API design — one `anchor()`, two modes

You're right that adding `anchor_dataset()` would clutter the API. Cleaner:

```python
def anchor(
    proof_engine=None, arweave=None,
    *,
    dataset=None,                       # NEW
    artifact_path=None,
    metadata=None,
    capture_otel=True,
    allow_empty_dataset_inputs=False,
) -> dict:
    """Anchor a verifiable commitment.

    Two modes — the function dispatches based on whether `dataset` is
    passed:

    A) Standalone dataset (when `dataset=ds` is set):
       Anchors the dataset's identity (name, source, digest,
       schema_hash) as its own signed event with its own Arweave TX.
       Does NOT require an active MLflow run. Useful for anchoring a
       dataset before training, or by a publisher who shares the TX
       with downstream model trainers.

    B) Training (when `dataset` is None — current behaviour):
       Must be called inside `mlflow.start_run()`. Anchors a
       training-complete event over the run's params/metrics/
       artifacts/dataset_inputs. For each dataset_input on the run,
       the plugin auto-anchors it as a standalone dataset event (a
       recursive `anchor(dataset=...)` call) and includes the
       resulting TX in the training payload's dataset_inputs entry.
    """
```

**Internal dispatch** keeps the public API simple while letting the implementation be cleanly split between two private helpers (`_anchor_training_event(...)` and `_anchor_dataset_event(...)`).

The pattern matches the existing event-type-per-context split in the rest of the plugin (`anchor()` for training, `ArioMlflowClient.create_model_version()` for registration, `VerifiedModel.predict()` for prediction) — the *existing* plugin already has multiple anchoring entry points; we're not introducing one, we're factoring one parameter into an existing one.

### Caller workflows

**Implicit (auto-anchor, recommended for typical use):** caller's MLflow code is unchanged from input-side v1. Plugin transparently anchors each logged dataset before signing the training event.

```python
ds = mlflow.data.from_pandas(df, source="...", name="train_q1")
with mlflow.start_run():
    mlflow.log_input(ds, context="training")
    model.fit(...)
    anchor()  # dataset gets its own TX; training references it
```

**Explicit (publisher / governance pattern):** anchor the dataset standalone, hand the TX off, training references it later.

```python
ds = mlflow.data.from_pandas(df, source="...", name="train_q1")
result = anchor(dataset=ds)
print(result["tx_id"])  # the dataset's own anchored proof
# ... later, in training, the same dataset gets auto-referenced via
# its TX (or, in v3, looked up via a registry of pre-anchored datasets)
```

---

## In scope for this expansion (v2)

1. **New event type `dataset`** with its own canonical payload schema (name, source, source_type, digest, schema_hash, anchored_at, public_key, signature). Subject type `mlflow_dataset`. Its own envelope, signed and uploaded to Arweave.
2. **`anchor()` accepts optional `dataset=ds`** — internal dispatch to `_anchor_dataset_event()` for that mode.
3. **Training's auto-anchor of inputs.** When `anchor()` is called from training context, each entry in `run.inputs.dataset_inputs` is anchored via the dataset path; each entry in the resulting training payload's `dataset_inputs` list gains an `anchor_tx` field pointing at the dataset's TX.
4. **Verify path for dataset events.** Same four-check structure as training/registration/prediction: signature, anchored bytes, source-of-truth re-derivation against MLflow's live dataset metadata, ar.io attestation.
5. **Demo lineage updates.** Dataset node on `model_chain.html` shows its own TX, its own mini-verify (proof found / record matches / signature confirmed), its own "View on ar.io" link.
6. **Tamper button retarget.** The tamper button (already designed for Piece C of v1) now targets the dataset proof's `payload.json` artifact in MLflow rather than the run's `meta.yaml`. Verify catches via the dataset event's SoT check, AND training's SoT (since training still inlines the metadata for redundancy).
7. **README + ROADMAP updates.**

## Out of scope (deferred)

- **Cross-run dataset reuse / dedup.** Today every training run that references the same dataset re-anchors it. Avoiding the duplicate Arweave upload requires a registry (file in `~/.ario-mlflow/`, MLflow tag, or explicit kwarg). Real but a v3-shape optimization, not a correctness issue.
- **Cross-org dataset publishing UX.** The plumbing supports it (publisher anchors, hands off TX), but a polished workflow (publish-and-share helpers, signed publisher attestation, etc.) is its own work.
- **Dataset versioning chain.** Dataset events would chain via `previous_hash` to the previous version of the same dataset, mirroring how training events chain to the previous training of the same registered model. Useful for "dataset v1 → v2 → v3" provenance, but adds chain-head bookkeeping that doesn't exist for standalone datasets today. v3 follow-up.

## Architecture decisions (locked)

- **Inlined metadata stays in canonical bytes; `anchor_tx` is navigation, NOT canonical.** Each `dataset_inputs` entry retains its inlined metadata (name, source, digest, schema_hash, context). The dataset's anchor TX is stored separately as a run-level MLflow tag (`ario.dataset_anchor_tx.<name>`) and surfaced through the demo's lifecycle store for UI / chain walking. Same separation the plugin already uses for registration events: `previous_hash` (in canonical bytes, for cryptographic chain integrity) vs `previous_tx` (in lifecycle store, for navigation).

  Why: chain integrity comes from the inlined `digest` (cryptographic binding to the dataset content), not from `anchor_tx`. Putting the TX in canonical bytes would force the verify-side refetcher to read it from MLflow at verify time, adding plumbing for no real security benefit. An attacker switching a run tag could only point at a different dataset proof with the same digest — still represents the same data, can't forge.

  Distinction worth keeping in mind for future readers: chain walking (an auditor following `anchor_tx` from training to dataset to prove lineage) and single-event integrity verification (does this event's signed bytes still match the live MLflow state?) are different operations. Chain walking is always a feature — that's why `anchor_tx` references exist on run tags. Inline metadata exists so a single event's *integrity* can be checked without cascading dependencies on every referenced event being reachable. Lineage walks the chain; integrity does not need to.
- **Auto-anchor by default.** Training's `anchor()` auto-anchors any logged inputs that aren't already anchored. Caller doesn't need a separate explicit call. The explicit `anchor(dataset=ds)` path remains for advanced workflows (publishers, pre-anchoring).
- **Same fingerprinting rule.** Schema and any structured fields go through `canonical_json()` (JCS) before hashing. Identical to the rest of the codebase.
- **Same fail-closed posture.** `allow_empty_dataset_inputs=True` escape hatch still applies to training-mode `anchor()`. Dataset-mode `anchor(dataset=ds)` requires `dataset` to be a valid MLflow Dataset object (raises otherwise).

---

## Plan structure

Four pieces, mirroring the cadence we used for input-side anchoring v1.

### Piece A — Plugin: dataset event type + auto-anchor in training

| Task | Summary |
|---|---|
| A1 | New `_anchor_dataset_event()` private helper. Builds canonical payload for a dataset, signs envelope, uploads to Arweave. Returns the same shape `anchor()` already returns (envelope, payload, payload_bytes, payload_hash, anchor_result, etc.). 5 failing tests including signature shape, payload schema, JCS canonicalization on schema, fail-on-bad-input. |
| A2 | `anchor()` public function dispatches on `dataset=` kwarg → `_anchor_dataset_event()`. Existing training-mode behaviour unchanged when no kwarg. 2 failing tests. |
| A3 | Training-mode `anchor()` auto-anchors each `dataset_input` and adds `anchor_tx` to the corresponding canonical-payload entry. 3 failing tests covering auto-anchor invocation, multiple inputs, behaviour when arweave is disabled. |
| A4 | Verify path: new `_refetch_dataset_live_fields` for dataset events; `_LIVE_FIELD_REFETCHERS` gains `"dataset"`; `_REQUIRES_FULL_MLFLOW_VERIFICATION` includes it. 3 failing tests for SoT pass/fail/legacy-skip. |

🛑 **Pause after Piece A** — surface plugin behaviour with a brief diagnostic showing the new event type's payload, then wait for approval.

### Piece B — Demo: dataset node has its own TX and verification

| Task | Summary |
|---|---|
| B1 | `app/main.py` lifecycle hydration captures the dataset event's TX (returned alongside training's anchor result) and persists it in lifecycle_store as a new `dataset_anchored` entry per dataset. |
| B2 | `model_chain.html` dataset node renders the lifecycle entry (not just inlined-from-training): its own TX, its own mini-verify card, "View on ar.io" link in the footer matching how training/registration nodes look. |
| B3 | `run_detail.html` "Training Inputs" section gains a "Anchor TX" row per dataset linking to ar.io. |

🛑 **Pause after Piece B** — manual smoke check: lineage page shows the dataset node with its own TX + green dot independent of training's status, verify chain flips both. Then wait for approval.

### Piece C — Demo: tamper button targets dataset proof

| Task | Summary |
|---|---|
| C1 | New tamper backend `tamper_dataset_proof(event_id, ...)` that mutates the dataset event's `ario/payload.json` artifact in MLflow (the dataset's own canonical bytes, not the run's). Same atomic snapshot+restore pattern as the existing `saved` tamper for predictions / training. |
| C2 | Tamper button on `model_chain.html` dataset node and `run_detail.html` Training Inputs section targets the new tamper kind. Mirrors the v1-original tamper button location decision. |
| C3 | Regression test: anchor → tamper dataset proof → verify dataset event SoT FAILs → reset → verify PASSes. |

🛑 **Pause after Piece C** — full demo smoke test (anchor, verify, tamper, observe both training and dataset rows flip, reset).

### Piece D — Documentation

| Task | Summary |
|---|---|
| D1 | README: explain dual-mode `anchor()`, the publisher/consumer workflow with an example, the auto-anchor default, deferral of cross-run dedup. |
| D2 | ROADMAP: mark standalone dataset anchoring shipped; add deferred items (cross-run reuse, cross-org publishing, dataset versioning chain). |

---

## File structure

| File | Role |
|---|---|
| `ario_mlflow/anchoring.py` | `anchor()` extended with `dataset=` kwarg; `_anchor_dataset_event()` helper; auto-anchor of inputs in training mode |
| `ario_mlflow/proof.py` | (no changes — existing `create_commitment()` handles arbitrary subjects) |
| `ario_mlflow/verify.py` | New `_refetch_dataset_live_fields`; updated `_LIVE_FIELD_REFETCHERS` and `_REQUIRES_FULL_MLFLOW_VERIFICATION`; download helper for the dataset's payload artifact |
| `ario_mlflow/__init__.py` | (no new exports — `anchor()` is the same name) |
| `app/main.py` | Lifecycle store gains `dataset_anchored` entries; cache-record builder includes anchor_tx per dataset |
| `app/ui.py` | Routes surface dataset's lifecycle entry to templates |
| `app/tamper.py` | New `tamper_dataset_proof` backend + reset path |
| `templates/run_detail.html` | Anchor TX row in Training Inputs |
| `templates/model_chain.html` | Dataset node uses lifecycle's dataset_anchored entry: own TX, own mini-verify, own footer link |
| `tests/test_input_anchoring.py` | Extended with dataset-event tests (already where the input-side anchoring tests live) |
| `tests/test_tamper_endpoints.py` | New regression test for dataset-proof tamper |
| `README.md` | Dual-mode `anchor()` workflow + publisher/consumer pattern |
| `ROADMAP.md` | Standalone dataset anchoring shipped; deferred items added |

---

## Constraints

- Use opus 4.7 (highest thinking mode) throughout. Direct execution by main agent — no subagent dispatch (same as v1).
- **Pause for user review after each Piece (A, B, C, D)**, surface evidence at each checkpoint, wait for explicit approval.
- Each task: one small commit, clear message. TDD where it makes sense (each new behavior in plugin + tamper backend gets failing tests first).
- No backwards-compat shims; pre-prod codebase. Existing input-side-anchoring v1 proofs (already on this branch) verify cleanly under v2 — they don't have an `anchor_tx` in their `dataset_inputs` entries, and the verify path treats that as legacy/absent (mirrors the v1 → legacy compatibility we already added for proofs without `dataset_inputs`).
- Match existing code style.

---

## Open questions to resolve at task time

- **Auto-anchor failure handling.** If the dataset upload to Arweave fails (transient error), should training-mode `anchor()` still proceed with `anchor_tx=None` and inlined metadata only, or refuse? Today the plugin's pattern for upload failures is "signed but not anchored" — the proof exists with a signature but no Arweave TX. I'd default to that pattern for consistency: `anchor_tx` field is empty, training proof still ships, verification of training works (inlined metadata still anchored), but the dataset's separate proof is "signed only" until the next successful upload retry. To resolve in Task A3.
- **Lifecycle store deduplication.** When an auto-anchor reuses a previously-anchored dataset (digest match), do we add a new `dataset_anchored` lifecycle entry or reference the existing one? Right now we don't dedupe — every training run gets its own dataset_anchored entry. Cleanup in v3. To document explicitly in Task B1.
