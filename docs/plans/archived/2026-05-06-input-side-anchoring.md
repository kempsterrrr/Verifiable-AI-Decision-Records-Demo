# Input-side Anchoring — Implementation Plan

> **STATUS (2026-05-06):** Active. Created during a strategic pivot from Piece C of `2026-05-05-verification-correctness-piece-b-c.md`. That plan's Piece C (Tasks 4–8 — trusted-issuer-key + swap-signer tamper) is paused and resumes after this work ships.
>
> **For agentic workers:** Execute directly with the main agent using opus 4.7 (highest thinking mode). Pause-for-review checkpoints between each Piece. Each task gets one small commit; regression test for every new tamper or verification path.

---

## Goal

Close the headline "input-side gap" in the verification story: training proofs commit to the **dataset(s) used** to train the model, alongside the existing source-code commit and runtime-environment fingerprints (which are already anchored).

After this ships, the demo can show a verifiable cryptographic chain from **dataset → training run → registration → prediction**, end-to-end, with each link signed and tamper-detectable.

---

## Why this priority change (context)

ROADMAP's near-term focus item #2 — "Our current proof hashes outputs, not inputs — a bad actor could change training data and produce a model with identical metrics but different weights" — is the biggest stated honesty gap. Input-side anchoring closes the dataset half of it.

Mapping to the EU AI Act assessment:

- **Annex IV §1** (training-data documentation, datasets, provenance) — directly addressed.
- **Annex IV §3** (data governance) — supported.
- **Article 43** (notified-body conformity assessment) — auditor can walk dataset → decision as a signed chain.
- **Article 53** (GPAI training data summaries) — provides the structured input for that summary.

---

## In scope for v1

1. **Plugin auto-captures dataset inputs from MLflow's `mlflow.log_input()` API.** No new caller-side code required — the plugin reads `run.inputs.dataset_inputs` at anchor time and includes them in the training canonical payload.
2. **Fail-closed on missing dataset inputs.** `anchor()` raises `ValueError` when called on a run that has no logged inputs. An explicit `allow_empty_dataset_inputs: bool = False` escape hatch is provided for the rare legitimate case (early-stage research, GPAI workflows with no single dataset). The demo eats its own dog food: `app/model.py::load_model` updated to call `mlflow.log_input()` with the synthetic credit-scoring dataset before anchoring.
3. **Schema is anchored as a hash, not plaintext.** The canonical payload contains `dataset_inputs[].schema_hash` (SHA-256 of the JCS-canonicalized schema JSON). Privacy-preserving — column names not exposed in the proof or any bundle exported from it. Same fingerprinting pattern as `artifact_checksums`.
4. **All fingerprinting goes through `canonical_json()` (RFC-8785 JCS) before hashing.** Locks byte-determinism across MLflow versions, whitespace variations, and dict-ordering quirks. Standard rule across the codebase from this point.
5. **Verify-side source-of-truth re-derivation** extended to re-fetch `dataset_inputs` from MLflow and detect post-anchor mutations of any field (digest, schema, name, source, context).
6. **Demo UI surfaces training inputs** on the run-detail page and adds a dataset node at the head of the model-chain lineage diagram.
7. **One demo tamper button: "Tamper with the dataset metadata"** on both `run_detail.html` and `model_chain.html`. Mutates the `digest` field in `mlruns/<exp>/datasets/<id>/meta.yaml`; verify catches via the source-of-truth check. Regression test included.
8. **README + ROADMAP updates** documenting the new capability, the MLflow `log_input()` workflow, the strict-mode behavior + escape hatch, and the documented limitation around MLflow's default sampled-row digest. ROADMAP gains a new "Chain-confirmation gating" item under Provenance depth (deferred).

## Out of scope for v1 (deferred to follow-up)

- **Container image digest** — anchoring `$IMAGE_DIGEST` or equivalent. Environment-specific; revisit when a customer asks.
- **Auto-detection of git commit beyond what MLflow already does.** MLflow's `mlflow.source.git.commit` tag is good enough for v1. (We already anchor it.)
- **Helpers for hashing very large datasets** (Merkle trees, streaming hashing). Caller's responsibility for v1; documented.
- **Chain-confirmation gating.** Refusing to register a model until its training event is *confirmed anchored*, refusing to predict until registration is confirmed, etc. Strong compliance story but a multi-week design pass touching async/retry semantics, customer SLAs, and the broader continuous-verification ROADMAP item. **Logged as a new ROADMAP entry under Provenance depth.**
- **Demo UI to capture dataset metadata via the training form.** UX polish, not foundational.
- **Multi-dataset structured groups** (train/val/test as a typed relationship). Treat all dataset inputs uniformly via MLflow's existing `context` tag for now.
- **Profile field** in the canonical payload. MLflow's `Dataset.profile` is large and noisy; we anchor `name`, `source`, `source_type`, `digest`, `schema_hash`, and the `mlflow.data.context` tag — that's enough for the audit story without bloating the proof.

## Documented limitation (must surface in README)

MLflow's default `Dataset.digest` is computed from a **sample** of the data, not the full bytes (this is for performance — large dataframes would be too expensive to hash row-by-row). That's enough to detect "someone swapped to a totally different dataset" but does **not** detect single-row poisoning attacks.

For airtight dataset integrity, callers can override the digest:

```python
import hashlib
true_digest = hashlib.sha256(open("training_data.csv", "rb").read()).hexdigest()
ds = PandasDataset.from_pandas(df, source="...", name="...", digest=true_digest)
mlflow.log_input(ds, context="training")
```

README must explain this distinction so adopters aren't misled about what v1 guarantees.

---

## File structure

| File | Role |
|---|---|
| `ario_mlflow/anchoring.py` | Read `run.inputs.dataset_inputs` in `anchor()`; serialize into canonical payload's new `dataset_inputs` field; raise on empty unless `allow_empty_dataset_inputs=True` |
| `ario_mlflow/verify.py` | Extend `_refetch_training_live_fields` to re-fetch dataset inputs; same serialization helper |
| `ario_mlflow/__init__.py` | (No new exports needed; existing API unchanged from caller side) |
| `app/model.py` | Demo's auto-train (`load_model`) updated to call `mlflow.log_input()` with the synthetic credit-scoring dataset before anchoring |
| `app/ui.py` | Surface `dataset_inputs` from the anchored payload on run-detail + model-chain routes |
| `app/tamper.py` | New tamper kind that mutates the `digest` field in `mlruns/<exp>/datasets/<id>/meta.yaml` (atomic replace pattern, snapshot for reset) |
| `app/main.py` | (Possibly) wire a new tamper route if the URL shape needs extending |
| `templates/run_detail.html` | New "Training Inputs" section + new tamper button |
| `templates/model_chain.html` | Dataset node at head of lineage diagram + new tamper button |
| `tests/test_plugin_smoke.py` | Plugin tests: anchor includes dataset_inputs; refetcher re-derives them; fail-closed when empty; escape hatch works; multi-input ordering deterministic; schema_hash uses canonical_json |
| `tests/test_tamper_endpoints.py` | Regression test: dataset-metadata tamper → source-of-truth FAIL → reset → PASS |
| `README.md` | Usage section: `mlflow.log_input()` workflow, fail-closed behavior + escape hatch, the digest-sampling limitation + override workaround, schema-hash privacy reasoning |
| `ROADMAP.md` | Mark dataset-input anchoring shipped; note container-digest deferred; add new "Chain-confirmation gating" item under Provenance depth |

---

## Piece A — Plugin: anchor + verify learn about `dataset_inputs`

### Task A1 — Plugin: `anchor()` includes `dataset_inputs` (fail-closed on empty)

**Files:**
- Modify: `ario_mlflow/anchoring.py` — extend `anchor()` and the training-payload builder to include `dataset_inputs`; raise when empty unless `allow_empty_dataset_inputs=True`
- New helper: `_serialize_dataset_inputs(run)` returning the canonical-friendly list

**Implementation sketch:**

```python
from ario_mlflow.proof import canonical_json, hash_data


def _serialize_dataset_inputs(run) -> list[dict]:
    """Read run.inputs.dataset_inputs and produce a deterministic list
    suitable for inclusion in canonical bytes.

    Schema is fingerprinted (SHA-256 of JCS-canonicalized schema JSON)
    rather than included verbatim — column names can be sensitive in
    regulated domains. Other identifier fields (name, source, digest,
    context) are plaintext, matching the existing pattern used by
    artifact_checksums (hashes for content, plaintext for identifiers).

    Returns an empty list when the run has no logged inputs. Sorted by
    (name, source, context, digest) for determinism even when multiple
    inputs share a name.
    """
    inputs = getattr(getattr(run, "inputs", None), "dataset_inputs", None) or []
    out = []
    for di in inputs:
        ds = di.dataset
        context = next(
            (t.value for t in (di.tags or []) if t.key == "mlflow.data.context"),
            None,
        )
        # Fingerprint the schema string. Run through canonical_json
        # first so MLflow's internal whitespace / field order can't
        # cause spurious hash drift.
        schema_str = ds.schema or ""
        if schema_str:
            try:
                schema_canonical = canonical_json(json.loads(schema_str))
            except (ValueError, TypeError):
                # Schema isn't valid JSON — fall back to hashing the
                # raw bytes. Surface in tests if this ever fires.
                schema_canonical = schema_str.encode("utf-8")
            schema_hash = hash_data(schema_canonical)
        else:
            schema_hash = ""

        out.append({
            "name":        ds.name,
            "source":      ds.source,
            "source_type": ds.source_type,
            "digest":      ds.digest,
            "schema_hash": schema_hash,
            "context":     context,
        })
    out.sort(key=lambda d: (
        d.get("name", "") or "",
        d.get("source", "") or "",
        d.get("context", "") or "",
        d.get("digest", "") or "",
    ))
    return out


def anchor(*, proof_engine, arweave, allow_empty_dataset_inputs=False, ...):
    ...
    dataset_inputs = _serialize_dataset_inputs(run)
    if not dataset_inputs and not allow_empty_dataset_inputs:
        raise ValueError(
            "anchor(): training run has no logged dataset inputs. Call "
            "mlflow.log_input(dataset, context=...) before training, or "
            "pass allow_empty_dataset_inputs=True to override (only do "
            "this for workflows that genuinely have no single dataset; "
            "see README on input-side anchoring)."
        )
    payload["dataset_inputs"] = dataset_inputs
    ...
```

The training canonical payload gains a new field `dataset_inputs: list[dict]`. With strict default, every new training proof has at least one dataset reference.

**Steps:**
- [ ] Write failing test `test_anchor_raises_when_no_dataset_inputs_logged`
- [ ] Write failing test `test_anchor_succeeds_with_empty_inputs_when_escape_hatch_set`
- [ ] Write failing test `test_anchor_includes_dataset_inputs_when_run_has_log_input`
- [ ] Write failing test `test_anchor_serializes_schema_as_jcs_hash_not_plaintext`
- [ ] Write failing test `test_anchor_dataset_inputs_serialization_is_deterministic` (same inputs in different log order → byte-identical canonical bytes; covers Q3 sort stability)
- [ ] Write failing test `test_anchor_schema_hash_stable_across_calls` (pin the JCS canonicalization behavior — the Q2 belt-and-braces check)
- [ ] Implement `_serialize_dataset_inputs`
- [ ] Wire into `anchor()`'s payload builder + add the strict-mode validation
- [ ] Run tests, confirm pass
- [ ] Commit

### Task A1.5 — Demo: auto-train uses `mlflow.log_input()`

**Files:**
- Modify: `app/model.py::load_model` — call `mlflow.log_input(dataset, context="training")` before the model fit, with the synthetic credit-scoring DataFrame as a `mlflow.data.from_pandas(...)` dataset

**Implementation sketch:**

```python
import mlflow
import mlflow.data

# Inside load_model's auto-train branch, before mlflow.start_run():
dataset = mlflow.data.from_pandas(
    train_df,
    source="synthetic://credit-scorer-demo",
    name="credit_scorer_demo_training_data",
)

with mlflow.start_run() as run:
    mlflow.log_input(dataset, context="training")
    # ... rest of existing training flow
```

Without this, the demo can't anchor under the new strict default. This task is the "demo eats its own dog food" change.

**Steps:**
- [ ] Update `load_model` to log the synthetic dataset
- [ ] Run the existing demo-fixture test suite to confirm the auto-train still completes end-to-end
- [ ] Manual smoke test: blank `data/` + `mlruns/`, boot uvicorn, confirm a fresh model trains and anchors
- [ ] Commit

### Task A2 — Plugin: `_refetch_training_live_fields` re-fetches dataset inputs

**Files:**
- Modify: `ario_mlflow/verify.py::_refetch_training_live_fields` — add `dataset_inputs` to the fresh dict and to `_TRAINING_REQUIRED_LIVE_FIELDS`

**Implementation sketch:**

The helper already re-fetches params, metrics, artifact_checksums, source_name, git_commit. Add:

```python
fresh["dataset_inputs"] = _serialize_dataset_inputs(run)
```

(Reuse the same helper from A1 — extract to a shared location if needed.)

Add `"dataset_inputs"` to `_TRAINING_REQUIRED_LIVE_FIELDS` so a refetcher that fails to produce it raises `LiveRefetchError`.

**Steps:**
- [ ] Write failing test `test_verify_source_of_truth_passes_when_dataset_inputs_match`
- [ ] Write failing test `test_verify_source_of_truth_fails_when_dataset_inputs_mutated_in_mlflow`
- [ ] Write failing test `test_verify_source_of_truth_fails_when_dataset_input_added_after_anchor`
- [ ] Implement the refetcher extension
- [ ] Run tests, confirm pass
- [ ] Commit

**🛑 PAUSE for user review here.** Piece A is complete. Surface evidence (test output, brief example of an anchored payload showing the new field) and wait for explicit approval before starting Piece B.

---

## Piece B — Demo: surface `dataset_inputs` in the UI

### Task B1 — Demo: `run_detail.html` "Training Inputs" section

**Files:**
- Modify: `templates/run_detail.html`
- Modify: `app/ui.py::run_detail` route — pass `dataset_inputs` to the template (read from the anchored payload's canonical bytes or from the training envelope's record cache)

**Implementation sketch:**

New section block on the run-detail page, immediately after the existing params/metrics summary and before the verification card. Renders one card per dataset input:

```jinja
{% if dataset_inputs %}
<div class="section">
  <h3>Training Inputs</h3>
  <div class="section-body">
    {% for di in dataset_inputs %}
      <div class="dataset-input-row">
        <strong>{{ di.name }}</strong>
        <span class="muted">({{ di.context or '—' }})</span>
        <div>Source: <code>{{ di.source }}</code></div>
        <div>Digest: <code class="mono mono-sm">{{ di.digest[:16] }}…</code> <button class="copy-btn" data-copy="{{ di.digest }}">copy</button></div>
        {% if di.schema_hash %}
        <div title="SHA-256 of the canonical-JSON-serialized schema. Schema column names are not stored in the proof for privacy — only this fingerprint.">Schema fingerprint: <code class="mono mono-sm">{{ di.schema_hash[:16] }}…</code></div>
        {% endif %}
      </div>
    {% endfor %}
  </div>
</div>
{% else %}
<div class="section muted">
  <h3>Training Inputs</h3>
  <p>No dataset inputs recorded for this run. With v1's strict mode this shouldn't happen for newly-anchored runs — likely a legacy proof from before input-side anchoring shipped, or a run anchored with the <code>allow_empty_dataset_inputs=True</code> escape hatch. See README.</p>
</div>
{% endif %}
```

Source of `dataset_inputs` in the route: read the training envelope's anchored payload (already cached on the lifecycle envelope as `record`).

**Steps:**
- [ ] Update `app/ui.py::run_detail` to extract `dataset_inputs` from the training envelope
- [ ] Add the section to `run_detail.html`
- [ ] Manual smoke test: train a fresh model with `log_input`, view run detail, confirm the section renders
- [ ] Manual smoke test: legacy training run without `log_input` → "No dataset inputs recorded" copy renders
- [ ] Commit

### Task B2 — Demo: `model_chain.html` dataset node

**Files:**
- Modify: `templates/model_chain.html`
- Modify: `app/ui.py::model_chain` route — extract dataset inputs from the training envelope

**Implementation sketch:**

Add a new node at the head of the existing lineage diagram (training → registration → predictions) showing the dataset(s). Visual treatment: same node style as the others (anchored badge if anchored, ar.io link, etc.), labeled "Dataset(s)". When multiple inputs are present, render them stacked or as a grouped sub-list.

The dataset node doesn't have its own Arweave TX — it lives *inside* the training event's canonical payload — so the badge says something like "Anchored within Training Run" with a link to the training proof on ar.io. This is the verifiable-chain visualization moment.

**Steps:**
- [ ] Update `app/ui.py::model_chain` to surface dataset inputs to the template
- [ ] Add the dataset node to the lineage diagram
- [ ] Manual smoke test: model with `log_input` shows dataset node at the head; chain visually clear
- [ ] Manual smoke test: legacy model without `log_input` shows an "informational" node noting no dataset captured
- [ ] Commit

**🛑 PAUSE for user review here.** Piece B (display) is complete. Surface evidence and wait for approval before starting Piece C.

---

## Piece C — Demo: tamper button for dataset metadata

### Task C1 — Backend: tamper kind that mutates a dataset input in MLflow

**Files:**
- Modify: `app/tamper.py` — new branch in `tamper_live` (or a new `tamper_dataset_metadata` function) that mutates a dataset input's digest on MLflow's file backend
- Modify: `app/tamper.py` — corresponding reset path

**Storage layout (resolved empirically against MLflow 3.11.1):**

```
mlruns/<exp_id>/
├── <run_id>/inputs/<input_uuid>/meta.yaml    ← input → dataset link (small)
└── datasets/<dataset_id>/meta.yaml           ← actual dataset metadata
      (digest, name, profile, schema, source, source_type)
```

**Tamper target:** `mlruns/<exp_id>/datasets/<dataset_id>/meta.yaml`. Mutate the `digest` field.

**Resolving dataset_id from a run_id:**
- Use `MlflowClient.get_run(run_id).inputs.dataset_inputs[0].dataset` and look up its file path. Or read `<run_id>/inputs/<input_uuid>/meta.yaml`'s `source_id` field.

**Mutation pattern (in-place truncate, consistent with the model.pkl tamper):**
1. Snapshot the original `meta.yaml` bytes for reset
2. Parse YAML, mutate `digest` to a tampered value (e.g., `TAMPERED-DIGEST-DEADBEEF`)
3. `open(path, "wb")` and write the new YAML
4. Reset writes the snapshot back via the same in-place pattern

(Atomic replace via `os.replace` was considered and explicitly declined for consistency with the model.pkl tamper — same race-window argument applies, both are single-user demo tampers on small files. If we ever want atomic-replace semantics, apply to both tampers in one change.)

**Honest limitation to document in the docstring:** the dataset YAML is *shared across all runs that reference the same dataset_id*. Tampering one mutates the digest seen by every run pointing at it. In the demo this is fine — each fresh auto-train logs its own dataset entry — but worth flagging for callers running multiple training jobs against the same dataset.

**Steps:**
- [ ] Add `tamper_dataset_metadata` branch (or new function) using the resolution + atomic-replace pattern above
- [ ] Add reset path mirroring the existing `artifact_swap_path` reset
- [ ] Smoke check: tamper a real run's dataset meta.yaml; confirm `MlflowClient.get_run(run_id).inputs.dataset_inputs[0].dataset.digest` reflects the change after mutation and again after reset
- [ ] Commit (tamper backend only; tests + UI follow in C2 and C3)

### Task C2 — Demo: tamper button on `run_detail` and `model_chain`

**Files:**
- Modify: `templates/run_detail.html` — add new tamper row
- Modify: `templates/model_chain.html` — add new tamper row (in the training-side tamper section, adjacent to the existing buttons)
- Modify: `app/main.py` — register a tamper-route variant if a new event_type or kind name is introduced

**Implementation sketch:**

New tamper row in the existing tamper sections:

```html
<div class="tamper-row">
  <div class="tamper-row-title"><span class="num">N</span>Tamper with the dataset metadata</div>
  <div class="tamper-row-desc">Mutate the digest of a dataset input recorded on the training run — simulates the "what data trained this model?" lie. <em>Modifies <code>inputs.yaml</code> on the run.</em> <strong>Breaks → Training Record Matches</strong> on this run (and on the Model Lineage page's training node).</div>
  <button class="btn-tamper" data-tamper-kind="dataset" data-event-type="training" data-event-id="{{ envelope.record.run_id }}">Tamper</button>
</div>
```

Same auto-reverify behavior already wired in for the existing tamper buttons.

**Steps:**
- [ ] Add the tamper row to `run_detail.html`
- [ ] Add the equivalent row to `model_chain.html`
- [ ] Verify the existing JS tamper handler routes correctly (POST to `/tamper/dataset/training/<run_id>`)
- [ ] Manual smoke test: train fresh model with `log_input`, click tamper button → page reloads with verify → "Training Record Matches" shows FAIL; reset → PASS
- [ ] Commit

### Task C3 — Regression test

**Files:**
- Modify: `tests/test_tamper_endpoints.py`

**Implementation sketch:**

Mirror `test_swap_artifact_tamper_breaks_registration_source_of_truth`. Train a model with `mlflow.log_input` in the test, then call the dataset-tamper backend directly, then run `verify_source_of_truth` and assert it flips PASS→FAIL. Then reset and assert PASS again.

**Steps:**
- [ ] Add test fixture helper to log a dataset input on the training run after the demo's auto-train
- [ ] Write `test_tamper_dataset_metadata_breaks_training_source_of_truth` (full cycle: anchor → verify PASS → tamper → verify FAIL → reset → verify PASS)
- [ ] Run the full tamper-endpoints suite to confirm no intra-file regressions
- [ ] Commit

**🛑 PAUSE for user review here.** Piece C (demo tamper) is complete. Surface test results + a manual smoke test confirmation and wait for approval before Piece D.

---

## Piece D — Documentation

### Task D1 — README

**Files:**
- Modify: `README.md`

**Content additions:**

A new "Input-side anchoring" section under usage, covering:

1. **The headline:** training proofs now commit to dataset references via MLflow's `log_input()` API.
2. **Code example** showing the user workflow (load + log_input + train + anchor).
3. **Strict-mode behavior:** `anchor()` now raises `ValueError` when called on a run with no logged dataset inputs. Documented escape hatch (`allow_empty_dataset_inputs=True`) for the rare legitimate case (research, GPAI workflows with no single dataset). Recommendation: always log inputs unless you have a specific reason not to.
4. **Schema-hash privacy reasoning:** column names of the dataset's schema are not stored in the proof — only the SHA-256 fingerprint of the JCS-canonicalized schema string. Tamper-detectable, but doesn't expose potentially-sensitive column names in any bundle exported from the proof.
5. **Documented limitation:** MLflow's default digest for `PandasDataset` samples rows for performance; it catches "different dataset entirely" but not single-row poisoning. For airtight integrity, override with a true SHA-256 of the full bytes.
6. **Override workaround example** (compute SHA-256 manually, pass as `digest` to the dataset constructor).
7. **What's deferred** (container digest, Merkle dataset hashing, demo UI capture form, chain-confirmation gating) — pointer to ROADMAP.

**Steps:**
- [ ] Draft the new README section
- [ ] Confirm the workflow code example actually runs (paste into a scratch script, execute against a local MLflow + plugin)
- [ ] Commit

### Task D2 — ROADMAP

**Files:**
- Modify: `ROADMAP.md`

**Edits:**

- Move "Input-side anchoring" from "Near-term focus" to the recent-work bullet list (update with the PR number once merged).
- Add a "Container digest" item under "Provenance depth" backlog with a short rationale for why it was deferred (environment-specific; revisit when a customer asks).
- Add a "Stronger dataset hashing" item under "Provenance depth" noting the Merkle / streaming follow-up for callers with very large datasets.
- **Add a new "Chain-confirmation gating" item under "Provenance depth"** with rationale: today the chain links via `previous_hash` regardless of whether the parent event has *confirmed* anchoring on Arweave. Strong audit story would refuse to register a model until its training is confirmed-anchored, etc. Multi-week design pass; logged here so it doesn't get lost.

**Steps:**
- [ ] Update ROADMAP entries
- [ ] Verify ROADMAP still reads cleanly end-to-end
- [ ] Commit

**🛑 PAUSE for user review and final sign-off before opening the PR.**

---

## Constraints

- Use opus 4.7 (highest thinking mode) throughout. Direct execution by main agent — no subagent dispatch.
- **Pause for user review after each Piece (A, B, C, D).** Surface evidence at each checkpoint; wait for explicit approval before continuing.
- Each task: one small commit, clear message.
- No backwards-compat shims; pre-prod codebase. Existing anchored proofs without `dataset_inputs` continue to verify (the field is absent, not invalid).
- Match existing code style (`dict | None`, snake_case, minimal comments, comments only when WHY is non-obvious).
- Tests use `tmp_path` and `monkeypatch` for filesystem and environment isolation. No live network calls.

## Decisions locked (open questions resolved in planning)

- **Q1a — File-backend path:** RESOLVED EMPIRICALLY against MLflow 3.11.1. Dataset metadata lives at `mlruns/<exp_id>/datasets/<dataset_id>/meta.yaml`; the run-to-dataset link lives at `mlruns/<exp_id>/<run_id>/inputs/<input_uuid>/meta.yaml`.
- **Q1b — Tamper target file:** the dataset's own `meta.yaml`. Reset is symmetric and clean. Documented limitation: shared across runs that reuse the same dataset_id (irrelevant in demo).
- **Q1c — Tamper target field:** `digest`. Strongest dataset-integrity narrative for the demo.
- **Q1d — Mutation pattern:** in-place truncate, consistent with the existing model.pkl tamper. Atomic replace explicitly declined for consistency; if we want to switch, apply to both tampers together.
- **Q2 — Schema serialization stability:** RESOLVED STRUCTURALLY by routing all hash inputs through `canonical_json()` (RFC-8785 JCS) before hashing. Independent of MLflow's serialization quirks. Locked as a codebase rule.
- **Q3 — Multi-input ordering:** sort by `(name, source, context, digest)`. Bulletproof against name collisions and edge cases. See `_serialize_dataset_inputs` in Task A1.

## Deferred / out of scope (recap)

See "Out of scope for v1" above. None of the deferred items are blockers for the headline value (verifiable dataset → decision chain).
