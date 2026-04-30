# Phase 3 — Demo UX polish (design spec)

**Date:** 2026-04-30
**Author:** Will Kempster (with Claude)
**Branch:** `phase3/demo-ux-polish` (off `main`)
**Ships as:** PR #9

This spec captures the design decisions for Phase 3 of the redesign trilogy
(Phase 1 = plugin redesign, PR #7 merged; Phase 2 = demo migration, PR #8
merged; Phase 3 = demo UX polish, this PR).

The starting brief is Part 7–8 of `~/.claude/plans/conduct-an-analysis-of-zany-kahn.md`.
Phase 3 was originally scoped as four tamper buttons + split proof viewer +
"what gets hashed" viewer + frontend-design polish. The brainstorm refined
that scope significantly — what's below is the canonical Phase 3 design.

Mockups produced during the brainstorm live in
`.superpowers/brainstorm/34033-1777551244/content/`:

- `decision-detail-full-page-v11.html` — canonical decision detail
- `index-page-v1.html` — dashboard / Decisions page
- `run-detail-page-v1.html` — training run detail
- `model-chain-page-v1.html` — model lineage
- `model-registry-page-v1.html` — Models list
- `who-this-is-for-page-v1.html` — personas explainer

These are the visual ground truth. When this spec and the mockups disagree,
the mockups win for visual structure; this spec wins for vocabulary and
behavior.

## 1. Goal

Bring the demo to sales-call polish, with three concrete outcomes:

1. **The verification story is legible.** Every page that shows a
   verification check uses the same three-row structure and the same
   user-language labels (`Proof Found` / `{event} Record Matches` /
   `Signature Confirmed`). A non-technical viewer can read the demo
   top-to-bottom and not encounter unexplained jargon.

2. **Tamper demonstrations are concrete and consistent.** Each tamperable
   page shows two tamper actions that mutate live MLflow data — the
   headline governance tamper. Each tamper breaks exactly one verification
   row, and the system catches it.

3. **The page architecture is shorter on first load.** Two collapsible
   sections (`Click to tamper`, `How verification works`) keep the demo's
   secondary surfaces out of the way until the presenter expands them.

## 2. Scope

### In scope

- **Six template files**, language pass + structure changes:
  - `templates/decision_detail.html`
  - `templates/index.html`
  - `templates/run_detail.html`
  - `templates/model_chain.html`
  - `templates/model_registry.html`
  - `templates/who_this_is_for.html`
- **CSS additions** in `templates/base.html` (and per-template inline
  styles) for: `collapsible`, `tamper-section`, `mini-verify`, split proof
  viewer, audit grid, section labels.
- **New backend endpoints** for the two tamper actions.
- **Removal of legacy panels** (most notably the `Proof Layer` panel on
  `run_detail.html`).
- **Discoverability links** so `run_detail` is reachable from more
  surfaces.
- **README updates** so the documented vocabulary matches the UI.
- **ROADMAP.md update** for the deferred trusted-issuer-key fast-follow.

### Out of scope (deferred to future polish passes or strategic backlog)

- **Trusted-issuer-key check** (would unlock a third tamper:
  *"Use a proof signed by someone else"*). Captured in `ROADMAP.md` under
  *External identity binding* as a Phase-3 fast-follow opportunity.
- **Two additional persona cards** (P4 ML Platform Lead, P6 Compliance
  engineer at frontier lab) for `who_this_is_for.html`. Strategic backlog.
- **Hosted verifier portal**, **continuous verification**,
  **input-side anchoring**, **framework-agnostic core** — all per
  `ROADMAP.md`.
- **Plugin API changes.** Phase 3 is a UI polish pass; the plugin's
  contract (`source_of_truth_ok`, `attestation_level`, `signature_valid`,
  `permanent_copy_found`, `hash_match`) does not change. Only how the UI
  labels and presents the data.

## 3. Vocabulary changes (cross-cutting)

These rename rules apply across all six pages, the README, and the plugin's
CLI output. Internal field names in `ario_mlflow/verify.py` and
`ario_mlflow/cli.py` stay unchanged for API stability — only user-facing
labels update.

### Status badge values (used everywhere)

| Today | Phase 3 |
|---|---|
| `Verified` | `Verified` *(unchanged)* |
| `Anchored` | `Pending verification` |
| `Anchoring` *(in-flight upload)* | `Anchoring` *(unchanged)* |
| `Failed` | `Tampered` |
| `Local Only` | `Not anchored` |
| `Permanent on ar.io` *(Turbo finalized state)* | `Confirmed` |
| `Confirmed on ar.io` *(Turbo confirmed state)* | `Confirmed` *(consolidated)* |
| `Uploading to ar.io…` | `Anchoring` *(consolidated)* |

Note: the Turbo statuses currently distinguish `FINALIZED` / `CONFIRMED` /
`NOT_FOUND` — Phase 3 collapses the user-facing display into two states
(`Confirmed` for both FINALIZED and CONFIRMED, `Anchoring` for NOT_FOUND).
The internal status strings stay; only the rendered labels change.

### Verify-row labels (per event type)

The three-row verify card is the same shape on every page that shows
verification. Row 2's noun is event-type-specific:

| Event type | Row 1 | Row 2 | Row 3 |
|---|---|---|---|
| Decision (prediction) | `Proof Found` | **`Decision Record Matches`** | `Signature Confirmed` |
| Training | `Proof Found` | **`Training Record Matches`** | `Signature Confirmed` |
| Registration | `Proof Found` | **`Registration Record Matches`** | `Signature Confirmed` |

Plus `Attested by` (operator + timestamp) and `Overall: PASS|FAIL`, plus a
collapsible `▶ How to verify independently` block.

### Decision card field renames

Applied on `decision_detail.html`'s top-left card:

| Today | Phase 3 |
|---|---|
| Card header `Prediction` | `Decision` |
| Field `Class` | `Result` |
| Field `Index` | `Confidence` (renders the highest probability as a percent) |
| Result value: `<span class="badge badge-primary">approve</span>` | `<span class="badge badge-green"><span class="badge-dot"></span> Approved</span>` (or `badge-red` + `Denied`) |
| Probability bar labels: `deny` / `approve` | `Deny` / `Approve` (Title Case) |

**Class-name-to-display-string mapping** for the Result chip
(past-tense, since it describes the decision that *was* made):

```python
RESULT_DISPLAY = {
    "approve": ("Approved", "badge-green"),
    "deny":    ("Denied",   "badge-red"),
}
```

The probability bars keep present-tense class-name labels (`Deny` /
`Approve`) since they label the underlying class probabilities, not the
outcome. This mixed usage is intentional.

### Brand: Arweave vs ar.io

User-facing copy: `ar.io` (lowercase brand token). The underlying network
is Arweave; the demo uses ar.io as the gateway/access layer. Where pages
say things like *"anchored to Arweave"*, switch to *"anchored to ar.io"*.

The CSS class `.pill-arweave` (yellow palette in the proof viewer) keeps
its name — it's the network color, used as a pill that displays `ar.io`.
Don't rename the class.

### Other phrase replacements

| Today | Phase 3 |
|---|---|
| *"audit trail"* (model_registry context banner, who_this_is_for) | *"verifiable record"* |
| *"changing the local record directly"* (model_registry banner) | *"tampering with MLflow data directly"* |
| *"Ed25519 signature"* (auditor persona answer) | *"the signature"* (drop algorithm jargon for non-technical reads) |
| `Cryptographically verified · Level N` (attestation badge values) | `Verified` (drop level numbers from user-facing copy; internal `attestation_level` field stays) |

## 4. Per-page designs

### 4.1. `decision_detail.html`

**Reference mockup:** `decision-detail-full-page-v11.html`

**Page header:**
- Title: `Decision Record`
- Page id: full `decision_id` in mono
- Actions: `Verify with ar.io` (existing primary button) and **NEW**
  `View Proof ↗` (outline button) → `https://turbo-gateway.com/{tx_id}`,
  `target="_blank"`. Hidden when `envelope.arweave_tx_id` is unset (same
  conditional pattern as Verify).

**Top row — 2-col grid:**

- **Decision** card (left, replaces `Prediction`):
  - Headline row: `Result` → green/red badge (`✓ Approved` / `✗ Denied`)
    using standard `.badge.badge-green` / `.badge.badge-red`.
  - `Confidence` row: `{highest_probability * 100}%`
  - `Features` row: comma-separated feature names (existing)
  - "Class probabilities" sub-section: prob bars with Title Case labels
    (`Deny` / `Approve`).

- **ar.io Verification** card (right):
  - Header: `ar.io Verification` (section-header CSS uppercase will render
    as `AR.IO VERIFICATION`; this is the established brand presentation
    inside uppercase contexts).
  - Intro: *"Independent verification of MLflow data integrity. These
    verify this decision record matches the original proof anchored with
    ar.io at runtime — they do **not** speak to whether the decision
    itself was correct."*
  - Three verify-rows: `Proof Found`, `Decision Record Matches`,
    `Signature Confirmed`. Values are `PASS`/`FAIL`/`Pending`/`Not checked`
    using the existing `.check` / `.cross` / `.badge.badge-yellow` /
    `.unchecked` styles.
  - Tooltips per row, written for a non-technical reader with a
    `FAIL means…` follow-on. Exact copy:
    - **Proof Found:** *"The proof exists on Arweave and ar.io can locate
      it. FAIL means the on-chain anchor is missing or unreachable."*
    - **Decision Record Matches:** *"The data in MLflow (both the saved
      canonical bytes and the live state) hashes to the same value that
      was anchored on Arweave. ar.io independently re-verifies the bytes.
      FAIL means MLflow data has been tampered with since anchoring."*
    - **Signature Confirmed:** *"The proof carries a valid signature from
      the user who issued it. ar.io independently re-verifies the
      signature. FAIL means the proof was altered after signing."*
  - `Attested by` row: operator name (e.g. `vilenarios.com`) +
    *"independent ar.io operator"* + timestamp.
  - `Overall` row (with top border): `PASS` / `FAIL` / `Pending`.
  - Collapsible `▶ How to verify independently` (existing pattern, copy
    updated to drop level numbers and reference the three checks).

**Below the top grid — 2x2 audit grid (always visible):**

The four cards are: **Model**, **Inference**, **Trace**, **ar.io Anchor**
in that reading order (left-to-right, top-to-bottom). All values render
in full — **no truncation**.

| Card | Fields |
|---|---|
| Model | Name, Version, Run ID *(linkable to `/ui/runs/{run_id}`)*, Lineage *(linkable to model_chain)* |
| Inference | Timestamp, Latency, Input Hash, Output Hash |
| Trace | Service, Trace ID, Span ID *(no `OTel: Connected` row — that was filler)* |
| ar.io Anchor | Transaction, Status *(`Confirmed` badge)*, Signer Key, Block |

**Below the audit grid — `Demonstrate Tampering` section label, then the
collapsible:**

- Section: `<details class="tamper-section">` (closed by default)
- Summary: `<summary>` containing `<div class="tamper-header-inner">` with
  chevron icon + title `Click to tamper` + hint `simulate two attacks`.
  (The wrapping `tamper-header-inner` div is required to avoid the
  `display: flex` summary click-area bug — see § 6.)
- Body intro: *"Two ways someone could try to alter this decision record
  after the fact — both hit MLflow data, both get caught by the same
  verification check. Click to see how. Tampers auto-reverse after a short
  window, or click Reset all."*
- Two tamper rows:
  1. **Tamper with the saved record** — *"Edit the locally-saved copy of
     the decision data. (Overwrites `ario/payload.json` in MLflow.)
     Breaks → Decision Record Matches."*
  2. **Tamper with the live data** ★ — *"Mutate MLflow's underlying
     record — the same thing an admin with registry access could do.
     (Overwrites the `ario.payload_json` trace tag.) Breaks → Decision
     Record Matches."* — with the lavender headline pill `★ Headline
     tamper for governance audiences`.
- Footer: italic note *"Both tampers hit different surfaces but the system
  catches each one — data tampering anywhere in MLflow shows up as the
  same broken check."* + `Reset all` button.

**Below tamper — `How verification works` section label, then the
collapsible:**

- Section: `<details class="collapsible">` (closed by default)
- Summary: `Show the canonical bytes ↔ signed commitment` + hint
  `click to expand`.
- Body intro: *"The proof has two parts. Ar.io anchors a tiny **signed
  commitment** — just hashes and a signature, no source data. MLflow
  stores the **canonical bytes** that produced the hash. Anyone with both
  can verify they match."*
- Two-column proof viewer (`.proof-split`, equal-height panels via flex):
  - **Left panel (Canonical bytes, MLflow):** rendered JSON of the
    canonical bytes. Footer: `ario/predictions/<id>/payload.json` +
    `Download →` link to MLflow artifact.
  - **Right panel (Signed commitment, ar.io):** rendered JSON of the
    signed envelope. The `payload_hash` line gets a strong yellow
    background slab (full-row, bleeding to panel edges, with thick yellow
    border-left). Footer: `TX <full_tx>` + `View on ar.io →` link to
    `https://turbo-gateway.com/{tx_id}`.
  - Pills in panel headers: `MLflow` (blue, `.pill-mlflow`) for
    canonical, `ar.io` (yellow, `.pill-arweave`) for signed. Both render
    uppercase via existing pill CSS.
  - Both panels have aligned footers via `display: flex; flex-direction:
    column; flex: 1; margin-top: auto` on `.panel-footer`.
- Equality bar (full width below the panels): gradient blue → lavender →
  yellow background; reads
  `Verifier recomputes: SHA-256(canonical bytes) = payload_hash ✓ matches "4b2c…e9f1"`.
  When tamper #1 fires and breaks check 2, the green chip flips to red
  *doesn't match*.
- Two-column legend below the equality:
  - *"What you can't do.* Change either side without breaking the
    equality. The signature on the signed commitment binds `payload_hash`
    to the signer's key.*"*
  - *"What this isn't.* Canonical bytes contain hashes of input/output,
    not the values themselves — predictions stay private.*"*

**Sections removed:**

- `Decision Record` panel (decision_id / timestamp / input_hash /
  output_hash / latency) — folded into the Inference audit card.
- `ar.io anchoring` standalone section (TX / Status / Receipt) — folded
  into the ar.io Anchor audit card. Receipt fields surface there as
  Signer Key + Block.
- `Model Lineage` standalone section — folded into the Model audit card.
- `Trace Context` standalone section — folded into the Trace audit card.

**Section label `What was decided` is NOT used** — user explicitly removed
it. Page goes from `page-header` directly to the 2-col grid.

### 4.2. `index.html` (Decisions dashboard)

**Reference mockup:** `index-page-v1.html`

**Page header subtitle:** *"Every decision below carries a signed proof
anchored to ar.io. Anyone can verify the live record matches what was
anchored — without trusting this app."*

**Stats cards (4):**
- `Verified` (green)
- `Pending verification` (yellow, replaces `Anchored`)
- `Tampered` (red, replaces `Failed`)
- `Not anchored` (gray, replaces `Local Only`)

The click-to-filter behavior keeps its existing logic; only labels change.

**Provenance card status badges:**
- `Verified` / `Pending verification` / `Anchoring` / `Not anchored`
  per the cross-cutting status vocabulary.

**Prediction form button:** `Predict & Record` → **`Make a decision`**.

**Records table:**
- Section heading: `Records` → **`Recent decisions`**
- Column renames:
  - `Prediction` → **`Result`** (renders as `● Approved` / `● Denied`
    badge using `.badge.badge-green` / `.badge.badge-red`)
  - `Arweave TX` → **`ar.io Anchor`**
- Status badge values per the cross-cutting vocabulary.
- Filters (version dropdown, date from/to, clear) **stay unchanged** —
  layout and behavior preserved.

### 4.3. `run_detail.html` (Training run detail)

**Reference mockup:** `run-detail-page-v1.html`

**Page header:**
- Title: `Training Run` *(unchanged)*
- Page id: full `run_id` in mono
- Actions: `Verify with ar.io` *(unchanged)* + **NEW** `View Proof ↗`

**Top row — 2-col grid:**

- **Training** card (left, replaces `Parameters`):
  - Headline row: `Status: ● Trained` badge + `Accuracy: 91.3%`
  - Sub-section `Parameters`: Algorithm, Max iter, Random state, Samples
  - Sub-section `Metrics`: Accuracy (and any others)

- **ar.io Verification** card (right): same three-row structure as
  `decision_detail.html`, but row 2 is `Training Record Matches`. Intro
  adapted: *"Independent verification of MLflow data integrity. These
  verify this training run matches the original proof anchored with ar.io
  at training time — they do not speak to whether the model itself is
  good."*

**Below the top grid — 2x2 audit grid:**

| Card | Fields |
|---|---|
| Model | Name, Version, Run ID *(linkable to `/ui/runs/{run_id}` — this page, but the consistent pattern matters)*, Lineage |
| Run | Started timestamp, Duration, Train samples, Test samples |
| Artifact Integrity | Artifact Hash *(full SHA-256)*, per-file checksums (`model.pkl`, `conda.yaml`), git commit *(full hash)* |
| ar.io Anchor | Transaction, Status, Signer Key, Block |

**No `Trace` card** on this page — training runs don't generate OTel
traces in the same way decisions do. (If we ever instrument training with
OTel, we add the card then.)

**Below the audit grid — `Demonstrate Tampering` collapsible:**

Two tampers, both → `Training Record Matches`:

1. **Tamper with the saved record** — Overwrites the training run's
   `ario/payload.json` in MLflow.
2. **Tamper with the live data** ★ — *"Mutate MLflow's underlying
   training run — for example, change a metric or replace the model
   artifact. The same thing an admin with registry access could do.
   (Overwrites a logged metric or rewrites `model.pkl`.)"*

**`How verification works` collapsible:** same split-proof viewer, with
training-specific canonical bytes content (event_type: `training`, params,
metrics, artifact_checksums) on the left and the matching signed
commitment on the right.

**`Live MLflow tags` collapsible** (third section, training-specific):

A new collapsible section showing the live `ario.*` tags pulled from
MLflow via `MlflowClient.get_run()`. Existing functionality wrapped in
the same collapsible pattern. Includes the `mlflow ui --backend-store-uri
./mlruns` snippet for users who want to see the native MLflow UI.

**Sections removed (currently in run_detail.html):**

- `Parameters` standalone section — folded into Training card.
- `Metrics` standalone section — folded into Training card.
- `ar.io anchoring` standalone section — folded into ar.io Anchor audit
  card.
- `Artifact Integrity` standalone section — folded into the audit grid as
  one of the four cards.
- **`Proof Layer` panel** (record_hash / previous_hash / signature /
  public_key) — **deleted entirely**. This content now lives in the
  Signed commitment panel of the `How verification works` viewer.
- `Model` standalone section — folded into Model audit card.
- `ario.* tags (live from MLflow)` always-visible section — wrapped in
  the new `Live MLflow tags` collapsible.

### 4.4. `model_chain.html` (Model lineage)

**Reference mockup:** `model-chain-page-v1.html`

**Page header:**
- Title: `Model Lineage` *(unchanged)*
- Subtitle: *"{model} / v{version} — every training run, model version,
  and prediction in this model's history, each cryptographically
  verifiable on ar.io."* (replaces *"a cryptographically verifiable audit
  trail of every training run, model version, and prediction"*)
- Action: `Verify All` → **`Verify chain`** (clearer scope)

**Chain visualization:** vertical chain of three nodes connected by a thin
border-color line, each with a colored chain-dot indicator.

**Each chain card** uses a 2-col body layout: event details on the left,
**mini verify card** on the right (using `.mini-verify` styles from the
mockup — compact 3-row PASS column, `Attested by` line, `Overall` row).

#### Training Run node

- Header: `Training Run` (h3, linkable to `/ui/runs/{run_id}`) + status
  badge.
- Left half: Run ID *(full)*, Algorithm, Max iter, Random state, Accuracy,
  Artifact Hash *(full SHA-256 with `sha256:` prefix)*, Git Commit *(full
  hash)*.
- Right half (mini-verify): `Proof Found` / `Training Record Matches` /
  `Signature Confirmed` / `Attested by` / `Overall`.
- Footer: TX link *(full Arweave TX)* + `Confirmed` badge.

#### Model Registration node

- Header: `Model Registration` + status badge.
- Left half: Model, Source Run *(full)*, Artifact Hash *(full SHA-256)*,
  Artifact Integrity *(`Matches training` / `Mismatch` badge)*,
  Training TX *(full)*.
- Right half (mini-verify): `Proof Found` / `Registration Record Matches`
  / `Signature Confirmed` / `Attested by` / `Overall`.
- Footer: TX link *(full)* + `Confirmed` badge.

#### Decisions node

- Header: `Decisions` + record count badge.
- Body (single column): row labels using new vocabulary:
  - `Total: N`
  - `Verified: V/N`
  - `Pending verification: P/N` (was `Anchored`)
  - `Tampered: T/N` (new — surfaces the tamper count, was implicit)
- Footer: `View all decisions →` link to `/ui/decisions`.

**Below the chain — `Demonstrate Tampering` collapsible:**

Two **live-data tampers** (matches the headline-tamper framing on the
other pages — see § 5 for the rationale):

1. **Tamper the training run's live data** ★ — *"Mutate the training
   run's metrics or artifacts in MLflow — for example, change the
   recorded accuracy from 91.3% to 99.9% to make the model look better
   than it was. (Overwrites a logged metric or rewrites `model.pkl` in
   MLflow's run store.)"* → Breaks `Training Record Matches`.

2. **Tamper the registration's live data** — *"Mutate the model version's
   metadata in MLflow — for example, point this v3 registration at a
   different training run, claiming a different model produced it.
   (Overwrites the model version's `source_run_id` tag.)"* → Breaks
   `Registration Record Matches`.

Footer note: *"Each tamper changes the live MLflow data at a different
link in the chain. The verifier re-derives canonical bytes from the
current state and compares to what was anchored — any divergence breaks
the matching record's check, and the chain breaks at exactly the link
that was attacked."*

**`How verification works` collapsible:** lighter version on this page —
explains the chain pattern (`previous_hash` links each event) and points
to the detail pages for the full split-proof viewer. Avoids three full
viewers stacked.

### 4.5. `model_registry.html` (Models list)

**Reference mockup:** `model-registry-page-v1.html`

**Page header:** unchanged.

**Context banner copy:**
- Heading unchanged: *"How would you prove, 18 months from now, which
  model made a specific decision?"*
- Body rewrite: *"Train a model below, make a prediction, then try to
  break the verifiable record — either by tampering with MLflow data
  directly, or by imagining the runs get deleted, models get silently
  retrained, or institutional memory is lost. Because every proof is
  anchored to ar.io, a third party can verify what actually happened
  without trusting this app or any records it stores locally."*

**Hero training section subtitle:**
*"Train, register, and anchor provenance to ar.io"* (was Arweave).

**Train progress steps:**
- `Anchoring training proof to Arweave...` → `Anchoring training proof to ar.io...`
- `Anchoring registration proof to Arweave...` → `Anchoring registration proof to ar.io...`

**Result text:** *"…anchored to Arweave"* → *"…anchored to ar.io"*.

**Versions table:**
- Run ID column: **show full 32-char run ID**, no truncation. Wrap via
  `word-break: break-all`. Smaller mono font (`font-size: 0.74rem`) so
  the value fits without dominating the row.
- **Run ID renders as `<a class="mono" href="/ui/runs/{run_id}">…</a>`** —
  primary-color, underline-on-hover, links directly to the run detail
  page.
- Training/Registration status badges per cross-cutting vocabulary.

### 4.6. `who_this_is_for.html` (personas explainer)

**Reference mockup:** `who-this-is-for-page-v1.html`

**Layout:** 4-column persona grid (was 2x2). Container `max-width` raises
to `1500px` so cards fit comfortably side-by-side. Responsive breakpoints:
2-col at 1200px, 1-col under 700px.

**Wording updates** (light touch):

| Location | Today | Phase 3 |
|---|---|---|
| Intro paragraph | *"…inference to Arweave via ar.io."* | *"…inference to ar.io."* |
| ML engineer answer | *"every run mints a signed proof on Arweave."* | *"every run mints a signed proof anchored to ar.io."* |
| Auditor answer | *"check the Ed25519 signature, fetch the permanent copy from Arweave."* | *"verify the signature, fetch the permanent copy from ar.io."* |
| OS publisher answer | *"SHA-256 artifact hash + Ed25519 signature anchored on Arweave"* | *"SHA-256 artifact hash + signed commitment anchored to ar.io"* |
| Disclaimer | *(unchanged)* | *(unchanged)* |

## 5. Tamper feature — design and backend

### Tamper buttons (UI)

Three pages get a `Demonstrate Tampering` section: `decision_detail.html`,
`run_detail.html`, `model_chain.html`. Each section is a `<details
class="tamper-section">` collapsible, closed by default.

**Tamper count per page (and which gets the ★ headline pill):**

| Page | Tamper #1 | Tamper #2 | Headline pill on |
|---|---|---|---|
| `decision_detail.html` | Tamper with the saved record | Tamper with the live data | **#2** (live data — governance fear) |
| `run_detail.html` | Tamper with the saved record | Tamper with the live data | **#2** (live data) |
| `model_chain.html` | Tamper the training run's live data | Tamper the registration's live data | **#1** (training live data — metrics-fudging is the most viscerally familiar governance scenario) |

Both tampers on a page break the same verification row (Decision Record
Matches / Training Record Matches), except on `model_chain.html` where
each tamper breaks the row on its corresponding chain node (training →
Training Record Matches; registration → Registration Record Matches).

### Tamper mechanism (backend)

Two new endpoints in `app/main.py`:

- **`POST /tamper/saved/{event_type}/{event_id}`** — overwrites the
  `ario/payload.json` artifact for the specified event in MLflow with
  garbage bytes. Returns `{"tampered": true, "kind": "saved", "expires_at":
  "..."}`.
- **`POST /tamper/live/{event_type}/{event_id}`** — mutates a live MLflow
  field. Behavior per event type:
  - `decision`: overwrites the trace's `ario.payload_json` tag.
  - `training`: overwrites a logged metric (`accuracy` from real value
    to `0.999`).
  - `registration`: overwrites the model version's `source_run_id` tag
    to a fake value.
- **`POST /tamper/reset/{event_type}/{event_id}`** — restores both saved
  and live state. Reset all = call this for the page's events.

**Auto-reverse:** each tamper endpoint schedules a background task
(`asyncio.create_task` or `BackgroundTasks`) that calls the reset
internally after `TAMPER_TTL_SECONDS` (env var, default 60 seconds, but
the UI never names the number — it says *"after a short window"*).

**Tamper state persistence:** the tampers should mutate real MLflow state
(via `MlflowClient.set_tag`, file-write to artifact store, etc.) so that
re-running `Verify with ar.io` actually catches the tamper. This is the
demo's whole point — the verification check needs to fail organically,
not via a UI flag.

**Pre-tamper backup:** before each tamper, snapshot the original value to
in-memory state keyed by event_id. Reset writes the original back. If the
process restarts mid-tamper, the auto-reverse fails silently — acceptable
for a demo.

### Tamper UI flow

1. User clicks the `Tamper` button → POST to the corresponding endpoint.
2. Button shows loading state (`data-loading-text` pattern, existing).
3. On success, the tamper card highlights (e.g., `tampered` class with a
   pulsing yellow border) so the user can see something happened.
4. User clicks `Verify with ar.io` (the existing button) → re-runs
   verification → corresponding row turns red.
5. After 60s (default), background task auto-reverts. UI doesn't refresh
   automatically — user has to re-verify or refresh.
6. `Reset all` button on the tamper card POSTs a reset to all events on
   the page.

**Removed code:** the legacy `POST /tamper/{decision_id}` endpoint
(commented as removed in Phase 2.C — confirm it's gone from `app/main.py`)
and any `/api/chain-integrity` artifacts.

## 6. CSS additions

These styles need to land in `templates/base.html` (or as shared classes
near the top of each template). Names match what's in the mockups so the
mockups can be lifted directly.

### Section labels (page-narrative spine)

Used on `decision_detail.html`, `run_detail.html`, `model_chain.html` to
delimit the demo-specific sections.

```css
.section-label {
  font-family: 'Besley', serif;
  font-weight: 800;
  font-size: 1rem;
  color: var(--black);
  margin: 2.25rem 0 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.section-label::before {
  content: "";
  height: 1px;
  background: var(--border);
  width: 1.5rem;
}
.section-label::after {
  content: "";
  height: 1px;
  background: var(--border);
  flex: 1;
}
```

### Generic collapsible (used for "How verification works", "Live MLflow tags")

```css
.collapsible { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; transition: border-color 0.2s; }
.collapsible[open] { border-color: var(--lavender); }
.collapsible summary {
  padding: 0.85rem 1rem;
  background: var(--surface);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  transition: background 0.2s;
}
.collapsible[open] summary { background: var(--lavender); border-bottom: 1px solid var(--lavender); }
/* Chevron icon — 90deg rotate when open */
.collapsible-summary-icon { width: 20px; height: 20px; border-radius: 50%; background: var(--white); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; transition: transform 0.2s, border-color 0.2s; flex-shrink: 0; color: var(--primary); }
.collapsible[open] .collapsible-summary-icon { transform: rotate(90deg); border-color: var(--primary); }
.collapsible-summary-title { font-size: 0.85rem; font-weight: 700; color: var(--black); }
.collapsible[open] .collapsible-summary-title { color: var(--primary); }
.collapsible-summary-hint { font-size: 0.78rem; color: var(--muted); margin-left: auto; padding-right: 0.2rem; }
.collapsible[open] .collapsible-summary-hint { display: none; }
.collapsible-body { padding: 1.25rem; background: white; }

/* Globally hide the default disclosure marker */
summary { cursor: pointer; }
summary::-webkit-details-marker { display: none; }
summary::marker { content: ""; }
```

### Tamper section (yellow theme variant of collapsible)

```css
.tamper-section { border: 1px solid var(--yellow); border-radius: 8px; overflow: hidden; }
.tamper-header {
  padding: 0.75rem 1rem;
  background: var(--yellow-bg);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--yellow);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  cursor: pointer;
  list-style: none;     /* remove implicit list-marker space */
}
/* IMPORTANT: do NOT use display: flex on the summary itself.
   Wrap flex content in `.tamper-header-inner` to keep the entire
   padded box clickable. See § 6.1 below. */
.tamper-header-inner { display: flex; align-items: center; gap: 0.6rem; }
.tamper-section[open] .tamper-header { border-bottom: 1px solid var(--yellow); }
.tamper-summary-icon { /* yellow variant of collapsible-summary-icon */ }
.tamper-section[open] .tamper-summary-icon { transform: rotate(90deg); }
.tamper-summary-title { flex: 1; }
.tamper-summary-hint { font-size: 0.7rem; font-weight: 400; text-transform: none; letter-spacing: normal; color: var(--yellow); }
.tamper-section[open] .tamper-summary-hint { display: none; }
.tamper-body { padding: 1rem; background: var(--white); }
/* tamper-row, btn-tamper, tamper-headline, tamper-footer styles per mockup */
```

#### 6.1. The summary click-area gotcha

`<summary>` with `display: flex` has a known quirk in some browsers: the
clickable region collapses to the inline content area, leaving most of
the padded box unclickable. The fix is to keep `<summary>` block-level
(no `display: flex`) and put the flex layout in a child `<div>`. See the
`.tamper-header-inner` pattern. The `.collapsible summary` rule uses
`display: flex` directly and works in our testing — keep it as-is, but
if click-area issues surface, apply the same wrapper pattern.

### Audit grid (always-visible 2x2 below verify card)

```css
.grid-2-audit { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 700px) { .grid-2-audit { grid-template-columns: 1fr; } }
```

Cards inside use the existing `.section`/`.section-header`/`.section-body`
styles — no new card chrome.

### Mini verify card (used in `model_chain.html` chain nodes)

```css
.mini-verify {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.55rem 0.75rem;
}
.mini-verify-row { display: flex; justify-content: space-between; align-items: center; padding: 0.32rem 0; font-size: 0.78rem; }
.mini-verify-row + .mini-verify-row { border-top: 1px solid var(--border); }
.mini-verify-meta { font-size: 0.7rem; color: var(--muted); padding-top: 0.45rem; margin-top: 0.3rem; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 0.15rem; }
```

### Split proof viewer

`.proof-split` — 2-col grid; `.panel` — flex column with header/body/footer
(footer pinned via `margin-top: auto`); `.panel-source-pill` —
`.pill-arweave` (yellow) and `.pill-mlflow` (blue); `.payload-hash-line` —
yellow background slab with thick yellow border-left, bleeds to panel
edges via negative margin; `.equality` — full-width gradient bar.

Full styles per `decision-detail-full-page-v11.html`.

### Result chip (decision card)

No new CSS class — use the existing `.badge.badge-green` (with new
`.badge-red` variant for `Denied`):

```css
.badge-red { background: var(--red-bg); color: var(--red); }
.badge-red .badge-dot { background: var(--red); }
```

## 7. Discoverability fixes

`run_detail.html` is currently only reachable from `model_chain.html`'s
training-node header. Phase 3 adds entry points:

| Page | Change |
|---|---|
| `model_registry.html` (versions table) | Run ID column wrapped in `<a class="mono" href="/ui/runs/{run_id}">` |
| `decision_detail.html` (Model audit card) | Run ID row's value wrapped in `<a class="mono" href="/ui/runs/{run_id}">` |
| `run_detail.html` (Model audit card) | Run ID row's value wrapped (links to itself, but consistent pattern) |
| `model_chain.html` (existing) | Already links — no change |

## 8. Files removed in Phase 3

- **`templates/decision_detail.html`** — `Decision Record`, `ar.io
  anchoring`, `Model Lineage`, `Trace Context` standalone sections
  (folded into the audit grid).
- **`templates/run_detail.html`** — `Parameters`, `Metrics`, `ar.io
  anchoring`, `Artifact Integrity`, **`Proof Layer`**, `Model`, and the
  always-visible `ario.* tags` sections (folded or wrapped in
  collapsibles).
- **`app/main.py`** — confirm legacy `POST /tamper/{decision_id}` and
  `/api/chain-integrity` endpoints are fully removed (Phase 2 commented
  them as removed; double-check).
- **`/api/export/{decision_id}`** — replaced by direct ar.io gateway link
  in `View Proof ↗`. Endpoint can be deleted; check for any other
  callers first.

## 9. Validation

### Automated tests

- `pytest` passes including any new tests for tamper endpoints.
- Add unit tests for the tamper backend functions: assert that after a
  tamper, the corresponding plugin verification check actually returns
  `False`; assert reset restores correctly.
- Smoke test: boot demo, train, predict, verify, tamper each tamper
  type, verify, confirm row fails, reset, verify, confirm row passes.

### Manual walkthrough (sales-call resolution)

A non-technical viewer can follow the demo top to bottom:

1. Land on `/` → see the dashboard with the new vocabulary.
2. Click into a decision → see the Decision card, ar.io Verification card,
   audit grid.
3. Expand `Click to tamper` → click `Tamper with the live data` →
   re-verify → see `Decision Record Matches` flip to FAIL → click
   `Reset all` → re-verify → flip back to PASS.
4. Expand `How verification works` → see the canonical bytes ↔ signed
   commitment side-by-side, the `payload_hash` slab highlighted, the
   equality bar at the bottom.
5. Click `View Proof ↗` → opens turbo-gateway.com with the raw envelope.
6. Navigate to `/ui/models/credit-decision-model/3` → see the chain
   visualization with three nodes.
7. Expand the chain page's tamper section → run the live-data tampers →
   see the corresponding chain node break.
8. Navigate from the chain page's training-node header to
   `/ui/runs/{run_id}` → see the same patterns applied to a training
   event.

### Frontend-design polish bar

The original Phase 3 plan called for using the `frontend-design` skill.
The brainstorm has already produced detailed mockups; the implementation
should match the mockups visually. Any pixel-level polish that comes from
running `frontend-design` on the live page should be folded into Phase 3
before merge.

### Manual review checkpoint (required, per `feedback_per_phase_manual_review.md`)

After validation gates pass, hand off to the user with:

- Side-by-side screenshots/recording of every UI surface that changed
  (Decision detail, Records dashboard, Run detail, Model lineage, Models,
  Who-this-is-for) — both default state and with each tamper triggered.
- Confirmation that the implementation matches the mockups visually.
- Recorded walkthrough demonstrating each tamper triggering its
  corresponding row failure and reset restoring it.
- Any deviations from this spec, surfaced explicitly.

User reviews and approves before PR is opened. **Do not ship without
explicit user approval at this gate.**

## 10. Out of scope / roadmap

- **Trusted-issuer-key check** unlocking *"Use a proof signed by someone
  else"* tamper. Captured in `ROADMAP.md` under *External identity
  binding*. Estimated effort: 10–20 lines in `app/ui.py::_verify_envelope`
  to compare `envelope["public_key"]` against an env-configured expected
  key. Doesn't change the plugin's contract.
- **Persona cards for P4 (ML Platform Lead) and P6 (Compliance engineer
  at frontier lab)** on `who_this_is_for.html`. Strategic backlog.
- **Hosted verifier portal**, **continuous verification service**,
  **input-side anchoring**, **framework-agnostic core**, **HuggingFace
  plugin**, **economic tiers / Merkle batching**, **coverage dashboard**.
  All in `ROADMAP.md`.

## 11. Cascading documentation updates

- **`README.md`** at repo root — every reference to the verification
  flow uses the new vocabulary (`Proof Found` / `Decision Record Matches`
  / `Signature Confirmed`); attestation level numbers dropped from
  user-facing copy; status badges renamed.
- **`ario_mlflow/README.md`** — same vocabulary sweep. Internal CLI
  output (`ario-mlflow verify run <run_id>`, `ario-mlflow verify trace
  <trace_id>`) should print the new labels.
- **`AGENTS.md`** — no changes needed; it's at a different abstraction
  level.
- **Tag-table sections of READMEs** — the `Runs` row's listing of which
  tags `anchor()` writes vs which the verify CLI writes was already
  cleaned up in Phase 2 PR feedback rounds; verify the language is still
  consistent.

## 12. Implementation order suggestion

Not prescriptive — but a reasonable execution order to keep the demo
functional throughout:

1. **CSS additions** in `base.html` (new utility classes — low risk,
   doesn't break anything until used).
2. **Vocabulary sweep** across all six templates and READMEs (no
   structural changes, just label renames). Test: page renders, status
   badges show new labels.
3. **`decision_detail.html` restructure** — apply the new layout
   (Decision card, audit grid, collapsibles). Test: page renders, tamper
   buttons present (but inactive, no backend yet).
4. **Tamper backend endpoints** (`/tamper/saved/...`, `/tamper/live/...`,
   `/tamper/reset/...`) with auto-reverse. Test: each tamper actually
   mutates MLflow state, verify catches it, reset restores.
5. **`run_detail.html` restructure** — apply the same patterns. Test:
   training tampers work end-to-end.
6. **`model_chain.html` restructure** — mini verify cards, chain-page
   tampers (training live, registration live). Test: chain tampers work.
7. **`index.html`, `model_registry.html`, `who_this_is_for.html`
   wording-only sweeps + discoverability links** (Run ID linkability).
8. **README + plugin docs** vocabulary sweep.
9. **Frontend-design polish pass** on the running app (per the original
   Phase 3 plan).
10. **Validation + manual walkthrough.**
11. **Manual review checkpoint with the user.**
12. **Open PR #9.**
