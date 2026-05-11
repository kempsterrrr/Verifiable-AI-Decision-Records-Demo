# Roadmap

This document tracks what we're building next and what we've explicitly parked. It's the team-facing mirror of the strategic backlog we keep in internal notes.

## Where we are today

The repo contains two parts: **`ario_mlflow/`** is the product (a standalone MLflow plugin), and **`app/`** is a sales-facing demo of what the plugin enables.

We're at an **early-stage proof of a verifiable-provenance pipeline** — not yet a production compliance product. Core primitives (hash, sign, anchor to Arweave via ar.io, verify) are in place and end-to-end working. The plugin installs cleanly, the demo runs on Railway with proper persistence, and 104 tests pass.

Recent work:

- **Plugin redesign — Phase 1** (PR #7, merged 2026-04-28): pure-commitment ~500-byte envelopes, canonical bytes preserved as `ario/payload.json` in MLflow, four-check verification flow, RFC-8785 (JCS) canonicalization, conservative Arweave tag policy. The plugin's three integration points (`anchor()`, `ArioMlflowClient`, `VerifiedModel`) are stable.
- **Demo migration to plugin's headline API** (PR #8, merged 2026-04-30): the demo dropped its hand-rolled lifecycle/decision/verify code and now wraps the plugin as a thin presentation layer. The demo is no longer ahead of or duplicating plugin behavior.
- **Demo UX polish** (PR #9, merged 2026-04-30): three-row verification cards, tamper backend with snapshot/TTL/auto-revert, split proof viewer, ar.io vocabulary sweep across UI + CLI + docs.
- **Reset feature for sales workflow** (PR #10, merged 2026-05-05): one-click `/demo/admin` page that wipes all demo state and auto-trains a fresh v1.
- **Production persistence config** (2026-05-05): Railway volume at `/app/persistent` covers all demo state; `VAIDR_*` env vars route every path there. See `docs/deployment.md`.
- **Verification correctness Piece B + tamper audit** (PR #12, merged 2026-05-06): `verify_record` (auditor-shaped primitive) and `verify_proof_by_tx` (operator-side wrapper). Demo's "Proof Found" row now uses an explicit `proof_found` boolean. End-to-end regression test for every shipped tamper button (1-of-6 found silently broken in MLflow 3.x, fixed). Tamper / reset auto-trigger re-verify so the rows reflect current state.
- **Input-side anchoring v1 + standalone dataset anchoring** (PR #13, merged 2026-05-08): training proofs now commit to dataset references, with each `mlflow.log_input()` dataset auto-anchored as its own first-class signed event with its own Arweave TX. `anchor(dataset=ds)` is the standalone publisher path; `anchor()` inside training auto-anchors each input. Dataset's identity (name, source, digest, schema fingerprint) lives in the canonical training payload AND as its own dataset event. Schema is hashed (privacy: column names never enter the proof). Strict-by-default — `anchor()` raises if the run has no logged dataset inputs, with `allow_empty_dataset_inputs=True` escape hatch.
- **Demo UX redesign + Phase E persisted display JSON** (PR #14, merged 2026-05-11): nav restructured to walk the verification chain left to right (Datasets → Training Runs → Models → Decisions → Lineage); two-column editorial detail layout shared via Jinja partials (`_verify_card`, `_status_badge`, `_active_badge`, `_proof_viewer`); 5-state canonical status enum; tamper relocated to a dedicated `/demo/tamper` page (one button per chain link); always-on anchored-proof viewer showing canonical MLflow bytes side-by-side with the signed ar.io commitment; `canonical_bytes_json` + `signed_commitment_json` persisted on each envelope at anchor time so the viewer renders without a gateway round-trip; focused-chain Lineage page with chip picker. Demo-only — no plugin behaviour changed.

Active work in `docs/plans/active/`. What follows is what we chose not to build yet, grouped by theme.

## Near-term focus

When we come back to this project, the highest-leverage next moves are:

1. **Dataset SoT via deeper MLflow dataset registry integration** *(fast follow — closes the v1 deferral on standalone dataset anchoring)*. Today the dataset event is verified via signature + ar.io attestation. The dataset's own "Record Matches" row is deferred — post-anchor mutation of dataset metadata is currently caught one link down (training's source-of-truth re-derives the inlined dataset identity and flips on tamper), but the dataset proof itself doesn't have a live re-derivation check. v2 wires `_refetch_dataset_live_fields` into `verify.py`, looks up the live dataset state via MLflow's dataset registry API, and re-derives the canonical bytes the same way `_anchor_dataset_event` produced them. The deeper goal is a clean tie-in to MLflow's dataset registry — the plugin's dataset events become first-class citizens alongside MLflow's own dataset entries (dataset_id stored on the proof for direct lookup, anchor TX surfaced as registry metadata, verification path uses MLflow's API rather than disk reads). Foundation for the cross-run dedup + dataset-versioning items in the backlog.
2. **Receipts vs. attestation as a two-stage verify UX** — the plugin currently treats ar.io Verify as a pass/fail check, but the underlying reality is two separate things: (a) the **Turbo receipt** comes back synchronously at anchor time and proves the upload was received; (b) the **ar.io Verify attestation level** matures asynchronously over hours/days as the network settles and gateways pick it up. The verify UX should surface both: receipt as a synchronous "did it land" check, attestation level as an explicit maturity gradient (Level 1 → 2 → 3) with appropriate context. Phase 1 ships a configurable threshold (default Level 2) as an interim — this is the cleaner refactor that follows.
3. **External identity binding** — sign with something linkable to a real organization, not an auto-generated key at `~/.ario-mlflow/keys/`. Even a registered-public-keys directory is a step up. Plan paused at `docs/plans/active/2026-05-05-verification-correctness-piece-b-c.md` (Piece C, Tasks 4-8): trusted-issuer-key check threaded through the plugin's `verify_signature`, with a CLI flag, demo env var, and a third tamper button — *"Use a proof signed by someone else"* — that demonstrates the check catching forged proofs. Single-key for v1; multi-key trusted lists are a follow-up. **Status: unblocked** — paused pending the input-side anchoring work, which has now shipped (PR #13).
4. **Continuous verification** — a background job that re-checks every anchored proof on a cadence and alerts when attestation doesn't land. Today "anchored" can silently mean "we called an API that may or may not have worked three weeks ago."

Each of these would be its own branch + plan. The plugin's current API surface doesn't need to change to accommodate any of them.

## The full backlog

Grouped by theme. Each item notes which audience (see *Audiences* below) it most directly serves.

### Provenance depth

| Item | What | Why | Serves |
|---|---|---|---|
| Dataset SoT + deeper MLflow dataset registry integration | Wire `_refetch_dataset_live_fields` into `verify.py` so the dataset event's own Record Matches check fires on tamper. Tie standalone dataset proofs into MLflow's dataset registry as first-class citizens — store the dataset_id on the dataset proof for direct lookup, surface the anchor TX as registry metadata, use the registry API rather than disk reads at verify time. | v1 ships dataset events with signature + ar.io attestation only; the live re-derivation against MLflow's dataset registry is deferred. Closing this gap is the headline v2 follow-up — it makes each dataset proof independently integrity-verifiable end-to-end and turns the plugin's dataset events into proper MLflow registry citizens. Foundation for cross-run dedup + dataset versioning chain. | P1, P2, P3, P6 |
| Cross-run dataset reuse / dedup | Avoid re-anchoring the same dataset across multiple training runs. A registry of pre-anchored datasets (digest → TX) so subsequent training runs reuse the existing dataset proof rather than producing a fresh one. | Today every training run that references the same dataset re-anchors it (one dataset event per training run, even if the dataset hasn't changed). Real but a v3-shape optimization, not a correctness issue. Comes after the dataset SoT work above lands. | P2, P4 |
| Dataset versioning chain | Dataset events chain via `previous_hash` to the previous version of the same dataset name. Mirrors how training proofs chain to the previous training of the same registered model. | Useful "dataset v1 → v2 → v3" provenance. Adds chain-head bookkeeping that doesn't exist for standalone datasets today. | P2, P3 |
| Container image digest | Anchor `$IMAGE_DIGEST` (or equivalent) for the runtime environment that ran training. Caller-supplied for v2; auto-detection helpers later. | Closes the third leg of the input-side trio (dataset + source code + runtime). Deferred from input-side anchoring v1 because it's environment-specific and hard to demo without a real production CI/CD context. | P2, P6 |
| Model-to-usage binding | Verify that the model serving production traffic matches the one anchored in the registry | Prevents registry/serving drift after deployment. `VerifiedModel` handles this at load time; production serving layers don't yet. | P1, P2 |
| Registration-hash enforcement | Block model promotion when training→registration artifact hashes don't match | Today the mismatch is tagged but doesn't block. Needs to be configurable (fail-closed with an escape hatch) because enabling it changes behavior. | P2 |

### Identity

| Item | What | Why | Serves |
|---|---|---|---|
| External identity binding | OIDC / KMS / registered public keys instead of auto-generated local keys | "Signed by Ed25519 key `0xabc…`" means nothing to a regulator. Signatures need to bind to a real organizational identity before the compliance story lands. | P2, P3, P6 |

### Verifier surface

| Item | What | Why | Serves |
|---|---|---|---|
| Hosted verifier portal | A no-install site where anyone can paste a decision_id or TX ID and get a certified report | Auditors need to verify claims about models without being handed client credentials or a Python environment. | P3 |
| Continuous verification service | Background re-verification of every anchored proof + alerts when Level 3 doesn't land within *N* hours | Today verification is opt-in and almost never runs. Without continuous verification, "anchored" and "maybe anchored" are indistinguishable after the fact. | All |
| Certified report format | Auditor-consumable PDF + raw-evidence bundle, in the shape of a Big-4 audit opinion | Auditors work in a world of signed, templated reports. Our current HTML is for humans, not for records retention. | P3, P6 |
| Auditor self-verify section | Expandable block on the decision-detail page showing the exact `curl` + `openssl` commands to independently verify this record | Cheap win — lets auditors convince themselves without trusting our app. | P3, P6 |

### Framework breadth

| Item | What | Why | Serves |
|---|---|---|---|
| Framework-agnostic core | Extract the proof engine + anchor client + verifier from MLflow-specific code. Ship thin adapters for SageMaker, Vertex, Databricks, Kubeflow, plus a framework-free `ario-cli`. | MLflow is a narrow slice of the market. The core primitive is framework-independent and should live that way. | P4 |
| `huggingface-cli` plugin + public registry | A drop-in for HF model publishers plus a lookup service ("here's the anchored hash for this model card") | Virality. Every HF model card with an "Anchored" badge is marketing. | P5 |

### Economics

| Item | What | Why | Serves |
|---|---|---|---|
| Economic tiers — sampled / Merkle-batched inference anchoring | Budgeted anchoring, statistical sampling, or Merkle-batching so high-volume inference doesn't blow the Turbo bill | Per-inference anchoring at scale is real Arweave spend. Enterprise adopters will do the math. | P2 |
| Coverage dashboard | Governance-level view: "73% of production decisions in the last 30 days are fully auditable, Level 3 attested, within policy." | Governance leads need a single number they can quote to their CRO. | P2 |

### Demo maturation

| Item | What | Why | Serves |
|---|---|---|---|
| MLflow UI mount alongside FastAPI on Railway | A reverse-proxied MLflow UI on the same Railway service at `/mlflow/*` so technical evaluators see the native MLflow UI | The current implementation shows the live MLflow tags on the run detail page, which covers the "prove it's not a facade" signal. Running an actual MLflow server alongside uvicorn is a separate infra lift (subprocess management, `--static-prefix`, proxying). | P1, P4 |

## Audiences

Four audiences we've written about. The demo includes a "Who this is for" page that names each one.

- **P1 — ML engineer in a regulated industry.** "Which model made this decision 14 months ago?"
- **P2 — Head of AI governance.** "What fraction of our AI decisions are actually auditable?"
- **P3 — External ML auditor.** "Is the model they evaluated the model they deployed?"
- **P5 — Open-source model publisher.** "How do I prove the weights match my paper?"

Two further audiences appear in the strategic notes but are not yet addressed by the demo:

- **P4 — ML platform lead at a large enterprise.** Multiple stacks, regulator coming, needs framework-agnostic primitives.
- **P6 — Compliance engineer at a frontier lab.** Preparing for third-party safety audits; needs publish-commitments mode.

## What we're explicitly **not** doing

Listed so nobody's confused about why these aren't on the roadmap:

- **Pivoting to a single persona.** The current phase deliberately broadened the demo for all four visible audiences. Picking one vertical becomes the next strategic question only if the current shape fails to get traction.
- **Building an MLflow competitor.** We plug into MLflow, not replace it.
- **Storing model binaries on Arweave.** We anchor hashes. Storing models themselves is prohibitively expensive and solves the wrong problem.
- **Proprietary verification tooling.** Verification is always reducible to standard-library `SHA-256` and `openssl ed25519`. No "call our API to verify" lock-in.

## When this document changes

Update `ROADMAP.md` and the mirrored internal notes whenever:

- An item moves from backlog into an active branch.
- A new item gets added (either from external input or from a post-work critique like the one that produced this list).
- An item is explicitly dropped (archive it at the bottom of this doc with the reason, don't silently delete).

Archive section is below — empty for now.

## Archive

### Demo persistence — done 2026-05-05
Originally listed as "use a Railway volume or small Postgres so `data/records.json` survives restarts." Solved by: Railway volume `verifiable-ai-demo-volume` mounted at `/app/persistent`; all `VAIDR_*` path env vars route demo state (mlruns, records, lifecycle, keys) to the volume; `/demo/admin` Reset button wipes on demand for the sales workflow. See `docs/deployment.md`.

### Vocabulary pass — done 2026-04-22
"Chain of custody" replaced by "Model lineage" (primary) with "audit trail" as a secondary phrase for GRC-leaning readers. Informed by a desk audit of ML (MLflow, W&B, Databricks, ClearML), GRC (Vanta, Drata, OneTrust), audit (Splunk, Datadog, CloudTrail), and AI-governance (Brundage et al., METR, Apollo) vocabulary: no surveyed tool uses "chain of custody" as a primary label; "lineage" dominates ML, "audit trail" is the cross-audience workhorse.
