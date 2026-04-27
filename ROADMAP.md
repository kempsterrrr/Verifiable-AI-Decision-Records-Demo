# Roadmap

This document tracks what we're building next and what we've explicitly parked. It's the team-facing mirror of the strategic backlog we keep in internal notes.

## Where we are today

The `ario-mlflow` plugin and the Railway-hosted demo are an **early-stage proof of a verifiable-provenance pipeline** — not a production compliance product. Core primitives (hash, sign, anchor to Arweave via ar.io, verify) are in place and end-to-end working. The plugin installs, the demo runs, all 33 smoke tests pass.

The goal of the current phase was *not* to build more features. It was to:

- **Goal A — Make the plugin usable.** Zero-config spin-up, clear errors, honest status when things go wrong, a README that pip-installed users actually see.
- **Goal B — Make the demo speak broadly.** Replace Iris with a relatable credit-decision scenario. Honest copy about what each verification level actually proves. One doorway for each of four plausible audiences.

Both goals landed on branch `phase3/harden-plugin-broaden-demo`. What follows is what we chose not to build yet, grouped by theme.

## Near-term focus

When we come back to this project, the highest-leverage next moves are:

1. **Input-side anchoring** — include dataset hash, source-code SHA, and container digest in the training proof. This closes the biggest honesty gap in the current offering ("we hash the model output, not the training inputs").
2. **External identity binding** — sign with something linkable to a real organization, not an auto-generated key at `~/.ario-mlflow/keys/`. Even a registered-public-keys directory is a step up.
3. **Continuous verification** — a background job that re-checks every anchored proof on a cadence and alerts when attestation doesn't land. Today "anchored" can silently mean "we called an API that may or may not have worked three weeks ago."

Each of these would be its own branch + plan. The plugin's current API surface doesn't need to change to accommodate any of them.

## The full backlog

Grouped by theme. Each item notes which audience (see *Audiences* below) it most directly serves.

### Provenance depth

| Item | What | Why | Serves |
|---|---|---|---|
| Input-side anchoring | Hash dataset + code commit + container digest into the training proof | Our current proof hashes outputs, not inputs — a bad actor could change training data and produce a model with identical metrics but different weights. | P1, P2, P3, P6 |
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
| Demo persistence | Use a Railway volume or small Postgres so `data/records.json` survives restarts | Today every Railway dyno restart wipes the records table. Fine for the core "this works" demo, bad for anyone returning to a URL they bookmarked. | All |
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

### Vocabulary pass — done 2026-04-22
"Chain of custody" is replaced by "Model lineage" (primary) with "audit trail" as a secondary phrase for GRC-leaning readers. Decision was informed by a short desk audit of ML (MLflow, W&B, Databricks, ClearML), GRC (Vanta, Drata, OneTrust), audit (Splunk, Datadog, CloudTrail), and AI-governance (Brundage et al., METR, Apollo) vocabulary: no surveyed tool uses "chain of custody" as a primary label; "lineage" dominates the ML world and "audit trail" is the cross-audience workhorse. See branch `phase3/harden-plugin-broaden-demo`.
