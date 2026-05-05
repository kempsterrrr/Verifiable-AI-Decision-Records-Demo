# Claude Code instructions

> Auto-loaded on every Claude Code session in this repo.

## What this project is

This is a **two-part repository** with one product and one sales tool:

- **`ario_mlflow/` — the actual product.** A standalone MLflow plugin that anchors training runs, model registrations, and inference predictions to ar.io / Arweave for cryptographic verification. External MLflow users `pip install` this and adopt it inside their own training pipelines. The plugin's API surface (`anchor()`, `ArioMlflowClient`, `VerifiedModel`) is the value being shipped.

- **`app/` — a sales-facing demonstration.** A FastAPI + Jinja2 demo app that uses the plugin to make the verification flow tangible (UI for predictions, decision records, tamper buttons, three-row verify cards, model lineage view). Hosted on Railway. Sales / pre-sales use the demo to show prospective adopters what the plugin makes possible. The demo is **not the product** — it's a working showcase.

When deciding where new behavior should live, default to the plugin. The demo only contains code that's specific to *demonstrating* — UI rendering, sales-friendly tamper buttons, the `/demo/admin` reset flow. Anything that's a real verification capability belongs in the plugin so any external consumer benefits.

## Where things live

| Question | Where to look |
|---|---|
| What is this? (public-facing) | `README.md` |
| How does the plugin work? | `docs/architecture.md` |
| How is the demo deployed? | `docs/deployment.md` |
| What's next strategically? | `ROADMAP.md` |
| What are we working on now? | Latest file in `docs/plans/active/` |
| What shipped? | `git log --oneline -20` + `docs/plans/archived/` |
| Plugin's standalone docs | `ario_mlflow/README.md` |

## Conventions

- **Plugin-first.** Any new verification capability lands in `ario_mlflow/` first; the demo wraps it. CLI flags, env vars, and verification semantics belong in the plugin so external `pip install ario-mlflow` users get the same capability.
- **Demo-only code goes in `app/`.** UI rendering, tamper buttons, sales-flow features (`/demo/admin`, the Reset feature, demo-mode gating) live demo-side. Don't reach into the plugin for these.
- **Production paths via `VAIDR_*` env vars.** Defaults under `data/` and `mlruns/` for local dev. Production overrides to `/app/persistent/*` per `docs/deployment.md`.
- **Tests:** `pytest`. Use `tmp_path` and `monkeypatch` for filesystem and environment isolation. No live network calls in tests.
- **No backwards-compat shims.** Pre-prod codebase. Change things cleanly. `ProofEngine.create_proof` / `verify_local` were removed in Phase 2; don't reintroduce that pattern.

## How agents should work here

- **Pause for user review at each phase before opening a PR.** Multi-phase work needs explicit user approval at each phase, not just at the end.
- **Validate end-to-end, not just unit tests.** Boot the demo, click through the user-facing flow, watch for regressions in adjacent features.
- **Report before fixing.** When triaging review feedback or production issues, present analysis and let the user accept the verdict before making changes.
- **Destructive operations require explicit user confirmation.** Wiping data, force-pushing, deleting branches, resetting state — confirm first, even in auto mode.
- **Update existing PRs / feature branches.** Don't push directly to `main`; don't create a new PR for follow-up work that belongs in the active one.
- **Replace in same PR.** Deleting user-facing functionality requires shipping the replacement in the SAME PR — "follow-up" creates rabbit-hole risk.
- **Plugin and demo on separate feature branches.** Don't push plugin/core changes to a demo PR or vice versa.

## ar.io / Arweave terminology

User leads with **ar.io** over Arweave in verification UX and copy. Use "ar.io Verify," "anchored to ar.io," "ar.io gateway" in user-facing strings; "Arweave" is acceptable for the underlying network when precision matters.
