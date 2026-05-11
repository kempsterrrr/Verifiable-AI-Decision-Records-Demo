# Claude Code instructions

> Auto-loaded on every Claude Code session in this repo.

## What this project is

A **FastAPI + Jinja2 sales-facing demo** that uses the
[`ar-io-mlflow`](https://github.com/ar-io/ar-io-mlflow) MLflow plugin
(installed from PyPI as `ar-io-mlflow`) to make verifiable ML
provenance tangible — UI for predictions, decision records, tamper
buttons, three-row verify cards, dataset and model lineage views.
Hosted on Railway at
[verifiable-ai-demo-production.up.railway.app](https://verifiable-ai-demo-production.up.railway.app).
Sales / pre-sales use the demo to show prospective adopters what the
plugin makes possible. The demo is **not the product** — it's a working
showcase.

When deciding where new behavior should live: real verification
capabilities belong in the plugin (over at `ar-io/ar-io-mlflow`), not
here. This repo only contains code that's specific to *demonstrating* —
UI rendering, sales-friendly tamper buttons, the `/demo/admin` reset
flow, demo-mode gating.

## Where things live

| Question | Where to look |
|---|---|
| What is this? (public-facing) | `README.md` |
| How does the plugin work? | [ar-io/ar-io-mlflow](https://github.com/ar-io/ar-io-mlflow) repo |
| How is the demo deployed? | `docs/deployment.md` |
| What's next strategically? | `ROADMAP.md` |
| What are we working on now? | Latest file in `docs/plans/active/` |
| What shipped? | `git log --oneline -20` + `docs/plans/archived/` |
| Known demo issues | `docs/known-issues.md` |

## Commands

```bash
# Install (pulls ar-io-mlflow from PyPI + demo deps)
pip install -r requirements.txt

# Demo dev server (auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Train a model outside the demo flow (auto-trains on first prediction otherwise)
python scripts/train_model.py

# Tests
pytest                                            # full demo suite
pytest tests/test_tamper_endpoints.py             # one file
pytest tests/test_tamper_endpoints.py::test_name  # one test
pytest -k tamper                                  # by keyword

# Plugin CLI (installed by ar-io-mlflow as `ar-io-mlflow`)
ar-io-mlflow verify run <run_id>
ar-io-mlflow verify model <name>/<version>
ar-io-mlflow verify trace <trace_id>
ar-io-mlflow audit <name>/<version>
```

The plugin registers an MLflow `RunContextProvider` entry point — just
having `ar-io-mlflow` installed auto-tags every run with `ario.enabled`
/ `ario.version`; the rich proof layer requires an explicit `anchor()`
inside the run.

## Conventions

- **Plugin work happens in `ar-io/ar-io-mlflow`.** If a feature needs to
  change in the plugin (`anchor()`, `VerifiedModel`, `ArioMlflowClient`,
  CLI, verification semantics, env vars), do it there and bump the
  pinned version in this repo's `requirements.txt` when it lands. Don't
  fork plugin behavior into the demo.
- **Demo-only code lives in `app/`** — UI rendering, tamper buttons,
  sales-flow features (`/demo/admin`, the Reset feature, demo-mode
  gating). The demo wraps the plugin; it doesn't extend or duplicate
  plugin behavior.
- **Production paths via `VAIDR_*` env vars.** Defaults under `data/`
  and `mlruns/` for local dev. Production overrides to
  `/app/persistent/*` per `docs/deployment.md`.
- **Tests:** `pytest`. Use `tmp_path` and `monkeypatch` for filesystem
  and environment isolation. No live network calls in tests.
- **No backwards-compat shims.** Pre-prod codebase. Change things
  cleanly.

## How agents should work here

- **Pause for user review at each phase before opening a PR.**
  Multi-phase work needs explicit user approval at each phase.
- **Validate end-to-end, not just unit tests.** Boot the demo, click
  through the user-facing flow, watch for regressions in adjacent
  features.
- **Report before fixing.** When triaging review feedback or
  production issues, present analysis and let the user accept the
  verdict before making changes.
- **Destructive operations require explicit user confirmation.** Wiping
  data, force-pushing, deleting branches, resetting state — confirm
  first, even in auto mode.
- **Update existing PRs / feature branches.** Don't push directly to
  `main`; don't create a new PR for follow-up work that belongs in the
  active one.
- **Replace in same PR.** Deleting user-facing functionality requires
  shipping the replacement in the SAME PR — "follow-up" creates
  rabbit-hole risk.

## ar.io / Arweave terminology

User leads with **ar.io** over Arweave in verification UX and copy.
Use "ar.io Verify," "anchored to ar.io," "ar.io gateway" in
user-facing strings; "Arweave" is acceptable for the underlying network
when precision matters.
