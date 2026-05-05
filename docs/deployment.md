# Deployment — Demo on Railway

This document covers the **demo's** production deployment. The plugin (`ario_mlflow/`) is distributed via `pip install -e .`; it has no deployment of its own.

## Production URL

The demo is publicly hosted at:

**https://verifiable-ai-demo-production.up.railway.app**

Railway auto-deploys on push to `main`. The build uses the Dockerfile in the repo root (`python:3.13-slim` + `uvicorn`).

## Persistence model

Railway provides a single mounted volume:

- **Volume name:** `verifiable-ai-demo-volume`
- **Mount path:** `/app/persistent`
- **Capacity:** 500MB

All persistent demo state must live under `/app/persistent`. The container's filesystem outside that path is ephemeral and gets wiped on every deploy.

The demo's `Settings` paths default to relative paths under `data/` and `mlruns/` — fine for local dev, but **not persistent on Railway**. Production overrides every relevant path via env vars to point at the volume.

## Required environment variables (production)

| Variable | Production value | Why |
|---|---|---|
| `VAIDR_MLFLOW_TRACKING_URI` | `/app/persistent/mlruns` | MLflow tracking store on volume |
| `VAIDR_RECORDS_FILE` | `/app/persistent/data/records.json` | Decision records cache on volume |
| `VAIDR_LIFECYCLE_FILE` | `/app/persistent/data/lifecycle.json` | Lifecycle events cache on volume |
| `VAIDR_ED25519_PRIVATE_KEY_PATH` | `/app/persistent/keys/ed25519_private.json` | Demo signing key on volume |
| `VAIDR_ED25519_PUBLIC_KEY_PATH` | `/app/persistent/keys/ed25519_public.json` | Demo public key on volume |
| `VAIDR_ARWEAVE_WALLET_PATH` | `/app/persistent/keys/arweave_wallet.json` | Arweave wallet on volume |
| `VAIDR_DEMO_MODE` | `true` | Registers `/tamper/*` and `/demo/*` routes (defaults true) |

**Critical:** if any path-related `VAIDR_*` env var is missing, the demo falls back to the default relative path which lands on the ephemeral filesystem. That data gets wiped on every deploy.

When adding new path-based settings to `app/config.py`, set the corresponding env var on Railway in the same change. Compare against `railway variables --kv | grep VAIDR` to verify.

## Inspecting / updating production config

Use the Railway CLI from the repo root (project is auto-linked):

```bash
# Show all env vars
railway variables

# Show just VAIDR_*, key=value format
railway variables --kv | grep VAIDR | sort

# List volumes
railway volume list

# Set or update a variable (triggers a redeploy)
railway variables --set "VAIDR_FOO=/app/persistent/foo"

# Set without auto-redeploy (takes effect on next deploy)
railway variables --set "VAIDR_FOO=/app/persistent/foo" --skip-deploys

# Recent deploys
railway deployment list

# Redeploy current
railway redeploy
```

`railway variables --set` triggers an automatic redeploy unless you pass `--skip-deploys`. After a path-related env var change, the demo's existing data may need a `/demo/admin` Reset for clean state.

## Sales / pre-sales workflow

The demo has a built-in admin page for the sales workflow:

**https://verifiable-ai-demo-production.up.railway.app/demo/admin**

What it does:
- Wipes all decisions, training runs, and model versions
- Auto-trains a fresh v1
- Anchored proofs on Arweave are not affected (they're permanent on the network)

When to use it:
- **Before a customer call** — pre-seed by running a few predictions with realistic data, then leave the demo in that state for the call
- **After a customer call** — wipe so the next session starts clean
- **After a deploy** that left the demo in an inconsistent state

Available only when `VAIDR_DEMO_MODE=true` (the default in production). Set `VAIDR_DEMO_MODE=false` for any non-demo deployment of this code.

## Ar.io credentials

The demo's signing key (`ed25519_private.json`) and Arweave wallet (`arweave_wallet.json`) live on the volume at `/app/persistent/keys/`. These were uploaded once via Railway's volume initialization; the auto-detect logic in `app/config.py` reads from the configured path.

Anchored proofs on Arweave are owned by the demo's wallet. If the wallet file is lost, future proofs would be signed by a different identity — existing proofs remain valid (signatures verify against their embedded public key) but the issuer-key check would fail for the new wallet's proofs.

## Lessons learned

- **2026-05-05** — `VAIDR_LIFECYCLE_FILE` was missing from production env vars for several weeks. `data/lifecycle.json` fell back to the ephemeral container filesystem while every other path was on the volume. Symptom: every deploy wiped lifecycle entries while preserving decisions and models, causing orphaned "Model Lineage" pages where decisions referenced run_ids whose lifecycle entries had vanished. Fix: set `VAIDR_LIFECYCLE_FILE=/app/persistent/data/lifecycle.json` and click `/demo/admin` Reset for clean state.

When adding new persistence paths to `Settings`, always set the corresponding production env var in the same change. The demo's path defaults are dev-only.
