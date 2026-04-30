# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python demo for verifiable AI decision records. `app/` contains the FastAPI app, UI routes, stores, lifecycle helpers, and decision-record logic. `ario_mlflow/` is the reusable MLflow plugin package for anchoring, proof generation, verification, CLI, and reports. `templates/` holds Jinja2 dashboard pages. `scripts/` contains local utilities such as model training and verification. `tests/` contains pytest coverage. `examples/sklearn-quickstart/` shows standalone plugin usage. Generated state in `data/`, `keys/`, and `mlruns/` is ignored.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the plugin in editable mode:

```bash
pip install -e .
```

Train or refresh the demo model:

```bash
python scripts/train_model.py
```

Run the local dashboard:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run tests:

```bash
pytest
```

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, 4-space indentation, and descriptive snake_case names for modules, functions, variables, and tests. Keep route handlers in `app/main.py` or `app/ui.py`, and reusable proof, anchoring, and verification behavior in `ario_mlflow/`. Prefer typed public helpers and keep JSON/proof fields explicit and stable, since they are part of the audit surface. Follow the existing style: small helper functions, direct proof dictionaries, and concise comments only where flow is non-obvious.

## Testing Guidelines

Tests use pytest. Add new tests under `tests/` as `test_*.py`, with focused names such as `test_proof_engine_rejects_tampered_record`. Prefer deterministic tests that avoid live network, real Arweave uploads, or external MLflow servers. Use `tmp_path` and `monkeypatch` for filesystem and environment isolation. Run `pytest` before opening a PR; for CLI changes, include parser or command-level coverage.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive summaries such as `Add button loading states with inline spinners` and phase summaries such as `Phase 3: harden plugin, broaden demo, honest verification UI`. Keep commits focused and use an imperative or clearly descriptive subject.

Pull requests should include a concise description, user-visible behavior changed, test commands run, and relevant issue links. Include screenshots or recordings for dashboard/template changes. Call out changes to proof schema, environment variables, anchoring behavior, or verification semantics because those affect audit compatibility.

## Security & Configuration Tips

Do not commit `.env`, Arweave wallets, signing keys, MLflow runs, or local record data. Runtime configuration uses `VAIDR_` variables for the demo app and `ARIO_MLFLOW_` variables for the plugin. Keep local-only credentials in ignored files such as `keys/arweave_wallet.json`.
