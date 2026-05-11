# Test suite speedup — investigation, proposal, and shipped result

> **Status**: shipped on branch `test-infra/speedup` (off `main`,
> after PR #15 merged).
> **Result**: full suite 431s → **59.66s** (7.2× speedup),
> 149 passed, 0 failed.

## Why this exists

The full pytest suite takes **~7 minutes** wall time. PR #15's
verification cycle ran the suite three times (after each fix in the
chain), costing ~21 minutes of just-watching-pytest. We need this
to be roughly an order of magnitude faster.

## Where the time goes — measured (2026-05-11, current `main`)

Captured via `pytest --durations=20`. Total suite: **431s (7:11)**
for 149 tests.

### Catastrophic outliers — three tests sleep for 60s each

| Time  | Phase | Test |
|---|---|---|
| 60.49s | call | `test_tamper_endpoints::test_tamper_saved_record_returns_ok` |
| 60.49s | call | `test_tamper_endpoints::test_tamper_reset_restores_state` |
| 60.31s | call | `test_tamper_endpoints::test_tamper_live_data_returns_ok` |

**Root cause**: `app/main.py::_scheduled_revert` is registered as a
FastAPI `BackgroundTask` by the `/tamper/saved/*` and `/tamper/live/*`
routes. It calls `time.sleep(tamper_mod.TAMPER_TTL_SECONDS)`
(`app/main.py:1211`), and `TAMPER_TTL_SECONDS` defaults to **60**
(`app/tamper.py:26`). Starlette's `TestClient` waits for all
background tasks attached to a response before returning from
`client.post(...)`, so each tamper-route POST blocks for 60s.

The four *training/registration*-tamper tests aren't affected
because they call `tamper_saved()` / `tamper_live()` directly as
functions (bypassing the FastAPI route) — no BackgroundTask is
registered, no sleep happens.

**~181s of the 431s total is literally `time.sleep(60)` × 3.**

### Persistent baseline — lifespan auto-train

After the 3 outliers, the remaining ~17 entries in the slowest-20
are all in `setup` phase, ranging 7.6–10.2s. That's the lifespan
handler running on every `TestClient(app)` fixture boot:

1. `_ensure_default_datasets_seeded` (3 standalone dataset anchors)
2. `load_model` → `train_and_register_with_params` (LR fit + ~14
   MLflow file-backend writes + `mlflow.sklearn.log_model` artifact
   serialization + plugin's anchor + `ArioMlflowClient.create_model_version`)

Cost dominated by MLflow's file backend doing a synchronous write
per `log_param` / `log_metric` / `log_input` / `log_model` call.
31 tests boot a TestClient: `test_demo_reset.py` (7), `test_tamper_endpoints.py` (11),
`test_dataset_first.py` (13).

### Other slow non-boot test

`13.79s test_dataset_first::test_lifespan_seeding_is_idempotent_on_reboot`
— call phase. This is *my* test that explicitly boots two
`TestClient`s with the same `tmp_path` to verify idempotency.
2× lifespan = ~14s. Working as intended; not optimisable without
losing the test's point.

## Recommended fix — three small stages

### Stage 0 — set `VAIDR_TAMPER_TTL_SECONDS=0` in test fixtures (recommended *first* fix)

Set the env var to `0` in each `TestClient`-booting fixture
(`tests/test_tamper_endpoints.py`, `tests/test_demo_reset.py`,
`tests/test_dataset_first.py`). With TTL = 0, `_scheduled_revert`
returns essentially instantly; the BackgroundTask completes before
the test continues; the 60s sleep evaporates.

- **Effort**: ~3 one-line additions to existing `monkeypatch.setenv`
  blocks.
- **Speedup**: ~181s → ~0s for the 3 affected tests. **Suite drops
  from 7:11 to roughly 4:10.**
- **Risk**: minimal. None of the 3 affected tests assert anything
  about TTL behaviour — they only check status codes / response
  bodies of the initial tamper POST. The auto-revert mechanism is
  exercised separately (without the TTL race) by the function-level
  tests that call `tamper_saved()` / `tamper_live()` directly.
- **Fidelity**: preserved.

### Stage 1 — `pytest-xdist` parallel runner (recommended *second*; biggest remaining win)

Add `pytest-xdist` to dev deps and `addopts = -n auto` to
`pyproject.toml` (or `pytest.ini`). Workers run in separate
processes, each with its own MLflow caches; the 31 boot-style tests
fan out across cores.

- **Effort**: ~5 lines of config + one dep.
- **Speedup after Stage 0**: 4:10 → roughly 1:00–1:30 on an 8-core
  machine with `-n auto` (4–8 workers; Amdahl's law dampens the
  ideal).
- **Risk**: low. Every test already uses pytest's `tmp_path`
  (unique dir per test), and we just fixed the only known
  `sys.modules` polluter in PR #15. MLflow's
  `_get_store_with_resolved_uri` cache is per-process, so cross-
  worker collisions are impossible by construction.
- **Fidelity**: no change.
- **No CI to worry about**: `.github/` is empty; this is a
  developer-machine optimisation only.

### Stage 2 (optional follow-up; only if 1:30 isn't fast enough) — session-cached `mlruns` fixture

Build the auto-trained v1 **once** per pytest session, store the
resulting `mlruns/` + `lifecycle.json` + `keys/` tree at a session
scratch path. Each test's setup fixture `shutil.copytree`-s the
tree into its own `tmp_path` instead of re-training.

- **Effort**: ~30–50 lines in `tests/conftest.py`. The cached
  tree's `meta.yaml` files carry absolute `artifact_location`
  values, so the copy step rewrites those for each new `tmp_path`,
  *or* the session fixture builds the cache against a
  `file://`-relative URI MLflow happily accepts.
- **Speedup on top of Stages 0+1**: 31 × ~7s of auto-training
  collapses to 1 × ~7s + 31 × (~200ms `copytree`). With 4 parallel
  workers that's ~10s of shared setup + sub-second per-test
  overhead. **Estimated combined wall-time: ~30s.**
- **Risk**: medium. Path-rewriting must be airtight, or MLflow
  rejects the copied store on read. The Arweave wallet and Ed25519
  keys are independent of paths and can be re-used as-is.
- **Fidelity**: preserved.

### Not recommending — sqlite tracking backend in tests

MLflow itself recommends `sqlite:///` over the file backend
(`mlflow/tracking/_tracking_service/utils.py:184` deprecation
warning), and the sqlite backend handles small writes faster.
But: `tests/test_tamper_endpoints.py` literally tampers files on
disk in `mlruns/<exp>/<run>/artifacts/` to simulate attacks. A
backend swap would either silently mask those tests (sqlite
artifacts live in a blob) or require a parallel tracking config.
Defer until file-backend has been replaced in the demo too —
that's a separate, larger discussion.

### Not recommending — mocking auto-train

Replacing `train_and_register_with_params` with a stub that returns
a canned result would be fast, but the tamper tests verify
real-canonical-bytes-vs-rebuilt-canonical-bytes against MLflow's
actual storage. Mocking would make the tests prove nothing about
the property they're supposed to prove.

## One-line recommendation

**Stage 0 + Stage 1 ships almost all of the speedup with under 20
lines of code total. Suite drops from 7:11 → ~1:30. Treat Stage 2
as backlog.**

## Sequencing

1. PR #15 (dataset-first lifecycle) merges to `main`.
2. New branch `test-infra/speedup` off `main`:
   - **Stage 0**: add `monkeypatch.setenv("VAIDR_TAMPER_TTL_SECONDS", "0")`
     to the three `TestClient`-booting fixtures.
   - **Stage 1**: add `pytest-xdist` to dev requirements
     (`requirements.txt` or a new `requirements-dev.txt`); add
     `[tool.pytest.ini_options]` with `addopts = -n auto --tb=short`
     to `pyproject.toml`.
   - Verify: full suite green at ~1:30.
3. Decide whether Stage 2 is needed based on the new baseline.

## Files involved (Stages 0 + 1)

- `tests/test_tamper_endpoints.py` — add `VAIDR_TAMPER_TTL_SECONDS=0`
  in the `client` fixture's `monkeypatch.setenv` block (~line 27).
- `tests/test_demo_reset.py` — same in `_reload_app` (~line 32).
- `tests/test_dataset_first.py` — same in `_reload_app` (~line 33).
- `requirements-dev.txt` (new) — `-r requirements.txt` include plus
  `pytest>=8.0` and `pytest-xdist>=3.0`. Kept separate from
  `requirements.txt` so the Dockerfile (Railway prod image) stays slim.
- `pytest.ini` (new) —
  `addopts = -n auto --dist=loadfile --tb=short`. `loadfile` keeps
  tests in a single file on the same worker; see "What actually
  shipped" below for the reason.

## Verification

- `time python3 -m pytest tests/ -q` before and after each stage.
- Target after Stage 0: ~4:10.
- Target after Stage 1: ~1:30.
- Each stage should preserve `149 passed`; no green-to-red
  regressions.

## What actually shipped

Measured wall time: **59.66s** (from 431s; 7.2× speedup), `149 passed`.

One adjustment from the original proposal: `pytest-xdist`'s default
`-n auto` distributes individual tests across workers, which
unmasked a pre-existing test-pollution bug in `test_plugin_smoke.py`
— four `VerifiedModel`-related tests rely on hidden state from an
earlier test in the same file (the same pattern as the
`turbo_sdk.bundle` polluter we fixed in PR #15, just with a different
shared resource). Two options were viable:

1. **Fix the four tests** individually so they're independently
   bootable. Out-of-scope work for this PR, and the failure is
   pre-existing on `main` (verified by running the same test in
   isolation on `main` — same error).
2. **`--dist=loadfile`** — keep all tests in a single file on the
   same worker, preserving intra-file order. Files still
   parallelise; only within-file parallelism is lost.

Shipped option 2 because `test_plugin_smoke.py` is the only file
with these dependencies, its 92 tests are pure-Python (fast even
serial), and the bottleneck files (`test_tamper_endpoints`,
`test_demo_reset`, `test_dataset_first`) each end up on their own
worker. Fixing the four polluted plugin tests stays a backlog item.
