# Demo polish — invert the lifecycle so datasets come before training runs

## Context

In the current demo a dataset only materializes as a *side-effect* of a
training run. `app/model.py:131` (`_generate_credit_data`) is called
inside `train_and_register_with_params`, the DataFrame is logged via
`mlflow.log_input` at `app/model.py:171`, the plugin auto-anchors it,
and the `dataset_anchored` lifecycle event appears in the Datasets list
only after the run completes. There is no way to look at the demo and
say "here is a dataset" before a model has been trained against it.

This is backwards relative to the real-world lifecycle and relative to
the verification chain the rest of the demo already presents
left-to-right (Datasets → Training Runs → Models → Decisions →
Lineage). The fix is purely a demo orchestration change — the plugin
already supports standalone dataset anchoring via `anchor(dataset=ds)`
(shipped in PR #13). The demo just doesn't use it.

User decisions captured during planning:

1. **Dataset creation**: pre-seed 2–3 synthetic variants at boot/reset
   *and* expose a "Create dataset" form on the Datasets page (params:
   name, sample size, random seed). No CSV upload in v1.
2. **Train CTA**: dataset detail gets a "Train a model with this
   dataset" CTA *and* the Training Runs hero gets a dataset selector.
   Both entry points work.
3. **Reset**: `/demo/admin` Reset wipes, seeds the default datasets,
   then auto-trains v1 against the default dataset. Preserves the
   current full-state sales-demo behaviour.

## Scope

- Touch `app/` and `templates/` and tests only.
- **No changes to `ario_mlflow/`.** The plugin's `anchor(dataset=ds)`
  standalone path is already what we need.
- One PR on a new feature branch off `main`.

## Design

### Datasets are first-class, persisted by source URL

Each dataset is identified by its content digest. The synthetic
generator is deterministic given `(n_samples, random_state)`, so we
encode those in the dataset's `source` string —
`synthetic://credit_scorer?n_samples=800&seed=42` — and re-derive the
DataFrame at training time. No parquet file management; the source
string + seed is the persistence.

Three seeded variants on boot/reset:

| Name | n_samples | seed |
|---|---|---|
| Credit scoring — small | 300 | 0 |
| Credit scoring — default | 800 | 42 |
| Credit scoring — large | 2000 | 7 |

Different digests, different anchored TXs, useful spread in the
Datasets list.

### Lifecycle change

Today: train → plugin auto-anchors input → demo writes
`dataset_anchored` event.

After: seed/create → demo calls `anchor(dataset=ds)` standalone →
`dataset_anchored` event written. Later, when a training run picks
that dataset, the plugin auto-anchors it again (it has no dedup yet —
backlog item "Cross-run dataset reuse / dedup"). The duplicate event
has the same digest, so the Datasets list (`app/ui.py:476–545`
grouped by digest) shows one row. Detail page uses the *first*
event's TX as canonical. Documented caveat; not a correctness issue.

### Train route gains a required `dataset_id`

`POST /api/train` (`app/main.py:584`) accepts
`{dataset_id, max_iter, random_state}`. `train_and_register_with_params`
in `app/model.py:131` is refactored to take a dataset reference,
look it up in the lifecycle store by digest, parse its source URL to
recover `(n_samples, seed)`, regenerate the DataFrame, assert the
digest matches what's anchored, log the input, and proceed.

The dataset preselection on the Train hero comes from a query param
(`/ui/runs?dataset=<digest>`) populated by the dataset-detail CTA. If
no dataset is preselected, the dropdown defaults to "Credit scoring —
default".

### Reset flow

`/demo/admin` reset handler: wipe → call `seed_default_datasets()` →
call existing auto-train pathway with `dataset_id` = default dataset's
digest.

## Files to change

| File | Change |
|---|---|
| `app/model.py` | Refactor `train_and_register_with_params` to take `dataset_digest` and use the looked-up dataset. Extract `_generate_credit_data` call site into a `_materialize_synthetic(n_samples, seed)` helper. Add `seed_default_datasets(...)` (calls `anchor(dataset=ds)` three times). Add `create_dataset_from_params(name, n_samples, seed)` (single anchor). |
| `app/main.py` | New route `POST /api/datasets` → calls `create_dataset_from_params`. Modify `POST /api/train` (line 584) to require `dataset_id`. Modify `/demo/admin` reset path to call `seed_default_datasets` before auto-train. |
| `app/ui.py` | Update the runs list route to pass `available_datasets` to the template (for the selector). Update the datasets list route (line 476) to pass form context for "Create dataset". Update dataset detail (line 548) to compute the train URL. |
| `templates/runs_list.html` | Add dataset `<select>` to the train hero form (lines 156–187). Update the inline JS at line 286 to send `dataset_id`. Read `?dataset=<digest>` from the URL and preselect. |
| `templates/datasets_list.html` | Add a "Create dataset" form/modal at the top (fields: name, n_samples, random_state). Submit to `POST /api/datasets`. |
| `templates/dataset_detail.html` | Add a "Train a model with this dataset" primary CTA above the identity card. Links to `/ui/runs?dataset={{ digest }}`. |
| `tests/` | New tests: `seed_default_datasets` writes 3 events with distinct digests; `POST /api/datasets` happy/sad paths; `POST /api/train` rejects missing `dataset_id`; train-with-existing-dataset produces a duplicate `dataset_anchored` event with matching digest (documents the known caveat); `/demo/admin` reset seeds + auto-trains. |

Existing utilities to reuse (no rewriting):

- `_generate_credit_data` (`app/model.py:131`) — keep, wrap.
- `mlflow.data.from_pandas` construction at `app/model.py:152–162` —
  keep, extract into the new helper.
- `ario_mlflow.anchor(dataset=ds, ...)` — plugin entry point (PR #13).
- Lifecycle-store conversion at `app/main.py:119–164` — feeds the
  `dataset_anchored` event into the store; reuse from the new
  standalone-anchor path.
- Datasets list grouping by digest in `app/ui.py:476–545` — already
  handles dedup visually.

## Verification

End-to-end manual run on the demo (boot, click through, monitor for
regressions in adjacent features — per the
*validate-end-to-end* memory):

1. Boot demo cold → Datasets list shows three seeded datasets, each
   anchored standalone (each with its own TX, no "used by" runs yet).
2. Click "Credit scoring — small" → detail page → "Train a model
   with this dataset" → arrives at `/ui/runs?dataset=<digest>` with
   the dataset preselected in the hero. Submit. New training run +
   model produced.
3. Lineage page: chain shows dataset → run → registration →
   decisions. Verify each link.
4. Datasets list → "Create dataset" form → enter name, size, seed →
   new dataset appears in list, anchored, no runs yet.
5. Train against the user-created dataset. Verify chain works.
6. `/demo/admin` Reset → list clears → seeds three datasets →
   auto-trains v1 against the default → models page shows v1.
7. All existing tamper buttons on `/demo/tamper` still work against
   the seeded chain.
8. `pytest` — all existing tests (104) plus the new ones pass.

Known caveat to verify the UI gracefully handles: after step 2, the
training run's auto-anchor produces a *second* `dataset_anchored`
event with the same digest. Datasets list should still show one row
for that dataset; detail page should show the original (seeded) TX as
canonical.
