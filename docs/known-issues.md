# Known issues

## Upstream: `ario_mlflow.anchor(dataset=ds)` standalone path doesn't accept a freshly-built `PandasDataset`

**Affects**: `ario_mlflow/anchoring.py::_anchor_dataset_event` — the
helper backing the standalone `anchor(dataset=...)` API shipped in
PR #13.

**Symptoms**: when a caller passes the live object returned by
`mlflow.data.from_pandas(...)`, the plugin crashes with one of:

- `AttributeError: 'Schema' object has no attribute 'encode'`
  — `dataset.schema` is an `mlflow.types.Schema` instance; the plugin
  tries `json.loads(schema)`, then falls back to `schema.encode("utf-8")`.
- `AttributeError: 'PandasDataset' object has no attribute 'source_type'`
  — `PandasDataset` doesn't expose `source_type` as an attribute at all;
  it lives under `dataset.source.source_type` and is materialized by
  `dataset.to_dict()`.

**Root cause**: `_anchor_dataset_event` was written against the
**training-mode** shape, where MLflow's `log_input` → run-input
round-trip stores the dataset via `to_dict()` and rehydrates it as a
dict-like with **stringified** `schema`, `source`, and `source_type`
fields. The standalone path hands the plugin the live in-memory
`PandasDataset`, whose attribute layout is different:

| Attribute     | Training-mode (post `to_dict()`) | Standalone (live `PandasDataset`) |
|---------------|----------------------------------|------------------------------------|
| `name`        | string                           | string                             |
| `digest`      | string                           | string                             |
| `source`      | string (`'{"uri": "..."}'`)      | `LocalArtifactDatasetSource` obj   |
| `source_type` | string (`'local'`)               | **not present** (no attribute)     |
| `schema`      | string (JSON)                    | `mlflow.types.Schema` object       |

**Suggested upstream fix**: normalize `dataset` through `to_dict()`
at the entry of `_anchor_dataset_event`, then read all fields as
strings from the dict. ~5 lines, no API change, makes the standalone
path produce the same digest + schema_hash as the training-mode path.

**Demo-side workaround** (in this repo): see
`app/model.py::_dataset_view_for_plugin`. It wraps the live
`PandasDataset` in a `SimpleNamespace` built from `dataset.to_dict()`,
exposing exactly the string-valued attributes the plugin expects.
Called from `anchor_synthetic_dataset` before
`ario_mlflow.anchor(dataset=ds)`. Remove the helper and pass the live
`PandasDataset` directly once the upstream fix lands.

**Tracking**:

- File an upstream issue at <https://github.com/ar-io/ar-io-mlflow>
  describing both symptoms, the attribute-shape table above, and the
  suggested `to_dict()`-based fix. Once the patch ships, bump the
  `ar-io-mlflow` version in `requirements.txt` and delete
  `_dataset_view_for_plugin` from `app/model.py`.
