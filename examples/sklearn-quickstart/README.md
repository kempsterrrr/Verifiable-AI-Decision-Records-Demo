# sklearn quickstart

Smallest possible example of `ario-mlflow`: train a toy sklearn model, log it
to a local MLflow store, and anchor a signed proof.

## Run

From the repo root:

```bash
pip install -e .                    # installs the plugin
pip install scikit-learn            # if not already installed
python examples/sklearn-quickstart/train_and_anchor.py
```

The script creates `examples/sklearn-quickstart/mlruns-quickstart/` alongside
itself, trains a Logistic Regression on iris, logs the model + metric, and
calls `ario_mlflow.anchor()`.

## What you should see

- One log line from the plugin telling you where the auto-generated Arweave
  wallet was persisted (first run only).
- The run ID, accuracy, wallet mode, verify status, artifact hash, and (if
  Arweave anchoring succeeded) the Arweave TX.
- Two copy-paste commands — one to verify the proof with the CLI, one to open
  the MLflow UI to see the `ario.*` tags on the run.

## What's happening under the hood

1. `mlflow.sklearn.log_model(model, "model")` writes the pickled model and a
   conda env to the run's artifacts.
2. `ario_mlflow.anchor()` reads the run's params/metrics, hashes the logged
   artifacts, creates a signed proof record, uploads it to Arweave via ar.io
   Turbo, and writes a set of `ario.*` tags on the run.
3. `ario-mlflow verify run <run_id>` fetches the proof back from Arweave,
   re-checks the signature and hash locally, and (if `ARIO_MLFLOW_ARIO_VERIFY_URL`
   is set) requests an ar.io Verify attestation.

## Reset between runs

Delete the local MLflow store to start fresh:

```bash
rm -rf examples/sklearn-quickstart/mlruns-quickstart
```
