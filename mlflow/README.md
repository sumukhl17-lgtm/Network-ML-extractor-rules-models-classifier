# MLflow: tracking server and model registry

MLflow records every training run (settings, metrics, the model file) and keeps a **registry**:
a named list of model versions with a movable label (`champion`) on the one that should be live.

## 1. Start the server

```bash
pip install -r backend/requirements.txt
./mlflow/start_mlflow.sh          # UI at http://127.0.0.1:5000
```

In every other terminal (training, backend):

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Data lives in `mlflow/data/` (a sqlite file + an artifacts folder). Delete that folder to start clean.
Running in a container later: set `MLFLOW_HOST=0.0.0.0` and mount `MLFLOW_DATA_DIR` as a volume.

When using the root Docker Compose stack, the MLflow UI is available at
`http://127.0.0.1:5001` by default (or the configured `MLFLOW_PORT` host port).
Containers continue to connect to the service at `http://mlflow:5000`.

## 2. Registered model names (agreed between branches)

| Branch | Registered name | Alias | Served from the registry today? |
|---|---|---|---|
| Text (Person 2) | `text-classifier` | `champion` | **Yes**. `text.model` loads `models:/text-classifier@champion` |
| Network (Person 1) | `network-classifier` (placeholder) | `champion` | **No**. `classify_network.py` loads `rf.joblib` from the repo |

Names can be changed with `TEXT_MODEL_NAME` / `NETWORK_MODEL_NAME`.

## 3. Put models into the registry

**Text:** already built in. Make a few versions to see the registry view working:

```bash
cd backend
python -m text.data.generate_sample_dataset --out data/sample --per-class 60
python -m text.train --data-dir data/sample --data-source synthetic-demo --register force
python -m text.train --data-dir data/sample --data-source synthetic-demo --register never   # a 2nd version, not promoted
```

**Network:** Sumukh's `train_compare.py` already logs runs to the experiment `network-known-unknown`
(artifact path `model`) when `MLFLOW_TRACKING_URI` is set. The registry page shows those runs read-only.
To register the model, Person 1 adds (after the run):

```python
mlflow.register_model("runs:/<run_id>/model", "network-classifier")
```

and then, to make promotion actually change what is served, loads `models:/network-classifier@champion`
in `classify_network.py` (keeping `rf.joblib` as a fallback). Until then the app says so on screen.

## 4. Promote and roll back

From the app (Registry page), or with the API:

```bash
curl -X POST localhost:8000/registry/models/text-classifier/promote \
     -H 'content-type: application/json' \
     -d '{"version":"2","confirm":true,"actor":"your-name"}'
```

Promotion moves the `champion` alias and reloads the live text model. Rollback moves it back to the
previous champion (from the activity log), or to `to_version` if you give one. Both refuse without `"confirm": true`.

## Retraining runbook

1. Collect or label new documents; run `python -m text.train ... --register never`.
2. Open the Registry page, compare the new version with the champion (F1, accuracy).
3. If it is better on the held-out set, promote it. If anything looks wrong afterwards, roll back.
4. Record what changed in the report (the activity log keeps who and when).
