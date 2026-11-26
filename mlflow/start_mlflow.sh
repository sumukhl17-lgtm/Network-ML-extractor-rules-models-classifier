#!/usr/bin/env bash
# Start the shared MLflow tracking server + model registry.
#   ./mlflow/start_mlflow.sh            (from the repo root)
# Then, in EVERY terminal that trains or serves a model:
#   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${MLFLOW_HOST:-127.0.0.1}"     # 0.0.0.0 only inside a container / trusted network
PORT="${MLFLOW_PORT:-5000}"
DATA="${MLFLOW_DATA_DIR:-$HERE/data}"

mkdir -p "$DATA/artifacts"
echo "MLflow UI + registry: http://$HOST:$PORT   (data in $DATA)"

# sqlite backend: the model registry (versions, aliases) needs a database, not just files.
# --serve-artifacts: clients upload/download model files through the server, so they do not need
# direct access to the artifact folder (this matters once the backend runs in a container).
exec mlflow server \
  --host "$HOST" --port "$PORT" \
  --backend-store-uri "sqlite:///$DATA/mlflow.db" \
  --artifacts-destination "$DATA/artifacts" \
  --serve-artifacts
