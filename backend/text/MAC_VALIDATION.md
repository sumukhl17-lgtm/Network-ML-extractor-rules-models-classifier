# Validate the text branch on your MacBook

This reference checklist predates the unified repository layout. Run commands from the repository root.
Dependency installation and model retraining are intentionally deferred; the shipped text model still
requires scikit-learn 1.9.1, while the network model requires 1.6.1.

## 0. Prerequisites

```bash
python3 --version            # must be 3.11 or newer.  If not:  brew install python@3.13
docker --version             # optional: needed only for the Tika / compose steps (install Docker Desktop)
```

## 1. Unzip and install

The shared environment must be Python 3.12.14 with scikit-learn 1.6.1. Do not install
`backend/text/requirements.txt` or `requirements-lock-py313.txt` into it: those select
scikit-learn 1.9.1 or 1.9.1-era dependencies.

## 2. Local tests (no external services) — expect all green

```bash
pytest backend/text/tests -v -rs -p no:cacheprovider 2>&1 | tee text_tests_local.txt
tail -3 text_tests_local.txt
```
Expected on the build machine (Linux, Python 3.13, no Tika / MLflow / FastAPI / Docker daemon): `143 passed, 14 skipped`.
Expected on a Mac where `requirements.txt` installed FastAPI, httpx and MLflow (no Tika, no Docker): `155 passed, 17 skipped`
(the 12 real HTTP tests and 5 real MLflow tests now run; the skips are 9 Tika + 3 Docker + 5 compose-config).
The skips are the Tika, MLflow, Docker and FastAPI tests; they stop being skipped as you install those (steps 4–6). Without the Docker CLI
on your Mac, the 5 compose-config tests also skip. A different Python / numpy may change timings but not the pass count. **Failures in step 2 matter most**: send me `text_tests_local.txt`.

## 3. Model compatibility check

```bash
TEXT_MODEL_DIR=/path/to/retrained/model python - <<'EOF'
import warnings, json, pathlib, sklearn
warnings.simplefilter("error")          # any scikit-learn version warning becomes an error
from backend import text
print("installed scikit-learn:", sklearn.__version__)
print("model trained with:   ", json.loads(pathlib.Path(text.config.meta_file()).read_text())["sklearn_version"])
print(text.model_info()["version"], "loaded OK")
EOF
python -m backend.text backend/text/samples/*/*          # classifies the six sample files
```
Use only a separately retrained model directory for `TEXT_MODEL_DIR`; do not replace the shipped model.

## 4. FastAPI HTTP tests *(never run so far)*

```bash
pytest backend/text/tests/test_api.py -v -rs
```

```bash
uvicorn backend.platform_api.main:app --port 8000 &
curl -s http://localhost:8000/classify/text/health
curl -s -F "file=@backend/text/samples/invoice/invoice_0000.pdf" http://localhost:8000/classify/text
kill %1
```

## 5. Real Apache Tika *(never run so far)*

```bash
docker compose -f backend/text/docker-compose.tika.yml up -d tika
sleep 20 && curl -s http://localhost:9998/tika          # a greeting means it is up
pytest backend/text/tests/test_integration_tika.py -v -rs -s     # -s prints the extraction timings
TIKA_URL=http://localhost:9998 python -m backend.text backend/text/samples/invoice/invoice_0000.pdf
docker compose -f backend/text/docker-compose.tika.yml down
```
On an Apple-silicon Mac, if Docker reports a platform mismatch, run the first command with `DOCKER_DEFAULT_PLATFORM=linux/amd64` in front (slower, emulated).
The next test needs Docker running: `pytest backend/text/tests/test_integration_docker.py -v -rs` (starts and stops the container itself).

## 6. Real MLflow *(never run so far)*

```bash
pip install mlflow
pytest backend/text/tests/test_integration_mlflow.py -v -rs      # uses a temporary SQLite store, no server needed
```
Then against a real server:

```bash
mlflow server --host 127.0.0.1 --port 5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m backend.text.data.generate_sample_dataset --out /tmp/sample --per-class 60 --seed 7
python -m backend.text.train --data-dir /tmp/sample --data-source synthetic-demo --register      # twice, to get v1 and v2
open http://127.0.0.1:5000                                                              # Models -> text-classifier -> alias "champion"
python -c "from backend import text; i = text.model_info(); print(i['source'], i['version'])"
kill %1
```

## 7. Send me

* `text_tests_local.txt` (step 2) and the output of steps 3–6, especially every failure.
* Which Python version and chip (Intel / Apple silicon) you have.
