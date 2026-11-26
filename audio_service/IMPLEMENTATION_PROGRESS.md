# Audio implementation progress

> **Document status:** This is a chronological development and validation log. Some dated sections describe earlier Compose-profile and isolated-registry configurations. For the current integrated deployment, use the root `README.md` and `docker-compose.yml`. The current default Compose stack starts the Audio API and worker without an `audio` profile.

## Repository baseline

- Branch: `integration/full-platform`.
- Before implementation, `git status --short --branch` showed `.gitignore`
  modified and these paths untracked: `.dockerignore`, `LICENSE`, `app/`,
  `audio_service/`, `backend/`, `backups/`, `datasets/`, `docker-compose.yml`,
  `frontend/`, `generate_test_pcap.py`, `generate_unknown_pcap.py`, `k8s/`,
  `mlflow/`, and `test_app.py`.
- The original Audio tree contained the decoder, API, worker, Compose
  integration, codec fixtures, and four test modules. It had no Audio
  preprocessing, YAMNet, ESC-50, or classifier implementation.
- Baseline validation:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv312/bin/python -m pytest audio_service/tests -q -p no:cacheprovider`
  reported 82 passed, with one Starlette/AnyIO deprecation warning.
- Baseline `docker compose config --quiet` succeeded.
- Existing `audio-stage211-validation` API and worker containers were
  healthy. They were not stopped or recreated during this work.
- No existing model or dataset files were overwritten. The root `LICENSE`
  was already untracked at baseline and has not been modified.

## Stage 2.2: Audio preprocessing — implemented

Implemented `preprocess_audio(DecodedAudio)` using PyAV `AudioResampler` to
produce a mono, 16 kHz, finite float32 waveform.

The implementation handles packed and planar input, clips finite
out-of-range values without peak normalization, preserves silence and
low-level signals, enforces decoder limits, and rejects clips shorter than
YAMNet's documented 975 ms minimum input.

Files added:
- `audio_service/preprocessing.py`
- `audio_service/tests/test_preprocessing.py`

Files modified:
- `audio_service/requirements.txt`
- `audio_service/worker_runtime.py`
- `audio_service/worker_client.py`

Validation:
- Preprocessing tests: 14 passed.
- Full Audio suite after preprocessing: 96 passed.
- Full Audio suite after YAMNet wrapper work: 108 passed.

## Stage 2.3: YAMNet integration — implemented

Implemented a lazy, per-process `YAMNetEmbedder` with input/output
validation, temporal mean pooling of frame scores and 1024-dimensional
embeddings, AudioSet class names, version reporting, and a SHA-256 digest
of the resolved model asset tree.

Added isolated inference dependencies and a worker-only Dockerfile. The
Platform API and lightweight Audio gateway image do not install TensorFlow.

Files:
- `audio_service/yamnet.py`
- `audio_service/tests/test_yamnet.py`
- `audio_service/requirements-inference.txt`
- `audio_service/Dockerfile.worker`

Dependencies:
- Base Audio: `numpy==2.0.2`, `starlette==0.49.3`.
- Worker inference: `tensorflow==2.19.0`, `tf-keras==2.19.0`,
  `tensorflow-hub==0.16.1`, `setuptools==80.9.0`, `scipy==1.13.1`,
  `scikit-learn==1.6.1`, `SQLAlchemy==2.0.54`, `joblib==1.5.3`,
  `pandas==2.3.3`, `mlflow==3.18.0`, and `skops==0.13.0`.
- `setuptools==80.9.0` is pinned because TensorFlow Hub 0.16.1 imports
  `pkg_resources`, which is absent in newer Setuptools. Its deprecation
  warning remains visible.
- No packages were installed into the developer virtual environment.

YAMNet verification and provenance:
- Official handle: https://tfhub.dev/google/yamnet/1
- Upstream implementation and model notes:
  https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- The TensorFlow Models repository's research tree is under Apache-2.0;
  attribution and license notice must be retained in Audio documentation.
- Upstream documents Keras 2 compatibility and a 975 ms minimum waveform.
  The worker uses TensorFlow 2.19.0 with `tf-keras==2.19.0` and
  `TF_USE_LEGACY_KERAS=1`.
- A verified Python 3.12 Linux/ARM64 worker image loaded the official model
  and returned finite `(521,)` class scores and `(1024,)` pooled embeddings.
- Observed model asset-tree SHA-256:
  `596bd1c6fd800af3d7e9ad78cc1b08ed69d97326a051ae08bc3f09f5d8ee7b45`.
- Model files used for that probe were cached in ephemeral test containers
  and were not added to the repository.

Resource validation:
- At 512 MiB, a 5-second YAMNet inference plus worker-parent and duplicated
  20 MiB upload peaked at 517726208 bytes (cgroup maximum 536870912 bytes).
- At 512 MiB, a 120-second waveform test was OOM-killed:
  `memory_peak=536870912`, `oom_kill=1`.
- The user approved raising only the Audio worker memory cap to 1 GiB.
- At 1 GiB, the same parent/child, 20 MiB upload, 120-second inference
  test succeeded with 249 score frames and 249 embeddings:
  `memory_peak=726360064`, `memory_max=1073741824`, `oom_kill=0`.
- These are direct YAMNet/container probes, not a production workload
  benchmark.

## Stage 2.3A: Persistent inference process runtime — implemented

Added a persistent child-process supervisor that initializes its runtime
once, processes sequential requests, enforces startup and request timeouts,
and terminates/reaps failed or cancelled children.

Safe `AudioServiceError` responses are transported across IPC with their
status code, error code, and safe message preserved. Ordinary application
errors do not unnecessarily terminate a healthy child.

The supervisor supports explicit startup, exposes child readiness and model
version, and shuts down its child process cleanly.

Files:
- `audio_service/persistent_runtime.py`
- `audio_service/tests/test_persistent_runtime.py`

Validation:
- Persistent runtime tests: 8 passed.
- Persistent runtime and inference runtime focused tests: 20 passed.

## Stage 2.3B: Persistent audio inference pipeline — implemented

Added eager YAMNet loading for child startup and the persistent inference
pipeline in `audio_service/inference_runtime.py`.

The pipeline:
- Loads and validates the MLflow `audio-classifier` champion once at
  child startup.
- Loads YAMNet once at child startup.
- Removes the service authentication token from child settings.
- Validates inference payloads and upload metadata.
- Runs decoding, preprocessing, YAMNet embedding, and classification inside
  the persistent child.
- Returns serialized classification response data through IPC.

Files:
- `audio_service/inference_runtime.py`
- `audio_service/tests/test_inference_runtime.py`
- `audio_service/yamnet.py`
- `audio_service/persistent_runtime.py`

Validation:
- YAMNet tests: 12 passed.
- Inference runtime tests: 12 passed.
- Persistent runtime tests: 8 passed.

## Stage 2.3C: Worker and API integration — implemented

Integrated the persistent inference supervisor into the isolated worker
lifecycle and connected the Platform Audio API to the worker.

Worker integration:
- The worker creates and starts the inference supervisor during application
  startup.
- Startup waits for the inference child to initialize and report its model
  version.
- The worker exposes readiness based on the running child and loaded model
  version.
- Classification requests are handled by the persistent child rather than
  reloading models per request.
- Client disconnects cancel the in-flight inference request.
- Shutdown closes the inference supervisor and the existing decode
  supervisor.
- The existing internal decode route remains available.
- Internal service authentication and safe error propagation are retained.

Platform API integration:
- The Platform API uses the worker client for classification and readiness.
- The Platform API does not import TensorFlow or load inference models.
- Worker errors are mapped to safe API responses.
- Upload limits, capacity handling, disconnect cancellation, and file
  cleanup remain covered by tests.

Compose configuration:
- The `audio-api` and `audio-worker` services use the `audio` profile.
- The API waits for the worker to become healthy.
- The worker waits for MLflow to become healthy.
- The worker is attached to both `default` and `audio-internal` networks.
  This permits access to MLflow and external access needed for an uncached
  TensorFlow Hub model download.
- The worker uses the persistent `audio_model_cache` volume at
  `/var/cache/audio-models`, with `TFHUB_CACHE_DIR` set to
  `/var/cache/audio-models/tfhub`.
- The worker memory limit defaults to 1 GiB.
- The worker health check uses `/ready`; the API health check also uses
  `/ready` so readiness reflects worker availability.
- `docker compose --profile audio config --quiet` succeeds.

Files added or integrated:
- `audio_service/worker_api.py`
- `audio_service/worker_client.py`
- `audio_service/tests/test_worker.py`
- `audio_service/tests/test_worker_client.py`
- `audio_service/tests/test_api.py`

Files also involved:
- `audio_service/api.py`
- `audio_service/config.py`
- `audio_service/persistent_runtime.py`
- `audio_service/inference_runtime.py`
- `docker-compose.yml`

Automated validation on 2026-11-24:
- Full Audio suite: **139 passed, 1 warning**.
- Python compilation of Audio modules: passed.
- `docker compose --profile audio config --quiet`: passed.
- `git diff --check`: clean.
- The remaining warning is a Starlette TestClient / AnyIO deprecation:
  `anyio.abc.BlockingPortal` is deprecated in the installed dependency stack.

Live integration verification using separately started isolated containers:
- A real isolated MLflow registry, worker container, and API container were
  started separately from the standard Compose deployment.
- The isolated registry used a copy of the MLflow database and artifacts.
  Its `audio-classifier` model version `1` was assigned the `champion` alias.
- The isolated worker loaded that model and YAMNet and reported ready with
  model version `1`.
- A real ESC-50 WAV upload passed through the Platform API, worker, and
  persistent inference process and returned HTTP 200.
- The authenticated result identified `1-100032-A-0.wav` as `dog`, with
  sound group `animals`, model version `1`, source `mlflow`, 5.0 seconds,
  16 kHz, and mono audio. The returned top prediction confidence was
  approximately 0.6494. The response explicitly states that confidence
  scores are not calibrated.
- Eight response assertions passed, including category, sound group,
  model version, source, confidence range, duration, sample rate, and
  channel count.
- An invalid service token returned HTTP 401 with the expected safe
  unauthorized response.
- Both isolated API and worker health/readiness checks succeeded.
- This was a real isolated container end-to-end inference test. It did not
  use the production MLflow registry.

## Stage 2.4: ESC-50 training and isolated model serving — implemented and validated

Dataset:
- Dataset README identifies the upstream project:
  https://github.com/karoldvl/ESC-50
- Local dataset directory: `datasets/ESC-50`.
- The local directory contains `audio/`, `esc50.csv`, `README.md`, and
  `LICENSE`.
- CSV validation found 2,000 metadata records, 50 classes, 40 recordings
  per class, and 400 recordings in each of the five official folds.
- The metadata contains the expected columns:
  `filename`, `fold`, `target`, `category`, `esc10`, `src_file`, and `take`.
- The dataset occupies approximately 844 MiB on disk.
- The README states that the main dataset is CC BY-NC 3.0, while the
  ESC-10 subset is CC BY. Clip-level attributions are provided in `LICENSE`.
  The dataset's academic citation is K. J. Piczak, “ESC: Dataset for
  Environmental Sound Classification,” ACM Multimedia 2015,
  DOI `10.1145/2733373.2806390`.
- The user authorized use for a non-commercial research/prototype only.
  Do not represent the resulting model or dataset use as commercially
  cleared.

Verified dataset provenance on 2026-11-24:
- Compared the local dataset against the official upstream Git checkout
  at `https://github.com/karoldvl/ESC-50`.
- Verified upstream commit:
  `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
- Upstream commit date: `2023-10-13T13:18:14+02:00`.
- All 2,000 local WAV files match the upstream WAV files byte-for-byte:
  zero missing paths, zero extra paths, and zero content differences.
- `README.md`, `LICENSE`, and the CSV also match upstream byte-for-byte.
  The upstream CSV is at `meta/esc50.csv`; the local copy is at
  `datasets/ESC-50/esc50.csv`.
- The local directory is not an exact full-repository copy: it omits
  eight ancillary upstream files:
  `.circleci/config.yml`, `.github/stale.yml`, `.gitignore`, `esc50.gif`,
  `meta/esc50-human.xlsx`, `pytest.ini`, `requirements.txt`, and
  `tests/test_dataset.py`.
- Local dataset tree fingerprint (relative paths and file SHA-256 digests):
  `cb1b83bbc9f7200469e95b18dabea9379b6f1122b43bbca074c77a9273101ba6`.
  This is a local snapshot fingerprint, not an upstream archive checksum.
- No original source archive was available locally, so an original archive
  SHA-256 checksum has not been established. The verified upstream Git
  commit and exact per-file comparisons establish the provenance of the
  local audio recordings and matching metadata.

Embedding and training:
- YAMNet embeddings were extracted for all 2,000 recordings.
- The resulting embedding matrix contains 2,000 rows of 1,024-dimensional
  float32 values; values were checked for finiteness.
- Embeddings were generated with `google/yamnet/1`.
- The cache is stored in the Docker named volume
  `network-ml_audio_model_cache`, including
  `/var/cache/audio-models/esc50-yamnet-embeddings.npz` and the TensorFlow
  Hub cache directory.
- Training evaluated each official fold once as held-out data using
  five-fold ESC-50 cross-validation.
- MLflow run ID:
  `267cbf3fb46e4b28ad466fd3171f70ab`
- MLflow run status: `FINISHED`.
- Registered model: `audio-classifier`, version `1`, status `READY`.
- Cross-validation accuracy: mean `0.8500`, standard deviation `0.0172`.
- Cross-validation macro F1: mean `0.8488`, standard deviation `0.0172`.
- These are results on ESC-50's official five-fold evaluation. They do not
  establish external-dataset generalization or calibrated confidence.

Registry isolation and promotion:
- The `champion` alias was assigned to version `1` only in the isolated
  MLflow registry used for validation.
- The production MLflow registry at `http://localhost:5001` was checked:
  `audio-classifier` had no aliases.
- Production registry isolation was verified. No Audio champion alias was
  promoted in production.
- Do not promote an Audio model alias in production without the user's
  explicit approval.

Live inference checks using separately started isolated containers:
- The isolated worker and API used the isolated MLflow copy, not the
  production MLflow registry.
- A valid authenticated upload returned HTTP 200 with a real model
  prediction.
- An invalid token returned HTTP 401.
- The isolated API and worker remained healthy after the terminal session
  used to start them ended.
- The live test verified API → worker → persistent inference process →
  MLflow model and YAMNet, using an actual ESC-50 recording.

### Standard Docker Compose Audio profile — validated on 2026-11-24

The standard Compose Audio API and worker services were started using a
temporary validation override, without starting the Compose `mlflow`
service. The override connected them to the already-running isolated
MLflow container and used the existing external Docker network and model
cache volume. The API was published on host port `8003`.

The validation used project name `audio-compose-validation` and the
existing isolated images:
- API: `network-ml-audio-api-isolated:latest`
- Worker: `network-ml-audio-worker-stage24:latest`

Results:
- Compose configuration validation passed.
- Only `audio-worker` and `audio-api` were started with `--no-deps`.
  No other Compose services or MLflow dependencies were started.
- Worker became healthy; its `/ready` check returned HTTP 200.
- API became healthy; `/health` returned HTTP 200 and `/ready` returned
  HTTP 200 with `loaded: true` and model version `1`.
- Authenticated upload of
  `datasets/ESC-50/audio/1-100032-A-0.wav` to
  `POST /internal/v1/classify` returned HTTP 200.
- Nine response assertions passed: filename, category, sound group,
  source, model version, duration, sample rate, channels, and non-empty
  top-k predictions.
- The response classified the recording as `dog`, sound group `animals`,
  source `mlflow`, model version `1`, duration `5.0` seconds, sample rate
  `16000`, and `1` channel. Top prediction confidence was `0.6494334`.
- An invalid service token returned HTTP 401 with error code `unauthorized`.
- The production MLflow container `network-ml-mlflow-1` remained running
  and healthy on host port `5001`.
- The production registry's `audio-classifier` aliases remained empty:
  `{}`. No production alias was created or changed.
- The isolated Compose API and worker were both healthy at the end of
  validation.

This verifies the standard Compose Audio API/worker service definitions
against an isolated MLflow registry. It does **not** verify a deployment
against production MLflow, production traffic, or production capacity.
The temporary override and isolated validation containers are retained;
they have not been removed.

## Probability calibration experiment

Evaluation design:
- Evaluated sigmoid probability calibration using nested cross-validation.
- Used the official five ESC-50 folds as outer holdouts. Each outer fold
  trained on 1,600 examples and evaluated on 400 examples.
- Used three-fold stratified inner calibration splits, with random state
  `42` and `ensemble=False`.
- Compared the uncalibrated classifier against the sigmoid-calibrated
  classifier using accuracy, macro F1, log loss, multiclass Brier score,
  and top-label expected calibration error (ECE).
- ECE used 10 equal-width confidence bins. Brier score is the mean
  per-example sum of squared errors across all 50 classes.

Five-fold mean results (lower is better for log loss, Brier score, and ECE):

| Metric | Raw | Sigmoid calibrated |
| --- | ---: | ---: |
| Accuracy | 0.8500 | 0.8425 |
| Macro F1 | 0.8485 | 0.8396 |
| Log loss | 0.6326 | 0.8906 |
| Brier score | 0.2275 | 0.3601 |
| Top-label ECE | 0.0460 | 0.3001 |

Outcome:
- Sigmoid calibration worsened all three probability-quality metrics and
  did not improve accuracy or macro F1.
- The calibration optimizer emitted NumPy/scikit-learn runtime warnings
  during sigmoid fitting, including invalid, divide-by-zero, and overflow
  matrix operations. The reported aggregate metrics were finite, but the
  warnings are an additional reason not to rely on this fitted calibrator.
- **Decision: reject this sigmoid calibrator.** Keep the existing raw
  classifier and its explicit response statement that confidence scores
  are not calibrated.
- No production inference behavior was changed. No model was registered,
  no MLflow alias was changed, and no production promotion occurred.
- This experiment does not establish calibrated confidence or justify
  interpreting the raw scores as probabilities.

## Remaining work and limitations

1. **Independent evaluation:** Evaluate the classifier on independent
   data outside ESC-50's five-fold evaluation before making claims about
   external-dataset generalization.
2. **Confidence calibration:** The sigmoid calibration experiment was
   rejected because it worsened probability metrics and emitted optimizer
   runtime warnings. Investigate a numerically stable alternative and
   validate it on held-out data before relying on confidence as a
   probability or setting confidence thresholds.
3. **Resource/deployment validation:** The 1 GiB worker memory limit is a
   measured safety adjustment based on controlled tests, not a validated
   production capacity target. Reassess with representative production
   workloads before changing it.
4. **Dataset licensing:** Retain ESC-50 attribution, citation, and per-clip
   license notices. Commercial use is not cleared by the current
   non-commercial authorization.
5. **Production promotion:** Do not promote the Audio model to the
   production `champion` alias without explicit user approval.
6. **Scope protection:** Do not modify Network, Text, Video, or unrelated
   frontend code as part of Audio implementation unless a separately
   scoped integration task requires it.
7. **Validation records:** Update this tracker after each completed stage
   and before starting the next one.

## Validation environment and preservation notes

- The standard Compose Audio API is exposed on host port `8003` under
  project `audio-compose-validation`.
- The Compose worker is internal to the project network; its container
  listens on port `8002`.
- The temporary Compose override is at
  `/tmp/audio-compose-validation.override.yml`.
- The temporary environment file is at
  `/tmp/audio-compose-validation.env`; it contains the service token and
  must not be printed or committed.
- The isolated MLflow test service is reachable at host port `5002` and
  Docker network name `audio-mlflow-isolated`.
- The existing production MLflow service remains on host port `5001`.
- Existing manual isolated containers and
  `audio-stage211-validation` containers were preserved.
- Do not stop, remove, recreate, or modify these containers, volumes,
  registries, or model aliases unless the user explicitly requests it.

## Final implementation audit — 2026-11-24

Final audit results:
- Full Audio test suite: **144 passed, 0 failed**.
- Python compilation: passed using `.venv312`.
- `git diff --check`: passed.
- No `TODO`, `FIXME`, `NotImplementedError`, or
  `IMPLEMENTATION REQUIRED` markers found in Audio Python files.
- One third-party Starlette/AnyIO deprecation warning remains:
  `anyio.abc.BlockingPortal` is deprecated. No Audio test failures
  resulted from this warning.
- The older `IMPLEMENTATION_PROGRESS.md.save` backup differs from the
  current tracker and has been preserved without modification.

Internship-scope completion assessment:
- Core Audio classification, preprocessing, inference, API, worker
  isolation, authentication, MLflow integration, and automated tests
  are implemented and validated.
- Isolated Compose API/worker inference and authenticated upload have
  been verified.
- The sigmoid calibration experiment was rejected; confidence scores
  remain explicitly uncalibrated.
- The implementation is suitable to be presented as an internship
  prototype within the documented ESC-50 and resource limitations.

Deferred work — not claimed as completed:
- Independent external-dataset evaluation.
- A validated probability-calibration method.
- Production-scale capacity and deployment testing.
- Production model promotion, which requires explicit approval.

No production model alias, registry, container, volume, or unrelated
Network, Text, Video, or frontend component was changed during this
final audit.
