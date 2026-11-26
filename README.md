# Network ML — Unified Classification Platform

A multi-modal classification platform integrating Network Traffic, Text,
Audio and Video classification behind a unified FastAPI backend and React
frontend.

The repository is designed to be cloned and deployed as a complete
Docker Compose application. A teammate can use a clean clone to validate
the deployment and prepare a portable VM for handover.

## 1. Platform architecture

| Module | Implementation | Deployment |
|---|---|---|
| Network | Protocol rules with advisory Random Forest / XGBoost | Unified backend |
| Text | TF-IDF and Linear SVM document classification | Unified backend |
| Audio | YAMNet embeddings and trained classifier | Audio API and worker |
| Video | FFmpeg frame extraction and pretrained CLIP zero-shot classification | Video API |
| Frontend | React and Vite, served by Nginx | Frontend container |
| Model tracking | MLflow | Shared MLflow service |
| Document extraction | Apache Tika | Tika container |

Audio and Video run in separate containers to isolate their heavyweight
inference dependencies from the unified backend.

The default Docker Compose deployment starts all platform services.
Audio and Video are not optional Compose profiles.

## 2. Requirements

- Git
- Docker Engine with Docker Compose v2
- Internet access for initial image builds and pretrained model downloads
- Sufficient memory and CPU for the inference containers

Python 3.12 and Node.js/npm are needed only for local development and tests.
They are not required to run the Docker deployment.

### Resource configuration

The default Compose configuration sets:

- Video API: 4 GiB memory and 2 CPU quota
- Audio worker: 1 GiB memory and 1 CPU quota

These are initial deployment limits, not production capacity guarantees.
Docker Desktop must also have sufficient memory and CPU allocated to its VM.

## 3. First-time setup

Clone the repository and enter its root directory:

```bash
git clone https://github.com/sumukhl17-lgtm/Network-ML-extractor-rules-models-classifier.git
cd network-ml
```

Create the local environment file:

```bash
cp .env.example .env
```

Generate a private token:

```bash
openssl rand -hex 32
```

Set the generated value as `AUDIO_SERVICE_TOKEN` in `.env`.
Do not commit `.env` or share its contents. The token is used for
authenticated communication between the platform backend and Audio API.

The supplied `.env.example` contains the default ports, upload limits,
resource settings and pinned Tika image configuration. Review it before
changing deployment settings.

## 4. Start the complete platform

From the repository root:

```bash
docker compose up --build -d
```

This starts:

- Unified FastAPI backend
- React frontend served through Nginx
- MLflow tracking server
- Apache Tika document extraction service
- Audio API
- Audio inference worker
- Video API

Check service status:

```bash
docker compose ps
```

Check the unified backend readiness:

```bash
curl http://localhost:8000/ready
```

The readiness response reports the status of the Network, Text, Audio
and Video integrations. Wait until all four checks report healthy.

Access the platform:

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| Interactive API documentation | http://localhost:8000/docs |
| Backend readiness | http://localhost:8000/ready |
| MLflow | http://localhost:5001 |

The Audio worker and Video API are internal Compose services. They are
not intended to be accessed directly from the host.

Stop the platform without deleting its persistent volumes:

```bash
docker compose down
```

To rebuild after source changes:

```bash
docker compose up --build -d
```

Do not use `docker compose down -v` unless you intentionally want to
delete persistent application and model data.

## 5. Classifier API endpoints

The unified backend exposes all four classification routes:

| Classifier | Method and endpoint | Frontend proxy endpoint |
|---|---|---|
| Network | `POST /classify/network` | `POST /api/classify/network` |
| Text | `POST /classify/text` | `POST /api/classify/text` |
| Audio | `POST /classify/audio` | `POST /api/classify/audio` |
| Video | `POST /classify/video` | `POST /api/classify/video` |

The frontend proxy routes requests through Nginx to the unified backend.
The backend communicates with the isolated Audio and Video services
over the internal Docker network.

Use `http://localhost:8000/docs` to inspect the API schemas and upload
requirements.

## 6. Audio classification

Audio inference uses a dedicated API and worker. The worker loads the
included model bundle by default:

```text
audio_service/model_bundle/
```

The bundle contains the model, MLflow model metadata, dependency
information and a SHA-256 checksum manifest.

The default configuration is:

```text
AUDIO_MODEL_SOURCE=bundle
```

An MLflow registry source can be configured separately using
`AUDIO_MODEL_SOURCE=registry`, subject to the required model and registry
configuration.

Audio implementation and resource notes:

- `audio_service/IMPLEMENTATION_PROGRESS.md`
- `audio_service/RESOURCE_VALIDATION.md`

The ESC-50 dataset has non-commercial licensing restrictions. Review
its license and all applicable model and dependency licenses before
redistribution or commercial use.

## 7. Video classification

Video inference runs in the dedicated Video API container.

The implementation:

- Accepts MP4, AVI, MOV, MKV and WEBM uploads.
- Enforces upload and duration limits.
- Uses FFmpeg/FFprobe to extract a bounded set of frames.
- Applies pretrained `openai/clip-vit-base-patch32` zero-shot classification.
- Uses a fixed category prompt set and an unknown threshold.

The pretrained model may need to be downloaded during initial
initialization. Internet access and sufficient storage are required.

**Video model limitation:** this is a pretrained zero-shot baseline.
Its category-level performance has not yet been validated on a
representative, held-out video dataset. CLIP softmax confidence is not
a calibrated probability of correctness. Do not interpret the output
as validated accuracy or production-grade classification.

The Video service does not currently combine predictions from the
Audio classifier. Detecting an audio stream in a video does not mean
audio-assisted classification is performed.

See `backend/video/README.md` for Video implementation details and
limitations.

## 8. Run the test suites

Use Python 3.12 for the integrated Python test suites.

Create and activate an environment:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

Install the unified backend dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

Run the backend, Video and Audio tests:

```bash
python -m pytest \
  backend/platform_api/tests \
  backend/video/tests \
  audio_service/tests \
  -q
```

Run frontend tests:

```bash
npm test --prefix frontend
```

Build the frontend:

```bash
npm run build --prefix frontend
```

Compile the backend and service modules:

```bash
python -m compileall -q \
  backend/platform_api backend/video audio_service video_service
```

Validate the Compose configuration:

```bash
docker compose config -q
```

The Audio and Video inference stacks have separate dependency
requirements. Do not install their heavyweight dependencies into the
unified backend environment unless intentionally developing those
services.

## 9. Local development

### Backend

Install the shared backend dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

Start the backend locally:

```bash
uvicorn backend.platform_api.main:app --reload --port 8000
```

A complete local backend run may also require the configured MLflow,
Tika, Audio and Video services. Docker Compose is the recommended
integrated development and verification environment.

### Frontend

Install dependencies and start the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

The frontend `.env.example` documents the local API URL setting.
The Docker deployment builds the frontend with its `/api` proxy
configuration.

### Dependency files

Use the dependency file belonging to the component being developed:

- `backend/requirements.txt` — shared platform backend
- `backend/video-requirements.txt` — Video inference dependencies
- `audio_service/requirements.txt` — Audio inference dependencies
- `backend/text/requirements.txt` — standalone Text component environment,
  not the integrated shared backend environment

The root `requirements.txt` is not used by the Dockerfiles for the
integrated deployment. Do not use it as a substitute for the component
dependency files above.

## 10. Network classifier

The Network classifier combines protocol rules with an advisory
machine-learning model.

Recognised protocols are labelled by port rules. Unmatched flows
default to `Unknown`. A Random Forest / XGBoost model trained on
nDPI-labelled flow statistics provides an advisory prediction.

The original contract is:

```python
from classify_network import classify_network

classify_network(flow)
```

`flow` is a dictionary containing:

- `protocol`
- `src_port`
- `dst_port`
- `packet_count`
- `total_bytes`
- `avg_packet_size`
- `duration`
- `packet_rate`
- `byte_rate`
- `syn`, `fin`, `rst`, `psh`, `ack`

Rule matches return a rule-based label. Unmatched flows default to
`Unknown`, with the ML prediction included as advisory information.

### Network API smoke tests

The repository includes two small synthetic PCAP fixtures for testing the integrated Network API. They are test data, not real network captures.

Start the platform with `docker compose up --build -d` and wait for `/ready` to report healthy services.

Upload the HTTP fixture:

```bash
curl -sS -f -F "file=@backend/platform_api/tests/fixtures/network/test_network.pcap" http://localhost:8000/classify/network
```

Upload the unmatched-port fixture:

```bash
curl -sS -f -F "file=@backend/platform_api/tests/fixtures/network/test_unknown_network.pcap" http://localhost:8000/classify/network
```

The first fixture should classify as HTTP and the second as Unknown. These synthetic fixtures verify API behaviour, not real-world classification accuracy. Scapy is only needed to regenerate the fixtures, not to run the API.

### Network pipeline

1. `run_capture.sh` captures traffic on macOS using `tcpdump` on `en0`.
2. `extract_flows.py` extracts NFStream flows and uses nDPI application
   names as labels; guessed flows are dropped.
3. `build_dataset.py` creates train/test splits by capture file.
4. `train_compare.py` trains Random Forest and XGBoost models and logs
   results to MLflow.

The shipped model excludes potentially leaky features:

```bash
python train_compare.py --exclude-features protocol,syn,fin,rst,psh,ack
```

### Network evaluation findings

- On a held-out capture from the same synthetic odd-traffic generator,
  F1 was approximately 0.97 for both models.
- A different, slower generator did not generalise: Unknown recall was
  approximately zero for models trained on the first generator, and
  vice versa.
- Known-only novelty detection also produced approximately zero Unknown
  recall.
- Confidence analysis showed confidently incorrect predictions.
- Rules-only classification matched or outperformed rules plus ML on
  these captures.
- The Unknown class is synthetic random UDP traffic on random ports.
  These scores are not evidence of real-world network generalisation.

Network capture data and generated CSV files are not committed because
they contain traffic data. Regenerate them using the capture and
dataset pipeline.

Saved model compatibility depends on the relevant scikit-learn version.
Retrain models when the runtime version or feature pipeline changes.

## 11. Configuration and operational notes

Compose services support environment-variable overrides for ports,
upload limits, resource limits and service settings. Review
`docker-compose.yml` and `.env.example` before changing deployment
configuration.

The unified backend readiness endpoint checks all four classifier
integrations: Network, Text, Audio and Video.

Persistent Docker volumes store application data and model caches.
The default deployment includes:

- Backend application data
- Video model cache
- Audio model cache
- MLflow data

Do not expose internal worker services or MLflow publicly without
appropriate authentication, network restrictions and deployment
security.

For a portable VM, the teammate preparing the VM should:

1. Clone the complete GitHub repository into the VM.
2. Install Docker Engine and the Compose v2 plugin.
3. Create `.env` from `.env.example` and generate a new private
   `AUDIO_SERVICE_TOKEN`.
4. Run `docker compose up --build -d`.
5. Wait for all services to become healthy.
6. Verify `/ready`, the frontend and all four classifier endpoints.
7. Document the VM resource allocation, startup procedure and
   any model downloads required on first launch.

The VM should be built and tested from a clean clone, not from
untracked files or local-only configuration on the development machine.

## 12. Repository structure

```text
backend/
  platform_api/       Unified FastAPI application and API routes
  text/               Text classification implementation
  video/              Video classification implementation
  requirements.txt    Shared backend dependencies
  video-requirements.txt
audio_service/        Audio API, worker, model bundle and tests
video_service/        Video API container entry point
frontend/             React/Vite application and Nginx configuration
mlflow/               MLflow container configuration
datasets/             Dataset assets and license notices
docker-compose.yml    Complete multi-service deployment
.env.example          Documented local deployment configuration
```

## 13. Licensing

See the root `LICENSE` file and the individual dataset, model and
dependency license notices.

The root project license does not override third-party model, dataset
or dependency licenses. In particular, review the ESC-50 dataset's
non-commercial restriction before any commercial use.
