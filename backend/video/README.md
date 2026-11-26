# Video classification branch

This branch is integrated into the unified FastAPI application at `POST /classify/video`.
It samples a bounded number of frames with FFmpeg and applies pretrained CLIP zero-shot
classification against the seven specified labels. Low top-category probability returns
`unknown`. The confidence is CLIP's softmax probability over this fixed prompt set; it is
not a calibrated probability of correctness.

## System requirements

Install FFmpeg and FFprobe and ensure both commands are on `PATH`. For a standalone
Python installation, install dependencies from `backend/video-requirements.txt`.
The integrated Docker deployment installs these dependencies in the Video API image. The first inference downloads the configured
Hugging Face CLIP checkpoint into `VIDEO_MODEL_CACHE` (default: `backend/video/model_cache`).

## Configuration

- `VIDEO_MAX_BYTES` (default 200 MiB)
- `VIDEO_MAX_DURATION_SECONDS` (default 600)
- `VIDEO_MAX_FRAMES` (default 12)
- `VIDEO_SAMPLE_INTERVAL_SECONDS` (default 2)
- `VIDEO_UNKNOWN_THRESHOLD` (default 0.22; requires validation/tuning)
- `VIDEO_CLIP_MODEL` (default `openai/clip-vit-base-patch32`)
- `VIDEO_MODEL_CACHE`

## Audio bridge

The integrated platform runs Audio and Video classification as separate services. Video
detects whether an audio stream exists, but it does not call the Audio classifier or fuse
Audio predictions. Therefore `audio_assisted` remains false and video classification
is visual-only.

## Validation status

This is a pretrained zero-shot baseline, not a dataset-trained video classifier. Do not
claim category accuracy until evaluated on a representative, held-out video dataset. Model
weights are not committed. Check the Hugging Face model card and applicable licence/terms
before redistribution or deployment.

## MLflow champion configuration

Video uses the shared MLflow registry name `video-classifier` and alias `champion`.
The registered version stores a small configuration (CLIP model ID, category prompts,
and threshold metadata); the large pretrained weights are cached by Hugging Face and
are not copied into MLflow or committed to the repository.

Set `MLFLOW_TRACKING_URI` and run from `backend/`:

```bash
python -m video.register_model
```

The integrated Docker Compose deployment sets `VIDEO_MODEL_SOURCE=pretrained`, which
loads the configured Hugging Face CLIP checkpoint directly. Outside Compose, the
application configuration defaults to `auto`, which tries the registry and falls back
to the pretrained checkpoint when the registry is unavailable. Set
`VIDEO_MODEL_SOURCE=registry` to require a champion and fail clearly if one is absent.
After promoting a different champion, call the shared model reload endpoint for
`video-classifier` so the cached CLIP runtime is refreshed.
