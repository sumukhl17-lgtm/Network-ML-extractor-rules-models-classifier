from __future__ import annotations

import asyncio
import os

import httpx
from fastapi import APIRouter, Response

from ..core import network_adapter
from ..core.errors import AppError
from ..core.metrics import metrics

router = APIRouter(tags=["ops"])


@router.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


async def _get_service_status(url: str, path: str):
    """Fetch a dependent service's status without exposing internal errors."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(2.5, connect=0.8)
        ) as client:
            result = await client.get(f"{url.rstrip('/')}{path}")

        try:
            payload = result.json()
        except ValueError:
            payload = None

        return result.status_code, payload

    except httpx.HTTPError:
        return None, None


@router.get("/ready")
async def ready(response: Response):
    """Readiness: report whether all four classifiers can accept requests."""
    checks = {}

    try:
        network_adapter.model_features()
        checks["network"] = {"ok": True}
    except AppError as exc:
        checks["network"] = {"ok": False, "error": exc.message}
    except Exception:
        checks["network"] = {
            "ok": False,
            "error": "Network classifier is unavailable",
        }

    try:
        from backend import text

        info = text.model_info()
        checks["text"] = {
            "ok": True,
            "model_source": info.get("source"),
            "model_version": info.get("version"),
        }
    except Exception:
        checks["text"] = {
            "ok": False,
            "error": "Text classifier is unavailable",
        }

    video_url = os.getenv(
        "VIDEO_SERVICE_URL", "http://video-api:8003"
    )
    audio_url = os.getenv(
        "AUDIO_SERVICE_URL", "http://audio-api:8001"
    )

    (video_result, audio_result) = await asyncio.gather(
        _get_service_status(video_url, "/classify/video/health"),
        _get_service_status(audio_url, "/ready"),
    )
    video_status, video_payload = video_result
    audio_status, audio_payload = audio_result

    video_ok = (
        video_status == 200
        and isinstance(video_payload, dict)
        and video_payload.get("status") == "ok"
    )

    if video_ok:
        model = video_payload.get("model")
        model = model if isinstance(model, dict) else {}
        checks["video"] = {
            "ok": True,
            "model_loaded": bool(model.get("loaded", False)),
            "model_version": model.get("version"),
        }
    else:
        checks["video"] = {
            "ok": False,
            "error": "Video classification service is unavailable",
        }

    audio_model = (
        audio_payload.get("model")
        if isinstance(audio_payload, dict)
        else {}
    )
    audio_model = audio_model if isinstance(audio_model, dict) else {}

    audio_loaded = bool(audio_model.get("loaded", False))
    audio_ok = (
        audio_status == 200
        and isinstance(audio_payload, dict)
        and audio_payload.get("status") == "ready"
        and audio_loaded
    )

    if audio_ok:
        checks["audio"] = {
            "ok": True,
            "model_loaded": True,
            "model_version": audio_model.get("version"),
        }
    else:
        checks["audio"] = {
            "ok": False,
            "model_loaded": audio_loaded,
            "error": "Audio classification service is not ready",
        }

    ok = all(check["ok"] for check in checks.values())

    if not ok:
        response.status_code = 503

    return {
        "status": "ready" if ok else "not_ready",
        "checks": checks,
    }


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
