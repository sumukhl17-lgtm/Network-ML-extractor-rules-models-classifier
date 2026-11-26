"""FastAPI application: wires Network, Text, Audio and Video classification, registry and ops endpoints.

Run from the repository root:   uvicorn backend.platform_api.main:app --reload --port 8000
Interactive API docs:           http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.text.api import router as text_router

from . import config
from .core.errors import AppError, code_for_status, error_body
from .core.metrics import metrics
from .routers import audio, health, network, registry, video

log = logging.getLogger("app")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Unified Network, Text, Audio & Video Classification API",
        version="0.1.0",
        description=(
            "Network flow classification with rule-first advisory ML; "
            "document classification with TF-IDF and Linear SVM; "
            "audio classification through the dedicated Audio service; "
            "and video classification using pretrained CLIP."
        ),
    )

    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins(),
                       allow_methods=["*"], allow_headers=["*"])

    # ---- one error format for everything ------------------------------------------
    @app.exception_handler(AppError)
    async def on_app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        # Normalize both string details and structured reports (e.g. text /health)
        # without flattening dictionaries into Python repr strings.
        if isinstance(exc.detail, dict):
            detail = dict(exc.detail)
            code = str(detail.pop("code", code_for_status(exc.status_code)))
            message = str(detail.pop("message", detail.pop("error", "Request could not be completed")))
            return JSONResponse(status_code=exc.status_code,
                                content=error_body(code, message, detail))
        return JSONResponse(status_code=exc.status_code,
                            content=error_body(code_for_status(exc.status_code), str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        problems = [{"field": ".".join(str(p) for p in e.get("loc", [])), "problem": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content=error_body("INVALID_REQUEST", "The request is invalid",
                                                                {"problems": problems}))

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=error_body("INTERNAL_ERROR", "Unexpected server error"))

    # ---- request metrics ---------------------------------------------------------------
    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "unmatched"   # route template, not the raw URL
            metrics.observe_request(request.method, path, status, time.perf_counter() - start)

    app.include_router(health.router)
    app.include_router(network.router)
    app.include_router(text_router)
    app.include_router(registry.router)
    app.include_router(video.router)
    app.include_router(audio.router)
    return app


app = create_app()
