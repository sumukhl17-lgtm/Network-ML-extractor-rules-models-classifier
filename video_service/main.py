"""Standalone FastAPI application for the isolated Video classifier."""

from fastapi import FastAPI

from backend.video.api import router

app = FastAPI(
    title="Video Classification Service",
    description="Isolated CLIP-based video classification service",
    version="1.0.0",
)

app.include_router(router)
