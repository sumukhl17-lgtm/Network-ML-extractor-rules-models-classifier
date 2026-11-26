from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..core import activity
from ..core.errors import AppError
from ..core.registry_service import RegistryService
from ..schemas import PromoteRequest, RollbackRequest

router = APIRouter(prefix="/registry", tags=["registry"])

# One service object; tests replace it with a fake-client version via set_service().
_service = RegistryService()


def set_service(service: RegistryService) -> None:
    global _service
    _service = service


@router.get("/models")
def list_models():
    """Registered models, their versions and metrics, which version is champion, what is live."""
    return _service.list_models()


@router.get("/models/{name}/compare")
def compare(name: str, versions: str = Query(..., description="comma separated, e.g. 1,2")):
    wanted = [v.strip() for v in versions.split(",") if v.strip()]
    return _service.compare(name, wanted)


@router.post("/models/{name}/promote")
def promote(name: str, body: PromoteRequest):
    return _service.promote(name, body.version, body.actor, body.confirm, body.reload)


@router.post("/models/{name}/rollback")
def rollback(name: str, body: RollbackRequest):
    return _service.rollback(name, body.actor, body.confirm, body.to_version, body.reload)


@router.post("/models/{name}/reload")
def reload(name: str):
    return _service.reload(name)


@router.get("/activity")
def activity_log(model: Optional[str] = None, limit: int = Query(100, ge=1, le=1000)):
    """Who promoted / rolled back which version, newest first."""
    return {"entries": activity.read(model=model, limit=limit)}


@router.get("/network/runs")
def network_runs(limit: int = Query(50, ge=1, le=200)):
    """Read-only: MLflow runs of the network experiment (the model is not registered yet)."""
    try:
        return _service.network_runs(limit)
    except AppError:
        raise
    except Exception as e:
        raise AppError(503, "REGISTRY_UNAVAILABLE", "Could not read runs from MLflow: {}".format(e)) from e
