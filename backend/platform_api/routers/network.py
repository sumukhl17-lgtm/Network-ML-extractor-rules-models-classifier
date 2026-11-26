from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from .. import config
from ..core import network_service
from ..core.metrics import metrics
from ..schemas import NetworkResponse

router = APIRouter(prefix="/classify/network", tags=["network"])


@router.post("", response_model=NetworkResponse)
def classify_network_upload(file: UploadFile = File(...)):
    """Upload a .pcap / .pcapng capture or a CSV of flows; get one result per flow.

    Plain `def` (not async): FastAPI runs it in a worker thread, so NFStream and the
    model call do not block the event loop.
    """
    data = file.file.read(config.max_upload_bytes() + 1)
    result = network_service.classify_upload(file.filename, data)
    for source, count in result["summary"]["by_source"].items():
        metrics.count_classifications("network", str(source), count)
    return result
