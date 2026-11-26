"""Pydantic response / request models. They document the API (see /docs) and validate output.

Python 3.9 compatible on purpose: Optional[...] / List[...] instead of `X | None`,
because pydantic evaluates these annotations at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorDetail


class NetworkFlowResult(BaseModel):
    index: int
    kind: str = "network"
    src_port: int
    dst_port: int
    label: str                          # HTTP | HTTPS | DNS | SSH | Unknown
    is_unknown: bool                    # absorbs the Unknown (network) / unknown (text) casing difference
    source: str                         # rule | default | ml
    confidence: Optional[float] = None  # only when source == "ml"
    advisory: bool                      # True = the answer comes from / is accompanied by the ML model
    ml_label: Optional[str] = None      # model's opinion when source == "default"
    ml_confidence: Optional[float] = None
    protocol: Optional[float] = None
    packet_count: Optional[float] = None
    total_bytes: Optional[float] = None
    duration: Optional[float] = None


class NetworkSummary(BaseModel):
    n_flows: int
    by_label: Dict[str, int]
    by_source: Dict[str, int]
    n_unknown: int
    n_advisory: int


class NetworkResponse(BaseModel):
    filename: str
    input_kind: str                     # pcap | csv
    mode: str                           # rules_with_advisory_ml | ml_overrides_unmatched
    summary: NetworkSummary
    results: List[NetworkFlowResult]


class ConfirmedAction(BaseModel):
    confirm: bool = False               # must be true, or the server refuses
    actor: str = "anonymous"            # who is doing it (shown in the activity log)
    reload: bool = True                 # reload the live model right after moving the alias


class PromoteRequest(ConfirmedAction):
    version: str


class RollbackRequest(ConfirmedAction):
    to_version: Optional[str] = None    # default: the champion before the last promotion
