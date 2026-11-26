"""One error format for every endpoint (network, text, registry).

Every error the API returns looks like:

    {"error": {"code": "EMPTY_FILE", "message": "The uploaded file is empty", "details": {...}}}

`code` is stable and meant for programs (the React UI switches on it); `message`
is meant for people.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def body(self) -> Dict[str, Any]:
        return error_body(self.code, self.message, self.details)


def error_body(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


# The text branch raises HTTPException(status, detail=str). We cannot edit that
# package, so main.py maps its status codes onto the same {code, message} format.
_STATUS_CODES = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "FILE_TOO_LARGE",
    415: "UNSUPPORTED_FILE_TYPE",
    422: "UNPROCESSABLE",
    503: "SERVICE_UNAVAILABLE",
}


def code_for_status(status: int) -> str:
    return _STATUS_CODES.get(status, "INTERNAL_ERROR" if status >= 500 else "ERROR")
