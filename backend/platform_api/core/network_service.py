"""Framework-free logic for POST /classify/network.

Turns an upload (PCAP or CSV of flows) into a list of flow dicts, runs Sumukh's
classify_network on each, and returns results in one stable shape.

Honest-by-design: the deck shows the ML layer does not generalise (Unknown recall
0.0 on traffic from a different generator). So any answer that comes from the model
is flagged `advisory: true`; the rule-based answer is the one to trust.
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional

from .. import config
from . import network_adapter as adapter
from .errors import AppError

PCAP_MAGICS = (
    b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",   # classic pcap (little / big endian)
    b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d",   # pcap with nanosecond timestamps
    b"\x0a\x0d\x0d\x0a",                          # pcapng
)
PORT_FIELDS = ("src_port", "dst_port")
# Extra fields shown in the table when present (never sent to the model by us).
INFO_FIELDS = ("protocol", "packet_count", "total_bytes", "duration")


def detect_kind(filename: Optional[str], data: bytes) -> str:
    """Return 'pcap' or 'csv'. Raises AppError for anything else."""
    ext = os.path.splitext((filename or "").lower())[1]
    if ext in (".ipfix", ".nf", ".netflow"):
        raise AppError(415, "UNSUPPORTED_FILE_TYPE",
                       "IPFIX / NetFlow files are not supported yet: the network branch only has a PCAP extractor. "
                       "Upload a .pcap or a CSV of flows.", {"received": ext})
    if data[:4] in PCAP_MAGICS:
        return "pcap"
    if ext in (".pcap", ".pcapng", ".cap"):
        raise AppError(422, "INVALID_PCAP", "The file has a PCAP extension but is not a valid PCAP / PCAPNG capture")
    if ext == ".csv" or (ext == "" and b"," in data[:2000]):
        if b"\x00" in data[:4096]:
            raise AppError(422, "INVALID_CSV", "The file looks binary, not a CSV")
        return "csv"
    raise AppError(415, "UNSUPPORTED_FILE_TYPE", "Upload a .pcap / .pcapng capture or a .csv of flows",
                   {"received": ext or "no extension"})


def _to_number(value: Any, row: int, column: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AppError(422, "INVALID_FLOW_VALUE",
                       "Row {}: column '{}' is not a number: {!r}".format(row, column, value),
                       {"row": row, "column": column}) from None
    if number != number or number in (float("inf"), float("-inf")):
        raise AppError(422, "INVALID_FLOW_VALUE", "Row {}: column '{}' is not a finite number".format(row, column),
                       {"row": row, "column": column})
    return number


def required_columns() -> List[str]:
    return list(PORT_FIELDS) + adapter.model_features()


def flows_from_csv(data: bytes) -> List[Dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AppError(422, "INVALID_CSV", "The CSV is not valid UTF-8 text") from None
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    if not header:
        raise AppError(422, "INVALID_CSV", "The CSV has no header row")
    reader.fieldnames = header
    needed = required_columns()
    missing = [c for c in needed if c not in header]
    if missing:
        raise AppError(422, "MISSING_COLUMNS", "The CSV is missing required column(s): {}".format(", ".join(missing)),
                       {"missing": missing, "required": needed, "found": header})
    flows: List[Dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):  # row 1 is the header
        if len(flows) >= config.max_flows():
            raise AppError(413, "TOO_MANY_FLOWS", "More than {} flows in one request".format(config.max_flows()),
                           {"max_flows": config.max_flows()})
        flow: Dict[str, Any] = {}
        for column in needed:
            number = _to_number(row.get(column), row_number, column)
            flow[column] = int(number) if column in PORT_FIELDS else number
        for column in INFO_FIELDS:
            if column in row and column not in flow and row[column] not in (None, ""):
                try:
                    flow[column] = float(row[column])
                except ValueError:
                    pass
        flows.append(flow)
    if not flows:
        raise AppError(422, "NO_FLOWS", "The CSV has a header but no flows")
    return flows


def flows_from_pcap(data: bytes) -> List[Dict[str, Any]]:
    fd, path = tempfile.mkstemp(suffix=".pcap")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            frame = adapter.extract_pcap(path)
        except AppError:
            raise
        except Exception as e:
            raise AppError(422, "PCAP_EXTRACTION_FAILED", "Could not read flows from the capture: {}".format(e)) from e
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if len(frame) == 0:
        raise AppError(422, "NO_FLOWS", "No flows were found in the capture")
    if len(frame) > config.max_flows():
        raise AppError(413, "TOO_MANY_FLOWS", "{} flows is more than the limit of {}".format(len(frame), config.max_flows()),
                       {"max_flows": config.max_flows()})
    needed = required_columns()
    flows = []
    for record in frame.to_dict(orient="records"):
        flow = {c: record[c] for c in needed}
        for column in INFO_FIELDS:
            if column in record:
                flow[column] = record[column]
        for column in PORT_FIELDS:
            flow[column] = int(flow[column])
        flows.append(flow)
    return flows


def classify_flows(flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run classify_network per flow and normalise the three possible output shapes."""
    results = []
    for index, flow in enumerate(flows):
        raw = adapter.classify(flow)
        source = raw.get("source")
        label = raw["label"]
        item: Dict[str, Any] = {
            "index": index,
            "kind": "network",
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "label": label,
            "is_unknown": str(label).lower() == "unknown",
            "source": source,                       # rule | default | ml
            "confidence": raw.get("confidence"),    # only when source == "ml"
            "advisory": source in ("default", "ml") and ("ml_label" in raw or "confidence" in raw),
            "ml_label": raw.get("ml_label"),
            "ml_confidence": raw.get("ml_confidence"),
        }
        for column in INFO_FIELDS:
            item[column] = flow.get(column)
        results.append(item)
    return results


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(r["label"] for r in results)
    sources = Counter(r["source"] for r in results)
    return {
        "n_flows": len(results),
        "by_label": dict(labels),
        "by_source": dict(sources),
        "n_unknown": sum(1 for r in results if r["is_unknown"]),
        "n_advisory": sum(1 for r in results if r["advisory"]),
    }


def classify_upload(filename: Optional[str], data: bytes) -> Dict[str, Any]:
    """Validate an upload, extract flows, classify them. Raises AppError."""
    if not data:
        raise AppError(400, "EMPTY_FILE", "The uploaded file is empty")
    if len(data) > config.max_upload_bytes():
        raise AppError(413, "FILE_TOO_LARGE", "File is larger than {} MB".format(config.max_upload_bytes() // (1024 * 1024)),
                       {"max_bytes": config.max_upload_bytes()})
    kind = detect_kind(filename, data)
    flows = flows_from_pcap(data) if kind == "pcap" else flows_from_csv(data)
    results = classify_flows(flows)
    return {
        "filename": filename or "",
        "input_kind": kind,
        "mode": "ml_overrides_unmatched" if adapter.ml_overrides_unmatched() else "rules_with_advisory_ml",
        "summary": summarise(results),
        "results": results,
    }
