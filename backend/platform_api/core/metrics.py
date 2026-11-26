"""A tiny Prometheus metrics registry (standard library only).

Prometheus scrapes GET /metrics, a plain-text page in a fixed format. For three
metric types that format is simple enough to write by hand, which keeps this
testable without extra packages. It can be swapped for `prometheus_client`
later without changing the metric names (Person 4's dashboards keep working).

Metrics:
  app_http_requests_total{method,path,status}     counter  -> request rate and error rate
  app_http_request_duration_seconds{method,path}  histogram -> latency
  app_classifications_total{branch,source}        counter  -> how many answers came from rules vs ML
"""
from __future__ import annotations

import threading
from typing import Dict, List, Tuple

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _labels(pairs: Tuple[Tuple[str, str], ...]) -> str:
    if not pairs:
        return ""
    body = ",".join('{}="{}"'.format(k, str(v).replace("\\", "\\\\").replace('"', '\\"')) for k, v in pairs)
    return "{" + body + "}"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[Tuple, int] = {}
        self._latency: Dict[Tuple, List[float]] = {}   # key -> [bucket counts..., sum, count]
        self._classifications: Dict[Tuple, int] = {}

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            key = (("method", method), ("path", path), ("status", str(status)))
            self._requests[key] = self._requests.get(key, 0) + 1
            lkey = (("method", method), ("path", path))
            row = self._latency.setdefault(lkey, [0.0] * (len(BUCKETS) + 2))
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    row[i] += 1
            row[-2] += seconds
            row[-1] += 1

    def count_classifications(self, branch: str, source: str, n: int = 1) -> None:
        with self._lock:
            key = (("branch", branch), ("source", source))
            self._classifications[key] = self._classifications.get(key, 0) + n

    def render(self) -> str:
        lines: List[str] = []
        with self._lock:
            lines += ["# HELP app_http_requests_total HTTP requests by method, route and status.",
                      "# TYPE app_http_requests_total counter"]
            for key, value in sorted(self._requests.items()):
                lines.append("app_http_requests_total{} {}".format(_labels(key), value))
            lines += ["# HELP app_http_request_duration_seconds Request latency in seconds.",
                      "# TYPE app_http_request_duration_seconds histogram"]
            for key, row in sorted(self._latency.items()):
                for i, bound in enumerate(BUCKETS):
                    lines.append("app_http_request_duration_seconds_bucket{} {}".format(
                        _labels(key + (("le", repr(bound)),)), int(row[i])))
                lines.append("app_http_request_duration_seconds_bucket{} {}".format(
                    _labels(key + (("le", "+Inf"),)), int(row[-1])))
                lines.append("app_http_request_duration_seconds_sum{} {}".format(_labels(key), row[-2]))
                lines.append("app_http_request_duration_seconds_count{} {}".format(_labels(key), int(row[-1])))
            lines += ["# HELP app_classifications_total Classifications by branch and answer source.",
                      "# TYPE app_classifications_total counter"]
            for key, value in sorted(self._classifications.items()):
                lines.append("app_classifications_total{} {}".format(_labels(key), value))
        return "\n".join(lines) + "\n"


metrics = Metrics()
