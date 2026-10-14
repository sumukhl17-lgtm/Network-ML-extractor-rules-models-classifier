"""Port-based flow classification rules."""

from typing import Optional


PORT_LABELS = {
    80: "HTTP",
    443: "HTTPS",
    53: "DNS",
    22: "SSH",
}


def classify_by_rule(flow: dict) -> Optional[str]:
    """Classify a flow when either endpoint uses a recognized port."""
    for port in (flow.get("src_port"), flow.get("dst_port")):
        if port in PORT_LABELS:
            return PORT_LABELS[port]
    return None
