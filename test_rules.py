import pytest

from rules import classify_by_rule


@pytest.mark.parametrize(
    ("port", "expected"),
    [(80, "HTTP"), (443, "HTTPS"), (53, "DNS"), (22, "SSH")],
)
@pytest.mark.parametrize("port_field", ["src_port", "dst_port"])
def test_classifies_recognized_port_on_either_side(port, expected, port_field):
    flow = {"src_port": 12345, "dst_port": 12345}
    flow[port_field] = port

    assert classify_by_rule(flow) == expected


def test_returns_none_when_neither_port_matches():
    assert classify_by_rule({"src_port": 12345, "dst_port": 54321}) is None
