import struct
import sys
import types

import pandas as pd

from extract_flows import OUTPUT_COLUMNS, extract, make_label


def write_tiny_pcap(path):
    global_header = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,
        2,
        4,
        0,
        0,
        65535,
        1,
    )
    ethernet = bytes.fromhex("00112233445566778899aabb0800")
    ipv4 = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        40,
        1,
        0,
        64,
        6,
        0,
        b"\xc0\xa8\x00\x01",
        b"\xc0\xa8\x00\x02",
    )
    tcp = struct.pack("!HHIIHHHH", 1234, 80, 0, 0, 0x5002, 65535, 0, 0)
    packet = ethernet + ipv4 + tcp
    record_header = struct.pack("<IIII", 1, 0, len(packet), len(packet))
    path.write_bytes(global_header + record_header + packet)


def test_extract_and_make_label_from_tiny_pcap(tmp_path, monkeypatch):
    pcap_path = tmp_path / "tiny.pcap"
    write_tiny_pcap(pcap_path)

    flows = pd.DataFrame(
        {
            "src_port": [1234, 2345, 3456],
            "dst_port": [80, 443, 53],
            "protocol": [6, 6, 17],
            "bidirectional_packets": [4, 2, 1],
            "bidirectional_bytes": [400, 200, 64],
            "bidirectional_mean_ps": [100.0, 100.0, 64.0],
            "bidirectional_duration_ms": [2000, 0, 500],
            "bidirectional_syn_packets": [1, 1, 0],
            "bidirectional_fin_packets": [0, 0, 0],
            "bidirectional_rst_packets": [0, 0, 0],
            "bidirectional_psh_packets": [2, 0, 0],
            "bidirectional_ack_packets": [4, 2, 0],
            "application_name": ["HTTPS", "Unknown protocol", "DNS"],
            "application_is_guessed": [0, 0, 1],
        }
    )
    calls = {}

    class FakeNFStreamer:
        def __init__(self, *, source, statistical_analysis):
            calls["source"] = source
            calls["statistical_analysis"] = statistical_analysis

        def to_pandas(self):
            return flows

    monkeypatch.setitem(
        sys.modules,
        "nfstream",
        types.SimpleNamespace(NFStreamer=FakeNFStreamer),
    )

    extracted = extract(str(pcap_path))
    assert calls == {
        "source": str(pcap_path),
        "statistical_analysis": True,
    }
    assert extracted.columns.tolist() == OUTPUT_COLUMNS
    assert extracted.loc[0, "duration"] == 2
    assert extracted.loc[0, "packet_rate"] == 2
    assert extracted.loc[0, "byte_rate"] == 200
    assert extracted.loc[1, "duration"] == 0.001

    labeled = make_label(extracted)
    assert labeled["label"].tolist() == ["Known", "Unknown"]
    assert labeled["application_name"].tolist() == ["HTTPS", "Unknown protocol"]
