"""Extract bidirectional flow features from a PCAP with NFStream."""

import argparse

import pandas as pd


OUTPUT_COLUMNS = [
    "src_port",
    "dst_port",
    "protocol",
    "packet_count",
    "total_bytes",
    "avg_packet_size",
    "duration",
    "packet_rate",
    "byte_rate",
    "syn",
    "fin",
    "rst",
    "psh",
    "ack",
    "application_name",
    "application_is_guessed",
]


def extract(pcap_path):
    """Return the requested NFStream features for each flow in ``pcap_path``."""
    from nfstream import NFStreamer

    flows = NFStreamer(
        source=pcap_path,
        statistical_analysis=True,
    ).to_pandas()

    columns = {
        "bidirectional_packets": "packet_count",
        "bidirectional_bytes": "total_bytes",
        "bidirectional_mean_ps": "avg_packet_size",
        "bidirectional_duration_ms": "duration",
        "bidirectional_syn_packets": "syn",
        "bidirectional_fin_packets": "fin",
        "bidirectional_rst_packets": "rst",
        "bidirectional_psh_packets": "psh",
        "bidirectional_ack_packets": "ack",
    }
    result = flows.loc[
        :,
        [
            "src_port",
            "dst_port",
            "protocol",
            *columns,
            "application_name",
            "application_is_guessed",
        ],
    ].rename(columns=columns)

    result["duration"] = (result["duration"] / 1000).clip(lower=0.001)
    result["packet_rate"] = result["packet_count"] / result["duration"]
    result["byte_rate"] = result["total_bytes"] / result["duration"]
    return result.loc[:, OUTPUT_COLUMNS]


def make_label(df):
    """Add Known/Unknown labels and remove flows with guessed applications."""
    result = df.copy()
    unknown = (
        result["application_name"]
        .astype("string")
        .str.startswith("Unknown", na=False)
    )
    result["label"] = unknown.map({True: "Unknown", False: "Known"})
    guessed = result["application_is_guessed"].eq(1).fillna(False)
    return result.loc[~guessed].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap_path", help="input PCAP file")
    parser.add_argument("csv_path", help="output CSV file")
    args = parser.parse_args()

    labeled_flows = make_label(extract(args.pcap_path))
    labeled_flows.to_csv(args.csv_path, index=False)
    print(labeled_flows["label"].value_counts())


if __name__ == "__main__":
    main()
