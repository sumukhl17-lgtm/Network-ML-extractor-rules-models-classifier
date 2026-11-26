"""Generate seeded, chunk-written flow CSVs for scale benchmarks.

The requested byte count is a minimum across train.csv and test.csv. Output may
exceed it by at most one generated row, except that at least 15 rows are emitted
so both splits contain both labels for the existing training workflow.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


FLOW_COLUMNS = [
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
    "label",
    "source_file",
]
INTEGER_COLUMNS = {
    "src_port",
    "dst_port",
    "protocol",
    "packet_count",
    "total_bytes",
    "syn",
    "fin",
    "rst",
    "psh",
    "ack",
    "application_is_guessed",
}
FLOAT_COLUMNS = {
    "avg_packet_size",
    "duration",
    "packet_rate",
    "byte_rate",
}
MINIMUM_ROWS = 15
APPLICATIONS = {
    "normal": ("TLS.Google", "HTTP.Apache", "DNS.Cloudflare", "SSH.OpenSSH"),
    "suspicious": ("Unknown.Proxy", "Unknown.Tunnel"),
    "anomalous": ("Unknown.Scanner", "Unknown.Bot"),
}
SERVICE_PORTS = (53, 80, 443, 22, 123, 443, 8080, 8443)


def _traffic_category(index: int, rng: random.Random, block: List[str]) -> str:
    if index % 4 == 0:
        block[:] = ["normal", "normal", "suspicious", "anomalous"]
        rng.shuffle(block)
    return block[index % 4]


def _make_record(
    index: int,
    category: str,
    rng: random.Random,
    train_rows: int,
) -> Dict[str, object]:
    if category == "normal":
        dst_port = rng.choice(SERVICE_PORTS)
        packet_count = max(1, int(rng.lognormvariate(2.4, 1.0)))
        duration = max(0.001, rng.lognormvariate(1.0, 1.1))
    elif category == "suspicious":
        dst_port = rng.choice((4444, 5555, 8081, 9001, 31337, 49152, 65534))
        packet_count = max(1, int(rng.lognormvariate(3.2, 1.25)))
        duration = max(0.001, rng.lognormvariate(-0.2, 1.0))
    else:
        dst_port = rng.choice((0, 1, 7, 19, 2323, 4444, 65000))
        packet_count = max(1, int(rng.lognormvariate(4.0, 1.4)))
        duration = max(0.001, rng.lognormvariate(-1.0, 0.9))

    protocol = rng.choices((6, 17, 1), weights=(0.72, 0.25, 0.03), k=1)[0]
    src_port = rng.randint(1024, 65535)
    average_size = min(1500.0, max(40.0, rng.lognormvariate(5.4, 0.65)))
    total_bytes = max(packet_count, int(round(packet_count * average_size)))
    average_size = total_bytes / packet_count

    if protocol == 6:
        syn = rng.randint(0, min(packet_count, 3 if category == "normal" else 10))
        fin = rng.randint(0, min(packet_count, 2))
        rst = rng.randint(0, min(packet_count, 1 if category == "normal" else 6))
        psh = rng.randint(0, min(packet_count, 5))
        ack = rng.randint(0, packet_count)
    else:
        syn = fin = rst = psh = ack = 0

    if index < train_rows:
        source_file = "flows{}.csv".format(index % 4 + 1)
    else:
        source_file = "flows5.csv"
    return {
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "avg_packet_size": average_size,
        "duration": duration,
        "packet_rate": packet_count / duration,
        "byte_rate": total_bytes / duration,
        "syn": syn,
        "fin": fin,
        "rst": rst,
        "psh": psh,
        "ack": ack,
        "application_name": rng.choice(APPLICATIONS[category]),
        "application_is_guessed": 0,
        "label": "Known" if category == "normal" else "Unknown",
        "source_file": source_file,
    }


def _format_row(record: Dict[str, object]) -> str:
    values = []
    for column in FLOW_COLUMNS:
        value = record[column]
        if column in FLOAT_COLUMNS:
            values.append("{:.6f}".format(float(value)))
        else:
            values.append(str(value))
    return ",".join(values) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(
    target_bytes: int,
    output_dir: Path,
    seed: int = 20261112,
    chunk_rows: int = 50_000,
    train_fraction: float = 0.8,
) -> Dict[str, object]:
    """Write train/test flow CSVs incrementally and return their manifest."""
    if target_bytes <= 0:
        raise ValueError("target_bytes must be greater than zero")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be greater than zero")
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    header = ",".join(FLOW_COLUMNS) + "\n"
    train_bytes = len(header.encode("utf-8"))
    test_bytes = len(header.encode("utf-8"))
    estimated_rows = max(
        MINIMUM_ROWS,
        int(math.ceil(max(0, target_bytes - 2 * len(header)) / 180)),
    )
    train_target_rows = max(4, int(round(estimated_rows * train_fraction)))
    train_target_rows = min(train_target_rows, estimated_rows - 3)

    rng = random.Random(seed)
    category_block: List[str] = []
    row_count = 0
    category_counts: Counter = Counter()
    label_counts = {"train": Counter(), "test": Counter()}
    train_chunk: List[str] = []
    test_chunk: List[str] = []

    with train_path.open("w", encoding="utf-8", newline="") as train_file, test_path.open(
        "w", encoding="utf-8", newline=""
    ) as test_file:
        train_file.write(header)
        test_file.write(header)
        while (
            train_bytes + test_bytes < target_bytes
            or row_count < MINIMUM_ROWS
        ):
            category = _traffic_category(row_count, rng, category_block)
            record = _make_record(row_count, category, rng, train_target_rows)
            line = _format_row(record)
            label = str(record["label"])
            category_counts[category] += 1
            split = "train" if row_count < train_target_rows else "test"
            label_counts[split][label] += 1
            if split == "train":
                train_chunk.append(line)
                train_bytes += len(line.encode("utf-8"))
            else:
                test_chunk.append(line)
                test_bytes += len(line.encode("utf-8"))
            row_count += 1

            if len(train_chunk) >= chunk_rows:
                train_file.writelines(train_chunk)
                train_chunk.clear()
            if len(test_chunk) >= chunk_rows:
                test_file.writelines(test_chunk)
                test_chunk.clear()

        train_file.writelines(train_chunk)
        test_file.writelines(test_chunk)

    files = {}
    for name, path in (("train.csv", train_path), ("test.csv", test_path)):
        files[name] = {
            "path": name,
            "bytes": path.stat().st_size,
            "rows": label_counts["train" if name == "train.csv" else "test"][
                "Known"
            ]
            + label_counts["train" if name == "train.csv" else "test"]["Unknown"],
            "sha256": _sha256(path),
            "label_counts": dict(
                label_counts["train" if name == "train.csv" else "test"]
            ),
        }

    if any(
        not {"Known", "Unknown"}.issubset(item["label_counts"])
        for item in files.values()
    ):
        raise RuntimeError("Both splits must contain Known and Unknown labels")

    manifest: Dict[str, object] = {
        "format_version": 1,
        "seed": seed,
        "target_bytes_minimum": target_bytes,
        "actual_total_bytes": train_path.stat().st_size + test_path.stat().st_size,
        "target_overshoot_bytes": max(
            0, train_path.stat().st_size + test_path.stat().st_size - target_bytes
        ),
        "record_count": row_count,
        "split": {
            "train_fraction_requested": train_fraction,
            "train_rows": files["train.csv"]["rows"],
            "test_rows": files["test.csv"]["rows"],
        },
        "chunk_rows": chunk_rows,
        "schema": {
            "columns": FLOW_COLUMNS,
            "integer_columns": sorted(INTEGER_COLUMNS),
            "float_columns": sorted(FLOAT_COLUMNS),
            "string_columns": ["application_name", "label", "source_file"],
        },
        "synthetic_traffic_category_counts": dict(category_counts),
        "files": files,
        "generation_parameters": {
            "normal_mix_per_four_rows": 2,
            "suspicious_mix_per_four_rows": 1,
            "anomalous_mix_per_four_rows": 1,
            "application_is_guessed": 0,
            "csv_encoding": "UTF-8",
            "numeric_float_precision": 6,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    manifest = generate_dataset(
        args.target_bytes,
        args.output_dir,
        args.seed,
        args.chunk_rows,
        args.train_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
