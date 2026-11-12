import csv
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

import classify_network
from benchmarks.network_scale import benchmark_inference
from benchmarks.network_scale.generate_flow_csv import generate_dataset
from benchmarks.network_scale.optimized_inference import classify_batch


def _load_flows(path, count):
    return pd.read_csv(path, nrows=count).to_dict(orient="records")


def test_rule_result_and_unmatched_inference_match_baseline(tmp_path):
    output = tmp_path / "flows"
    generate_dataset(8_000, output, seed=12, chunk_rows=4)
    flows = _load_flows(output / "train.csv", 40)
    flows[0]["dst_port"] = 443
    flows[1]["dst_port"] = 5555

    expected = [classify_network.classify_network(flow) for flow in flows]
    actual = classify_batch(
        flows,
        classify_network._MODEL,
        classify_network._MODEL_FEATURES,
        classify_network.ML_OVERRIDES_UNMATCHED,
    )
    assert actual == expected
    assert actual[0] == {"label": "HTTPS", "source": "rule"}
    assert actual[1]["source"] == "default"
    assert "ml_label" in actual[1] and "ml_confidence" in actual[1]


def test_ml_override_mode_parity(tmp_path, monkeypatch):
    output = tmp_path / "flows"
    generate_dataset(8_000, output, seed=14)
    flows = _load_flows(output / "test.csv", 12)
    monkeypatch.setattr(classify_network, "ML_OVERRIDES_UNMATCHED", True)

    expected = [classify_network.classify_network(flow) for flow in flows]
    actual = classify_batch(
        flows,
        classify_network._MODEL,
        classify_network._MODEL_FEATURES,
        True,
    )
    assert actual == expected
    assert all(result["source"] in ("rule", "ml") for result in actual)


def test_streaming_benchmark_bounds_batches_and_counts_results(tmp_path, monkeypatch):
    output = tmp_path / "flows"
    generate_dataset(30_000, output, seed=17, chunk_rows=11)
    observed_batch_sizes = []
    original = benchmark_inference.classify_batch

    def checked_batch(flows, *args, **kwargs):
        observed_batch_sizes.append(len(flows))
        return original(flows, *args, **kwargs)

    monkeypatch.setattr(benchmark_inference, "classify_batch", checked_batch)
    report = benchmark_inference.run_benchmark(
        [output / "train.csv", output / "test.csv"],
        REPOSITORY_ROOT / "rf.joblib",
        mode="optimized",
        chunk_rows=7,
        latency_sample_limit=3,
        max_records=21,
    )

    assert report["records_seen"] == 21
    assert report["records_processed"] == 21
    assert report["records_errored"] == 0
    assert report["model_sha256"]
    assert observed_batch_sizes
    assert max(observed_batch_sizes) <= 7
    assert report["latency_sample_count"] == 3
    assert report["latency_p50_seconds"] is not None
    assert report["peak_rss_bytes"] > 0


def test_baseline_benchmark_reports_rule_and_ml_counts(tmp_path):
    output = tmp_path / "flows"
    generate_dataset(8_000, output, seed=21)
    report = benchmark_inference.run_benchmark(
        [output / "train.csv"],
        REPOSITORY_ROOT / "rf.joblib",
        mode="baseline",
        chunk_rows=5,
        latency_sample_limit=8,
        max_records=8,
    )

    assert report["records_processed"] == 8
    assert report["records_errored"] == 0
    assert sum(report["rule_classifications_by_label"].values()) + sum(
        report["ml_predictions_by_label"].values()
    ) == 8
    assert report["latency_measurement"].startswith("per-record")
