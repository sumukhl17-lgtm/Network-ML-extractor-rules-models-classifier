"""Stream flow CSVs through baseline or batch network classification."""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

if __package__:
    from .optimized_inference import classify_batch
else:
    from optimized_inference import classify_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _percentile(samples: List[float], percentile: float) -> Optional[float]:
    if not samples:
        return None
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _sample_latency(
    samples: List[float],
    value: float,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(samples) < limit:
        samples.append(value)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        samples[replacement] = value


def _load_classifier(model_path: Path):
    import classify_network

    model = joblib.load(model_path)
    names = list(
        getattr(model, "feature_names_in_", classify_network.MODEL_FEATURES)
    )
    original_model = classify_network._MODEL
    original_features = classify_network._MODEL_FEATURES
    classify_network._MODEL = model
    classify_network._MODEL_FEATURES = names
    return classify_network, model, names, original_model, original_features


def run_benchmark(
    input_files: Sequence[Path],
    model_path: Path,
    mode: str = "baseline",
    chunk_rows: int = 10_000,
    latency_sample_limit: int = 10_000,
    max_records: Optional[int] = None,
) -> Dict[str, object]:
    if mode not in ("baseline", "optimized"):
        raise ValueError("mode must be baseline or optimized")
    if chunk_rows <= 0 or latency_sample_limit < 0:
        raise ValueError("chunk_rows must be positive and latency sample non-negative")
    for path in input_files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    model_path = Path(model_path).resolve()
    classifier, model, model_features, original_model, original_features = (
        _load_classifier(model_path)
    )
    latency_samples: List[float] = []
    amortized_latency_samples: List[float] = []
    latency_calls_seen = 0
    latency_rng = random.Random(0)
    amortized_rng = random.Random(0)
    records_seen = records_processed = records_errored = 0
    rules_by_label: Counter = Counter()
    ml_predictions_by_label: Counter = Counter()
    error_examples: List[str] = []
    process_start = time.perf_counter()
    process_cpu_start = time.process_time()

    try:
        for input_path in input_files:
            for frame in pd.read_csv(input_path, chunksize=chunk_rows):
                if max_records is not None:
                    remaining = max_records - records_seen
                    if remaining <= 0:
                        break
                    frame = frame.iloc[:remaining]
                flows = frame.to_dict(orient="records")
                records_seen += len(flows)
                if mode == "baseline":
                    for flow in flows:
                        start = time.perf_counter()
                        try:
                            result = classifier.classify_network(flow)
                        except Exception as error:
                            records_errored += 1
                            if len(error_examples) < 20:
                                error_examples.append(repr(error))
                        else:
                            records_processed += 1
                            if result["source"] == "rule":
                                rules_by_label[result["label"]] += 1
                            else:
                                ml_predictions_by_label[
                                    result.get("ml_label", result["label"])
                                ] += 1
                        elapsed = time.perf_counter() - start
                        latency_calls_seen += 1
                        _sample_latency(
                            latency_samples,
                            elapsed,
                            latency_calls_seen,
                            latency_sample_limit,
                            latency_rng,
                        )
                else:
                    start = time.perf_counter()
                    try:
                        results = classify_batch(
                            flows,
                            model,
                            model_features,
                            classifier.ML_OVERRIDES_UNMATCHED,
                        )
                    except Exception as error:
                        records_errored += len(flows)
                        if len(error_examples) < 20:
                            error_examples.append(repr(error))
                    else:
                        records_processed += len(results)
                        for result in results:
                            if result["source"] == "rule":
                                rules_by_label[result["label"]] += 1
                            else:
                                ml_predictions_by_label[
                                    result.get("ml_label", result["label"])
                                ] += 1
                        batch_elapsed = time.perf_counter() - start
                        latency_calls_seen += 1
                        _sample_latency(
                            latency_samples,
                            batch_elapsed,
                            latency_calls_seen,
                            latency_sample_limit,
                            latency_rng,
                        )
                        _sample_latency(
                            amortized_latency_samples,
                            batch_elapsed / max(1, len(flows)),
                            latency_calls_seen,
                            latency_sample_limit,
                            amortized_rng,
                        )
                if max_records is not None and records_seen >= max_records:
                    break
            if max_records is not None and records_seen >= max_records:
                break
    finally:
        classifier._MODEL = original_model
        classifier._MODEL_FEATURES = original_features

    elapsed = time.perf_counter() - process_start
    cpu_seconds = time.process_time() - process_cpu_start
    throughput = records_processed / elapsed if elapsed else 0.0
    batch_mode = mode == "optimized"
    report: Dict[str, object] = {
        "mode": mode,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "input_files": [
            {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size}
            for path in input_files
        ],
        "chunk_rows": chunk_rows,
        "records_seen": records_seen,
        "records_processed": records_processed,
        "records_errored": records_errored,
        "rule_classifications_by_label": dict(rules_by_label),
        "ml_predictions_by_label": dict(ml_predictions_by_label),
        "elapsed_seconds": elapsed,
        "throughput_records_per_second": throughput,
        "latency_measurement": (
            "batch wall time; percentile sample is bounded to the first sampled batches"
            if batch_mode
            else "per-record classify_network call wall time; bounded first-record sample"
        ),
        "latency_sample_count": len(latency_samples),
        "latency_calls_seen": latency_calls_seen,
        "latency_p50_seconds": _percentile(latency_samples, 50),
        "latency_p95_seconds": _percentile(latency_samples, 95),
        "latency_p99_seconds": _percentile(latency_samples, 99),
        "amortized_latency_per_record_seconds": (
            _percentile(amortized_latency_samples, 50)
            if batch_mode and amortized_latency_samples
            else None
        ),
        "amortized_latency_per_record_p95_seconds": (
            _percentile(amortized_latency_samples, 95)
            if batch_mode and amortized_latency_samples
            else None
        ),
        "amortized_latency_per_record_p99_seconds": (
            _percentile(amortized_latency_samples, 99)
            if batch_mode and amortized_latency_samples
            else None
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_of_one_core": (
            cpu_seconds / elapsed * 100 if elapsed else 0.0
        ),
        "error_examples": error_examples,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model", type=Path, default=REPOSITORY_ROOT / "rf.joblib"
    )
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--latency-sample-limit", type=int, default=10_000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        args.input,
        args.model,
        args.mode,
        args.chunk_rows,
        args.latency_sample_limit,
        args.max_records,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        if args.output.exists():
            parser.error("refusing to overwrite report: {}".format(args.output))
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
