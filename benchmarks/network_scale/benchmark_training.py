"""Run train_compare.py in a new, isolated directory (execution is opt-in)."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPOSITORY_ROOT / "train_compare.py"
METRICS_NAME = "training_stage_metrics.json"


def _validate_paths(train_csv: Path, test_csv: Path, run_dir: Path) -> None:
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError("train and test CSV files must exist")
    resolved_run = run_dir.resolve()
    if resolved_run == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved_run.parents:
        raise ValueError("run_dir must be outside the repository")
    if run_dir.exists():
        raise FileExistsError("refusing to reuse training directory: {}".format(run_dir))


def _write_driver(path: Path) -> None:
    source = r'''import json
import os
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["NETWORK_ML_REPOSITORY"])
import train_compare
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

metrics = {
    "cross_validation_wall_seconds": 0.0,
    "cross_validation_fit_seconds": 0.0,
    "final_fit_seconds": 0.0,
    "final_prediction_seconds": 0.0,
    "final_metrics_seconds": 0.0,
}
stage = ["setup"]
original_cross_validate = train_compare._cross_validate
def timed_cross_validate(*args, **kwargs):
    stage[0] = "cross_validation"
    start = time.perf_counter()
    try:
        return original_cross_validate(*args, **kwargs)
    finally:
        metrics["cross_validation_wall_seconds"] += time.perf_counter() - start
        stage[0] = "final"
train_compare._cross_validate = timed_cross_validate

original_make_models = train_compare._make_models
def timed_make_models(*args, **kwargs):
    models = original_make_models(*args, **kwargs)
    for model in models.values():
        for method_name in ("predict", "predict_proba"):
            original = getattr(model, method_name)
            def timed_prediction(*positional, _original=original, **named):
                start = time.perf_counter()
                try:
                    return _original(*positional, **named)
                finally:
                    if stage[0] == "final":
                        metrics["final_prediction_seconds"] += time.perf_counter() - start
            setattr(model, method_name, timed_prediction)
    return models
train_compare._make_models = timed_make_models

for model_class in (RandomForestClassifier, XGBClassifier):
    original_fit = model_class.fit
    def make_timed_fit(original):
        def timed_fit(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                key = "cross_validation_fit_seconds" if stage[0] == "cross_validation" else "final_fit_seconds"
                metrics[key] += time.perf_counter() - start
        return timed_fit
    model_class.fit = make_timed_fit(original_fit)

original_metrics = train_compare._print_metrics
def timed_metrics(*args, **kwargs):
    start = time.perf_counter()
    try:
        return original_metrics(*args, **kwargs)
    finally:
        metrics["final_metrics_seconds"] += time.perf_counter() - start
train_compare._print_metrics = timed_metrics

try:
    train_compare.main()
except BaseException as error:
    metrics["error"] = repr(error)
    Path(os.environ["NETWORK_ML_STAGE_METRICS"]).write_text(json.dumps(metrics, indent=2) + "\n")
    raise
Path(os.environ["NETWORK_ML_STAGE_METRICS"]).write_text(json.dumps(metrics, indent=2) + "\n")
'''
    path.write_text(source, encoding="utf-8")


def run_training_benchmark(
    train_csv: Path,
    test_csv: Path,
    run_dir: Path,
    python_executable: str = sys.executable,
) -> Dict[str, object]:
    """Execute the existing training pipeline only inside a fresh run directory."""
    train_csv, test_csv, run_dir = map(Path, (train_csv, test_csv, run_dir))
    _validate_paths(train_csv, test_csv, run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "train.csv").symlink_to(train_csv.resolve())
    (run_dir / "test.csv").symlink_to(test_csv.resolve())
    tracking_dir = run_dir / "mlruns"
    tracking_dir.mkdir()
    driver = run_dir / "isolated_training_driver.py"
    _write_driver(driver)

    stage_metrics_path = run_dir / METRICS_NAME
    stdout_path = run_dir / "training.stdout.log"
    stderr_path = run_dir / "training.stderr.log"
    env = os.environ.copy()
    env.update(
        {
            "MLFLOW_TRACKING_URI": tracking_dir.as_uri(),
            "NETWORK_ML_REPOSITORY": str(REPOSITORY_ROOT),
            "NETWORK_ML_STAGE_METRICS": str(stage_metrics_path),
        }
    )
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            [python_executable, str(driver)],
            cwd=run_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        peak_rss = 0
        while process.poll() is None:
            try:
                output = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(process.pid)],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if output:
                    peak_rss = max(peak_rss, int(output) * 1024)
            except (subprocess.CalledProcessError, ValueError):
                pass
            time.sleep(0.2)
    elapsed = time.perf_counter() - start

    if stage_metrics_path.exists():
        stage_metrics = json.loads(stage_metrics_path.read_text(encoding="utf-8"))
    else:
        stage_metrics = {}
    stage_metrics["other_wall_seconds"] = max(
        0.0,
        elapsed
        - float(stage_metrics.get("cross_validation_wall_seconds", 0.0))
        - float(stage_metrics.get("final_fit_seconds", 0.0)),
    )
    model_outputs = {}
    for name in ("rf.joblib", "xgb.json"):
        path = run_dir / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as model_file:
                for block in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(block)
            model_outputs[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        else:
            model_outputs[name] = None
    error = None
    if process.returncode:
        error = {
            "exit_code": process.returncode,
            "stderr_tail": stderr_path.read_text(encoding="utf-8")[-4000:],
        }
    report: Dict[str, object] = {
        "run_dir": str(run_dir.resolve()),
        "train_csv": str(train_csv.resolve()),
        "test_csv": str(test_csv.resolve()),
        "mlflow_tracking_uri": tracking_dir.as_uri(),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "exit_code": process.returncode,
        "error": error,
        "stage_metrics": stage_metrics,
        "model_outputs": model_outputs,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if process.returncode:
        raise RuntimeError(
            "training failed with exit code {}; inspect {}".format(
                process.returncode, stderr_path
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute training; without this flag only validate and print the plan",
    )
    args = parser.parse_args()
    _validate_paths(args.train, args.test, args.run_dir)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execution": "not started",
                    "train": str(args.train.resolve()),
                    "test": str(args.test.resolve()),
                    "run_dir": str(args.run_dir.resolve()),
                    "mlflow_uri": (args.run_dir.resolve() / "mlruns").as_uri(),
                    "next_step": "rerun with --execute after reviewing the plan",
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            run_training_benchmark(
                args.train, args.test, args.run_dir, args.python
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
