from pathlib import Path
import hashlib
import py_compile
import sys

import pytest

from benchmarks.network_scale.benchmark_training import (
    _install_prediction_timing_wrappers,
    _validate_paths,
    _write_driver,
    run_training_benchmark,
)


def test_training_directory_must_be_outside_repository(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("label\nKnown\n")
    test.write_text("label\nUnknown\n")
    repository_run = Path(__file__).resolve().parents[3] / "benchmarks" / "network_scale" / "run"

    with pytest.raises(ValueError, match="outside the repository"):
        _validate_paths(train, test, repository_run)


def test_existing_training_directory_is_never_reused(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    run_dir = tmp_path / "run"
    train.write_text("label\nKnown\n")
    test.write_text("label\nUnknown\n")
    run_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        _validate_paths(train, test, run_dir)


def test_isolated_training_driver_compiles(tmp_path):
    driver = tmp_path / "driver.py"
    _write_driver(driver)
    py_compile.compile(str(driver), doraise=True)


def test_timed_model_predictions_remain_serializable(tmp_path, monkeypatch):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    model_classes = (RandomForestClassifier, XGBClassifier)
    for model_class in model_classes:
        for method_name in ("predict", "predict_proba"):
            monkeypatch.setattr(
                model_class, method_name, getattr(model_class, method_name)
            )

    metrics = {"final_prediction_seconds": 0.0}
    stage = ["final"]
    _install_prediction_timing_wrappers(model_classes, metrics, stage)

    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    models = (
        RandomForestClassifier(n_estimators=3, random_state=1),
        XGBClassifier(
            n_estimators=2,
            max_depth=2,
            n_jobs=1,
            random_state=1,
            eval_metric="logloss",
        ),
    )
    for index, model in enumerate(models):
        model.fit(features, labels)
        expected_predictions = model.predict(features)
        expected_probabilities = model.predict_proba(features)
        path = tmp_path / "model-{}.joblib".format(index)
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(loaded.predict(features), expected_predictions)
        np.testing.assert_allclose(
            loaded.predict_proba(features), expected_probabilities
        )

    assert metrics["final_prediction_seconds"] > 0


def test_in_repository_run_cannot_overwrite_model_artifacts(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("label\nKnown\n")
    test.write_text("label\nUnknown\n")

    repository_root = Path(__file__).resolve().parents[3]
    run_dir = repository_root / "benchmarks" / "network_scale" / "protected-run"
    artifacts = (
        "rf.joblib",
        "xgb.json",
        "rf_6.joblib",
        "xgb_6.json",
        "rf_all12.joblib",
        "xgb_all12.json",
    )
    original_hashes = {
        name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
        for name in artifacts
    }

    with pytest.raises(ValueError, match="outside the repository"):
        run_training_benchmark(train, test, run_dir)

    assert not run_dir.exists()
    assert {
        name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
        for name in artifacts
    } == original_hashes


def test_training_subprocess_keeps_model_outputs_in_isolated_run_dir(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("label\nKnown\n")
    test.write_text("label\nUnknown\n")

    repository_root = Path(__file__).resolve().parents[3]
    artifacts = ("rf.joblib", "xgb.json")
    original_hashes = {
        name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
        for name in artifacts
    }

    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!{}\n"
        "from pathlib import Path\n"
        "Path('rf.joblib').write_bytes(b'isolated rf')\n"
        "Path('xgb.json').write_bytes(b'isolated xgb')\n".format(sys.executable)
    )
    fake_python.chmod(0o755)
    run_dir = tmp_path / "isolated-run"

    report = run_training_benchmark(train, test, run_dir, str(fake_python))

    assert report["exit_code"] == 0
    assert (run_dir / "rf.joblib").read_bytes() == b"isolated rf"
    assert (run_dir / "xgb.json").read_bytes() == b"isolated xgb"
    assert {
        name: hashlib.sha256((repository_root / name).read_bytes()).hexdigest()
        for name in artifacts
    } == original_hashes
