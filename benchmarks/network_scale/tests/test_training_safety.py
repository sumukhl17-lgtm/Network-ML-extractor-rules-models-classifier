from pathlib import Path
import py_compile

import pytest

from benchmarks.network_scale.benchmark_training import _validate_paths, _write_driver


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
