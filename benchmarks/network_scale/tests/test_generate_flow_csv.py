import csv
import hashlib
import json

import pandas as pd
import pytest

from benchmarks.network_scale.generate_flow_csv import (
    FLOAT_COLUMNS,
    FLOW_COLUMNS,
    INTEGER_COLUMNS,
    generate_dataset,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_schema_types_and_training_sources(tmp_path):
    output = tmp_path / "dataset"
    manifest = generate_dataset(8_000, output, seed=42, chunk_rows=3)
    train = pd.read_csv(output / "train.csv")
    test = pd.read_csv(output / "test.csv")

    assert train.columns.tolist() == FLOW_COLUMNS
    assert test.columns.tolist() == FLOW_COLUMNS
    assert set(train["source_file"]) == {
        "flows1.csv",
        "flows2.csv",
        "flows3.csv",
        "flows4.csv",
    }
    assert set(test["source_file"]) == {"flows5.csv"}
    for column in INTEGER_COLUMNS:
        assert pd.api.types.is_integer_dtype(train[column])
    for column in FLOAT_COLUMNS:
        assert pd.api.types.is_float_dtype(train[column])
    for data in (train, test):
        assert set(data["label"]) == {"Known", "Unknown"}
        assert data["application_is_guessed"].eq(0).all()

    assert manifest["record_count"] == len(train) + len(test)
    assert manifest["actual_total_bytes"] == sum(
        (output / name).stat().st_size for name in ("train.csv", "test.csv")
    )
    assert manifest["files"]["train.csv"]["sha256"] == _digest(output / "train.csv")
    assert manifest["files"]["test.csv"]["sha256"] == _digest(output / "test.csv")
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_same_seed_produces_identical_csv_files(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(12_000, first, seed=99, chunk_rows=5)
    generate_dataset(12_000, second, seed=99, chunk_rows=17)

    for filename in ("train.csv", "test.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_target_is_met_with_bounded_row_overshoot(tmp_path):
    target = 8_000
    output = tmp_path / "target"
    manifest = generate_dataset(target, output, seed=5, chunk_rows=4)

    assert manifest["actual_total_bytes"] >= target
    assert manifest["target_overshoot_bytes"] == (
        manifest["actual_total_bytes"] - target
    )
    assert manifest["target_overshoot_bytes"] < 2_000
    assert manifest["record_count"] >= 15


def test_generator_rejects_reuse_and_invalid_chunk_size(tmp_path):
    output = tmp_path / "dataset"
    generate_dataset(500, output, seed=1)
    with pytest.raises(FileExistsError):
        generate_dataset(500, output, seed=1)
    with pytest.raises(ValueError, match="chunk_rows"):
        generate_dataset(500, tmp_path / "bad", chunk_rows=0)
