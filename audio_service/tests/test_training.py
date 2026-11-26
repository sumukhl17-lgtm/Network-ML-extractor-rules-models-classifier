from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from audio_service import training
from audio_service.taxonomy import ESC50_CLASSES
from audio_service.yamnet import YAMNET_EMBEDDING_SIZE


MODEL_VERSION = "google/yamnet/1"
MODEL_SHA256 = "a" * 64


def make_dataset_dir(tmp_path):
    root = tmp_path / "ESC-50"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True)

    rows = []
    for class_index, label in enumerate(ESC50_CLASSES):
        for item_index in range(40):
            fold = item_index // 8 + 1
            filename = f"sample-{class_index:02d}-{item_index:02d}.wav"
            (audio_dir / filename).write_bytes(b"test")

            rows.append(
                {
                    "filename": filename,
                    "fold": str(fold),
                    "category": label,
                }
            )

    with (root / "esc50.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "fold", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return root


def fake_embedding(index):
    return SimpleNamespace(
        embedding=np.full(
            YAMNET_EMBEDDING_SIZE,
            index / 1000,
            dtype=np.float32,
        ),
        model_version=MODEL_VERSION,
        model_sha256=MODEL_SHA256,
    )


class FakeEmbedder:
    def __init__(self):
        self.load_calls = 0
        self.info_calls = 0

    def load(self):
        self.load_calls += 1

    def model_info(self):
        self.info_calls += 1
        return {
            "loaded": True,
            "version": MODEL_VERSION,
            "sha256": MODEL_SHA256,
        }


def make_records(tmp_path, count):
    return tuple(
        training.DatasetRecord(
            filename=f"sample-{index}.wav",
            label=ESC50_CLASSES[index % len(ESC50_CLASSES)],
            fold=index % 5 + 1,
            path=tmp_path / f"sample-{index}.wav",
        )
        for index in range(count)
    )


def patch_manifest(monkeypatch, records, csv_sha256="b" * 64):
    monkeypatch.setattr(
        training,
        "load_dataset_manifest",
        lambda _: (records, csv_sha256),
    )


def test_load_dataset_manifest_validates_official_dataset(tmp_path):
    root = make_dataset_dir(tmp_path)

    records, csv_sha256 = training.load_dataset_manifest(root)

    assert len(records) == 2000
    assert len(csv_sha256) == 64
    assert {record.fold for record in records} == {1, 2, 3, 4, 5}
    assert {record.label for record in records} == set(ESC50_CLASSES)


def test_load_dataset_manifest_rejects_missing_audio(tmp_path):
    root = make_dataset_dir(tmp_path)
    (root / "audio" / "sample-00-00.wav").unlink()

    with pytest.raises(ValueError, match="audio file is missing"):
        training.load_dataset_manifest(root)


def test_extract_embeddings_writes_complete_cache_and_metadata(
    tmp_path,
    monkeypatch,
):
    records = make_records(tmp_path, 3)
    patch_manifest(monkeypatch, records)

    embedded_filenames = []

    def fake_embed_record(record, **kwargs):
        embedded_filenames.append(record.filename)
        index = int(record.filename.rsplit("-", 1)[1].split(".")[0])
        return fake_embedding(index)

    monkeypatch.setattr(training, "_embed_record", fake_embed_record)

    embedder = FakeEmbedder()
    cache_path = tmp_path / "cache" / "embeddings.npz"

    dataset = training.extract_dataset_embeddings(
        tmp_path,
        cache_path=cache_path,
        embedder=embedder,
        checkpoint_interval=2,
    )

    assert dataset.embeddings.shape == (3, YAMNET_EMBEDDING_SIZE)
    assert dataset.embeddings.dtype == np.float32
    assert dataset.filenames == tuple(record.filename for record in records)
    assert dataset.metadata["samples"] == 3
    assert dataset.metadata["csv_sha256"] == "b" * 64
    assert dataset.metadata["yamnet_version"] == MODEL_VERSION
    assert dataset.metadata["yamnet_sha256"] == MODEL_SHA256
    assert dataset.metadata["embedding_size"] == YAMNET_EMBEDDING_SIZE
    assert embedded_filenames == [record.filename for record in records]
    assert cache_path.is_file()


def test_extract_embeddings_reuses_complete_cache_without_inference(
    tmp_path,
    monkeypatch,
):
    records = make_records(tmp_path, 2)
    patch_manifest(monkeypatch, records, "c" * 64)

    cache_path = tmp_path / "embeddings.npz"
    embeddings = np.ones(
        (2, YAMNET_EMBEDDING_SIZE),
        dtype=np.float32,
    )
    labels = np.asarray(
        [record.label for record in records],
        dtype=np.str_,
    )
    folds = np.asarray(
        [record.fold for record in records],
        dtype=np.int64,
    )
    metadata = {
        "csv_sha256": "c" * 64,
        "yamnet_version": MODEL_VERSION,
        "yamnet_sha256": MODEL_SHA256,
        "embedding_size": YAMNET_EMBEDDING_SIZE,
        "samples": 2,
    }

    training._save_cache(
        cache_path,
        embeddings,
        labels,
        folds,
        tuple(record.filename for record in records),
        metadata,
    )

    def unexpected_embedding(*args, **kwargs):
        raise AssertionError("A complete cache should not run inference.")

    monkeypatch.setattr(training, "_embed_record", unexpected_embedding)

    embedder = FakeEmbedder()
    dataset = training.extract_dataset_embeddings(
        tmp_path,
        cache_path=cache_path,
        embedder=embedder,
    )

    assert np.array_equal(dataset.embeddings, embeddings)
    assert dataset.metadata["samples"] == 2
    assert embedder.load_calls == 1
    assert embedder.info_calls == 1


def test_extract_embeddings_resumes_partial_cache_and_updates_metadata(
    tmp_path,
    monkeypatch,
):
    records = make_records(tmp_path, 4)
    patch_manifest(monkeypatch, records, "d" * 64)

    cache_path = tmp_path / "embeddings.npz"
    partial_embeddings = np.stack(
        [
            fake_embedding(0).embedding,
            fake_embedding(1).embedding,
        ]
    ).astype(np.float32)
    partial_labels = np.asarray(
        [record.label for record in records[:2]],
        dtype=np.str_,
    )
    partial_folds = np.asarray(
        [record.fold for record in records[:2]],
        dtype=np.int64,
    )
    partial_metadata = {
        "csv_sha256": "d" * 64,
        "yamnet_version": MODEL_VERSION,
        "yamnet_sha256": MODEL_SHA256,
        "embedding_size": YAMNET_EMBEDDING_SIZE,
        "samples": 2,
    }

    training._save_cache(
        cache_path,
        partial_embeddings,
        partial_labels,
        partial_folds,
        tuple(record.filename for record in records[:2]),
        partial_metadata,
    )

    embedded_filenames = []

    def fake_embed_record(record, **kwargs):
        embedded_filenames.append(record.filename)
        index = int(record.filename.rsplit("-", 1)[1].split(".")[0])
        return fake_embedding(index)

    monkeypatch.setattr(training, "_embed_record", fake_embed_record)

    dataset = training.extract_dataset_embeddings(
        tmp_path,
        cache_path=cache_path,
        embedder=FakeEmbedder(),
        checkpoint_interval=2,
    )

    assert embedded_filenames == [
        records[2].filename,
        records[3].filename,
    ]
    assert dataset.embeddings.shape == (4, YAMNET_EMBEDDING_SIZE)
    assert dataset.metadata["samples"] == 4

    cached = training._load_cache(
        cache_path,
        csv_sha256="d" * 64,
        yamnet_version=MODEL_VERSION,
        yamnet_sha256=MODEL_SHA256,
        records=records,
    )
    assert cached is not None
    assert cached[3] == 4
    assert cached[4]["samples"] == 4


def test_load_cache_rejects_wrong_dataset_fingerprint(tmp_path):
    cache_path = tmp_path / "embeddings.npz"
    records = make_records(tmp_path, 1)

    training._save_cache(
        cache_path,
        np.zeros((1, YAMNET_EMBEDDING_SIZE), dtype=np.float32),
        np.asarray([records[0].label], dtype=np.str_),
        np.asarray([records[0].fold], dtype=np.int64),
        (records[0].filename,),
        {
            "csv_sha256": "x" * 64,
            "yamnet_version": MODEL_VERSION,
            "yamnet_sha256": MODEL_SHA256,
            "embedding_size": YAMNET_EMBEDDING_SIZE,
            "samples": 1,
        },
    )

    result = training._load_cache(
        cache_path,
        csv_sha256="b" * 64,
        yamnet_version=MODEL_VERSION,
        yamnet_sha256=MODEL_SHA256,
        records=records,
    )

    assert result is None


def test_load_cache_rejects_stale_sample_count(tmp_path):
    cache_path = tmp_path / "embeddings.npz"
    records = make_records(tmp_path, 1)

    training._save_cache(
        cache_path,
        np.zeros((1, YAMNET_EMBEDDING_SIZE), dtype=np.float32),
        np.asarray([records[0].label], dtype=np.str_),
        np.asarray([records[0].fold], dtype=np.int64),
        (records[0].filename,),
        {
            "csv_sha256": "b" * 64,
            "yamnet_version": MODEL_VERSION,
            "yamnet_sha256": MODEL_SHA256,
            "embedding_size": YAMNET_EMBEDDING_SIZE,
            "samples": 99,
        },
    )

    result = training._load_cache(
        cache_path,
        csv_sha256="b" * 64,
        yamnet_version=MODEL_VERSION,
        yamnet_sha256=MODEL_SHA256,
        records=records,
    )

    assert result is None


def test_evaluate_five_fold_uses_each_official_fold_as_holdout(monkeypatch):
    labels = np.asarray(
        [
            label
            for label in ESC50_CLASSES
            for _ in range(40)
        ],
        dtype=np.str_,
    )
    folds = np.asarray(
        [
            fold
            for _label in ESC50_CLASSES
            for fold in range(1, 6)
            for _ in range(8)
        ],
        dtype=np.int64,
    )
    embeddings = np.zeros(
        (2000, YAMNET_EMBEDDING_SIZE),
        dtype=np.float32,
    )

    fitted_sizes = []

    class FakeClassifier:
        def fit(self, x, y):
            fitted_sizes.append((len(x), len(y)))
            self.prediction = y[0]
            return self

        def predict(self, x):
            return np.full(len(x), self.prediction, dtype=np.str_)

    monkeypatch.setattr(
        training,
        "build_classifier",
        FakeClassifier,
    )

    result = training.evaluate_five_fold(embeddings, labels, folds)

    assert set(result["folds"]) == {"1", "2", "3", "4", "5"}
    assert len(fitted_sizes) == 5
    assert all(size == (1600, 1600) for size in fitted_sizes)
    assert result["evaluation"] == "official five-fold ESC-50 cross-validation"
    assert 0 <= result["accuracy_mean"] <= 1
    assert 0 <= result["macro_f1_mean"] <= 1


def test_evaluate_five_fold_rejects_invalid_embedding_shape():
    with pytest.raises(ValueError, match="invalid shape or value"):
        training.evaluate_five_fold(
            np.zeros((10, 4), dtype=np.float32),
            np.asarray([], dtype=np.str_),
            np.asarray([], dtype=np.int64),
        )


def test_evaluate_five_fold_rejects_unbalanced_fold_distribution():
    labels = np.asarray(
        [
            label
            for label in ESC50_CLASSES
            for _ in range(40)
        ],
        dtype=np.str_,
    )
    folds = np.asarray(
        [
            fold
            for _label in ESC50_CLASSES
            for fold in range(1, 6)
            for _ in range(8)
        ],
        dtype=np.int64,
    )
    folds[0] = 2
    embeddings = np.zeros(
        (2000, YAMNET_EMBEDDING_SIZE),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="400 recordings"):
        training.evaluate_five_fold(embeddings, labels, folds)
