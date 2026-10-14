import pandas as pd

from build_dataset import build_dataset


def test_build_dataset_preserves_labels_sources_and_split_order(
    tmp_path,
    monkeypatch,
    capsys,
):
    train_one = tmp_path / "flows1.csv"
    train_two = tmp_path / "flows3.csv"
    test_file = tmp_path / "flows2.csv"
    pd.DataFrame(
        {"flow_id": [1, 2], "label": ["Known", "Unknown"]}
    ).to_csv(train_one, index=False)
    pd.DataFrame(
        {"flow_id": [3], "label": ["Known"]}
    ).to_csv(train_two, index=False)
    pd.DataFrame(
        {"flow_id": [4, 5], "label": ["Unknown", "Known"]}
    ).to_csv(test_file, index=False)
    monkeypatch.chdir(tmp_path)

    build_dataset([train_one, train_two], test_file)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    assert train["flow_id"].tolist() == [1, 2, 3]
    assert train["label"].tolist() == ["Known", "Unknown", "Known"]
    assert train["source_file"].tolist() == [
        "flows1.csv",
        "flows1.csv",
        "flows3.csv",
    ]
    assert test["flow_id"].tolist() == [4, 5]
    assert test["label"].tolist() == ["Unknown", "Known"]
    assert test["source_file"].tolist() == ["flows2.csv", "flows2.csv"]
    assert "train.csv label counts:" in capsys.readouterr().out
