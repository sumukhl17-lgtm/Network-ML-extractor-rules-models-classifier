"""Compare capture-specific generalization across flow datasets."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score


FEATURES = [
    "packet_count",
    "total_bytes",
    "avg_packet_size",
    "duration",
    "packet_rate",
    "byte_rate",
]
LABELS = ["Known", "Unknown"]
TRAIN_SETS = {
    "A (fast Unknown)": ("flows1.csv", "flows3.csv", "flows4.csv"),
    "B (slow Unknown)": ("flows1.csv", "flows5.csv"),
    "C (both)": ("flows1.csv", "flows3.csv", "flows4.csv", "flows5.csv"),
}
TEST_SETS = {
    "flows2.csv (fast Unknown)": "flows2.csv",
    "flows5.csv (slow Unknown)": "flows5.csv",
}


def _read_capture(data_dir, filename):
    capture = pd.read_csv(data_dir / filename)
    required = set(FEATURES + ["label"])
    missing = required.difference(capture.columns)
    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(
                filename,
                ", ".join(sorted(missing)),
            )
        )
    if not all(
        pd.api.types.is_numeric_dtype(capture[feature].dtype)
        for feature in FEATURES
    ):
        raise TypeError("{} contains non-numeric model features".format(filename))
    unexpected_labels = set(capture["label"].dropna()).difference(LABELS)
    if capture["label"].isnull().any() or unexpected_labels:
        raise ValueError(
            "{} labels must be Known or Unknown; found {}".format(
                filename,
                sorted(unexpected_labels),
            )
        )
    return capture


def main():
    data_dir = Path(__file__).resolve().parent
    captures = {}
    filenames = {
        filename
        for training_files in TRAIN_SETS.values()
        for filename in training_files
    }
    filenames.update(TEST_SETS.values())
    for filename in sorted(filenames):
        captures[filename] = _read_capture(data_dir, filename)

    rows = []
    for train_name, train_files in TRAIN_SETS.items():
        train_capture_names = set(train_files)
        train_data = pd.concat(
            [captures[filename] for filename in train_files],
            ignore_index=True,
        )
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        )
        model.fit(train_data.loc[:, FEATURES], train_data["label"])

        for test_name, test_filename in TEST_SETS.items():
            if test_filename in train_capture_names:
                continue

            test_data = captures[test_filename]
            predictions = model.predict(test_data.loc[:, FEATURES])
            recalls = recall_score(
                test_data["label"],
                predictions,
                labels=LABELS,
                average=None,
                zero_division=0,
            )
            rows.append(
                (
                    train_name,
                    test_name,
                    recalls[LABELS.index("Known")],
                    recalls[LABELS.index("Unknown")],
                )
            )

    print(
        "{:<20} {:<29} {:>14} {:>16}".format(
            "Train set",
            "Test set",
            "Known recall",
            "Unknown recall",
        )
    )
    print("-" * 83)
    for train_name, test_name, known_recall, unknown_recall in rows:
        print(
            "{:<20} {:<29} {:>14.4f} {:>16.4f}".format(
                train_name,
                test_name,
                known_recall,
                unknown_recall,
            )
        )


if __name__ == "__main__":
    main()
