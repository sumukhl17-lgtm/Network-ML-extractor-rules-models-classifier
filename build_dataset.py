"""Combine labeled flow CSV files into train.csv and test.csv."""

import argparse
from pathlib import Path

import pandas as pd


def _read_with_source(csv_path):
    flows = pd.read_csv(csv_path)
    flows["source_file"] = Path(csv_path).name
    return flows


def build_dataset(train_files, test_file, output_dir="."):
    """Write ordered training and test datasets with source-file metadata."""
    output_dir = Path(output_dir)
    train = pd.concat(
        [_read_with_source(csv_path) for csv_path in train_files],
        ignore_index=True,
    )
    test = _read_with_source(test_file)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    for name, flows in (("train.csv", train), ("test.csv", test)):
        counts = flows["label"].value_counts().reindex(
            ["Known", "Unknown"],
            fill_value=0,
        )
        print("{} label counts:".format(name))
        print(counts.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="one or more labeled CSV files to concatenate for training",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="labeled CSV file to use as the test dataset",
    )
    args = parser.parse_args()
    build_dataset(args.train, args.test)


if __name__ == "__main__":
    main()
