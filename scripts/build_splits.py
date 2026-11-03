"""
One-off step: walk the dataset, build the train/val/test split, and save
to outputs/{train,val,test}.csv so every model trains on identical data.

Run me first:
    python scripts/build_splits.py
"""
import sys
from pathlib import Path

# Allow `python scripts/build_splits.py` to find the project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS
from data.dataset_loader import build_index, stratified_split


def main():
    df = build_index()
    train, val, test = stratified_split(df)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train.to_csv(ARTIFACTS / "train.csv", index=False)
    val.to_csv(ARTIFACTS / "val.csv", index=False)
    test.to_csv(ARTIFACTS / "test.csv", index=False)

    print(f"\nWrote {ARTIFACTS / 'train.csv'}, val.csv, test.csv")
    print("\nPer-fruit distribution (train):")
    print(train.groupby(["fruit", "stage"]).size().rename("n").reset_index())


if __name__ == "__main__":
    main()
