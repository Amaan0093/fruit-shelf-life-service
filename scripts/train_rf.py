"""
Train Pipeline B: Random Forest on HSV + HOG + fruit-type.

    python scripts/train_rf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS
from models.random_forest import train_rf, save, grouped_importance


def main():
    train = pd.read_csv(ARTIFACTS / "train.csv")
    model, lengths = train_rf(train)
    save(model, lengths)

    # Aggregated importance answers the methods-section question:
    # how much do colour, texture, and fruit-type each contribute?
    print("\n[rf] Aggregated feature-group importance:")
    for k, v in grouped_importance(model, lengths).items():
        print(f"    {k:11s} {v:.3f}")


if __name__ == "__main__":
    main()
