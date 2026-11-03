"""
Train Pipeline A: SVR (RBF) on HSV histograms + one-hot fruit type.

    python scripts/train_svr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS
from models.svr_model import train_svr, save


def main():
    train = pd.read_csv(ARTIFACTS / "train.csv")
    val = pd.read_csv(ARTIFACTS / "val.csv")
    scaler, model, params = train_svr(train, val)
    save(scaler, model, params)


if __name__ == "__main__":
    main()
