"""
Train Pipeline C: MobileNetV2 transfer learning + fruit-type embedding.

    python scripts/train_mobilenet.py

Will use CUDA if available; falls back to CPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS
from models.mobilenet_model import train_mobilenet, save


def main():
    train = pd.read_csv(ARTIFACTS / "train.csv")
    val = pd.read_csv(ARTIFACTS / "val.csv")
    model = train_mobilenet(train, val)
    save(model)


if __name__ == "__main__":
    main()
