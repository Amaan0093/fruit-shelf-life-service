"""
Build the dataset index and produce stratified train / val / test splits.

Expected directory layout under config.DATA_ROOT:

    dataset/
      banana/
        stage_1/
          *.jpg
        stage_2/
          *.jpg
        ...
      apple/
        stage_1/
          ...

Folder names are case-insensitive and may use any of:
    stage_1, stage1, s1, 1, ripe1, raw1, ...
The trailing integer is what counts. Any folder that doesn't end in a digit
is skipped (with a warning).
"""
from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import DATA_ROOT, FRUIT_LIST, FRUITS, SEED, SPLIT_RATIOS
from data.label_mapping import stage_to_days

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_STAGE_RE = re.compile(r"(\d+)\s*$")          # last integer in the folder name


@dataclass(frozen=True)
class Sample:
    """A single labelled image."""
    path: Path
    fruit: str
    stage: int
    days: float


# Index building

def _parse_stage(folder_name: str) -> int | None:
    """Extract the trailing integer from a folder name, or None if absent."""
    m = _STAGE_RE.search(folder_name)
    return int(m.group(1)) if m else None


def build_index(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Walk `data_root`, return a DataFrame with one row per labelled image.

    Columns: path, fruit, stage, days.
    """
    if not data_root.exists():
        sys.exit(
            f"[dataset] data_root does not exist: {data_root}\n"
            f"          Place fruit subfolders there (see README.md)."
        )

    records: List[Sample] = []
    for fruit_dir in sorted(data_root.iterdir()):
        if not fruit_dir.is_dir():
            continue
        fruit = fruit_dir.name.lower()
        if fruit not in FRUITS:
            warnings.warn(f"Skipping unknown fruit folder: {fruit_dir}")
            continue

        n_stages = FRUITS[fruit]["stages"]
        for stage_dir in sorted(fruit_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            stage = _parse_stage(stage_dir.name)
            if stage is None:
                warnings.warn(
                    f"Skipping {stage_dir}: folder name has no trailing integer."
                )
                continue
            if not (1 <= stage <= n_stages):
                warnings.warn(
                    f"Skipping {stage_dir}: stage {stage} outside 1..{n_stages}."
                )
                continue

            days = stage_to_days(fruit, stage)
            for img in stage_dir.iterdir():
                if img.suffix.lower() in VALID_EXTS:
                    records.append(Sample(img, fruit, stage, days))

    if not records:
        sys.exit(f"[dataset] No images found under {data_root}.")

    df = pd.DataFrame(
        [(str(s.path), s.fruit, s.stage, s.days) for s in records],
        columns=["path", "fruit", "stage", "days"],
    )
    print(
        f"[dataset] Indexed {len(df):,} images across {df.fruit.nunique()} fruits."
    )
    return df


# Stratified split

def stratified_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test, stratified by (fruit, stage).

    Each (fruit, stage) bin is split independently so every cell is
    represented in every split (per the proposal's stratification choice).
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for (fruit, stage), group in df.groupby(["fruit", "stage"]):
        idx = group.index.to_numpy().copy()  # .copy(): pandas index arrays are read-only
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train : n_train + n_val])
        test_idx.extend(idx[n_train + n_val :])

    train, val, test = (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )
    print(
        f"[dataset] Split sizes  -  train: {len(train):,}  "
        f"val: {len(val):,}  test: {len(test):,}"
    )
    return train, val, test


# Encoding helpers

def fruit_to_index(fruit: str) -> int:
    """Map a fruit name to its index in the canonical FRUIT_LIST order."""
    return FRUIT_LIST.index(fruit)


def one_hot_fruit(fruits: pd.Series) -> np.ndarray:
    """Vectorised one-hot encoding for a series of fruit names.

    Returns an (n_samples, n_fruits) float32 array.
    """
    n = len(fruits)
    out = np.zeros((n, len(FRUIT_LIST)), dtype=np.float32)
    for i, f in enumerate(fruits):
        out[i, fruit_to_index(f)] = 1.0
    return out


if __name__ == "__main__":
    # Smoke test: index the dataset and print per-fruit/stage counts.
    df = build_index()
    print(df.groupby(["fruit", "stage"]).size().rename("n").reset_index())
