"""
Pipeline B from the proposal:
    HSV histogram + HOG + one-hot fruit type  ->  Random Forest regressor.

Random Forest is robust to mixed feature types, doesn't need scaling, and
exposes feature-importance scores -- which we use for the error analysis.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import ARTIFACTS, HSV_BINS, RF_PARAMS
from data.dataset_loader import one_hot_fruit
from features.color_histogram import extract_batch as hsv_batch
from features.hog_features import extract_batch as hog_batch


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, feature_lengths) where feature_lengths is [n_hsv, n_hog, n_fruit].

    Lengths are returned so we can later compute aggregated importance scores
    by feature group.
    """
    hsv = hsv_batch(df["path"].tolist(), desc="HSV")
    hog = hog_batch(df["path"].tolist(), desc="HOG")
    fruit = one_hot_fruit(df["fruit"])
    X = np.concatenate([hsv, hog, fruit], axis=1)
    lengths = np.array([hsv.shape[1], hog.shape[1], fruit.shape[1]], dtype=int)
    return X, lengths


def train_rf(train_df: pd.DataFrame) -> tuple[RandomForestRegressor, np.ndarray]:
    print("[rf] Building train features...")
    X_train, lengths = build_features(train_df)
    y_train = train_df["days"].to_numpy(dtype=np.float32)

    model = RandomForestRegressor(**RF_PARAMS)
    print(f"[rf] Fitting RandomForestRegressor on {X_train.shape[0]:,} samples, "
          f"{X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    return model, lengths


def grouped_importance(model: RandomForestRegressor, lengths: np.ndarray) -> dict[str, float]:
    """Aggregate sklearn feature importances by feature group.

    Returns the fraction of total importance attributable to colour, texture,
    and fruit-type respectively -- the question Section V.B of the proposal
    promises to answer.
    """
    imp = model.feature_importances_
    n_hsv, n_hog, _ = lengths
    return {
        "colour": float(imp[:n_hsv].sum()),
        "texture": float(imp[n_hsv : n_hsv + n_hog].sum()),
        "fruit_type": float(imp[n_hsv + n_hog :].sum()),
    }


def save(model: RandomForestRegressor, lengths: np.ndarray, out_dir: Path = ARTIFACTS):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "lengths": lengths}, out_dir / "rf.joblib")
    print(f"[rf] Saved -> {out_dir / 'rf.joblib'}")


def load(out_dir: Path = ARTIFACTS) -> tuple[RandomForestRegressor, np.ndarray]:
    bundle = joblib.load(out_dir / "rf.joblib")
    return bundle["model"], bundle["lengths"]


def predict(model: RandomForestRegressor, df: pd.DataFrame) -> np.ndarray:
    X, _ = build_features(df)
    return model.predict(X)
