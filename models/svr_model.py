"""
Pipeline A from the proposal:
    HSV histogram + one-hot fruit type  ->  SVR (RBF) with grid-search.

Trained on `train`, hyperparameters selected on `val`, final score on `test`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import ARTIFACTS, SEED, SVR_GRID
from data.dataset_loader import one_hot_fruit
from features.color_histogram import extract_batch as hsv_batch


def build_features(df: pd.DataFrame) -> np.ndarray:
    # Concat HSV histogram with one-hot fruit type.
    hsv = hsv_batch(df["path"].tolist(), desc="HSV")
    fruit = one_hot_fruit(df["fruit"])
    return np.concatenate([hsv, fruit], axis=1)


def train_svr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[StandardScaler, SVR, dict]:
    # Train SVR with manual grid search; pick best hyperparams by val MAE.
    print("[svr] Building train features...")
    X_train = build_features(train_df)
    y_train = train_df["days"].to_numpy(dtype=np.float32)

    print("[svr] Building val features...")
    X_val = build_features(val_df)
    y_val = val_df["days"].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_model: SVR | None = None
    best_mae = float("inf")
    best_params: dict = {}

    for C in SVR_GRID["C"]:
        for gamma in SVR_GRID["gamma"]:
            model = SVR(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_val_s)
            mae = float(np.mean(np.abs(preds - y_val)))
            print(f"[svr] C={C:>5} gamma={str(gamma):>6}  ->  val MAE = {mae:.4f}")
            if mae < best_mae:
                best_mae, best_model, best_params = mae, model, {"C": C, "gamma": gamma}

    assert best_model is not None
    print(f"[svr] Best params: {best_params}  (val MAE = {best_mae:.4f})")
    return scaler, best_model, best_params


def save(scaler: StandardScaler, model: SVR, params: dict, out_dir: Path = ARTIFACTS):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "params": params}, out_dir / "svr.joblib")
    print(f"[svr] Saved -> {out_dir / 'svr.joblib'}")


def load(out_dir: Path = ARTIFACTS) -> tuple[StandardScaler, SVR, dict]:
    bundle = joblib.load(out_dir / "svr.joblib")
    return bundle["scaler"], bundle["model"], bundle["params"]


def predict(scaler: StandardScaler, model: SVR, df: pd.DataFrame) -> np.ndarray:
    X = build_features(df)
    return model.predict(scaler.transform(X))
