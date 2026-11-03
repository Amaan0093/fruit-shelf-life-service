"""
Inference-latency benchmarking on CPU.

Per the proposal: report mean +/- std latency in milliseconds per image,
measured over (up to) 1,000 test images, plus the on-disk model size.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.color_histogram import hsv_histogram
from features.hog_features import hog_features
from data.dataset_loader import one_hot_fruit, fruit_to_index
from models.mobilenet_model import FruitImageDataset, FruitShelfLifeNet


# Per-image timers (we want PER-IMAGE latency, not batched throughput)

def time_svr(scaler, model, df: pd.DataFrame, n: int = 1000) -> dict:
    """SVR: per-image HSV extraction + scaler.transform + model.predict."""
    sub = df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
    times_ms: list[float] = []
    for _, row in sub.iterrows():
        t0 = time.perf_counter()
        hsv = hsv_histogram(row["path"])
        fruit_oh = np.zeros(len(scaler.mean_) - len(hsv), dtype=np.float32)
        fruit_oh[fruit_to_index(row["fruit"])] = 1.0
        x = np.concatenate([hsv, fruit_oh]).reshape(1, -1)
        x = scaler.transform(x)
        _ = model.predict(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summary("SVR", times_ms)


def time_rf(model, df: pd.DataFrame, n: int = 1000) -> dict:
    """Random Forest: per-image HSV + HOG + one-hot + model.predict."""
    sub = df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
    n_fruits = model.n_features_in_ - (
        len(hsv_histogram(sub.iloc[0]["path"])) + len(hog_features(sub.iloc[0]["path"]))
    )
    times_ms: list[float] = []
    for _, row in sub.iterrows():
        t0 = time.perf_counter()
        hsv = hsv_histogram(row["path"])
        hog = hog_features(row["path"])
        fruit_oh = np.zeros(n_fruits, dtype=np.float32)
        fruit_oh[fruit_to_index(row["fruit"])] = 1.0
        x = np.concatenate([hsv, hog, fruit_oh]).reshape(1, -1)
        _ = model.predict(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summary("RF", times_ms)


def time_mobilenet(model: FruitShelfLifeNet, df: pd.DataFrame, n: int = 1000) -> dict:
    """MobileNetV2: per-image preprocessing + forward pass on CPU."""
    sub = df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
    ds = FruitImageDataset(sub, training=False)
    model = model.cpu().eval()
    times_ms: list[float] = []
    with torch.no_grad():
        for i in range(len(ds)):
            t0 = time.perf_counter()
            img, fruit, _ = ds[i]
            img = img.unsqueeze(0)
            fruit_t = torch.tensor([fruit])
            _ = model(img, fruit_t)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summary("MobileNetV2", times_ms)


def _summary(name: str, times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    out = {
        "model": name,
        "n_images": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
    print(f"[latency] {name:12s}  mean={out['mean_ms']:.2f}+/-{out['std_ms']:.2f}ms  "
          f"p50={out['p50_ms']:.2f}  p95={out['p95_ms']:.2f}  (n={out['n_images']})")
    return out


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
