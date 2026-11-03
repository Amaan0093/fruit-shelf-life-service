"""
Histogram of Oriented Gradients (HOG) features.

Captures local edge / texture structure that pure colour histograms miss
(bruising patterns, mould flecks, peel wrinkling). Per Mazen & Nashat (2019)
[5], texture features add ~8% on top of colour for banana ripeness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from config import HOG_PARAMS, TRADITIONAL_IMG_SIZE


def hog_features(path: str | Path) -> np.ndarray:
    """Extract HOG features for one image. Returns a 1-D float32 array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    img = cv2.resize(img, TRADITIONAL_IMG_SIZE, interpolation=cv2.INTER_AREA)
    feats = hog(img, **HOG_PARAMS, feature_vector=True)
    return feats.astype(np.float32)


def extract_batch(paths: Iterable[str | Path], desc: str = "HOG") -> np.ndarray:
    """Extract HOG features for an iterable of image paths."""
    feats = [hog_features(p) for p in tqdm(list(paths), desc=desc)]
    return np.stack(feats).astype(np.float32)
