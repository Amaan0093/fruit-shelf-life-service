"""
HSV colour histogram features.

Per the proposal: 32 bins per channel -> 96-dimensional feature vector
per image. We use HSV (rather than RGB) because it separates luminance
from chrominance and is more robust to lighting variation, as Ropelewska
& Rutkowski (2021) [4] reported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from config import HSV_BINS, TRADITIONAL_IMG_SIZE


def hsv_histogram_from_bgr(img: np.ndarray, bins: int = HSV_BINS) -> np.ndarray:
    """Extract the HSV histogram from an already-decoded BGR image."""
    img = cv2.resize(img, TRADITIONAL_IMG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    parts = []
    # OpenCV uses H in [0,180), S and V in [0,256).
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        h = cv2.calcHist([hsv], [ch], None, [bins], [lo, hi]).flatten()
        s = h.sum()
        if s > 0:
            h = h / s
        parts.append(h)
    return np.concatenate(parts).astype(np.float32)


def hsv_histogram(path: str | Path, bins: int = HSV_BINS) -> np.ndarray:
    """Extract a normalised concatenated HSV histogram for one image file."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return hsv_histogram_from_bgr(img, bins)


def extract_batch(paths: Iterable[str | Path], desc: str = "HSV") -> np.ndarray:
    """Extract HSV histograms for an iterable of image paths.

    Returns an (n_samples, 3*bins) float32 array.
    """
    feats = [hsv_histogram(p) for p in tqdm(list(paths), desc=desc)]
    return np.stack(feats).astype(np.float32)
