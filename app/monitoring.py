"""Structured prediction logs, used later to check for drift. No image data is logged."""
import json
import logging
import sys

import numpy as np
from PIL import Image

logger = logging.getLogger("fruit_api.predictions")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def image_stats(img: Image.Image) -> dict:
    """Summarise an RGB image as mean saturation and brightness, each in [0, 1]."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return {
        "mean_saturation": round(float(hsv[:, :, 1].mean()), 4),
        "mean_brightness": round(float(hsv[:, :, 2].mean()), 4),
    }


def log_prediction(**fields) -> None:
    logger.info(json.dumps({"event": "prediction", **fields}))