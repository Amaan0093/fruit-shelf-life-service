"""
Build the drift reference from held-out test images and the deployed model's
saved predictions. Run again whenever the deployed model changes.

    python scripts/build_reference.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from app.monitoring import image_stats
from config import ARTIFACTS, FRUITS


def main():
    saved = pd.read_csv(ARTIFACTS / "predictions_mobilenet.csv")
    rows = []
    for row in tqdm(list(saved.itertuples()), desc="reference"):
        with Image.open(row.path) as img:
            stats = image_stats(img.convert("RGB"))
        days = min(max(row.y_pred, 0.0), FRUITS[row.fruit]["max_days"])
        rows.append({"fruit": row.fruit, "predicted_days": round(days, 2), **stats})
    out = ARTIFACTS / "monitoring_reference.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()