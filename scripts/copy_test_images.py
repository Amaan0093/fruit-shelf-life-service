"""
Copy one held-out test image per fruit into tests/fixtures/parity, so the
parity test can run without the full dataset.

    python scripts/copy_test_images.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS, PROJECT_ROOT


def main():
    out = PROJECT_ROOT / "tests" / "fixtures" / "parity"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ARTIFACTS / "predictions_mobilenet.csv").groupby("fruit").head(1)
    for row in rows.itertuples():
        shutil.copy(row.path, out)
        print("copied", Path(row.path).name)


if __name__ == "__main__":
    main()