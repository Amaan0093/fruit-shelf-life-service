"""The refactored feature code must still reproduce the saved test predictions."""
import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS, DATA_ROOT, FRUIT_LIST
from features.color_histogram import hsv_histogram

DATASET_FILES = {p.name: p for p in DATA_ROOT.rglob("*") if p.is_file()}


def test_matches_saved_predictions():
    saved = pd.read_csv(ARTIFACTS / "predictions_svr.csv")
    saved["name"] = saved["path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    rows = saved[saved["name"].isin(DATASET_FILES)]
    assert len(rows) > 0

    bundle = joblib.load(ARTIFACTS / "svr.joblib")
    for row in rows.itertuples():
        one_hot = np.eye(len(FRUIT_LIST), dtype=np.float32)[FRUIT_LIST.index(row.fruit)]
        x = np.concatenate([hsv_histogram(DATASET_FILES[row.name]), one_hot]).reshape(1, -1)
        pred = bundle["model"].predict(bundle["scaler"].transform(x))[0]
        assert abs(pred - row.y_pred) < 1e-3