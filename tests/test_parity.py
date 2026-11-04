"""The API's preprocessing must reproduce the saved test predictions."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from config import ARTIFACTS
from data.dataset_loader import fruit_to_index
from models.mobilenet_model import eval_transform, load

IMAGES = list((Path(__file__).parent / "fixtures" / "parity").iterdir())


def test_matches_saved_predictions():
    saved = pd.read_csv(ARTIFACTS / "predictions_mobilenet.csv")
    saved["name"] = saved["path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    expected = saved.set_index("name")

    assert len(IMAGES) == 3
    model = load(device="cpu")
    transform = eval_transform()
    for path in IMAGES:
        row = expected.loc[path.name]
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            pred = model(tensor, torch.tensor([fruit_to_index(row["fruit"])])).item()
        assert abs(pred - row["y_pred"]) < 1e-3