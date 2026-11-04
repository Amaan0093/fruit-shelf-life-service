"""HTTP API that serves the trained MobileNetV2 shelf-life model."""
from enum import Enum
from io import BytesIO
from time import perf_counter

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from config import FRUITS
from data.dataset_loader import fruit_to_index
from models.mobilenet_model import eval_transform, load

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MODEL_VERSION = "mobilenetv2-v1"

app = FastAPI(title="Fruit Shelf-Life API", version="2.0.0")
_model = load(device="cpu")
_transform = eval_transform()


class Fruit(str, Enum):
    apple = "apple"
    banana = "banana"
    orange = "orange"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(fruit: Fruit = Form(...), file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File is not a readable image")

    start = perf_counter()
    with torch.inference_mode():
        days = _model(_transform(img).unsqueeze(0), torch.tensor([fruit_to_index(fruit.value)])).item()
    inference_ms = (perf_counter() - start) * 1000

    return {
        "fruit": fruit.value,
        "predicted_days": round(min(max(days, 0.0), FRUITS[fruit.value]["max_days"]), 2),
        "model_version": MODEL_VERSION,
        "inference_ms": round(inference_ms, 1),
    }