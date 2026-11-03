"""HTTP API that serves the trained SVR shelf-life model."""
from enum import Enum

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import ARTIFACTS, FRUIT_LIST, FRUITS
from features.color_histogram import hsv_histogram_from_bgr

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MODEL_VERSION = "svr-v1"

app = FastAPI(title="Fruit Shelf-Life API", version="1.0.0")
_bundle = joblib.load(ARTIFACTS / "svr.joblib")
_scaler, _model = _bundle["scaler"], _bundle["model"]


class Fruit(str, Enum):
    apple = "apple"
    banana = "banana"
    orange = "orange"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(fruit: Fruit = Form(...), file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(status_code=400, detail="File is not a readable image")

    one_hot = np.zeros(len(FRUIT_LIST), dtype=np.float32)
    one_hot[FRUIT_LIST.index(fruit.value)] = 1.0
    features = np.concatenate([hsv_histogram_from_bgr(img), one_hot]).reshape(1, -1)
    days = float(_model.predict(_scaler.transform(features))[0])

    return {
        "fruit": fruit.value,
        "predicted_days": round(float(np.clip(days, 0, FRUITS[fruit.value]["max_days"])), 2),
        "model_version": MODEL_VERSION,
    }