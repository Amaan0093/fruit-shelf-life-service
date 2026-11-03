from pathlib import Path

from fastapi.testclient import TestClient

from app.main import MAX_UPLOAD_BYTES, app

client = TestClient(app)
IMAGE = next((Path(__file__).parent / "fixtures").glob("banana.*"))


def post(fruit, data, name="fruit.png"):
    return client.post("/predict", data={"fruit": fruit}, files={"file": (name, data)})


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_valid_image():
    body = post("banana", IMAGE.read_bytes()).json()
    assert body["fruit"] == "banana"
    assert 0 <= body["predicted_days"] <= 7


def test_rejects_unknown_fruit():
    assert post("mango", IMAGE.read_bytes()).status_code == 422


def test_rejects_non_image():
    assert post("banana", b"not an image").status_code == 400


def test_rejects_empty_file():
    assert post("banana", b"").status_code == 400


def test_rejects_oversized_file():
    assert post("banana", b"0" * (MAX_UPLOAD_BYTES + 1)).status_code == 413