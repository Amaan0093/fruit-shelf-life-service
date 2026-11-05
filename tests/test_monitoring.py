import json
import logging

import numpy as np
import pandas as pd
from PIL import Image

from app.monitoring import image_stats, log_prediction
from evaluation.drift import category_psi, drift_report, load_prediction_logs, psi


def make_frame(brightness, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "mean_brightness": rng.normal(brightness, 0.1, n),
        "mean_saturation": rng.normal(0.4, 0.1, n),
        "predicted_days": rng.normal(6, 2, n),
        "fruit": rng.choice(["apple", "banana", "orange"], n),
    })


def test_psi_is_low_for_same_distribution():
    assert psi(make_frame(0.5)["mean_brightness"], make_frame(0.5, seed=1)["mean_brightness"]) < 0.1


def test_psi_is_high_for_shifted_distribution():
    assert psi(make_frame(0.5)["mean_brightness"], make_frame(0.2, seed=1)["mean_brightness"]) > 0.25


def test_category_psi_detects_new_mix():
    reference = pd.Series(["apple", "banana", "orange"] * 100)
    assert category_psi(reference, reference) == 0
    assert category_psi(reference, pd.Series(["apple"] * 300)) > 0.25


def test_report_marks_major_shift():
    report = drift_report(make_frame(0.5), make_frame(0.2, seed=1)).set_index("feature")
    assert report.loc["mean_brightness", "status"] == "major"
    assert report.loc["mean_saturation", "status"] == "stable"


def test_image_stats_range():
    assert image_stats(Image.new("RGB", (4, 4), (255, 255, 255))) == {"mean_saturation": 0.0, "mean_brightness": 1.0}


def test_log_contains_only_given_fields(caplog):
    with caplog.at_level(logging.INFO, logger="fruit_api.predictions"):
        log_prediction(fruit="apple", predicted_days=3.5)
    assert json.loads(caplog.records[-1].getMessage()) == {"event": "prediction", "fruit": "apple", "predicted_days": 3.5}


def test_loads_azure_json_and_plain_logs(tmp_path):
    event = json.dumps({"event": "prediction", "fruit": "apple", "predicted_days": 1.0})
    azure = tmp_path / "azure.json"
    azure.write_text(json.dumps([{"Log_s": event}, {"Log_s": "INFO: 200 OK"}]))
    plain = tmp_path / "plain.log"
    plain.write_text(f"INFO: started\n{event}\n")
    assert len(load_prediction_logs(azure)) == 1
    assert len(load_prediction_logs(plain)) == 1