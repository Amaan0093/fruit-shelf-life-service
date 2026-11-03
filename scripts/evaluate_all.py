"""
Evaluate all three trained models on the held-out test set:

  - Overall MAE / RMSE / R^2
  - Per-fruit and per-(fruit, stage) breakdowns
  - CPU inference latency (1000 images)
  - Model size on disk

    python scripts/evaluate_all.py

Outputs:
  outputs/predictions_{svr,rf,mobilenet}.csv
  outputs/metrics_summary.csv
  outputs/per_fruit_metrics.csv
  outputs/latency.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS
from evaluation.metrics import overall_metrics, per_fruit, per_stage, report
from evaluation.benchmark import (
    model_size_mb,
    time_mobilenet,
    time_rf,
    time_svr,
)
from models import svr_model, random_forest, mobilenet_model


def main():
    test = pd.read_csv(ARTIFACTS / "test.csv")
    summary_rows = []
    per_fruit_rows = []
    latency_rows = []

    # SVR 
    if (ARTIFACTS / "svr.joblib").exists():
        scaler, svr, params = svr_model.load()
        preds = svr_model.predict(scaler, svr, test)
        out = test.assign(y_pred=preds)
        out.to_csv(ARTIFACTS / "predictions_svr.csv", index=False)
        report("SVR (test)", test, preds)
        m = overall_metrics(test["days"].to_numpy(), preds)
        summary_rows.append({"model": "SVR", **m,
                             "size_mb": model_size_mb(ARTIFACTS / "svr.joblib"),
                             "params": params})
        per_fruit_rows.append(per_fruit(test, preds).assign(model="SVR"))
        latency_rows.append(time_svr(scaler, svr, test, n=1000))

    # Random Forest 
    if (ARTIFACTS / "rf.joblib").exists():
        rf, lengths = random_forest.load()
        preds = random_forest.predict(rf, test)
        out = test.assign(y_pred=preds)
        out.to_csv(ARTIFACTS / "predictions_rf.csv", index=False)
        report("Random Forest (test)", test, preds)
        m = overall_metrics(test["days"].to_numpy(), preds)
        summary_rows.append({"model": "RandomForest", **m,
                             "size_mb": model_size_mb(ARTIFACTS / "rf.joblib"),
                             "params": dict(zip(["n_hsv","n_hog","n_fruit"], lengths.tolist()))})
        per_fruit_rows.append(per_fruit(test, preds).assign(model="RandomForest"))
        latency_rows.append(time_rf(rf, test, n=1000))

    # MobileNetV2
    if (ARTIFACTS / "mobilenet.pt").exists():
        net = mobilenet_model.load(device="cpu")
        preds = mobilenet_model.predict(net, test, device="cpu")
        out = test.assign(y_pred=preds)
        out.to_csv(ARTIFACTS / "predictions_mobilenet.csv", index=False)
        report("MobileNetV2 (test)", test, preds)
        m = overall_metrics(test["days"].to_numpy(), preds)
        summary_rows.append({"model": "MobileNetV2", **m,
                             "size_mb": model_size_mb(ARTIFACTS / "mobilenet.pt"),
                             "params": "ImageNet-V2 pretrained"})
        per_fruit_rows.append(per_fruit(test, preds).assign(model="MobileNetV2"))
        latency_rows.append(time_mobilenet(net, test, n=1000))

    if not summary_rows:
        sys.exit("No trained models found in outputs/. Train at least one first.")

    pd.DataFrame(summary_rows).to_csv(ARTIFACTS / "metrics_summary.csv", index=False)
    pd.concat(per_fruit_rows).to_csv(ARTIFACTS / "per_fruit_metrics.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(ARTIFACTS / "latency.csv", index=False)
    print(f"\nWrote summaries to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
