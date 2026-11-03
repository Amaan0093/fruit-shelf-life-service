"""
Ablation: Unified multi-fruit models vs. per-fruit specialists.

Trains an SVR and a Random Forest *per fruit* (using only that fruit's
training data) and compares test-set MAE to the unified models that were
trained on all six fruits at once. This is the methodology described in
Section V.D ("Coherent vs. Fruity Models") of the proposal.

We skip MobileNetV2 here because retraining six deep models is expensive;
extending to MobileNetV2 is a one-line change if you have GPU time.

    python scripts/ablation_study.py

Output:
  outputs/ablation.csv   -- per-fruit MAE for unified vs specialist
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import ARTIFACTS, FRUIT_LIST
from evaluation.metrics import overall_metrics
from models.svr_model import build_features as svr_build, train_svr
from models.random_forest import build_features as rf_build, train_rf


def _filter(df: pd.DataFrame, fruit: str) -> pd.DataFrame:
    return df[df["fruit"] == fruit].reset_index(drop=True)


def main():
    train = pd.read_csv(ARTIFACTS / "train.csv")
    val = pd.read_csv(ARTIFACTS / "val.csv")
    test = pd.read_csv(ARTIFACTS / "test.csv")

    rows: list[dict] = []

    for fruit in FRUIT_LIST:
        tr_f, va_f, te_f = _filter(train, fruit), _filter(val, fruit), _filter(test, fruit)
        if min(len(tr_f), len(va_f), len(te_f)) == 0:
            print(f"[ablation] Skipping {fruit} (insufficient data)")
            continue

        # Specialist SVR 
        sc_s, svr_s, _ = train_svr(tr_f, va_f)
        Xt = sc_s.transform(svr_build(te_f))
        m_svr_spec = overall_metrics(te_f["days"].to_numpy(), svr_s.predict(Xt))

        # Specialist RF
        rf_s, _ = train_rf(tr_f)
        Xt_rf, _ = rf_build(te_f)
        m_rf_spec = overall_metrics(te_f["days"].to_numpy(), rf_s.predict(Xt_rf))

        rows.append({"fruit": fruit, "model": "SVR_specialist",      **m_svr_spec})
        rows.append({"fruit": fruit, "model": "RF_specialist",       **m_rf_spec})

    # Unified models on the same per-fruit slices
    # Load unified models (trained earlier via train_svr.py / train_rf.py).
    from models import svr_model, random_forest
    if (ARTIFACTS / "svr.joblib").exists():
        sc_u, svr_u, _ = svr_model.load()
        for fruit in FRUIT_LIST:
            te_f = _filter(test, fruit)
            if len(te_f) == 0:
                continue
            preds = svr_model.predict(sc_u, svr_u, te_f)
            rows.append({"fruit": fruit, "model": "SVR_unified",
                         **overall_metrics(te_f["days"].to_numpy(), preds)})
    if (ARTIFACTS / "rf.joblib").exists():
        rf_u, _ = random_forest.load()
        for fruit in FRUIT_LIST:
            te_f = _filter(test, fruit)
            if len(te_f) == 0:
                continue
            preds = random_forest.predict(rf_u, te_f)
            rows.append({"fruit": fruit, "model": "RF_unified",
                         **overall_metrics(te_f["days"].to_numpy(), preds)})

    df = pd.DataFrame(rows).sort_values(["fruit", "model"]).reset_index(drop=True)
    df.to_csv(ARTIFACTS / "ablation.csv", index=False)
    print("\n=== Ablation: unified vs per-fruit specialists ===")
    pivot = df.pivot_table(index="fruit", columns="model", values="mae")
    print(pivot.round(3))
    print(f"\nWrote {ARTIFACTS / 'ablation.csv'}")


if __name__ == "__main__":
    main()
