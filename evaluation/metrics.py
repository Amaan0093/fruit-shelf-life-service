"""
Evaluation metrics and per-fruit / per-stage breakdowns.

Headline metric is MAE (interpretable -- "off by X days"). RMSE is the
secondary metric (penalises large errors). R^2 reports variance explained.
All three are reported on the held-out test set, both overall and broken
out per fruit, per stage, and per (fruit, stage) cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def breakdown(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    by: list[str],
) -> pd.DataFrame:
    """Group-wise metrics. `df` must include the columns named in `by` and `days`."""
    work = df.copy()
    work["y_pred"] = y_pred
    rows = []
    for keys, sub in work.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(by, keys)),
            **overall_metrics(sub["days"].to_numpy(), sub["y_pred"].to_numpy()),
        })
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def per_fruit(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    return breakdown(df, y_pred, ["fruit"])


def per_stage(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    return breakdown(df, y_pred, ["fruit", "stage"])


def report(name: str, df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Print a tidy report and return a dict of all metrics."""
    y_true = df["days"].to_numpy()
    overall = overall_metrics(y_true, y_pred)
    by_fruit = per_fruit(df, y_pred)
    by_stage = per_stage(df, y_pred)

    print(f"\n=== {name} ===")
    print(f"  overall  MAE={overall['mae']:.3f}d  "
          f"RMSE={overall['rmse']:.3f}d  R^2={overall['r2']:.3f}  "
          f"(n={overall['n']:,})")
    print("  per fruit:")
    for _, r in by_fruit.iterrows():
        print(f"    {r['fruit']:11s} MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
              f"R^2={r['r2']:+.3f}  (n={int(r['n']):,})")

    return {"overall": overall, "by_fruit": by_fruit, "by_stage": by_stage}
