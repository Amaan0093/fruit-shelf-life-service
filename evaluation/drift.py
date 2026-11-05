"""
Compare live prediction logs with a reference set using PSI (Population
Stability Index). Common rule of thumb: below 0.1 is stable, 0.1 to 0.25 is
worth watching, above 0.25 is a major shift.

This detects changes in inputs and predictions. It cannot measure accuracy,
because live requests have no true shelf-life labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["mean_brightness", "mean_saturation", "predicted_days"]
WATCH_PSI = 0.1
RETRAIN_PSI = 0.25
_EPS = 1e-4


def _psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, _EPS, None), np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of a numeric feature, using quantile bins taken from the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    return _psi_from_shares(ref, cur)


def category_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI of a categorical feature, such as the share of each fruit."""
    ref = reference.value_counts(normalize=True)
    cur = current.value_counts(normalize=True)
    index = ref.index.union(cur.index)
    return _psi_from_shares(
        ref.reindex(index, fill_value=0).to_numpy(),
        cur.reindex(index, fill_value=0).to_numpy(),
    )


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(reference[f], current[f])} for f in NUMERIC_FEATURES]
    rows.append({"feature": "fruit", "psi": category_psi(reference["fruit"], current["fruit"])})
    report = pd.DataFrame(rows)
    report["status"] = np.select(
        [report["psi"] > RETRAIN_PSI, report["psi"] > WATCH_PSI], ["major", "watch"], default="stable"
    )
    return report


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", errors="replace")


def load_prediction_logs(path: str | Path) -> pd.DataFrame:
    """Read prediction events from Azure query JSON, or from plain log lines."""
    text = _read_text(Path(path))
    try:
        lines = [row.get("Log_s", "") for row in json.loads(text)]
    except json.JSONDecodeError:
        lines = text.splitlines()

    events = []
    for line in lines:
        start = line.find("{")
        if start == -1:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "prediction":
            events.append(event)
    return pd.DataFrame(events)