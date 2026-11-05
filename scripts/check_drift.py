"""
Compare prediction logs with the reference and recommend retraining if needed.

    python scripts/check_drift.py outputs/production_logs.json

Exit codes: 0 = no action, 2 = retraining review recommended.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ARTIFACTS
from evaluation.drift import drift_report, load_prediction_logs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs")
    parser.add_argument("--min-samples", type=int, default=200)
    args = parser.parse_args()

    current = load_prediction_logs(args.logs)
    print(f"Prediction events found: {len(current)}")
    if len(current) < args.min_samples:
        print(f"NOT ENOUGH DATA: need at least {args.min_samples}, so no decision was made.")
        return 0

    report = drift_report(pd.read_csv(ARTIFACTS / "monitoring_reference.csv"), current)
    print(report.round(3).to_string(index=False))
    report.to_csv(ARTIFACTS / "drift_report.csv", index=False)

    if (report["status"] == "major").any():
        print("RETRAINING REVIEW RECOMMENDED: at least one feature shows a major shift.")
        return 2
    print("NO ACTION: no major shift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())