"""
statistical_detector.py
Rule-based statistical detector using rolling variance and rate-of-
change. Purpose-built to catch anomalies that ML models miss because
they look "too simple" rather than "too extreme": stuck sensors,
suspiciously low-variance (too_clean) windows, and slow drift.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/processed/labeled_data.parquet")
FEATURE = "Voltage"
ROLLING_WINDOW = 15  # minutes

# Learned from normal data below, not hardcoded guesses
LOW_VARIANCE_PERCENTILE = 1    # bottom 1% of normal rolling std = suspicious
RATE_OF_CHANGE_PERCENTILE = 99  # top 1% of normal rate-of-change = suspicious


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # Rolling standard deviation: how much natural "wiggle" is in each window
    df["rolling_std"] = df[FEATURE].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()

    # Rate of change: how much the value jumps between consecutive readings
    df["rate_of_change"] = df[FEATURE].diff().abs()

    normal_mask = df["label"] == "normal"

    std_threshold = df.loc[normal_mask, "rolling_std"].quantile(LOW_VARIANCE_PERCENTILE / 100)
    roc_threshold = df.loc[normal_mask, "rate_of_change"].quantile(RATE_OF_CHANGE_PERCENTILE / 100)

    print(f"Low-variance threshold (flatline/too-clean signal): std < {std_threshold:.4f}")
    print(f"Rate-of-change threshold (spike signal): change > {roc_threshold:.4f}")

    # Flag: suspiciously flat OR suspiciously jumpy
    df["predicted_anomaly"] = (
        (df["rolling_std"] < std_threshold) | (df["rate_of_change"] > roc_threshold)
    )
    df["actual_anomaly"] = df["label"] != "normal"

    print("\n--- Evaluation against injected labels ---")
    valid = df["rolling_std"].notna()  # first ROLLING_WINDOW rows have no rolling std yet
    print(classification_report(
        df.loc[valid, "actual_anomaly"], df.loc[valid, "predicted_anomaly"],
        target_names=["normal", "anomaly"],
    ))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(df.loc[valid, "actual_anomaly"], df.loc[valid, "predicted_anomaly"]))

    print("\nDetection rate by anomaly_type:")
    for atype in df["anomaly_type"].unique():
        if atype == "none":
            continue
        subset = df.loc[valid & (df["anomaly_type"] == atype)]
        caught = subset["predicted_anomaly"].mean()
        print(f"  {atype}: {caught:.1%} caught ({len(subset)} rows)")


if __name__ == "__main__":
    main()