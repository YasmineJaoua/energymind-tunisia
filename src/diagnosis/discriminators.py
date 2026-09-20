"""
discriminators.py
Given a flagged anomaly window, determines whether it's a genuine
grid event or a sensor/data-quality fault, using physical-consistency
and statistical checks - this is what turns "something's weird here"
into "here's what it actually is."
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/processed/labeled_data.parquet")


def physical_consistency_check(df, tolerance=0.05):
    """
    Compares actual active power to power expected from
    Voltage x Current x power_factor. A mismatch suggests one signal
    has drifted/biased away from physical reality - a sensor fault,
    not a real grid event.
    """
    expected_power = (
        df["Voltage"] * df["Global_intensity"] * df["power_factor"] / 1000
    )
    actual_power = df["Global_active_power"]
    denom = expected_power.abs().clip(lower=0.05)
    relative_error = (actual_power - expected_power).abs() / denom

    df["physical_inconsistency"] = relative_error
    df["physical_check_flag"] = relative_error > tolerance
    return df


def variance_check(df, feature="Voltage", window=15, low_pct=1):
    """
    Rolling std check - suspiciously low variance points to a
    sensor_fault (too_clean/stuck sensor signature), not a genuine
    grid event.
    """
    rolling_std = df[feature].rolling(window, min_periods=window).std()
    normal_mask = df["label"] == "normal"
    threshold = rolling_std[normal_mask].quantile(low_pct / 100)
    df["low_variance_flag"] = rolling_std < threshold
    return df


def outage_check(df, threshold=1.0):
    """
    A real outage means Voltage, power, AND current all drop to ~0
    together - a coordinated, physically consistent total loss.
    This distinguishes it from a single flatlined/spoofed sensor
    (which affects one signal while others stay normal).
    """
    df["outage_flag"] = (
        (df["Voltage"] < threshold)
        & (df["Global_active_power"] < threshold)
        & (df["Global_intensity"] < threshold)
    )
    return df


def classify(df):
    """
    Priority order matters:
      1. Outage check FIRST - a coordinated total-zero reading across
         all signals is a genuine event, checked before the
         low-variance rule (which would otherwise misread the exact
         zero as a spoofed flatline).
      2. Low variance -> sensor_fault (too_clean/stuck sensor).
      3. Physical inconsistency -> sensor_fault (bias signature).
      4. Otherwise -> genuine_grid_event.
    """
    df["diagnosed_as"] = np.select(
        [
            df["outage_flag"],
            df["low_variance_flag"],
            df["physical_check_flag"],
        ],
        [
            "genuine_grid_event",
            "sensor_fault",
            "sensor_fault",
        ],
        default="genuine_grid_event",
    )
    return df


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    df = physical_consistency_check(df)
    df = variance_check(df)
    df = outage_check(df)
    df = classify(df)

    anomalous = df[df["label"] != "normal"].copy()
    valid = anomalous["low_variance_flag"].notna()
    anomalous = anomalous[valid]

    print("\n--- Discriminator evaluation (genuine_grid_event vs sensor_fault) ---")
    print(classification_report(anomalous["label"], anomalous["diagnosed_as"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(
        anomalous["label"], anomalous["diagnosed_as"],
        labels=["genuine_grid_event", "sensor_fault"],
    ))

    print("\nDiagnosis accuracy by anomaly_type:")
    for atype in anomalous["anomaly_type"].unique():
        subset = anomalous[anomalous["anomaly_type"] == atype]
        true_label = subset["label"].iloc[0]
        correct = (subset["diagnosed_as"] == true_label).mean()
        print(f"  {atype} (true={true_label}): {correct:.1%} correctly diagnosed")


if __name__ == "__main__":
    main()