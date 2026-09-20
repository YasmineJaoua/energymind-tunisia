"""
inject_anomalies.py
Injects synthetic anomalies into the clean baseline sensor data,
producing labeled training data for the anomaly detector and
discriminators.

Anomaly categories:
  - genuine_grid_event: real physical grid disturbances
  - sensor_fault: data-quality problems originating at the sensor/
    communication layer (not a real grid event)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "baseline_clean.parquet"
OUTPUT_PATH = PROCESSED_DIR / "labeled_data.parquet"

WINDOW_MIN = 15   # shortest anomaly window (minutes)
WINDOW_MAX = 120  # longest anomaly window (minutes)
N_ANOMALIES_PER_TYPE = 40  # how many injected windows per anomaly type

rng = np.random.default_rng(seed=123)


def pick_windows(n_rows, n_windows, min_gap=500):
    """Pick non-overlapping random (start, length) windows in the data."""
    windows = []
    attempts = 0
    while len(windows) < n_windows and attempts < n_windows * 50:
        attempts += 1
        length = rng.integers(WINDOW_MIN, WINDOW_MAX)
        start = rng.integers(0, n_rows - length)
        # Reject if it overlaps (with a buffer) an existing window
        if any(abs(start - w[0]) < min_gap for w in windows):
            continue
        windows.append((start, length))
    return windows


# --- Genuine grid event injectors -----------------------------------

def inject_voltage_sag(df, start, length):
    drop = rng.uniform(0.10, 0.25)
    idx = slice(start, start + length)
    df.loc[df.index[idx], "Voltage"] *= (1 - drop)
    # Full compensation: a constant-power load draws more current as
    # voltage drops, keeping actual power roughly unchanged - this
    # matches the physics our discriminator checks against.
    df.loc[df.index[idx], "Global_intensity"] /= (1 - drop)
    return "voltage_sag", "genuine_grid_event"


def inject_voltage_swell(df, start, length):
    rise = rng.uniform(0.10, 0.20)
    idx = slice(start, start + length)
    df.loc[df.index[idx], "Voltage"] *= (1 + rise)
    df.loc[df.index[idx], "Global_intensity"] /= (1 + rise)
    return "voltage_swell", "genuine_grid_event"

def inject_outage(df, start, length):
    idx = slice(start, start + length)
    df.loc[df.index[idx], "Voltage"] = 0.0
    df.loc[df.index[idx], "Global_active_power"] = 0.0
    df.loc[df.index[idx], "Global_intensity"] = 0.0
    df.loc[df.index[idx], "power_factor"] = 0.0
    return "outage", "genuine_grid_event"


def inject_frequency_deviation(df, start, length):
    deviation = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
    idx = slice(start, start + length)
    df.loc[df.index[idx], "frequency"] += deviation
    return "frequency_deviation", "genuine_grid_event"


def inject_thd_spike(df, start, length):
    idx = slice(start, start + length)
    df.loc[df.index[idx], "thd"] += rng.uniform(8, 15)
    return "thd_spike", "genuine_grid_event"


# --- Sensor / data-quality fault injectors ---------------------------

def inject_drift(df, start, length):
    idx_range = np.arange(start, start + length)
    ramp = np.linspace(0, rng.uniform(0.15, 0.35), length)
    df.loc[df.index[idx_range], "Voltage"] *= (1 + ramp)
    return "drift", "sensor_fault"


def inject_stuck_sensor(df, start, length):
    idx = slice(start, start + length)
    frozen_value = df["Voltage"].iloc[start]
    df.loc[df.index[idx], "Voltage"] = frozen_value
    return "stuck_sensor", "sensor_fault"


def inject_bias(df, start, length):
    idx = slice(start, start + length)
    # Guarantee a meaningful offset (not too close to 0), both
    # directions possible - a real sensor calibration fault is
    # rarely a negligible 1-2V shift
    offset = rng.choice([-1, 1]) * rng.uniform(8, 20)
    df.loc[df.index[idx], "Voltage"] += offset
    return "bias", "sensor_fault"


def inject_falsified_spike(df, start, length):
    # Modify the full window (not just first 10 rows) so the label
    # correctly matches what was actually altered - simulates a
    # sustained comms glitch/spoofing burst rather than one instant
    idx = slice(start, start + length)
    df.loc[df.index[idx], "Voltage"] = rng.uniform(0, 400, size=length)
    return "falsified_spike", "sensor_fault"


def inject_too_clean(df, start, length):
    """
    Suspiciously low-variance window: values are smoothed to near-constant
    with almost no natural noise. Real sensor data always has some jitter;
    a sudden loss of it can indicate spoofed/replayed data (e.g. a
    cyberattack), not a genuine grid state. Labeled as a sensor_fault
    since it's a data-integrity issue, not a physical event.
    """
    idx = slice(start, start + length)
    mean_voltage = df["Voltage"].iloc[start:start + length].mean()
    tiny_noise = rng.normal(0, 0.01, size=length)  # near-zero noise
    df.loc[df.index[idx], "Voltage"] = mean_voltage + tiny_noise
    return "too_clean", "sensor_fault"


INJECTORS = [
    inject_voltage_sag,
    inject_voltage_swell,
    inject_outage,
    inject_frequency_deviation,
    inject_thd_spike,
    inject_drift,
    inject_stuck_sensor,
    inject_bias,
    inject_falsified_spike,
    inject_too_clean,
]


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)

    # Ground-truth labels: normal by default
    df["label"] = "normal"
    df["anomaly_type"] = "none"

    n_rows = len(df)
    total_injected = 0

    for injector in INJECTORS:
        windows = pick_windows(n_rows, N_ANOMALIES_PER_TYPE)
        for start, length in windows:
            anomaly_type, label = injector(df, start, length)
            idx = slice(start, start + length)
            df.loc[df.index[idx], "label"] = label
            df.loc[df.index[idx], "anomaly_type"] = anomaly_type
            total_injected += 1
        print(f"Injected {len(windows)} windows of '{injector.__name__}'")

    print(f"Total anomaly windows injected: {total_injected}")
    print(df["label"].value_counts())
    print(df["anomaly_type"].value_counts())

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved labeled data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()