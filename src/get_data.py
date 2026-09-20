"""
get_data.py
Downloads and cleans the UCI Individual Household Electric Power
Consumption dataset, used as our baseline "clean" sensor data.
"""

import zipfile
import io
import requests
import pandas as pd
from pathlib import Path

# --- Paths ---
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
RAW_ZIP_PATH = RAW_DIR / "household_power_consumption.zip"
RAW_TXT_PATH = RAW_DIR / "household_power_consumption.txt"
PROCESSED_PATH = PROCESSED_DIR / "baseline_clean.parquet"


def download_data():
    """Download the dataset zip if we don't already have it."""
    if RAW_TXT_PATH.exists():
        print(f"Raw data already exists at {RAW_TXT_PATH}, skipping download.")
        return

    print(f"Downloading dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL)
    response.raise_for_status()  # error out loudly if download failed

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(RAW_DIR)

    print(f"Downloaded and extracted to {RAW_DIR}")


def load_and_clean():
    """Load the raw semicolon-separated file and clean it up."""
    print("Loading raw data...")
    df = pd.read_csv(
        RAW_TXT_PATH,
        sep=";",
        low_memory=False,
        na_values=["?"],  # the dataset uses "?" for missing values
    )

    # Combine Date + Time into a single datetime column
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])

    # Convert measurement columns to numeric (they load as strings due to "?")
    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing readings
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    print(f"Dropped {before - after} rows with missing values ({after} remain)")

    df = df.sort_values("timestamp").reset_index(drop=True)
     # Derive apparent power (kVA) and power factor from voltage/current/active power
    df["apparent_power"] = (df["Voltage"] * df["Global_intensity"]) / 1000
    df["power_factor"] = df["Global_active_power"] / df["apparent_power"]
    # Guard against divide-by-zero when apparent power is ~0 (e.g. no load)
    df["power_factor"] = df["power_factor"].clip(upper=1.0)
    return df


def main():
    download_data()
    df = load_and_clean()
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")
    print(df.head())
    print(df.describe())


if __name__ == "__main__":
    main()