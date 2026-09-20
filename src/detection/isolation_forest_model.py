"""
isolation_forest_model.py
Trains an Isolation Forest anomaly detector on the sensor signals.
Unsupervised: does not use the injected labels during training —
labels are only used afterward to evaluate how well it did.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/processed/labeled_data.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

# The signals the model will actually look at
FEATURES = [
    "Voltage", "Global_active_power", "Global_reactive_power",
    "Global_intensity", "power_factor", "frequency", "thd",
]


def load_data():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    return df


def train_model(df):
    X = df[FEATURES]

    # contamination = expected proportion of anomalies in the data.
    # We know from injection: ~400 windows out of ~2.05M rows worth
    # of data, roughly 1.3% of rows are anomalous - we tell the model
    # roughly what fraction to expect flagging as outliers.
    contamination = (df["label"] != "normal").mean()
    print(f"Estimated contamination rate: {contamination:.4f}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )
    print("Training Isolation Forest...")
    model.fit(X)
    return model


def evaluate(model, df):
    X = df[FEATURES]
    # predict() returns 1 for normal, -1 for anomaly
    predictions = model.predict(X)
    df["predicted_anomaly"] = predictions == -1
    df["actual_anomaly"] = df["label"] != "normal"

    print("\n--- Evaluation against injected labels ---")
    print(classification_report(
        df["actual_anomaly"], df["predicted_anomaly"],
        target_names=["normal", "anomaly"],
    ))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(df["actual_anomaly"], df["predicted_anomaly"]))

    # Breakdown: how well did we catch each specific anomaly type?
    print("\nDetection rate by anomaly_type:")
    for atype in df["anomaly_type"].unique():
        if atype == "none":
            continue
        subset = df[df["anomaly_type"] == atype]
        caught = subset["predicted_anomaly"].mean()
        print(f"  {atype}: {caught:.1%} caught ({len(subset)} rows)")

    return df


def main():
    df = load_data()
    model = train_model(df)
    df = evaluate(model, df)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()