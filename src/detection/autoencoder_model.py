"""
autoencoder_model.py
Trains a neural network autoencoder on short windows of normal sensor
data. Reconstruction error on unseen windows becomes the anomaly
score - the autoencoder is good at catching pattern-based anomalies
(drift, stuck sensor, too_clean) that a row-level model like
Isolation Forest misses.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("data/processed/labeled_data.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "autoencoder.joblib"
SCALER_PATH = MODEL_DIR / "autoencoder_scaler.joblib"

FEATURES = [
    "Voltage", "Global_active_power", "Global_reactive_power",
    "Global_intensity", "power_factor", "frequency", "thd",
]
WINDOW_SIZE = 30  # minutes of context per sample
STEP = 5          # slide the window forward this many rows each time


def load_data():
    print(f"Loading {DATA_PATH} ...")
    return pd.read_parquet(DATA_PATH)


def make_windows(df, features, window_size, step):
    """
    Turn the time series into overlapping windows. Each window becomes
    one flat feature vector (window_size * n_features long). A window
    is labeled anomalous if ANY row inside it was injected as one -
    since a fault window's effects are visible throughout that stretch.
    """
    values = df[features].values
    labels = (df["label"] != "normal").values
    anomaly_types = df["anomaly_type"].values

    X, y, types = [], [], []
    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        X.append(values[start:end].flatten())
        y.append(labels[start:end].any())
        # Take the most common non-"none" type in the window, if any
        window_types = anomaly_types[start:end]
        non_none = window_types[window_types != "none"]
        types.append(non_none[0] if len(non_none) > 0 else "none")

    return np.array(X), np.array(y), np.array(types)


def train_autoencoder(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # A small "bottleneck" architecture: input -> 32 -> 8 -> 32 -> output
    # The 8-unit middle layer forces the network to compress the
    # window down to a compact summary, then reconstruct it - this
    # compression is what makes it learn "normal" structure instead
    # of just memorizing every input.
    n_features = X_train.shape[1]
    model = MLPRegressor(
        hidden_layer_sizes=(32, 8, 32),
        activation="relu",
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    print("Training autoencoder (this may take a few minutes)...")
    model.fit(X_scaled, X_scaled)  # target = input itself
    return model, scaler


def reconstruction_error(model, scaler, X):
    X_scaled = scaler.transform(X)
    X_reconstructed = model.predict(X_scaled)
    # Mean squared error per sample (row-wise)
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    return errors


def main():
    df = load_data()
    print(f"Building windows (size={WINDOW_SIZE}, step={STEP})...")
    X, y, types = make_windows(df, FEATURES, WINDOW_SIZE, STEP)
    print(f"Total windows: {len(X)}, anomalous: {y.sum()} ({y.mean():.2%})")

    # Train ONLY on windows with no injected anomaly - the autoencoder
    # should only ever see "normal" during training
    X_train = X[~y]
    print(f"Training on {len(X_train)} normal-only windows...")

    model, scaler = train_autoencoder(X_train)

    # Score every window (train + anomalous) to set a threshold and evaluate
    errors = reconstruction_error(model, scaler, X)

    # Threshold: flag the top N% highest-error windows as anomalous,
    # matching roughly the true anomaly rate in the data
    threshold = np.percentile(errors, 100 * (1 - y.mean()))
    predicted = errors > threshold
    print(f"\nReconstruction error threshold: {threshold:.4f}")

    print("\n--- Evaluation against injected labels ---")
    print(classification_report(y, predicted, target_names=["normal", "anomaly"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y, predicted))

    print("\nDetection rate by anomaly_type:")
    for atype in np.unique(types):
        if atype == "none":
            continue
        mask = types == atype
        caught = predicted[mask].mean()
        print(f"  {atype}: {caught:.1%} caught ({mask.sum()} windows)")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()