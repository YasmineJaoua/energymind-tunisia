"""
explain.py
Trains a small, interpretable classifier on the discriminator's
engineered signals, then uses SHAP to explain each flagged anomaly's
verdict (genuine_grid_event vs sensor_fault) in plain language, with
a confidence score and an auditable decision log entry.
"""
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for script use
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/labeled_data.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "explainer_classifier.joblib"

FEATURES = [
    "physical_inconsistency",
    "rolling_std",
    "outage_flag",
    "rate_of_change",
]

FEATURE_DESCRIPTIONS = {
    "physical_inconsistency": "mismatch between actual power and power expected from voltage x current",
    "rolling_std": "natural signal variance over the last 15 minutes",
    "outage_flag": "whether voltage, power, and current all dropped to zero together",
    "rate_of_change": "how much the reading jumped from the previous minute",
}

def build_features(df):
    """Recompute the engineered signals from discriminators.py, so
    this file can run standalone on the labeled dataset."""
    expected_power = (df["Voltage"] * df["Global_intensity"] * df["power_factor"] / 1000)
    actual_power = df["Global_active_power"]
    denom = expected_power.abs().clip(lower=0.05)
    df["physical_inconsistency"] = (actual_power - expected_power).abs() / denom
    df["rolling_std"] = df["Voltage"].rolling(15, min_periods=15).std()
    df["rate_of_change"] = df["Voltage"].diff().abs()
    df["outage_flag"] = (
        (df["Voltage"] < 1.0) & (df["Global_active_power"] < 1.0) & (df["Global_intensity"] < 1.0)
    ).astype(int)
    return df

def train_classifier(df):
    anomalous = df[df["label"] != "normal"].copy()
    anomalous = anomalous.dropna(subset=FEATURES)
    X = anomalous[FEATURES]
    y = anomalous["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Small, shallow forest - accurate enough, and shallow trees keep
    # SHAP explanations simple and fast to compute
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("--- Classifier evaluation ---")
    print(classification_report(y_test, model.predict(X_test)))
    return model, X_test, y_test

def explain_instance(model, explainer, X_row, prediction, confidence):
    """Turn one row's SHAP values into a plain-language explanation."""
    shap_values = explainer.shap_values(X_row.to_frame().T)
    
    # Extraire le tableau 1D pour la classe prédite selon le format retourne par SHAP
    class_idx = list(model.classes_).index(prediction)
    if isinstance(shap_values, list):
        # Format liste : [array_class_0, array_class_1]
        values = np.array(shap_values[class_idx]).squeeze()
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Format 3D : [samples, features, classes]
        values = shap_values[0, :, class_idx]
    else:
        values = np.array(shap_values).squeeze()

    contributions = sorted(
        zip(FEATURES, values), key=lambda x: abs(float(x[1])), reverse=True
    )
    top = contributions[0]
    feature_name, contribution = top
    direction = "pointed toward" if contribution > 0 else "pointed away from"
    explanation = (
        f"Verdict: {prediction} (confidence: {confidence:.0%}). "
        f"Main factor: {FEATURE_DESCRIPTIONS[feature_name]} "
        f"({feature_name}={X_row[feature_name]:.3f}) {direction} this verdict."
    )
    return explanation, contributions

def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)
    model, X_test, y_test = train_classifier(df)
    joblib.dump(model, MODEL_PATH)

    print("\nBuilding SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_test)

    print("\n--- Global feature importance (mean |SHAP value|) ---")
    # Standardisation en matrice 2D [n_echantillons, n_features] pour la classe d'interet
    if isinstance(shap_values_all, list):
        shap_array = np.array(shap_values_all[1])
    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        shap_array = shap_values_all[:, :, 1]
    else:
        shap_array = np.array(shap_values_all)

    # Calcul de la moyenne des valeurs absolues pour chaque variable (vecteur 1D)
    mean_abs = np.abs(shap_array).mean(axis=0)

    for feat, val in sorted(zip(FEATURES, mean_abs), key=lambda x: -float(x[1])):
        print(f"  {feat}: {float(val):.4f}")
 # Save a SHAP summary bar plot as a report figure
    figures_dir = Path("reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_array, X_test, feature_names=FEATURES,
        plot_type="bar", show=False,
    )
    plt.title("SHAP Feature Importance — Genuine Grid Event vs Sensor Fault")
    plt.tight_layout()
    fig_path = figures_dir / "shap_feature_importance.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved SHAP summary plot to {fig_path}")
    
    # Example: explain a few individual anomalies
    print("\n--- Example individual explanations ---")
    sample = X_test.sample(5, random_state=1)
    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)
    decision_log = []

    for i, (idx, row) in enumerate(sample.iterrows()):
        pred = predictions[i]
        conf = probabilities[i].max()
        explanation, contributions = explain_instance(model, explainer, row, pred, conf)
        print(f"\n[{idx}] {explanation}")
        decision_log.append({
            "row_index": idx,
            "verdict": pred,
            "confidence": conf,
            "explanation": explanation,
        })

    log_df = pd.DataFrame(decision_log)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    log_path = reports_dir / "decision_log_sample.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nSaved sample decision log to {log_path}")

if __name__ == "__main__":
    main()