"""
Model training script for TrackX railway monitoring.
Trains a RandomForestClassifier on railway measurements to predict condition classes
and calculate calibrated error percentages.
Saves model dictionary with metadata (features, classes) to trained_models/train_condition_model.joblib.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "railway_1km_measurements.csv"
OUTPUT_DIR = ROOT / "trained_models"
MODEL_PATH = OUTPUT_DIR / "train_condition_model.joblib"


def train_and_save():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    target_col = "Train condition"
    excluded_cols = {target_col, "Crack-vicinity label"}
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Classes: {y.unique().tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "features": feature_cols,
        "classes": clf.classes_.tolist(),
        "target": target_col,
        "metrics": {"accuracy": round(float(acc), 4)}
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved trained model artifact to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    train_and_save()
