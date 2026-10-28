"""
Step 2 - Train the ML model and register it in the local model registry.

Algorithm: Random Forest Classifier (predicts whether a field needs irrigation).
Run:  python train_model.py
Requires: data/sensor_data.csv (run generate_dataset.py first)
"""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import config
import model_registry


def load_dataset() -> pd.DataFrame:
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")
    df = pd.read_csv(config.DATASET_PATH, parse_dates=["timestamp"])
    # Clean: drop rows where the sensor failed (missing soil reading)
    df = df.dropna(subset=config.FEATURES + [config.TARGET])
    return df


def train():
    df = load_dataset()
    X = df[config.FEATURES]
    y = df[config.TARGET]

    # Stratified split keeps the same ratio of irrigation/no-irrigation in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)

    params = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5,
              "class_weight": "balanced", "random_state": config.RANDOM_SEED}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    entry = model_registry.register_model(model, metrics, config.FEATURES, params, promote=True)

    print("\n=== Training complete ===")
    print(f"Registered as version {entry['version']} (stage: {entry['stage']})")
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix [[TN FP] [FN TP]]:", cm)

    # Save feature importance for the dashboard
    importance = pd.DataFrame({
        "feature": config.FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nFeature importance:\n", importance.to_string(index=False))


if __name__ == "__main__":
    train()
