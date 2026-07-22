"""
BankShield AI — Module 1: Fraud Detection Modeling

Trains:
- Random Forest
- XGBoost
- Isolation Forest

Evaluates using:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

possible_files = [
    BASE_DIR / "data" / "processed" / "fraud_processed.csv",
    BASE_DIR / "data" / "processed" / "credit_card_fraud_10k_cleaned.csv",
    BASE_DIR / "data" / "processed" / "credit_card_fraud_10k.csv",
    Path.home() / "Downloads" / "credit_card_fraud_10k_cleaned.csv",
    Path.home() / "Downloads" / "credit_card_fraud_10k.csv",
]

DATA_PATH = None

for file in possible_files:
    if file.exists():
        DATA_PATH = file
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found.\n"
        "Place it inside:\n"
        "bankshield_ai/data/processed/\n"
        "or keep it in Downloads."
    )

MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "fraud_model_report.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Using dataset:\n{DATA_PATH}")

# ==========================================================
# FEATURES
# ==========================================================

FEATURES = [
    "amount",
    "amount_log",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
    "is_night_transaction",
    "high_amount_flag",
    "high_risk_merchant",
    "multi_signal_risk",
    "merchant_category_enc",
    "velocity_tier_enc",
]

TARGET = "is_fraud"

# ==========================================================
# LOAD DATA
# ==========================================================
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Create amount_log if it doesn't exist
    if "amount_log" not in df.columns:
        df["amount_log"] = np.log1p(df["amount"])

    # Use existing column if available
    if "high_amount_flag" not in df.columns:
        if "is_large_transaction" in df.columns:
            df["high_amount_flag"] = df["is_large_transaction"]
        else:
            df["high_amount_flag"] = (df["amount"] > 200).astype(int)

    # Encode merchant category
    le_merchant = LabelEncoder()
    df["merchant_category_enc"] = le_merchant.fit_transform(
        df["merchant_category"]
    )

    # Encode velocity tier
    le_velocity = LabelEncoder()
    df["velocity_tier_enc"] = le_velocity.fit_transform(
        df["velocity_tier"].astype(str)
    )

    joblib.dump(le_merchant, MODEL_DIR / "le_merchant.pkl")
    joblib.dump(le_velocity, MODEL_DIR / "le_velocity.pkl")

    return df



# ==========================================================
# EVALUATION
# ==========================================================


def evaluate(y_true, y_prob, model_name, threshold=0.5):

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    f1 = f1_score(y_true, y_pred)

    print(f"\n===== {model_name} =====")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(cm)

    print(classification_report(y_true, y_pred, digits=3))

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": f1,
        "confusion_matrix": cm,
    }


# ==========================================================
# MAIN
# ==========================================================


def main():

    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    rf_prob = rf.predict_proba(X_test)[:, 1]

    results.append(
        evaluate(y_test, rf_prob, "Random Forest")
    )

    joblib.dump(
        rf,
        MODEL_DIR / "rf_model.pkl",
    )

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    scale_pos_weight = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    xgb_model.fit(X_train, y_train)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results.append(
        evaluate(y_test, xgb_prob, "XGBoost")
    )

    xgb_model.save_model(
        MODEL_DIR / "xgboost_fraud.json"
    )

    # ------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------

    iso = IsolationForest(
        n_estimators=300,
        contamination=float(y_train.mean()),
        random_state=42,
        n_jobs=-1,
    )

    iso.fit(X_train)

    scores = -iso.decision_function(X_test)

    score_range = scores.max() - scores.min()

    if score_range == 0:
        scores = np.zeros_like(scores)
    else:
        scores = (scores - scores.min()) / score_range

    results.append(
        evaluate(
            y_test,
            scores,
            "Isolation Forest",
        )
    )

    joblib.dump(
        iso,
        MODEL_DIR / "isolation_forest.pkl",
    )

    # ------------------------------------------------------

    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    summary = pd.DataFrame(results)[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ]
    ]

    print(summary.to_string(index=False))

    print(f"\nModels saved in:\n{MODEL_DIR}")
    print(f"Report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()