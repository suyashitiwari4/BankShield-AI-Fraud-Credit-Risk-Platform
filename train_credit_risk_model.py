"""
BankShield AI — Module 2: Credit Risk Modeling
Trains Logistic Regression, Random Forest, and XGBoost to predict loan
default, then uses SHAP to explain individual predictions in plain,
business-readable terms (e.g. "flagged mainly due to high loan-to-income
ratio and low credit score").
 
Default rate is 24.65% (moderate imbalance, no synthetic sampling needed --
class_weight/scale_pos_weight handle it).
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score
)

try:
    import shap
except ModuleNotFoundError:
    shap = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT

possible_files = [
    SCRIPT_DIR / "data" / "processed" / "loan_cleaned.csv",
    SCRIPT_DIR / "data" / "processed" / "Loan_Default.csv",
    PROJECT_ROOT / "data" / "processed" / "loan_cleaned.csv",
    PROJECT_ROOT / "data" / "processed" / "Loan_Default.csv",
    Path.home() / "Downloads" / "loan_cleaned.csv",
    Path.home() / "Downloads" / "Loan_Default.csv",
    Path.cwd() / "data" / "processed" / "loan_cleaned.csv",
    Path.cwd() / "data" / "processed" / "Loan_Default.csv",
]

DATA_PATH = None

for file in possible_files:
    if file.exists():
        DATA_PATH = file
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Loan dataset not found. Tried:\n" + "\n".join(str(p) for p in possible_files)
    )

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using dataset:\n{DATA_PATH}")



TARGET="Status"
# dropped: 'age' (raw bucket string, replaced by age_numeric),
# 'credit_score_tier' (derived from Credit_Score, redundant -- avoid double counting)
DROP_COLS=[TARGET, "age", "credit_score_tier"]
def load_and_encode():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Target column
    y = df[TARGET].copy()

    # Convert target to integer if stored as string
    if y.dtype == "object":
        y = y.astype(str).str.strip()
        if set(y.dropna().unique()) <= {"0", "1"}:
            y = y.astype(int)

    # Features
    X = df.drop(columns=DROP_COLS)

    # ------------------------------
    # Fill missing values
    # ------------------------------

    # Numerical columns
    num_cols = X.select_dtypes(include=[np.number]).columns

    # Fill numeric columns with median
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Fill categorical columns with mode
    for col in cat_cols:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encoding
    X_encoded = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=True
    )

    # Safety check (shouldn't be needed, but prevents errors)
    X_encoded = X_encoded.fillna(0)

    print("\n========== DATA QUALITY ==========")
    print(f"Samples : {len(X_encoded)}")
    print(f"Features: {X_encoded.shape[1]}")
    print(f"Remaining NaNs: {X_encoded.isna().sum().sum()}")
    print("==================================\n")

    return X_encoded, y
def evaluate(y_true,y_proba,model_name,threshold=0.5):
    y_pred=(y_proba >= threshold).astype(int)
    roc_auc=roc_auc_score(y_true,y_proba)
    pr_auc=average_precision_score(y_true,y_proba)
    f1=f1_score(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred).tolist()
    report=classification_report(y_true,y_pred,output_dict=True)
     
    print(f"\n=== {model_name} (threshold={threshold}) ===")
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  F1: {f1:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print(classification_report(y_true, y_pred, digits=3))

     
    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "confusion_matrix": cm,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
    }
def main():
    x, y = load_and_encode()
    print(f"feature matrix:{x.shape}")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train: {x_train.shape}, default rate: {y_train.mean():.4f}")
    print(f"Test:  {x_test.shape}, default rate: {y_test.mean():.4f}")

    joblib.dump(list(x.columns), MODEL_DIR / "credit_risk_feature_names.pkl")

    results = []

    # LOGISTIC REGRESSION
   
    imputer = SimpleImputer(strategy="median")

    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    joblib.dump(imputer, MODEL_DIR / "credit_risk_imputer.pkl")

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, MODEL_DIR / "credit_risk_scaler.pkl")

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(x_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(x_test_scaled)[:, 1]
    results.append(evaluate(y_test, logreg_proba, "Logistic Regression"))
    joblib.dump(logreg, MODEL_DIR / "logreg_credit_risk.pkl")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(x_train, y_train)
    rf_proba = rf.predict_proba(x_test)[:, 1]
    results.append(evaluate(y_test, rf_proba, "Random Forest"))
    joblib.dump(rf, MODEL_DIR / "random_forest_credit_risk.pkl")

    # --- XGBoost ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(x_train, y_train)
    xgb_proba = xgb_model.predict_proba(x_test)[:, 1]
    results.append(evaluate(y_test, xgb_proba, "XGBoost"))
    xgb_model.save_model(str(MODEL_DIR / "xgboost_credit_risk.json"))

    with open(REPORT_DIR / "credit_risk_model_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    summary_df = pd.DataFrame(results)[["model", "roc_auc", "pr_auc", "precision", "recall", "f1"]]
    print(summary_df.to_string(index=False))

    # --- SHAP explainability on the best model (XGBoost) ---
    if shap is not None:
        try:
            print("\nComputing SHAP values (this can take a minute on 148k rows)...")
            explainer = shap.TreeExplainer(xgb_model)
            sample_idx = x_test.sample(n=min(2000, len(x_test)), random_state=42).index
            X_sample = x_test.loc[sample_idx]
            shap_values = explainer(X_sample)

            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(REPORT_DIR / "shap_summary_credit_risk.png", dpi=150)
            plt.close()
            print(f"Saved SHAP summary plot to {REPORT_DIR / 'shap_summary_credit_risk.png'}")

            print("\n=== Sample explanations for top 3 highest-risk test customers ===")
            top_risk_idx = np.argsort(-xgb_model.predict_proba(x_test)[:, 1])[:3]
            for rank, i in enumerate(top_risk_idx, 1):
                row = x_test.iloc[[i]]
                proba = xgb_model.predict_proba(row)[0, 1]
                sv = explainer(row)
                contributions = pd.Series(sv.values[0], index=row.columns).sort_values(key=abs, ascending=False)
                top_reasons = contributions.head(4)
                print(f"\nCustomer #{rank} — predicted default probability: {proba:.1%}")
                print("Top contributing factors:")
                for feat, val in top_reasons.items():
                    direction = "increases" if val > 0 else "decreases"
                    print(f"  - {feat} {direction} risk (SHAP contribution: {val:+.3f})")
        except Exception as e:
            print(f"\nSHAP explainability skipped: {e}")
    else:
        print("\nSHAP is not installed; skipping explainability plots.")

    print(f"\nSaved models to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
