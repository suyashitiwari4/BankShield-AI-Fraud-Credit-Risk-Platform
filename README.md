# 🛡️ BankShield AI — Banking Analytics & Risk Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

An end-to-end banking analytics platform integrating **Real-time Fraud Detection** and **Credit Risk Prediction**. The system processes transaction and loan data, trains production-grade Machine Learning models with SHAP-based explainability, stores data in an optimized local SQLite database, and exposes risk metrics through an interactive Streamlit dashboard.

---

## 📌 Executive Summary & Architecture

BankShield AI addresses two critical risk vectors in modern retail banking:
1. **Transaction Fraud Detection:** Identifying high-risk transactions instantly using velocity, location, device trust, and composite risk signals.
2. **Credit Risk & Default Prediction:** Predicting loan default likelihood while mitigating target leakage and providing explainable decision drivers for credit officers.

## ✨ Key Features

* **Data Engineering & Leakage Prevention:** Explicitly identified and removed systemic target leakage in credit datasets (e.g., fields populated only post-approval like `LTV`, `Interest_rate_spread`).
* **Feature Engineering Pipeline:** Engineered domain-specific flags including transaction velocity tiers, multi-signal composite risk metrics, debt-to-income quantiles, and logarithmic transaction transformations.
* **Multi-Model Machine Learning:**
  * **Fraud Module:** Isolation Forest (Unsupervised Anomaly), Random Forest, and XGBoost (Class-imbalance handled via `scale_pos_weight`).
  * **Credit Risk Module:** Logistic Regression (Standardized baseline), Random Forest, and XGBoost.
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to break down individual high-risk predictions into plain-English business justifications.
* **Local Backend Database:** Embedded **SQLite** database supporting structured data retrieval, query execution, and metric generation.
* **Interactive Dashboard:** Built with **Streamlit** to offer real-time risk assessment tools and high-level KPI visualizers.

---

## 📊 Dataset & Metrics Overview

| Feature Module | Dataset Size | Primary Target | Key Models Used | Key Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Fraud Detection** | ~10,000 transactions | `is_fraud` (Binary) | Random Forest, XGBoost, Isolation Forest | ROC-AUC, PR-AUC, F1-Score |
| **Credit Risk** | ~148,000 loans | `Status` (Binary Default) | Logistic Regression, Random Forest, XGBoost | ROC-AUC, PR-AUC, Recall |

---

## 📁 Repository Structure

```text
bankshield_ai/
├── data/
│   ├── raw/                      # Original transaction and loan datasets
│   └── processed/                # Cleaned CSV files ready for SQL/ML
├── models/                       # Trained models (.pkl, .json) & scalers/imputers
├── reports/                      # Output JSON reports & SHAP visual plots
├── scripts/
│   ├── module1_fraud_clean.py    # Fraud data cleaning & feature engineering
│   ├── module2_credit_clean.py   # Credit data cleaning & leakage handling
│   ├── db_loader.py              # SQLite creation and SQL verification queries
│   ├── train_fraud_model.py      # Fraud ML training pipeline
│   └── train_credit_model.py     # Credit ML training pipeline with SHAP
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
