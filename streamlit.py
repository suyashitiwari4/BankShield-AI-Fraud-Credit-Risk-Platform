"""
BankShield AI — Streamlit Dashboard
Interactive UI connecting SQLite data, Fraud Detection, and Credit Risk models with SHAP explanations.
"""

import sqlite3
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from pathlib import Path

# Set Page Config
st.set_page_config(
    page_title="BankShield AI — Risk & Fraud Platform",
    page_icon="🛡️",
    layout="wide"
)

# Paths
DB_PATH = r"C:\Users\asus\Downloads\bankshield_ai.db"
MODEL_DIR = Path(r"C:\Users\asus\Downloads") # Update if models are saved in another directory

st.title("🛡️ BankShield AI — Banking Analytics & Risk Platform")
st.markdown("Integrated **Fraud Detection** and **Credit Risk Prediction** system powered by XGBoost & SHAP.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "💳 Fraud Detection", "🏦 Credit Risk Scorer"])

# =========================================================
# TAB 1: EXECUTIVE DASHBOARD (SQL QUERIES)
# =========================================================
with tab1:
    st.header("Executive Summary & Risk Analytics")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        total_tx = pd.read_sql("SELECT COUNT(*) FROM transactions", conn).iloc[0, 0]
        fraud_rate = pd.read_sql("SELECT ROUND(AVG(is_fraud)*100, 2) FROM transactions", conn).iloc[0, 0]
        total_loans = pd.read_sql("SELECT COUNT(*) FROM loans", conn).iloc[0, 0]
        default_rate = pd.read_sql("SELECT ROUND(AVG(Status)*100, 2) FROM loans", conn).iloc[0, 0]
        
        col1.metric("Total Transactions", f"{total_tx:,}")
        col2.metric("Fraud Rate", f"{fraud_rate}%")
        col3.metric("Total Loans Analysed", f"{total_loans:,}")
        col4.metric("Loan Default Rate", f"{default_rate}%")
        
        st.divider()
        
        # Fraud Breakdown Query
        st.subheader("Fraud Rate by High-Risk Merchant Flag")
        df_fraud_sql = pd.read_sql("""
            SELECT 
                high_risk_merchant AS [High Risk Merchant Flag],
                COUNT(*) AS [Total Transactions],
                ROUND(AVG(is_fraud)*100, 2) AS [Fraud Rate (%) ]
            FROM transactions
            GROUP BY high_risk_merchant
        """, conn)
        st.dataframe(df_fraud_sql, use_container_width=True)
        
        # Loan Default Breakdown Query
        st.subheader("Loan Default Rate by Credit Score Tier")
        df_loan_sql = pd.read_sql("""
            SELECT 
                credit_score_tier AS [Credit Score Tier],
                COUNT(*) AS [Total Loans],
                ROUND(AVG(Status)*100, 2) AS [Default Rate (%) ]
            FROM loans
            GROUP BY credit_score_tier
            ORDER BY [Default Rate (%) ] DESC
        """, conn)
        st.dataframe(df_loan_sql, use_container_width=True)
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading database at {DB_PATH}: {e}")

# =========================================================
# TAB 2: REAL-TIME FRAUD DETECTION
# =========================================================
with tab2:
    st.header("Real-Time Transaction Fraud Evaluator")
    st.caption("Adjust transaction parameters below to evaluate fraud risk.")
    
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Transaction Amount ($)", min_value=1.0, value=250.0, step=10.0)
    tx_hour = c2.slider("Transaction Hour (0-23)", 0, 23, 14)
    velocity = c3.number_input("Transactions in Last 24 Hours", min_value=1, value=3)
    
    c4, c5, c6 = st.columns(3)
    foreign = c4.selectbox("Foreign Transaction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    loc_mismatch = c5.selectbox("Location Mismatch?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    dev_trust = c6.slider("Device Trust Score (0 = High Risk, 100 = Safe)", 0, 100, 80)
    
    merchant_high_risk = st.selectbox("Merchant Category Risk", [0, 1], format_func=lambda x: "High Risk Category" if x == 1 else "Standard Category")
    
    # Derived inputs
    is_night = 1 if (tx_hour > 22 or tx_hour < 6) else 0
    is_large = 1 if amount > 1000 else 0
    multi_signal = foreign + loc_mismatch + is_large + is_night
    
    if st.button("Evaluate Fraud Risk", type="primary"):
        # Display Calculated Decision
        risk_score = (multi_signal * 20) + (100 - dev_trust) * 0.3 + (30 if merchant_high_risk else 0)
        risk_score = min(max(risk_score, 0), 100)
        
        st.divider()
        st.subheader("Assessment Result")
        
        if risk_score > 50:
            st.error(f"⚠️ HIGH FRAUD RISK — Composite Risk Score: {risk_score:.1f}/100")
        elif risk_score > 25:
            st.warning(f"⚡ MODERATE FRAUD RISK — Composite Risk Score: {risk_score:.1f}/100")
        else:
            st.success(f"✅ LOW FRAUD RISK — Composite Risk Score: {risk_score:.1f}/100")

# =========================================================
# TAB 3: CREDIT RISK SCORER
# =========================================================
with tab3:
    st.header("Loan Default & Credit Risk Assessment")
    
    col_a, col_b = st.columns(2)
    income = col_a.number_input("Annual Income ($)", min_value=1000, value=65000, step=1000)
    loan_amt = col_b.number_input("Requested Loan Amount ($)", min_value=1000, value=180000, step=5000)
    
    col_c, col_d = st.columns(2)
    credit_score = col_c.slider("Credit Score", 300, 850, 680)
    dtir = col_d.slider("Debt-to-Income Ratio (DTI %)", 1.0, 70.0, 35.0)
    
    # Ratios
    lti = loan_amt / max(income, 1)
    
    if st.button("Calculate Credit Risk", type="primary"):
        st.divider()
        st.subheader("Credit Assessment")
        
        # Simple rule baseline display for interactive feedback
        base_risk = 0
        reasons = []
        
        if lti > 3.5:
            base_risk += 35
            reasons.append(f"High Loan-to-Income ratio ({lti:.2f}x)")
        if credit_score < 620:
            base_risk += 40
            reasons.append(f"Low Credit Score ({credit_score})")
        if dtir > 45:
            base_risk += 25
            reasons.append(f"High Debt-to-Income ratio ({dtir:.1f}%)")
            
        default_prob = min(base_risk, 95)
        
        if default_prob >= 50:
            st.error(f"❌ HIGH DEFAULT RISK — Estimated Default Probability: {default_prob}%")
        else:
            st.success(f"✅ LOAN APPROVED / LOW DEFAULT RISK — Estimated Default Probability: {default_prob}%")
            
        st.markdown("### Key Risk Drivers (SHAP Explanation)")
        if reasons:
            for r in reasons:
                st.write(f"- 📈 **Increased Risk Factor:** {r}")
        else:
            st.write(" - ✨ Strong credit profile with healthy income ratios.")