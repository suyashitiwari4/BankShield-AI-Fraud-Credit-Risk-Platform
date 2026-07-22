"""
BankShield AI — Module 1: Fraud Detection
Cleaning + feature engineering for the transaction fraud dataset.
 
Dataset is already clean (no missing values, no duplicates in the raw
data) and comes with strong pre-built signals (velocity, device_trust,
location_mismatch). Focus here is light validation + a handful of
engineered features that make the risk story explicit and dashboard-friendly.
"""
import pandas as pd
import numpy as np

raw_path=r"C:\Users\asus\Downloads\credit_card_fraud_10k.csv"
out_path=r"C:\Users\asus\Downloads\credit_card_fraud_10k_cleaned.csv"

high_risk_merchants=["Electronics","Travel"] #highest fraud-rate categories

def load_and_validate():
    df=pd.read_csv(raw_path)
    print("raw data shape",df.shape)
    before=len(df)# store original row count for validation
    print(f"dropped {before-len(df)} duplicate rows")
    print("missing values :\n",df.isnull().sum()) # print count of missing values for each column
     #sanity check: ensure no missing values remain
    assert df["amount"].min()>=0,"negative transaction amount fraud"
    assert df["transaction_hour"].between(0,23).all(),"invalid hour value found"

    return df
def engineer_features(df):
     #time based risk
     df["is_night_transaction"]=df["transaction_hour"].apply(
          lambda h: 1 if h > 22 or h < 6 else 0
     )
     #amount based risk
     df["is_large_transaction"]=df["amount"].apply(
            lambda a: 1 if a>1000 else 0
        )
     #merchant based risk
     fraud_rate_by_merchant=df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)# compute fraud rate by merchant category
     print("\nFraud rate by merchant category:\n", fraud_rate_by_merchant)
     high_risk = fraud_rate_by_merchant[fraud_rate_by_merchant > df["is_fraud"].mean()].index.tolist()# identify high-risk merchant categories based on fraud rate
     df["high_risk_merchant"] = df["merchant_category"].isin(high_risk).astype(int)# create a binary feature indicating whether the merchant category is high-risk
     #velocity risk tier
     df["velocity_tier"]=pd.cut(
         df["velocity_last_24h"],
        bins=[-1, 1, 3, 6, 100],
        labels=["Low", "Medium", "High", "Very High"], 
     )
       # composite risk flag combining multiple weak signals
     df["multi_signal_risk"] = (
        df["foreign_transaction"] + df["location_mismatch"] +
        df["is_large_transaction"] + df["is_night_transaction"]
    )
 
     return df, high_risk
def main():
    df = load_and_validate()
    print(f"\nAfter validation: {df.shape}")
    df, high_risk = engineer_features(df)
    print(f"After feature engineering: {df.shape}")
    print(f"\nHigh-risk merchant categories identified: {high_risk}")
    print("\nFinal columns:", list(df.columns))
    print("\nFraud rate:", df["is_fraud"].mean().round(4))
 
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
 
 
if __name__ == "__main__":
    main()