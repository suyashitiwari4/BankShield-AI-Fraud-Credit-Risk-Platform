"""
BankShield AI — Module 2: Credit Risk
Cleaning + feature engineering for the loan default dataset.

Key decision (documented for README/interviews):
  rate_of_interest, Interest_rate_spread, property_value, LTV, and
  Upfront_charges are missing in 90-100% of DEFAULT rows but present in
  90%+ of non-default rows. This is leakage baked into missingness
  (these fields are likely only populated once a loan closes normally).
  They are DROPPED rather than imputed to avoid an artificially perfect
  model that would not generalize to a real scoring scenario.
"""
import pandas as pd
import numpy as np

RAW_PATH = r"C:\Users\asus\Downloads\loan.csv"
OUT_PATH = r"C:\Users\asus\Downloads\loan_cleaned.csv"

LEAKY_COLS = [
    "rate_of_interest",
    "Interest_rate_spread",
    "property_value",
    "LTV",
    "Upfront_charges",
]

# age buckets -> numeric midpoint, needed for a usable numeric feature
AGE_MAP = {
    "<25": 22, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5,
    "55-64": 59.5, "65-74": 69.5, ">74": 78,
}


def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # constants / IDs with no predictive value
    df = df.drop(columns=["ID", "year"])

    # drop leakage-prone columns (see module docstring)
    df = df.drop(columns=LEAKY_COLS)

    # dtir1: missingness looked weaker at first glance (67.6% vs 24.6% default
    # rate) but a follow-up check showed its missing-flag correlates at 0.44
    # with the target -- too strong to be organic. Treated like the other
    # leakage columns: impute the value but do NOT keep a missing-indicator.
    df["dtir1"] = df["dtir1"].fillna(df["dtir1"].median())

    # income: missingness looks unrelated to default (13.5% vs 24.6%) —
    # safe to median-impute with no flag needed
    df["income"] = df["income"].fillna(df["income"].median())

    # low-cardinality categoricals with a handful of missing rows — mode-impute
    cat_cols_to_mode_impute = [
        "loan_limit", "approv_in_adv", "loan_purpose",
        "Neg_ammortization", "age", "submission_of_application",
    ]
    for col in cat_cols_to_mode_impute:
        df[col] = df[col].fillna(df[col].mode()[0])

    # duplicates (none found, but keep the check in the pipeline)
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    return df


def engineer_features(df):
    # numeric age from bucket, needed for ratio features
    df["age_numeric"] = df["age"].map(AGE_MAP)

    # core risk ratios
    df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_to_income"].fillna(df["loan_to_income"].median())

    df["term_years"] = df["term"] / 12

    # credit score bucket (business-readable risk tiers, not just raw score)
    df["credit_score_tier"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    )

    # high-risk flags aligned with roadmap's requested features
    df["high_dtir_flag"] = (df["dtir1"] > df["dtir1"].quantile(0.75)).astype(int)
    df["high_loan_to_income_flag"] = (df["loan_to_income"] > df["loan_to_income"].quantile(0.75)).astype(int)

    return df


def main():
    df = load_and_clean()
    print(f"After cleaning: {df.shape}")
    df = engineer_features(df)
    print(f"After feature engineering: {df.shape}")
    print("\nFinal columns:", list(df.columns))
    print("\nDefault rate:", df["Status"].mean().round(4))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()