"""
BankShield AI — load processed data into SQLite.
This IS the "backend" — a single local .db file, queried directly by
pandas / Streamlit. No server, no API needed for this project.
"""
import sqlite3
import pandas as pd
DB_PATH=r"C:\Users\asus\Downloads\bankshield_ai.db"

TABLES={
    "transactions":r"C:\Users\asus\Downloads\credit_card_fraud_10k_cleaned.csv",
    "loans":r"C:\Users\asus\Downloads\loan_cleaned.csv"
}
def main():
    conn=sqlite3.connect(DB_PATH)
    for table_name,csv_path in TABLES.items():
        df=pd.read_csv(csv_path)
        df.to_sql(table_name,conn,if_exists="replace",index=False)
        print(f"Loaded {len(df)} rows into table '{table_name}'")

    # sanity check with a real SQL query (good to show you can write SQL, not just pandas)
    print("\n--- Sample query: fraud rate by high_risk_merchant flag ---")
    result=pd.read_sql(
        """
        Select high_risk_merchant,
           count(*) as n_transactions,
           round(avg(is_fraud)*100,2)as fraud_rate_pct
        from transactions
        group by high_risk_merchant
     """  ,
       conn,# connection to the SQLite database


    )
    print(result)
    print("\n--- Sample query: default rate by high_loan_to_income_flag ---")
    result2=pd.read_sql(
        """
        select credit_score_tier,
            count(*) as n_loans,
            round(avg(status)*100,2) as default_rate_pct
        from loans
        group by credit_score_tier
        order by default_rate_pct desc
        """,
        conn,
    )
    print(result2)
    conn.close()
    print(f"\nDatabase saved to {DB_PATH}")
if __name__=="__main__":
    main()