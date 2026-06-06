import pandas as pd
import numpy as np
from src.integration import run_integration
from pathlib import Path

def feature_engineering(df):
    # financial features 
    df['debt_to_income_ratio'] = df['total_debt'] / df['yearly_income']
    
    # age-based features
    df["years_to_retirement"] = (df["retirement_age"] - df["current_age"]).clip(lower=0)

    df['is_retired'] = (df['current_age'] >= df['retirement_age']).to_numpy().astype(int)

    # card life-cycle features
    df["card_age_months"] = (
        (df["date"] - df["acct_open_date"])
    ).dt.days // 30

    df["days_until_expiry"] = (
        df["expires"] - df["date"]
    ).dt.days

    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    # clean up
    df.drop(columns=['id', 'date', 'client_id', 'card_id', 'current_age', 'retirement_age',
                     'per_capita_income', 'acct_open_date', 'year_pin_last_changed'], 
                     inplace=True)
    
    # save for testing purposes, remove from pipeline later
    df[df["target"].notna()].to_csv(Path.cwd().parent / "archive" / "labeled_data.csv", index=False)
    df[df["target"].isna()].to_csv(Path.cwd().parent / "archive" / "unlabeled_data.csv", index=False)

    df = df[df["target"].notna()]

    return df

if __name__ == "__main__":
    df = run_integration()
    df = feature_engineering(df)
    print(df.head())