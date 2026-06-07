import pandas as pd
import numpy as np
from pathlib import Path
from src.integration import run_integration

def feature_engineering(df):
    # financial features 
    df['debt_to_income_ratio'] = df['total_debt'] / df['yearly_income']
    
    # age-based features
    df["years_to_retirement"] = (df["retirement_age"] - df["current_age"]).clip(lower=0)

    df['is_retired'] = (df['current_age'] >= df['retirement_age']).to_numpy().astype(bool)

    # card life-cycle features
    df["card_age_months"] = (
        (df["date"] - df["acct_open_date"])
    ).dt.days // 30
    df["card_age_months"] = df["card_age_months"].astype(int)

    df["days_until_expiry"] = (
        df["expires"] - df["date"]
    ).dt.days.astype(int)

    df["hour"] = df["date"].dt.hour.astype(int)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(bool)

    # clean up
    df.drop(columns=['id', 'date', 'client_id', 'card_id', 'current_age', 'retirement_age',
                     'per_capita_income', 'acct_open_date', 'year_pin_last_changed', 
                     'expires', 'card_on_dark_web'], 
                     inplace=True)
    
    # save
    df[df["target"].notna()].to_csv(Path.cwd().parent / "archive" / "labeled_data.csv", index=False)
    df[df["target"].isna()].to_csv(Path.cwd().parent / "archive" / "unlabeled_data.csv", index=False)

    df = df[df["target"].notna()]

    return df

if __name__ == "__main__":
    from integration import run_integration
    
    df = run_integration()
    df = feature_engineering(df)
    print(df.head())