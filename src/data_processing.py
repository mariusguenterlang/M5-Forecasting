import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# Data Loaders
def load_csv(file):
    file = Path.cwd().parent / "archive" / file
    return pd.read_csv(file)

def load_json(file):
    file = Path.cwd().parent / "archive" / file
    with open(file, "r") as f:
        raw = json.load(f)
    return pd.DataFrame.from_dict(raw, orient="index")

# Data Cleaners
def clean_cards(cards):
    cards = cards.rename(columns={'id': 'card_id'})
    cards.drop(columns=['card_number', 'cvv'], inplace=True)
    cards['card_brand'] = cards['card_brand'].astype('category')
    cards['card_type'] = cards['card_type'].astype('category')
    cards['expires'] = pd.to_datetime(cards['expires'], format='%m/%Y').dt.to_period('M')
    cards['has_chip'] = cards['has_chip'].map({'YES': True, 'NO': False}).astype(int)
    cards['credit_limit'] = (cards['credit_limit'].str.replace('$', '').astype(float))
    cards['acct_open_date'] = pd.to_datetime(cards['acct_open_date'], format='%m/%Y').dt.to_period('M')
    cards['year_pin_last_changed'] = pd.to_datetime(cards['year_pin_last_changed'], format='%Y').dt.to_period('Y')
    cards['card_on_dark_web'] = cards['card_on_dark_web'].map({'Yes': True, 'No': False}).astype(int)
    return cards

def clean_users(users):
    users = users.rename(columns={'id': 'client_id'})
    if 'gender' in users.columns:
        clean_gender = users['gender'].astype(str).str.strip().str.lower()
        users['is_female'] = clean_gender.map({'female': 1}).fillna(0).astype(int)
        users = users.drop(columns=['gender'])
    users.drop(columns=['address', 'latitude', 'longitude'], inplace=True)
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        users[col] = users[col].str.replace('$', '').astype(float)
    users['is_retired'] = (users['current_age'] >= users['retirement_age']).astype(int)
    if 'total_debt' in users.columns and 'yearly_income' in users.columns:
        users['debt_to_income_ratio'] = np.where(
            users['yearly_income'] > 0, 
            users['total_debt'] / users['yearly_income'], 
            0.0
        )
    users.drop(columns=['per_capita_income', 'total_debt', 'retirement_age',
                        'birth_year', 'birth_month'], inplace=True) #for multicolinearity but check!!1!
    return users

def clean_transactions(transactions):
    transactions = transactions.copy()

    transactions['date'] = pd.to_datetime(transactions['date'])
    transactions['amount'] = transactions['amount'].str.replace('$', '').astype(float)
    transactions['use_chip'] = transactions['use_chip'].astype('category')
    transactions['merchant_state'] = transactions['merchant_state'].fillna('Online').astype('category')
    transactions['errors'] = transactions['errors'].fillna('None').astype('category')
    transactions.drop(columns=['merchant_id', 'merchant_city', 'merchant_state', 'zip'], inplace=True)

    # One-hot encode 'use_chip' column
    encoder = OneHotEncoder(sparse_output=False)
    use_chip_encoded = encoder.fit_transform(transactions[['use_chip']])
    encoded_cols = encoder.get_feature_names_out(['use_chip'])
    encoded_cols_df = pd.DataFrame(use_chip_encoded, columns=encoded_cols, index=transactions.index).astype(int)
    transactions = pd.concat([transactions, encoded_cols_df], axis=1)
    transactions.drop(columns=['use_chip'], inplace=True)
    
    # One-hot encode 'errors' column
    errors_encoded = encoder.fit_transform(transactions[['errors']])
    encoded_errors_cols = encoder.get_feature_names_out(['errors'])
    encoded_errors_cols_df = pd.DataFrame(errors_encoded, columns=encoded_errors_cols, index=transactions.index).astype(int)
    transactions = pd.concat([transactions, encoded_errors_cols_df], axis=1)
    transactions.drop(columns=['errors'], inplace=True)

    # Clean column names
    transactions.columns = (
        transactions.columns
        .str.replace('use_chip_', '', regex=False)
        .str.replace('errors_', '', regex=False)
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )

    return transactions

def clean_labels(labels):
    labels = labels.T
    labels.reset_index(inplace=True)
    labels.rename(columns={'index': 'id'}, inplace=True)
    labels['target'] = labels['target'].map({'Yes': 1, 'No': 0})
    labels['id'] = labels['id'].astype('int64')
    return labels

def clean_mcc(mcc):
    mcc = mcc.reset_index()
    mcc = mcc.rename(columns={'index': 'mcc'})
    mcc['mcc'] = mcc['mcc'].astype('int64')
    return mcc

# Merge and save DataFrames
def merge_and_save(users, cards, transactions, labels, mcc):
    merged = (transactions
        .merge(users, on="client_id", how="left")
        .merge(cards, on=["card_id", "client_id"], how="left")
        .merge(labels, on="id", how="left")
        .merge(mcc, on="mcc", how="left")
    )
    merged.to_csv(Path.cwd().parent / "archive" / "merged_data.csv", index=False)
    return merged

def main():
    transactions = load_csv("transactions_data.csv")
    users = load_csv("users_data.csv")
    cards = load_csv("cards_data.csv")
    labels = load_json("train_fraud_labels.json")
    mcc = load_json("mcc_codes.json")

    transactions = clean_transactions(transactions)
    users = clean_users(users)
    cards = clean_cards(cards)
    labels = clean_labels(labels)
    mcc = clean_mcc(mcc)

    merged = merge_and_save(users, cards, transactions, labels, mcc)

    df_train = merged[merged["target"].notna()].copy()
    df_test = merged[merged["target"].isna()].copy()

    df_train.to_csv(Path.cwd().parent / "archive" / "train_data.csv", index=False)
    df_test.to_csv(Path.cwd().parent / "archive" / "test_data.csv", index=False)

    print(f"Pipeline complete.")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"Fraud rate in train: {df_train['target'].mean():.4%}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = main()