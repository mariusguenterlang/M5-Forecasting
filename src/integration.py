import pandas as pd
from pathlib import Path
from ingestion import load_csv, load_json
from preprocessing import clean_cards, clean_users, clean_transactions, clean_labels

def main():
    # Load data
    transactions = load_csv("transactions_data.csv")
    users = load_csv("users_data.csv")
    cards = load_csv("cards_data.csv")
    labels = load_json("train_fraud_labels.json")

    # Clean data
    transactions = clean_transactions(transactions)
    users = clean_users(users)
    cards = clean_cards(cards)
    labels = clean_labels(labels)

    # Merge data
    data = transactions.merge(users, on='client_id', how='left')
    data = data.merge(cards, on=["card_id", "client_id"], how='left')
    data = data.merge(labels, left_on='client_id', right_on='id', how='left')

    # Save integrated data
    data.to_csv(Path.cwd().parent / "archive" / "integrated_data.csv", index=False)

    return data


if __name__ == "__main__":
    main()