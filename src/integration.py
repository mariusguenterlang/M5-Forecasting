from pathlib import Path

def run_integration():
    # Load data
    transactions = load_csv("transactions_data.csv")
    users = load_csv("users_data.csv")
    cards = load_csv("cards_data.csv")
    labels = load_json("train_fraud_labels.json")
    print("Data loaded successfully ...")

    # Clean data
    transactions = clean_transactions(transactions)
    users = clean_users(users)
    cards = clean_cards(cards)
    labels = clean_labels(labels)
    print("Data cleaned successfully ...")

    # Merge data
    data = (transactions
        .merge(users, on='client_id', how='left')
        .merge(cards, on=["card_id", "client_id"], how='left')
        .merge(labels, on='id', how='left')
    )
    print("Data merged successfully ...")

    # Save integrated data
    data.to_csv(Path.cwd().parent / "archive" / "integrated_data.csv", index=False)

    return data


if __name__ == "__main__":
    from ingestion import load_csv, load_json
    from preprocessing import clean_cards, clean_users, clean_transactions, clean_labels

    run_integration()