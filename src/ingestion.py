from pathlib import Path
import pandas as pd
import json

def load_csv(file):
    file = Path.cwd().parent / "archive" / file
    return pd.read_csv(file)

def load_json(file):
    file = Path.cwd().parent / "archive" / file
    with open(file, "r") as f:
        raw = json.load(f)
    return pd.DataFrame.from_dict(raw, orient="index")

if __name__ == "__main__":
    cards = load_csv("cards_data.csv")
    mcc = load_json("mcc_codes.json")
    print(cards.head())
    print(mcc.head())