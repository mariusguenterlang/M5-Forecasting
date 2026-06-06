import joblib
import pandas as pd
from pathlib import Path

def evaluate():
    pipeline = joblib.load("models/lgb_model.pkl")
    data = pd.read_csv(Path.cwd().parent / "archive" / "unlabeled_data.csv")

    # clean data, MOVE TO FEATURE ENGINEERING
    data.drop(columns=["expires", "card_on_dark_web", "target"], inplace=True)
    data[["is_retired", "is_night"]] = data[["is_retired", "is_night"]].astype(bool)

    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        data[col] = data[col].astype('category')

    # predictions
    preds = pipeline.predict(data)
    proba = pipeline.predict_proba(data)[:, 1]

    # attach to dataset
    data["preds"] = preds
    data["proba"] = proba

    # save predictions
    data.to_csv(Path.cwd().parent / "archive" / "predictions.csv", index=False)
    
    return data

if __name__ == "__main__":
    evaluate()