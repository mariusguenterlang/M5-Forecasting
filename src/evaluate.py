import joblib
import pandas as pd
from pathlib import Path

def evaluate(pipeline = joblib.load("models/lgb_model.pkl")):
    
    data = pd.read_csv(Path.cwd().parent / "archive" / "unlabeled_data.csv")
    data = data.drop(columns=["target"], errors="ignore")

    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        data[col] = data[col].astype('category')

    # predictions
    preds = pipeline.predict(data)
    proba = pipeline.predict_proba(data)[:, 1]

    # attach to dataset
    data["preds"] = preds
    data["proba"] = proba

    print(f'[Eval] Prediction-confidence range: {proba.min():.4f} - {proba.max():.4f}')

    # save predictions
    data.to_csv(Path.cwd().parent / "archive" / "predictions.csv", index=False)
    print('[Info] Predictions saved to archive/predictions.csv')

    
    return data

if __name__ == "__main__":
    evaluate()