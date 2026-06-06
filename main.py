import joblib
import pandas as pd
from pathlib import Path
from src.integration import run_integration
from src.features import feature_engineering
from src.train import train
from src.evaluate import evaluate

MODEL_PATH = Path.cwd() / "models" / "lgb_model.pkl"

def main(retrain=False):

    if not MODEL_PATH.exists() or retrain:
        print("Training model...")
        pipeline = run_integration()

        pipeline = feature_engineering(pipeline)
        pipeline = train(pipeline)

    else:
        print("Loading existing model...")
        pipeline = joblib.load(MODEL_PATH)

    evaluate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the credit card fraud detection pipeline.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model even if a saved model exists.")
    args = parser.parse_args()
    main(retrain=args.retrain)