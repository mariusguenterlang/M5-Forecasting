import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

def train(labeled_df):
    X = labeled_df.drop(columns=["target"], inplace=False)
    y = labeled_df["target"].astype(int)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
        )
    
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    bin_cols = [col for col in num_cols if X[col].nunique() == 2]
    num_cols = [col for col in num_cols if col not in bin_cols]

    # train model
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight='balanced'
    )

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col]  = X_test[col].astype('category')

    lgb_model.fit(X_train, y_train)

    # save model
    joblib.dump(lgb_model, Path.cwd() / "models" / "lgb_model.pkl")

    return labeled_df

#####

if __name__ == "__main__":
    from features import feature_engineering

    df = pd.read_csv(Path.cwd().parent / "archive" / "labeled_data.csv")

    df.drop(columns=["expires", "card_on_dark_web"], inplace=True)
    df[["is_retired", "is_night"]] = df[["is_retired", "is_night"]].astype(bool)

    labeled_df = df[df["target"].notna()]
    model = train(labeled_df)