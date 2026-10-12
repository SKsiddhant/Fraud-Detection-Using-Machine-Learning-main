"""Training and prediction helpers shared by train_model.py and the Streamlit app."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["type"]
NUMERIC = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "isFraud"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            # handle_unknown="ignore": an unseen transaction type no longer crashes prediction
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def split(df, test_size=0.3, random_state=42):
    """Stratified split with a fixed seed so results are reproducible."""
    x = df[FEATURES]
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)


def known_types(pipeline):
    """Transaction types the fitted model was trained on (read from the model, not hard-coded)."""
    encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
    return [str(c) for c in encoder.categories_[0]]


def predict_one(pipeline, transaction_type, amount, old_org, new_org, old_dest, new_dest):
    row = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }])
    return int(pipeline.predict(row)[0])
