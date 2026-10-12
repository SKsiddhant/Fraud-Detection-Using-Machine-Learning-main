"""Train the fraud pipeline from the Kaggle CSV and save it.

Usage: python train_model.py ["AIML Dataset.csv"] [fraud_detection_pipeline.pkl]
"""
import sys

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from fraud_model import FEATURES, TARGET, build_pipeline, split


def main(csv_path="AIML Dataset.csv", out_path="fraud_detection_pipeline.pkl"):
    df = pd.read_csv(csv_path, usecols=FEATURES + [TARGET])
    x_train, x_test, y_train, y_test = split(df)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1]), 4))

    joblib.dump(pipeline, out_path)
    print(f"saved {out_path} (scikit-learn {sklearn.__version__}; load it with the same version)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
