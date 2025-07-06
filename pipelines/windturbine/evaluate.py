"""Evaluation script for measuring mean squared error."""
import json
import logging
import os
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    roc_auc_score,
    recall_score,
    precision_score)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions.round())
    recall = recall_score(y_test, predictions.round())
    precision = precision_score(y_test, predictions.round())

    logger.debug("Calculating mean squared error.")
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {
                "value": auc,
                "standard_deviation": "NaN"
            },
            "recall": {
                "value": recall,
                "standard_deviation": "NaN"
            },
            "precision": {
                "value": precision,
                "standard_deviation": "NaN"
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(name=output_dir, mode=755, exist_ok=True)
    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
