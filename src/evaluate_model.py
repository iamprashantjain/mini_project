import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, 
    precision_score, recall_score, roc_auc_score
)
from utils import logging, CustomException
import sys
import mlflow
import dagshub

# Initialize Dagshub tracking
dagshub.init(repo_owner='iamprashantjain', repo_name='mini_project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/mini_project.mlflow")


try:
    # Load test data
    test_df = pd.read_csv(os.path.join("data", "feature_engineered", "test_data.csv"))
    X_test = test_df.drop(columns=["sentiment"]).values
    y_test = test_df["sentiment"].values
    logging.info("Loaded test data")

    # Load trained model
    model = joblib.load("models/xgb_model.joblib")
    logging.info("Loaded trained model")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    logging.info("Made predictions")

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    logging.info("Evaluation metrics computed")

    # Save metrics to JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(auc, 4)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved to 'metrics/metrics.json'")

except Exception as e:
    logging.info("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)

# Registering model to MLflow and logging experiment info
def main():
    mlflow.set_experiment("dvc-pipeline")
    
    with mlflow.start_run():
        try:
            # Log metrics
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)

            # Log model parameters if available
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_val in params.items():
                    mlflow.log_param(param_name, param_val)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log metrics file as artifact
            mlflow.log_artifact("metrics/metrics.json")

            logging.info("Model and metrics logged to MLflow")

        except Exception as e:
            logging.info("Exception occurred during MLflow logging")
            raise CustomException(e, sys)

