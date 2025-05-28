import os
import pandas as pd
import joblib
import xgboost as xgb
import yaml
from utils import logging, CustomException
import sys

learning_rate = yaml.safe_load(open('params.yaml','r'))['train_model']['learning_rate']
n_estimators = yaml.safe_load(open('params.yaml','r'))['train_model']['n_estimators']

try:
    # Load feature-engineered data (includes both X and y)
    train_df = pd.read_csv(os.path.join("data", "feature_engineered", "train_data.csv"))
    
    logging.info("reading data from feature_engineered folder")

    # Split features and labels
    X_train = train_df.drop(columns=['sentiment']).values
    y_train = train_df['sentiment'].values
    
    logging.info("Split features and labels")

    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', learning_rate=learning_rate, n_estimators = n_estimators)
    xgb_model.fit(X_train, y_train)
    
    logging.info("Train XGBoost model")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/xgb_model.joblib")

    logging.info("Model trained and saved to 'models/xgb_model.joblib'")
    
except Exception as e:
    logging.info("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)    