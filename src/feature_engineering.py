import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from utils import logging, CustomException
import sys

max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

try:
    # Load processed data
    processed_train_data = pd.read_csv(os.path.join("data", "processed", "train_data.csv"))
    processed_test_data = pd.read_csv(os.path.join("data", "processed", "test_data.csv"))
    logging.info("reading data from processed folder")

    X_train = processed_train_data['content'].values
    y_train = processed_train_data['sentiment'].values

    X_test = processed_test_data['content'].values
    y_test = processed_test_data['sentiment'].values

    logging.info("defining X_train, y_train, X_test, y_test")
    
    # Apply Bag of Words
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    logging.info("Applying BOW")

    # Convert sparse matrix to DataFrame
    X_train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
    X_train_bow_df['sentiment'] = y_train

    X_test_bow_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
    X_test_bow_df['sentiment'] = y_test

    # Save feature-engineered data
    os.makedirs("data/feature_engineered", exist_ok=True)
    X_train_bow_df.to_csv("data/feature_engineered/train_data.csv", index=False)
    X_test_bow_df.to_csv("data/feature_engineered/test_data.csv", index=False)

    logging.info("Saved feature-engineered data to 'data/feature_engineered/'")

except Exception as e:
    logging.info("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)