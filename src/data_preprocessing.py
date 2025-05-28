import os
import numpy as np
import pandas as pd
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from utils import logging, CustomException
import sys

nltk.download('stopwords')
nltk.download('wordnet')

try:
    train_data = pd.read_csv(os.path.join("data", "raw", "train_data.csv"))
    test_data = pd.read_csv(os.path.join("data", "raw", "test_data.csv"))
    logging.info("reading train & test data from raw folder")

except Exception as e:
    logging.exception("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)

# Text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    return re.sub('\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_small_sentences(df):
    df['content'] = df['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
    return df

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        df = remove_small_sentences(df)
        return df.dropna(subset=['content'])
    except Exception as e:
        logging.exception("Exception occurred during evaluation script execution")
        raise CustomException(e, sys)
        return df

try:
    # Apply normalization
    processed_train_data = normalize_text(train_data)
    processed_test_data = normalize_text(test_data)

    # Save processed data
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    processed_train_path = os.path.join("data", "processed", "train_data.csv")
    processed_test_path = os.path.join("data", "processed", "test_data.csv")

    processed_train_data.to_csv(processed_train_path, index=False)
    processed_test_data.to_csv(processed_test_path, index=False)

    logging.info("Saved processed_train and processed_test data to 'data/processed/'")

except Exception as e:
    logging.info("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)