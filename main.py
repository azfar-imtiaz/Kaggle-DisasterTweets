import os
import csv
import joblib
import pandas as pd
from typing import List
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from DataCleaner import DataCleaner
from FeatureExtractor import FeatureExtractor
from TextClassifier import TextClassifier


def read_data(filename: str, is_train_file: bool=True) -> List[dict]:
    data = defaultdict(lambda: [])
    with open(filename, 'r', encoding='utf-8') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            data['keyword'].append(row[1])
            data['location'].append(row[2])
            data['text'].append(row[3])
            if is_train_file:
                data['target'].append(int(row[4]))
    return dict(data)


if __name__ == '__main__':
    train_filename = 'data/train.csv'
    if not os.path.exists('data/train_cleaned.pkl'):
        print("Loading training data from file...")
        train_data = read_data(filename=train_filename)
        
        data_cleaner = DataCleaner(use_ner=True)
        print("Cleaning data...")
        train_data = data_cleaner.clean_data(train_data)
        joblib.dump(train_data, 'data/train_cleaned.pkl')
        # with open('train_cleaned.pkl', 'wb') as wfile:
        #     pickle.dump(train_data, wfile)
    else:
        train_data = joblib.load('data/train_cleaned.pkl')
        # with open('data/train_cleaned.pkl', 'rb') as rfile:
        #     train_data = pickle.load(rfile)        
    
    target_data = train_data.pop('target')
    train_df = pd.DataFrame.from_dict(train_data)
    print("Performing train test split...")
    X_train, X_val, y_train, y_val = train_test_split(train_df, target_data, test_size=0.2, random_state=42)

    feature_extractor = FeatureExtractor()
    print("Extracting features from training data...")
    features_train = feature_extractor.extract_features_from_text(X_train['text'], 
                                                            X_train['location'], 
                                                            X_train['keyword'], 
                                                            is_train=True)
    
    model = TextClassifier()
    print("Training model...")
    model.train_model(features_train, y_train)
    
    print("Extracting features from validation data...")
    features_val = feature_extractor.extract_features_from_text(X_val['text'], X_val['location'], X_val['keyword'], is_train=False)
    print("Getting predictions on validation data...")
    predictions = model.get_predictions(features_val)

    print(classification_report(y_val, predictions))