import csv
import pandas as pd
from typing import List
from collections import defaultdict
from sklearn.model_selection import train_test_split

from DataCleaner import DataCleaner
from FeatureExtractor import FeatureExtractor


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
    return data


if __name__ == '__main__':
    train_filename = 'data/train.csv'
    train_data = read_data(filename=train_filename)
    
    data_cleaner = DataCleaner(use_ner=True)
    train_data = data_cleaner.clean_data(train_data)

    target_data = train_data.pop('target')
    train_df = pd.DataFrame.from_dict(train_data)
    X_train, X_val, y_train, y_val = train_test_split(train_df, target_data, test_size=0.2, random_state=42)

    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features_from_text(X_train['text'], 
                                                            X_train['location'], 
                                                            X_train['keyword'], 
                                                            is_train=True)
    