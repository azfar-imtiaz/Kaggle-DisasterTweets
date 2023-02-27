import csv
from typing import List
from collections import defaultdict

from DataCleaner import DataCleaner


def read_data(filename: str, is_train_file: bool=True) -> List[dict]:
    data = defaultdict(lambda: [])
    with open(filename, 'r') as rfile:
        # field_names = ['id', 'keyword', 'location', 'text']
        # if is_train_file:
        #     field_names.append('target')
        # reader = csv.DictReader(rfile, fieldnames=field_names)
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
    