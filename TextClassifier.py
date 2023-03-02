import numpy as np
from typing import Union, List
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


def split_dataset_on_keywords(X: np.array, y: Union[np.array, None], keywords: List[str]) -> List:
    keyword_datasets = defaultdict(lambda: defaultdict(lambda: []))
    
    for index, keyword in enumerate(keywords):
        keyword_datasets[keyword]['X'].append(X[index])
        if y:
            keyword_datasets[keyword]['y'].append(y[index])
        
    # current_keyword = keywords[0]
    # current_kw_dataset_X = []
    # current_kw_dataset_y = []
    # for index, keyword in enumerate(keywords):
        # if keyword != current_keyword:
        #     if current_kw_dataset_X:
        #         keyword_datasets[current_keyword] = {
        #             'X': np.array(current_kw_dataset_X)
        #         }
        #         if y:
        #             keyword_datasets[current_keyword]['y'] = np.array(current_kw_dataset_y)
        #         current_kw_dataset_X = []
        #         current_kw_dataset_y = []
        #         current_keyword = keyword
        # current_kw_dataset_X.append(X[index])
        # if y:
        #     current_kw_dataset_y.append(y[index])
    
    # if current_kw_dataset_X:
    #     keyword_datasets[current_keyword] = {
    #         'X': np.array(current_kw_dataset_X)
    #     }
    #     if y:
    #         keyword_datasets[current_keyword]['y'] = np.array(current_kw_dataset_y)
        
    return keyword_datasets


class TextClassifier:
    def __init__(self, split_classifiers_on_label=False) -> None:
        self.multiple_clfs = split_classifiers_on_label
        self.model = None

    def train_model(self, X: np.array, y: np.array, keywords: List[str]=None) -> Union[List[RandomForestClassifier], RandomForestClassifier]:
        if not self.multiple_clfs:
            self.model = RandomForestClassifier(n_estimators=200)
            self.model.fit(X, y)
        else:
            keyword_datasets = split_dataset_on_keywords(X, y, keywords)
            self.model = {}
            for keyword, kw_dataset in keyword_datasets.items():
                kw_dataset_X = kw_dataset['X']
                kw_dataset_y = kw_dataset['y']
                print(f"\t- Training classifier for {keyword}...")
                print("\t\t- Length of training instances: {}".format(len(kw_dataset_X)))
                print("\t\t- Length of target instances: {}".format(len(kw_dataset_y)))
                clf = RandomForestClassifier(n_estimators=50)
                clf.fit(kw_dataset_X, kw_dataset_y)
                self.model[keyword] = clf


    def get_predictions(self, X_test: np.array, keywords: List[str]=None) -> np.array:
        if not self.multiple_clfs:
            return self.model.predict(X_test)
        else:
            keyword_datasets = split_dataset_on_keywords(X_test, None, keywords)
            predictions = []
            for keyword, kw_dataset in keyword_datasets.items():
                kw_dataset_X = kw_dataset['X']
                print("\t\t- Length of validation instances: {}".format(len(kw_dataset_X)))
                # print(f"\t-Getting predictions from classifier for {keyword}...")
                predictions.extend(self.model[keyword].predict(kw_dataset_X))
            return predictions
