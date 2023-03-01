import numpy as np
from scipy.sparse import hstack
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


class FeatureExtractor:
    def __init__(self) -> None:
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
        # self.label_encoder = LabelEncoder()

    def extract_features_from_text(self, ls_texts: List[str], ls_locations: List[int], is_train=True) -> np.array:
        if is_train:
            text_vectors: np.array = self.count_vectorizer.fit_transform(ls_texts)
            # keyword_labels: np.array = self.label_encoder.fit_transform(ls_keywords)
        else:
            text_vectors: np.array = self.count_vectorizer.transform(ls_texts)
            # keyword_labels: np.array = self.label_encoder.fit_transform(ls_keywords)
        ls_locations: List = [[x] for x in ls_locations]
        
        # This feature messes things up, because it doesn't make any sense
        # keyword_labels = [[x] for x in keyword_labels.tolist()]
        
        combined_features: np.array = hstack([text_vectors, ls_locations]).toarray()
        return combined_features