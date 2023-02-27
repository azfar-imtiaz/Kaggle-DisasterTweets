import numpy as np
from scipy.sparse import hstack
from typing import List
from sklearn.feature_extraction.text import CountVectorizer


class FeatureExtractor:
    def __init__(self) -> None:
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 3))

    def extract_features_from_text(self, ls_texts: List[str], ls_locations: List[int]) -> np.array:
        text_vectors: np.array = self.count_vectorizer.fit_transform(ls_texts)
        ls_locations: List = [[x] for x in ls_locations]
        combined_features: np.array = hstack([text_vectors, ls_locations]).toarray()
        return combined_features