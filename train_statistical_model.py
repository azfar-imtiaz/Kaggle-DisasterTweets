from pandas import DataFrame as df

from TextClassifier import TextClassifier
from FeatureExtractor import FeatureExtractor
from sklearn.metrics import classification_report


def train_model(X_train: df, y_train: df, X_val: df, y_val: df, split_classifiers_on_label: bool = False) -> None:
    feature_extractor = FeatureExtractor()
    print("Extracting features from training data...")
    features_train = feature_extractor.extract_features_from_text(X_train['text'], X_train['location'], is_train=True)

    model = TextClassifier(split_classifiers_on_label=split_classifiers_on_label)
    print("Training model...")
    model.train_model(features_train, y_train, list(X_train['keyword']))

    print("Extracting features from validation data...")
    features_val = feature_extractor.extract_features_from_text(X_val['text'], X_val['location'], is_train=False)
    print("Getting predictions on validation data...")
    predictions = model.get_predictions(features_val, keywords=list(X_val['keyword']))

    print(classification_report(y_val, predictions))
