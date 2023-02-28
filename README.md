# Kaggle-DisasterTweets

Link to challenge: https://www.kaggle.com/competitions/nlp-getting-started/

## Columns in dataset

| Column name | Column type |
| --- | --- |
| Text | Input, str |
| Location | Input, str [optional] |
| Keyword | Input, str [optional] |
| Target | Output, int [binary] |

## Approaches

### Data Cleaning

- Standardize or remove certain elements from text
    - Mentions (@mark)
    - Links
    - Hashtags (#thisIsAHashtag)
- Identify dates in text, replace them with a constant like DATE
    - [https://stackoverflow.com/questions/19994396/best-way-to-identify-and-extract-dates-from-text-python](https://stackoverflow.com/questions/19994396/best-way-to-identify-and-extract-dates-from-text-python)
    - Can use Spacy NER model for identifying dates and times too
    - Can use Spacy NER for other entities too!
- Replace numbers with a constant like NUMBER
- Remove punctuation marks
- Lowercase text
- Stemming/lemmatization?

### Feature Extraction

- The location can be turned into a binary/ternary feature, by validating it
    - if valid location → 1
    - if invalid location → 0
    - if no location → 0/-1
- Create a list of disaster related words. Create a binary feature that is
    - 1 if any disaster word occurs in the tweet
    - 0 otherwise
    - This might not be so helpful, since all tweets with a keyword usually/always have that disaster keyword in them…
- CountVectorizers/TF-IDF
- For tweets with keywords, use a pre-trained Word Sense Disambiguation model to identify sense of the keyword in that tweet. Then use this as an extra feature
    - https://github.com/BPYap/BERT-WSD
- Sentiment analysis of tweets as a feature?

### Data Classification

- Train SVC or RandomForest classifier
- Train a list/ensemble of classifiers,
    - one for each keyword, and
    - one for the tweets for no keyword.
    - For evaluation, either only use the corresponding classifier for the keyword in the tweet, or pass the text example through a sub-ensemble of classifiers containing only a/the general purpose classifier and the classifier corresponding to the keyword (with maybe more weight assigned to the keyword classifier)
- Train LSTM/RNN model for this using PyTorch
    - Try Torchtext!
- Try fine-tuning pre-trained BERT model for binary sentence classification on this dataset.