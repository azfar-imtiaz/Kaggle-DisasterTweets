import re
import spacy
from typing import List


class DataCleaner:
    def __init__(self, use_ner=False) -> None:
        self.ner_model = spacy.load('en_core_web_sm') if use_ner else None

    def remove_elements_from_text(self, text: str) -> str:
        # replace Twitter mentions with keyword
        text = re.sub(r'@\w+', 'MENTION', text)
        # replace hashtags with keywords
        text = re.sub(r'#\w+', 'HASHTAG', text)
        # replace links with keyword
        text = re.sub(r'https?:\/\/([\w\.\w]+\/[\w\.\w]*)+', 'LINK', text)

        if self.ner_model:
            # replace dates and times with corresponding keyword
            doc = self.ner_model(text)
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME']:
                    text = text[:ent.start_char] + ent.label_ + text[ent.end_char:]

        return text

    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    def clean_data(self, data: dict) -> List:
        # TODO: This can be improved; reduce 3 map calls to 1 if possible
        data['text'] = map(self.lowercase, data['text'])
        data['text'] = map(self.remove_elements_from_text, data['text'])
        data['text'] = map(self.remove_punctuation, data['text'])
        return data