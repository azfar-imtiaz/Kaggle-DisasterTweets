import re
import spacy
from typing import List, Union


class DataCleaner:
    def __init__(self, use_ner=False) -> None:
        self.ner_model = spacy.load('en_core_web_sm') if use_ner else None

    def validate_location(self, location: str) -> Union[int, str]:
        if not self.ner_model:
            return location
        doc = self.ner_model(location)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                return 1
        return 0

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
                    # text = text[:ent.start_char] + ent.label_ + text[ent.end_char:]
                    text = text.replace(ent.text, ent.label_)

        return text

    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_numbers(text: str) -> str:
        return re.sub(r'\d+', 'NUMBER', text)
    
    def apply_preprocessing_to_text(self, text: str) -> str:
        text = self.lowercase(text)
        text = self.remove_elements_from_text(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        return text

    def clean_data(self, data: dict) -> List:
        data['text'] = map(self.apply_preprocessing_to_text, data['text'])
        data['location'] = map(self.validate_location, data['location'])
        return data
