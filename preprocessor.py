import re
import arabic_reshaper
import seaborn as sns
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from nltk.tokenize import word_tokenize
from config import ARABIC_STOPWORDS

class TextPreprocessor:
    """Handles Arabic text preprocessing"""
    
    def __init__(self, normalize_unicode: bool = True):
        self.normalize_unicode = normalize_unicode
        
    def full_preprocess(self, text: str) -> str:
        
        """Complete text preprocessing pipeline"""
        if not isinstance(text, str):
            return ''
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Arabic
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = self._normalize_arabic(text)
        return text.strip()

    def tokenize(self, text: str) -> list:
        """Tokenize text with bidirectional support"""
        if self.normalize_unicode:
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            return word_tokenize(bidi_text)
        return word_tokenize(text)

    @staticmethod
    def remove_stopwords(tokens: list) -> list:
        """Remove Arabic stopwords"""
        return [token for token in tokens if token not in ARABIC_STOPWORDS]

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        """Normalize Arabic characters"""
        replacements = {
            '[٩٨٧٦٥٤٣٢١٠]': '',
            '[ٱإأآاﺂ]': 'ا',
            '_': ' '
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text