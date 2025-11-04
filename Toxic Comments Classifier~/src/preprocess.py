import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

stop_words = None
lemmatizer = None

def download_nltk_data():
    """Downloads necessary NLTK datasets if they aren't found."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

def initialize_nltk_components():
    """
    Initializes the NLTK components (stopwords, lemmatizer)
    This is called by clean_text() to ensure they are ready.
    """
    global stop_words, lemmatizer
    
    if stop_words is None:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
    
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Applies basic text cleaning steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove stopwords
    5. Lemmatize
    """

    initialize_nltk_components()
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

if __name__ == "__main__":
    raw_text = "This is a STUPID, 123 hateful comment!! But this one is nice."
    cleaned = clean_text(raw_text)
    print(f"Original: {raw_text}")
    print(f"Cleaned:  {cleaned}")

