# preprocessing.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

#function to help  tag the parts of the speech
def get_pos_tag(token):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV, "N": wordnet.NOUN}
    return tag_dict.get(tag, wordnet.NOUN)

def fast_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token, get_pos_tag(token)) for token in tokens]
    filtered = [word for word in lemmatized if word not in stop_words]
    return " ".join(filtered)

