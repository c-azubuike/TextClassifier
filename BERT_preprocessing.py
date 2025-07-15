import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

#function to help  tag the parts of the speech
def get_pos_tag(token):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
        "N": wordnet.NOUN
    }
    return tag_dict.get(tag, wordnet.NOUN)

# main preprocessing function
def fast_preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and digits
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token, get_pos_tag(token)) for token in tokens]
    filtered = [word for word in lemmatized if word not in stop_words]
    return " ".join(filtered)
