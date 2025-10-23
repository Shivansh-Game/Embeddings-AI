import nltk
import re
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def trim(word):
    return re.sub(r'(.)\1+$', r'\1', word.lower())
