import re
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
import nltk

# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('punkt')


def special_chars_removal(lst):
    lst1 = list()
    for element in lst:
        str = ""
        str = re.sub(r"[^\w\s]", "", element)
        lst1.append(str)
    return lst1


def lemmatize(lemmatizer, word):
    return lemmatizer.lemmatize(word)


def stopwords_removal_gensim_and_lemmatize(lemmatizer, lst):
    lst1 = list()
    for str in lst:
        text_tokens = word_tokenize(str)
        tokens_without_sw = [
            lemmatize(lemmatizer, word.lower())
            for word in text_tokens
            if ((not word.lower() in STOPWORDS) and len(word) > 1)
        ]
        #         str_t = ' '.join(tokens_without_sw)
        lst1.append(tokens_without_sw)

    return lst1
