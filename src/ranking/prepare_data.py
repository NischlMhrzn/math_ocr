import pandas as pd
from src.ranking.preprocessing import (
    special_chars_removal,
    stopwords_removal_gensim_and_lemmatize,
)


def prepare_data(lemmatizer, data):
    return stopwords_removal_gensim_and_lemmatize(
        lemmatizer, special_chars_removal(data)
    )
