import pandas as pd
from rank_bm25 import *
from nltk.stem import WordNetLemmatizer
import argparse

from src.ranking.prepare_data import prepare_data


def rank(text, csv_path):
    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv(csv_path)
    corpus = prepare_data(lemmatizer, df["Texts"].tolist())
    query = prepare_data(lemmatizer, text)[0]
    print("*" * 20)
    print(query)
    print("*" * 20)
    bm25 = BM25Okapi(corpus)
    df["scores"] = list(bm25.get_scores(query))
    df.sort_values(by=["scores"], axis=0, ascending=False, inplace=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="text for query")
    parser.add_argument("--csv_path", type=str, default=None, help="path to the csv")
    args = parser.parse_args()

    print(rank([args.text], args.csv_path))
