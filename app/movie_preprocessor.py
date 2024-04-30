from tqdm import tqdm
from dotenv import load_dotenv

import pandas as pd
import requests
import sys
import os

def add_url(row):
    return f"http://www.imdb.com/title/tt{row}/"


def add_rating(df):
    ratings_df = pd.read_csv('data/ratings.csv')
    ratings_df['movieId'] = ratings_df['movieId'].astype(str)
    agg_df = ratings_df.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_avg=('rating', 'mean')
    ).reset_index()

    rating_added_df = df.merge(agg_df, on='movieId')
    return rating_added_df


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    movies_df = pd.read_csv('data/movies.csv')
    movies_df['movieId'] = movies_df['movieId'].astype(str)
    links_df = pd.read_csv('data/links.csv', dtype=str)
    merged_df = movies_df.merge(links_df, on='movieId', how='left')
    merged_df['url'] = merged_df['imdbId'].apply(lambda x: add_url(x))
    result_df = add_rating(merged_df)

    result_df.to_csv("data/movies_sort.csv", index=None)
