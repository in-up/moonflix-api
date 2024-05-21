import pandas as pd
import re

item_fname = 'data/movies_final.csv'


def get_weighted_rating_df():
    def weighted_rating(x):
        v = x["rating_count"]
        R = x["rating_avg"]
        return ((v/(v+m) * R) + (m/(m+v) * C))
    
    def get_year(x):
        title = x["title"][-6::]
        get = re.compile('\(([^)]+)')
        year = get.findall(title)
        return year[0]
    
    movies_df = pd.read_csv(item_fname)
    movies_df["rating_count"] = movies_df["rating_count"].astype(int)
    rating_avg = movies_df["rating_avg"].astype(int)
    C = rating_avg.mean()
    m = movies_df["rating_count"].quantile(0.95)

    wr_df = movies_df[(movies_df["rating_count"] >= m)].copy()
    wr_df["weighted_rating"] = wr_df.apply(weighted_rating, axis=1)
    wr_df["year"] = wr_df.apply(get_year, axis=1)
    return wr_df


def random_items(wr_df=get_weighted_rating_df()):
    movies_df = pd.read_csv(item_fname)
    wr_df = wr_df.sort_values(by="year", ascending=False)[:10]

    movies_indexes = wr_df.index
    result_items = movies_df.loc[movies_indexes].to_dict("records")

    return result_items


def random_genres_items(genre: str, wr_df=get_weighted_rating_df()):
    movies_df = pd.read_csv(item_fname)
    genre_df = wr_df[wr_df["genres"].str.contains(genre)]
    genre_df = genre_df.sort_values(by="year", ascending=False)[:10]

    movies_indexes = genre_df.index
    result_items = movies_df.loc[movies_indexes].to_dict("records")

    return result_items