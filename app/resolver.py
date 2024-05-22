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
    wr_df["year"] = wr_df["year"].astype(int)
    q = wr_df["year"].quantile(0.9)
    df = wr_df[wr_df["year"] > q].copy()
    df = df.sample(n=10).sort_values(by="year", ascending=False)

    movies_indexes = df.index
    result_items = movies_df.loc[movies_indexes].to_dict("records")

    return result_items


def random_genres_items(genre: str, wr_df=get_weighted_rating_df()):
    movies_df = pd.read_csv(item_fname)
    genre_df = wr_df[wr_df["genres"].str.contains(genre)]
    genre_df["year"] = genre_df["year"].astype(int)
    m = genre_df["year"].mean()
    df = genre_df[genre_df["year"] > m].copy()
    if len(df) <= 10:
        df = df.sort_values(by="year", ascending=False)
    else:
        df = df.sample(n=10).sort_values(by="year", ascending=False)

    movies_indexes = df.index
    result_items = movies_df.loc[movies_indexes].to_dict("records")

    return result_items