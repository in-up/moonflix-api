import pandas as pd
import random

item_fname = 'data/movies_final.csv'


def random_items():

    def weighted_rating(x):
        v = x["rating_count"]
        R = x["rating_avg"]
        return ((v/(v+m) * R) + (m/(m+v) * C))
    
    movies_df = pd.read_csv(item_fname)
    movies_df["rating_count"] = movies_df["rating_count"].astype(int)
    rating_avg = movies_df["rating_avg"].astype(int)
    C = rating_avg.mean()
    m = movies_df["rating_count"].quantile(0.95)

    wr_df = movies_df[(movies_df["rating_count"] >= m)]
    wr_df["weighted_rating"] = wr_df.apply(weighted_rating, axis=1)
    wr_df = wr_df.sort_values(by="weighted_rating", ascending=False)[:250]

    random_movies = wr_df.sample(n=10)
    movies_indexes = random_movies.index
    result_items = movies_df.loc[movies_indexes].to_dict("records")

    return result_items


def random_genres_items(genre: str):
    movies_df = pd.read_csv(item_fname)
    movies_df["rating_count"] = movies_df["rating_count"].astype(int)

    tmp1, tmp2, min, max, df_len = 0, 0, 0, 0, 0
    while((tmp1 == tmp2) | df_len < 50):
        for i in range(0, 2):
            random_number = random.randint(10, 329)
            if (i == 0): tmp1 = random_number
            if (i == 1): tmp2 = random_number
        if (tmp1 > tmp2):
            max = tmp1
            min = tmp2
        else:
            min = tmp1
            max = tmp2
        genre_df = movies_df[movies_df["genres"].str.contains(genre)]
        random_df = genre_df[(genre_df["rating_count"] > min) & (genre_df["rating_count"] < max)]
        df_len = len(random_df)

    result_items = random_df.sort_values(by="rating_avg", ascending=False)[:10].to_dict("records")

    return result_items