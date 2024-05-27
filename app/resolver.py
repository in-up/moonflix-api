import pandas as pd
import random
import ast

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


def filter_movies(genre=None, year=None, sort_by_year=False, sort_by_rating=False):
    movies_df = pd.read_csv(item_fname)
    df = movies_df.copy()

    if genre:
        df['genres'] = df['genres'].apply(lambda x: x.split("|"))
        df = df[df['genres'].apply(lambda x: genre.lower() in [g.lower() for g in x])]

    if year:
        # year 컬럼이 문자열 타입인 경우 숫자형으로 변환
        if df['year'].dtype == 'object':
            df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # NaN 값을 0으로 대체 후 숫자형으로 변환
        df['year'] = df['year'].fillna(0).astype(int)
        df = df[df['year'] == int(year)]

    # 숫자형 컬럼의 NaN 값을 0으로, 문자열 컬럼의 NaN 값을 빈 문자열로 처리
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)  # 숫자형 컬럼 NaN 처리

    if sort_by_year:
        df = df.sort_values(by='year')
    elif sort_by_rating:
        df = df.sort_values(by='rating_avg', ascending=False)
    else:
        df = df.sort_values(by='genres')

    result_items = df.to_dict("records")
    return result_items