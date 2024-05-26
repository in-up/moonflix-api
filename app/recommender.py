import pandas as pd
import numpy as np
import os
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
from supabase import create_client, Client
import pickle

saved_model_fname = "model/finalized_model.sav"
data_fname = "data/ratings.csv"
item_fname = "data/movies_final.csv"
weight = 10
load_dotenv()
url = os.getenv("supabase_url")
key = os.getenv("supabase_key")
supabase = create_client(url, key)

def model_train():
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")

    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),
            (
                ratings_df["movieId"].cat.codes.copy(),
                ratings_df["userId"].cat.codes.copy(),
            ),
        )
    )

    als_model = AlternatingLeastSquares(
        factors=50, regularization=0.01, dtype=np.float64, iterations=50
    )

    als_model.fit(weight * rating_matrix)

    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model

def calculate_item_based(item, movies_df, index):
    movies_df[item] = movies_df[item].fillna('')
    t = TfidfVectorizer(stop_words="english")
    tfidf_matrix = t.fit_transform(movies_df[item])
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similar_list = list(enumerate(cosine_matrix[int(index)]))
    return similar_list

def item_based_recommendation(id):
    movies_df = pd.read_csv(item_fname)
    title_to_index = movies_df[movies_df["tmdbId"] == id].index.values
    idx = title_to_index[0]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_overview = executor.submit(calculate_item_based, "overview", movies_df, idx)
        future_genres = executor.submit(calculate_item_based, "genres", movies_df, idx)
        future_title = executor.submit(calculate_item_based, "title", movies_df, idx)

        overview_similar_movies = future_overview.result()
        genres_similar_movies = future_genres.result()
        title_similar_movies = future_title.result()

    similar_movies = [(x1, y1 + y2 + (0.5 * y3)) for (x1, y1), (x2, y2), (x3, y3) in zip(overview_similar_movies, genres_similar_movies, title_similar_movies)]
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]
    
    movies_indexes = [index[0] for index in similar_movies]
    result_items = movies_df.iloc[movies_indexes].to_dict("records")

    return result_items

def get_rating_df():
    response = supabase.table("useritem").select("*").execute()
    data = response.data
    df = pd.DataFrame(data)
    return df

def map_tmdbId_to_movieId(df, mapping_df):
    df = df.rename(columns={"user_id": "userId", "movie_id": "tmdbId"})
    df = pd.merge(df, mapping_df, on="tmdbId", how="inner")
    df = df[["userId", "movieId", "rating"]]
    return df

# Load the movie mapping data
movies_df = pd.read_csv("data/movies_final.csv")
# Create a mapping DataFrame
movie_id_mapping = movies_df[['movieId', 'tmdbId']].drop_duplicates()

def build_rating_matrix():
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype(str)
    ratings_df["movieId"] = ratings_df["movieId"].astype(str)
    
    df = get_rating_df()
    mapped_df = map_tmdbId_to_movieId(df, movie_id_mapping)
    
    merged_df = pd.concat([ratings_df, mapped_df], ignore_index=True)
    rating_matrix = merged_df.pivot_table(index="userId", columns="movieId", values="rating")
    rating_matrix = rating_matrix.fillna(0)

    rating_matrix.columns = rating_matrix.columns.astype(str)

    return rating_matrix

def calculate_user_based(auth_userId, matrix=build_rating_matrix(), n_neighbors=5):
    auth_user_idx = matrix.index.get_loc(auth_userId)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(matrix)
    
    distances, indices = model_knn.kneighbors(matrix.iloc[auth_user_idx, :].values.reshape(1, -1), n_neighbors=n_neighbors + 1)
    similar_users = [matrix.index[i] for i in indices.flatten()]
    
    return similar_users[1:]

def user_based_recommendation(auth_userId, matrix=build_rating_matrix()):
    movies_df = pd.read_csv(item_fname)
    similar_users = calculate_user_based(auth_userId)
    similar_users = matrix[matrix.index.isin(similar_users)]
    similar_users = similar_users.mean(axis=0)
    similar_users_df = pd.DataFrame(similar_users, columns=["user_similarity"])
    
    user_df = matrix[matrix.index == auth_userId]
    user_df = user_df.transpose()
    user_df.columns = ["rating"]

    unseen_df = user_df[user_df["rating"] == 0]
    unseen_list = unseen_df.index.tolist()

    filter_unseen_movies_df = similar_users_df[similar_users_df.index.isin(unseen_list)]
    sorted_filter_unseen_movies_df = filter_unseen_movies_df.sort_values(by="user_similarity", ascending=False)
    top = sorted_filter_unseen_movies_df.head(10)
    index = top.index.tolist()

    result = movies_df[movies_df["movieId"].astype(str).isin(index)].to_dict("records")

    return result

async def _search_movies_r(query):
    if not query:
        return []

    movies_df = pd.read_csv(item_fname)
    movies_df['title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    search_terms = query.lower().split()

    full_match_mask = movies_df['title'].str.lower() == query.lower()
    full_match_df = movies_df[full_match_mask]

    if full_match_df.empty:
        mask = movies_df['title'].str.lower().str.contains('|'.join(search_terms), regex=False)
        filtered_df = movies_df[mask]
    else:
        filtered_df = full_match_df
    result_df = pd.concat([filtered_df, full_match_df])
    result_df = result_df.drop_duplicates(subset=['movieId'])

    if not result_df.empty:
        search_genres = result_df['genres'].tolist()[0].split('|')  

        series_titles = []
        for title in result_df['title']: 
            base_title = title.split(':')[0].strip()  
            series_mask = movies_df['title'].str.startswith(base_title)
            series_titles.extend(movies_df[series_mask]['title'].tolist())
            
        # 유사 장르 영화 추천 (Top 5)
        genre_similarity = []
        for idx, row in movies_df.iterrows():
            if row['title'] not in series_titles and row['title'] not in result_df['title'].tolist():
                movie_genres = row['genres'].split('|')
                common_genres = len(set(search_genres) & set(movie_genres))  # 장르 교집합 개수 계산
                genre_similarity.append((idx, common_genres))

        genre_similarity.sort(key=lambda x: x[1], reverse=True)  # 장르 유사도 기준 내림차순 정렬
        top_similar_indices = [idx for idx, count in genre_similarity[:5]]  # 상위 5개 영화 인덱스 추출

        recommended_df = movies_df.iloc[top_similar_indices].sort_values(by='rating_count', ascending=False)  # rating_count 기준 내림차순 

        # 결과 합치기 (중복 제거 포함)
        result_df = pd.concat([result_df, 
                               movies_df[movies_df['title'].isin(series_titles)], 
                               recommended_df])
        result_df = result_df.drop_duplicates(subset=['movieId'])
        result_items = result_df.to_dict('records')
        
        print(result_items)

    return result_items

import pandas as pd
import numpy as np

def _search_movies(query):
    if not query:
        return []

    movies_df = pd.read_csv(item_fname)
    movies_df['title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    search_terms = query.lower().split()

    full_match_mask = movies_df['title'].str.lower() == query.lower()
    full_match_df = movies_df[full_match_mask]

    if full_match_df.empty:
        mask = movies_df['title'].str.lower().str.contains('|'.join(search_terms), regex=False)
        filtered_df = movies_df[mask]
    else:
        filtered_df = full_match_df
    result_df = pd.concat([filtered_df, full_match_df])
    result_df = result_df.drop_duplicates(subset=['movieId'])
    
    if not result_df.empty: 
        genre_mask = movies_df['genres'].str.contains('|'.join(result_df['genres'].tolist()[0].split('|')))
        
        series_titles = []
        for title in result_df['title']: 
            base_title = title.split(':')[0].strip()
            series_mask = movies_df['title'].str.startswith(base_title)
            series_titles.extend(movies_df[series_mask]['title'].tolist())
            
            # 결과 합치기 (중복 제거 포함)
        result_df = pd.concat([result_df, 
                               movies_df[movies_df['title'].isin(series_titles)], 
                               ])
        result_df = result_df.drop_duplicates(subset=['movieId'])
    
    # NaN 및 Inf 값을 None으로 대체
    result_df = result_df.replace({np.nan: None, np.inf: None, -np.inf: None})

    result_items = result_df.to_dict('records')
    
    return result_items


if __name__ == "__main__":
    model = model_train()